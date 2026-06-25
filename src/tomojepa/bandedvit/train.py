"""BandedViT self-supervised pre-training (SIGReg / SimMIM).

Examples:
    tomojepa train-bandedvit --data_dir /data --schedule configs/bandedvit/petiole_sigreg_224.yaml
    torchrun --nproc_per_node=4 -m tomojepa.bandedvit.train --data_dir /data ...
"""
from __future__ import annotations

import gc
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import tqdm
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

from tomojepa.bandedvit.config import BandedJEPAConfig, add_argparse_args, from_args
from tomojepa.bandedvit.bvit import weighted_band_counts
from tomojepa.bandedvit.model import BandedJEPA
from tomojepa.bandedvit.pca_viz import build_probe, run_pca_strip
from tomojepa.core import dist as D
from tomojepa.core.dataset import TomographyDataset
from tomojepa.swinjepa.train import (
    build_param_groups,
    maybe_resume,
    print_config_log_block,
    resolve_job_config,
    save_ckpt,
)


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="BandedViT SIGReg / SimMIM pre-training")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pattern", default="*.zarr")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="auto")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--augment", choices=["tomo", "tomo2"], default="tomo2")
    p.add_argument("--crop_mode", choices=["resized", "native", "resize", "crop_down"],
                   default="resize")
    p.add_argument("--random_rotate_deg", type=float, default=180.0)
    p.add_argument("--resize_jitter", type=float, nargs=2, default=(0.9, 1.1))
    p.add_argument("--global_views", type=int, default=1)
    p.add_argument("--local_views", type=int, default=0)
    p.add_argument("--global_scale", type=float, nargs=2, default=(0.4, 1.0))
    p.add_argument("--local_scale", type=float, nargs=2, default=(0.1, 0.4))
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--ckpt_dir", default="checkpoints_bandedvit")
    p.add_argument("--out_dir", default="outputs_bandedvit")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--pca_every", type=int, default=100)
    p.add_argument("--pca_samples", type=int, default=1)
    p.add_argument("--pca_terms", type=int, default=9,
                   help="Number of grayscale PCA band panels (plus RGB composite)")
    p.add_argument("--pca_max_tokens", type=int, default=4096,
                   help="Max patch tokens for PCA fit (subsample if larger)")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="BandedViT_Tomo")
    p.add_argument("--schedule", default="")
    p.add_argument("--aug_config", default="")
    add_argparse_args(p)
    return p.parse_args()


def _yaml_safe(value):
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def apply_model_yaml_overrides(cfg: BandedJEPAConfig, path: str) -> BandedJEPAConfig:
    """Merge optional ``model:`` block from a job YAML into the dataclass."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    model_block = data.get("model")
    if not model_block:
        return cfg
    kwargs = asdict(cfg)
    for key, val in model_block.items():
        if key not in kwargs:
            raise ValueError(f"unknown model YAML key {key!r}")
        if key in ("beta_sig", "sigreg_blocks", "block_scale_range", "band_weights") and isinstance(val, list):
            val = tuple(val)
        kwargs[key] = val
    return BandedJEPAConfig(**kwargs)


def main():
    args = parse_args()
    schedule, aug_cfg, aug_sched, job_meta = resolve_job_config(args)
    cfg = from_args(args)
    if args.schedule:
        cfg = apply_model_yaml_overrides(cfg, args.schedule)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cfg.pred_enabled and aug_cfg.global_views < 2:
        raise ValueError(
            f"pred_enabled requires global_views >= 2, got {aug_cfg.global_views}"
        )

    device, _ = D.init_distributed()
    ws = D.world_size()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if D.is_main():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
        print_config_log_block(args, cfg, schedule, aug_cfg, aug_sched, job_meta)

    log_wandb = args.wandb and D.is_main()
    if log_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config={**vars(args), **asdict(cfg)})

    shared_aug = aug_sched is not None and args.num_workers > 0
    train_ds = TomographyDataset(
        data_dir=args.data_dir,
        dataset_key=args.dataset_key,
        pattern=args.pattern,
        img_size=cfg.img_size,
        is_train=True,
        backend=args.backend,
        aug_config=aug_cfg,
        aug_schedule=aug_sched,
        shared_aug_state=shared_aug,
        foreground_mask=cfg.foreground_mask,
        fg_mode=cfg.fg_mode,
        fg_std_thresh=cfg.fg_std_thresh,
        fg_circle_diameter_frac=cfg.fg_circle_diameter_frac,
        fg_key=cfg.fg_key or None,
    )
    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if D.is_distributed() else None
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        sampler=sampler,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
        if args.persistent_workers:
            loader_kwargs["persistent_workers"] = True
    loader = DataLoader(train_ds, **loader_kwargs)
    steps_per_epoch = len(loader)
    train_ds.update_augmentations(0, steps_per_epoch * cfg.epochs, steps_per_epoch)

    if D.is_main():
        ddp = f" x {ws} GPUs" if D.is_distributed() else ""
        print(
            f"Dataset: {len(train_ds)} slices -> {steps_per_epoch} steps/epoch on {device}{ddp}; "
            f"aug: {aug_cfg.summary_line()}",
            flush=True,
        )

    model = BandedJEPA(cfg).to(device)
    if D.is_main():
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        band_mix = ""
        if cfg.band_weights:
            counts = weighted_band_counts(cfg.depth, cfg.band_weights)
            band_mix = (
                f"; band_mix near/mid/far={counts}"
                f" weights={list(cfg.band_weights)} mode={cfg.band_sample_mode}"
            )
        band_sched = ""
        if cfg.band_m0 > 0 and cfg.band_m1 > 0:
            band_sched = f"; band_schedule off={cfg.band_m0} on={cfg.band_m1}"
        sig_extra = ""
        if cfg.sigreg_queue_len > 0 or cfg.sigreg_min_token_dist > 0:
            sig_extra = (
                f"; sigreg_queue={cfg.sigreg_queue_len}"
                f" sigreg_min_dist={cfg.sigreg_min_token_dist}"
            )
        print(
            f"BandedViT depth={cfg.depth} embed={cfg.embed_dim} "
            f"pred_enabled={cfg.pred_enabled} band_K={cfg.band_K}{band_mix}{band_sched}{sig_extra}; "
            f"sigreg_blocks={model.sigreg_blocks}; trainable {n_train:.2f}M params"
            f"{('; foreground_mask fg_mode=' + cfg.fg_mode
               + (f' fg_circle_diameter_frac={cfg.fg_circle_diameter_frac}'
                  if cfg.fg_mode == 'circle' else '')
               if cfg.foreground_mask else '')}",
            flush=True,
        )

    opt = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_iters > 0:
        total_steps = min(total_steps, cfg.max_iters)
    warmup_steps = max(1, int(cfg.warmup_pct * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp_dtype]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"

    start_epoch, global_step = maybe_resume(args.ckpt_dir, model, opt, sched, device)
    pca_probe = None
    pca_dir = os.path.join(args.out_dir, "pca")
    if D.is_main() and args.pca_every > 0:
        pca_probe = build_probe(
            args.data_dir, cfg.img_size, args.pattern, args.backend,
            args.dataset_key, aug_cfg=aug_cfg, seed=args.seed,
            n_samples=args.pca_samples,
            foreground_mask=cfg.foreground_mask,
            fg_mode=cfg.fg_mode,
            fg_std_thresh=cfg.fg_std_thresh,
            fg_circle_diameter_frac=cfg.fg_circle_diameter_frac,
            fg_key=cfg.fg_key,
        )
        print(
            f"PCA probe: {len(pca_probe[0])} slice(s), {args.pca_terms} bands + RGB -> {pca_dir}",
            flush=True,
        )

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(total=steps_per_epoch, desc=f"epoch {epoch}") if D.is_main() else None
        for vs, fg_vs in loader:
            train_ds.update_augmentations(global_step, total_steps, steps_per_epoch)
            if cfg.pred_enabled:
                x = vs.to(device, non_blocking=True)
            else:
                x = vs[:, 0].to(device, non_blocking=True) if vs.ndim == 5 else vs.to(device, non_blocking=True)
            fg = fg_vs.to(device, non_blocking=True) if cfg.foreground_mask else None
            opt.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = model.compute_loss(
                    x, fg_px=fg, step=global_step, total_steps=total_steps, epoch=epoch,
                )
            if loss.requires_grad:
                loss.backward()
                D.average_grads_(trainable)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
            sched.step()

            if D.is_main():
                pbar.set_postfix(
                    loss=f"{logs['total']:.3f}",
                    pred=f"{logs.get('l_pred', 0):.3f}",
                    sig=f"{logs.get('l_sig', 0):.2f}",
                    **({"fg": f"{logs['fg_cov']:.2f}"} if "fg_cov" in logs else {}),
                )
                if global_step % args.log_every == 0:
                    pbar.write(
                        f"[step {global_step}] total={logs['total']:.4f} "
                        f"pred={logs.get('l_pred', 0):.4f} sig={logs.get('l_sig', 0):.4f}",
                        file=sys.stderr,
                    )
                if log_wandb and global_step % args.log_every == 0:
                    wandb.log(
                        {f"train/{k}": v for k, v in logs.items()}
                        | {"train/lr": sched.get_last_lr()[0]},
                        step=global_step,
                    )
            if D.is_main() and args.save_every and global_step > 0 and global_step % args.save_every == 0:
                save_ckpt(args.ckpt_dir, "last", epoch, global_step, model, opt, sched)
                save_ckpt(args.ckpt_dir, f"step_{global_step:06d}", epoch, global_step, model, opt, sched)
            if D.is_main() and pca_probe is not None and args.pca_every > 0 and global_step % args.pca_every == 0:
                try:
                    out = run_pca_strip(
                        model, pca_probe, global_step, pca_dir, device,
                        n_terms=args.pca_terms, max_tokens=args.pca_max_tokens,
                        fg_coverage=cfg.fg_coverage,
                    )
                    if pbar is not None:
                        pbar.write(f"[pca] saved {out}", file=sys.stderr)
                except Exception as e:
                    model.train()
                    if pbar is not None:
                        pbar.write(f"[pca] skipped step {global_step}: {e}", file=sys.stderr)
            global_step += 1
            if pbar is not None:
                pbar.update(1)
            if cfg.max_iters > 0 and global_step >= cfg.max_iters:
                stop = True
                break
        if pbar is not None:
            pbar.close()
        if D.is_main():
            save_ckpt(args.ckpt_dir, f"epoch_{epoch}", epoch, global_step, model, opt, sched)
            save_ckpt(args.ckpt_dir, "last", epoch, global_step, model, opt, sched)
        if stop:
            break
        gc.collect()

    if log_wandb:
        wandb.finish()
    if D.is_main():
        print("done.", flush=True)
    D.cleanup()


if __name__ == "__main__":
    main()
