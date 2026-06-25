"""SwinSimMIM training — dual-aug masked latent SimMIM (separate from SwinMSJEPA)."""
import argparse
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
import tqdm

from ..core.dataset import TomographyDataset
from ..core import dist as D
from .simmim_config import SwinSimMIMConfig, add_argparse_args, from_args
from .simmim_model import SwinSimMIM
from .train import (
    build_param_groups,
    build_pca_probe,
    maybe_resume,
    print_config_log_block,
    resolve_job_config,
    run_pca_viz,
    save_ckpt,
)


def parse_args():
    p = argparse.ArgumentParser(description="SwinSimMIM dual-aug pre-training")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pattern", default="*.zarr")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="auto")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--crop_size", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--ckpt_dir", default="checkpoints_swin_simmmim")
    p.add_argument("--out_dir", default="outputs_swin_simmmim")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--pca_every", type=int, default=25)
    p.add_argument("--pca_samples", type=int, default=4)
    p.add_argument("--pca_max_tokens", type=int, default=4096)
    p.add_argument("--pca_es_every", type=int, default=0)
    p.add_argument("--pca_es_terms", type=int, default=9)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="SwinSimMIM_Tomo")
    p.add_argument("--schedule", default="",
                   help="YAML job config: stage schedule + augmentations")
    p.add_argument("--aug_config", default="")
    p.add_argument("--global_views", type=int, default=1)
    p.add_argument("--local_views", type=int, default=0)
    p.add_argument("--global_scale", type=float, nargs=2, default=(0.4, 1.0))
    p.add_argument("--local_scale", type=float, nargs=2, default=(0.1, 0.4))
    add_argparse_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    schedule, aug_cfg, aug_sched, job_meta = resolve_job_config(args)
    cfg = from_args(args)
    if aug_cfg.global_views < 2:
        raise ValueError(
            f"SwinSimMIM requires global_views >= 2 (dual aug), got {aug_cfg.global_views}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, _local_rank = D.init_distributed()
    ws = D.world_size()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if D.is_main():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
        print_config_log_block(args, cfg, schedule, aug_cfg, aug_sched, job_meta)

    crop_size = args.crop_size if args.crop_size > 0 else None
    shared_aug = aug_sched is not None and args.num_workers > 0
    train_ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        img_size=cfg.img_size, is_train=True, backend=args.backend,
        crop_size=crop_size,
        foreground_mask=cfg.foreground_mask, fg_mode=cfg.fg_mode,
        fg_std_thresh=cfg.fg_std_thresh,
        fg_circle_diameter_frac=cfg.fg_circle_diameter_frac,
        fg_key=cfg.fg_key or None,
        aug_config=aug_cfg, aug_schedule=aug_sched, shared_aug_state=shared_aug,
    )
    sampler = (DistributedSampler(train_ds, shuffle=True, drop_last=True)
               if D.is_distributed() else None)
    loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=(sampler is None),
                         drop_last=True, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"), sampler=sampler)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
        if args.persistent_workers:
            loader_kwargs["persistent_workers"] = True
    loader = DataLoader(train_ds, **loader_kwargs)
    steps_per_epoch = len(loader)
    train_ds.update_augmentations(0, steps_per_epoch * cfg.epochs, steps_per_epoch)

    model = SwinSimMIM(cfg).to(device)
    if args.schedule:
        model.set_schedule(schedule)
    model.set_steps_per_epoch(steps_per_epoch)

    if D.is_main():
        ddp = f" x {ws} GPUs" if D.is_distributed() else ""
        embed = (f"embed_dim={cfg.backbone_embed_dim} -> "
                 if cfg.backbone_embed_dim is not None else "")
        print(f"Dataset: {len(train_ds)} slices -> {steps_per_epoch} steps/epoch{ddp}; "
              f"augment: {aug_cfg.summary_line()}", flush=True)
        print(f"SwinSimMIM {cfg.backbone_name} {embed}"
              f"stage dims {model.out_chans} -> lat_dims {model.lat_chans}; "
              f"mask_ratio={cfg.mask_ratio} s4 grid {model.grids[-1]}; "
              f"beta_sig={list(cfg.beta_sig)}; queue={cfg.sigreg_queue_len}; "
              f"stage_base_weights={list(cfg.stage_base_weights)}", flush=True)
        if model.schedule is not None:
            for line in model.schedule.summary_lines():
                print(line, flush=True)

    opt = torch.optim.AdamW(build_param_groups(model, cfg.weight_decay),
                            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
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
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.amp_dtype]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"

    start_epoch, global_step = maybe_resume(args.ckpt_dir, model, opt, sched, device)
    if start_epoch > 0 and model.schedule is not None:
        model._sync_freeze_from_schedule(global_step, total_steps)
        trainable = [p for p in model.parameters() if p.requires_grad]

    pca_probe = None
    pca_dir = os.path.join(args.out_dir, "pca")
    pca_es_dir = os.path.join(args.out_dir, "pca_es")
    if D.is_main() and (args.pca_every > 0 or args.pca_es_every > 0):
        pca_probe = build_pca_probe(args, cfg, aug_cfg)
        if args.pca_every > 0:
            grids = model.grids
            print(f"PCA probe: slices {pca_probe[0]} -> {pca_dir} every "
                  f"{args.pca_every} steps ({', '.join(f'{h}x{w}' for h, w in grids)})",
                  flush=True)

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        if model.schedule is not None and model.schedule.progress_scope == "epoch":
            model.reset_schedule_epoch()
            trainable = [p for p in model.parameters() if p.requires_grad]
        elif model.schedule is None:
            newly = model.apply_freeze_schedule(epoch)
            if newly:
                trainable = [p for p in model.parameters() if p.requires_grad]
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(total=steps_per_epoch, desc=f"epoch {epoch}") if D.is_main() else None

        for vs, fg_vs in loader:
            train_ds.update_augmentations(global_step, total_steps, steps_per_epoch)
            x = vs.to(device, non_blocking=True)
            fg = fg_vs.to(device, non_blocking=True) if cfg.foreground_mask else None

            opt.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = model.compute_loss(
                    x, fg_px=fg, step=global_step, total_steps=total_steps, epoch=epoch)
            if logs.get("mae/cos") is not None:
                model.note_mae_cos(logs["mae/cos"])
            loss.backward()
            if model._last_newly_frozen:
                trainable = [p for p in model.parameters() if p.requires_grad]
                model._last_newly_frozen = []
            D.average_grads_(trainable)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            sched.step()

            if D.is_main():
                sk = model.stage_keys
                postfix = dict(loss=f"{logs['total']:.3f}",
                               mae=f"{logs['l_mae']:.3g}",
                               sig=f"{logs['l_sig']:.2f}",
                               er4=f"{logs['effrank/s4']:.1f}")
                if logs.get("mae/cos") is not None:
                    postfix["cos"] = f"{logs['mae/cos']:.3g}"
                if cfg.s4_cosine_level > 0 and "sigreg/cos_gate" in logs:
                    postfix["sig_gate"] = f"{logs['sigreg/cos_gate']:.3g}"
                if logs.get("sig/raw/s4") is not None:
                    postfix["sig_r"] = f"{logs['sig/raw/s4']:.2f}"
                pbar.set_postfix(**postfix)
                if global_step % args.log_every == 0:
                    sig_str = " ".join(f"{logs.get(f'sig/{k}', 0):.3g}" for k in sk)
                    pbar.write(
                        f"[step {global_step}] mae={logs['l_mae']:.3g} "
                        f"cos={logs.get('mae/cos', 0):.3g} sig [{sig_str}] "
                        f"effrank {logs.get('effrank/s4', 0):.1f}",
                        file=sys.stderr)

            if (D.is_main() and args.save_every and global_step > 0
                    and global_step % args.save_every == 0):
                save_ckpt(args.ckpt_dir, "last", epoch, global_step, model, opt, sched)

            do_strip = pca_probe is not None and args.pca_every > 0 and global_step % args.pca_every == 0
            if do_strip and D.is_main():
                try:
                    run_pca_viz(model, pca_probe, global_step, pca_dir, pca_es_dir,
                                args.pca_es_terms, args.pca_max_tokens,
                                do_strip=True, do_es=False, device=device,
                                pca_mode="inference")
                except Exception as e:
                    pbar.write(f"[pca] skipped step {global_step}: {e}", file=sys.stderr)

            global_step += 1
            if pbar is not None:
                pbar.update(1)
            if global_step >= total_steps:
                stop = True
                break
        if pbar is not None:
            pbar.close()
        if stop:
            break

    if D.is_main():
        save_ckpt(args.ckpt_dir, "last", epoch, global_step, model, opt, sched)
        print("done.", flush=True)


if __name__ == "__main__":
    main()
