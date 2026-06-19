"""Swin multi-scale latent-JEPA self-supervised pre-training.

Mirrors this repo's conventions (``ssl/train.py`` / ``vitup/train.py``):
single-process / single-GPU by default (multi-GPU opt-in via ``torchrun``),
``argparse`` config, AMP (bf16), AdamW + warmup/cosine, manual checkpoint
save/resume, and the shared ``TomographyDataset`` loader. The method consumes
images only (no labels) and trains the *same* backbone the downstream ViT-Up
upsampler will consume.

Collapse control is per-stage SIGReg only -- there is no EMA teacher and no
pixel decoder (asserted at startup).

Examples:
    # from-scratch pretrain on a zarr stack (Swin-T, 224 tiles)
    python -m tomojepa.swinjepa.train --data_dir . --pattern 'soild_stack.zarr' \
        --backend zarr --dataset_key reconstruction --img_size 224 \
        --batch_size 16 --epochs 100 --beta_sig 0.05 0.05 0.05 0.01

    # data2vec ablation (no cross-scale predictor), s4 SIGReg queue on
    python -m tomojepa.swinjepa.train --data_dir . --pattern '*.zarr' \
        --no_predictor_enabled --sigreg_queue_len 8192

Multi-GPU (data-parallel gradient averaging; SIGReg is estimated per-rank, so
keep the per-rank batch large):
    torchrun --nproc_per_node=4 -m tomojepa.swinjepa.train --data_dir /data ...
"""
import os
import re
import sys
import glob
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.augmentations import build_slice_fg_mask
from ..core.dataset import TomographyDataset
from ..core import dist as D
from .config import add_argparse_args, from_args
from .model import SwinMSJEPA


def parse_args():
    p = argparse.ArgumentParser(description="Swin multi-scale latent-JEPA pre-training")
    # data
    p.add_argument("--data_dir", required=True, help="Directory of .h5/.zarr volumes")
    p.add_argument("--pattern", default="*.zarr", help="Glob for volume files")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="auto")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--augment", choices=["tomo", "tomo2"], default="tomo2")
    p.add_argument("--crop_mode", choices=["resized", "native", "resize"], default="resized",
                   help="'resized' = RandomResizedCrop to --img_size; 'native' = "
                        "RandomCrop a true --img_size window (no rescaling); 'resize' = "
                        "rescale the whole slice to --img_size (full FOV, e.g. 1024->512).")
    p.add_argument("--num_workers", type=int, default=8)
    # io / logging
    p.add_argument("--ckpt_dir", default="checkpoints_swinjepa")
    p.add_argument("--out_dir", default="outputs_swinjepa")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--pca_every", type=int, default=25,
                   help="render a multi-scale PCA probe (all 4 stages) every N "
                        "steps (0 = off)")
    p.add_argument("--pca_samples", type=int, default=3,
                   help="number of fixed slices in the PCA probe")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="SwinMSJEPA_Tomo")
    # all SwinMSJEPAConfig fields (backbone, mask, predictor, sigreg, optim)
    add_argparse_args(p)
    return p.parse_args()


def build_param_groups(model, weight_decay):
    """AdamW param groups with the standard ViT no-decay list.

    Routes ``mask_token``, pos/stage embeds, norms, and all 1-D params (biases)
    to a no-weight-decay group.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim <= 1 or "norm" in name or "mask_token" in name
                or "bg_token" in name or "bg_stage_tokens" in name
                or "mask_query" in name or "stage_embed" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def save_ckpt(ckpt_dir, tag, epoch, step, model, opt, sched):
    torch.save({"epoch": epoch, "step": step, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict()},
               os.path.join(ckpt_dir, f"ckpt_{tag}.pth"))


def maybe_resume(ckpt_dir, model, opt, sched, device):
    last = os.path.join(ckpt_dir, "ckpt_last.pth")
    cands = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pth"))
    if os.path.exists(last):
        path = last
    elif cands:
        path = max(cands, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    else:
        return 0, 0
    print(f"Resuming from {path}", flush=True)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    return ckpt["epoch"] + 1, ckpt["step"]


def build_pca_probe(args, cfg, device):
    """Load a fixed (deterministic) set of slices for the recurring PCA probe.

    Uses the same ``crop_mode`` as training (e.g. ``resize`` -> whole slice to
    ``img_size``) so the probe shows features at the resolution the model trains
    on.
    """
    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant=args.augment, img_size=cfg.img_size,
        is_train=False, backend=args.backend, crop_mode=args.crop_mode,
    )
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(ds), size=min(args.pca_samples, len(ds)),
                             replace=False).tolist())
    imgs = [ds[int(i)][0].unsqueeze(0).to(device) for i in idxs]   # [1,C,H,W] each
    return idxs, imgs


def _pca_stage_rgb(feat, out_size):
    """Top-3 PCA -> RGB for one stage map ``[C, h, w]``, nearest to ``out_size``."""
    c, h, w = feat.shape
    tok = feat.reshape(c, h * w).T.float()                        # [h*w, C]
    tok = tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    _, _, v = torch.pca_lowrank(tok, q=3)
    pcs = (tok @ v[:, :3]).reshape(h, w, 3).permute(2, 0, 1)[None]
    pcs = F.interpolate(pcs, size=(out_size, out_size), mode="nearest")[0]
    pcs = pcs.permute(1, 2, 0).cpu().numpy()
    for ch in range(3):
        lo, hi = np.percentile(pcs[..., ch], [1, 99])
        pcs[..., ch] = np.clip((pcs[..., ch] - lo) / (hi - lo), 0, 1) if hi > lo else 0.0
    return pcs


@torch.no_grad()
def run_pca_probe(model, probe, step, pca_dir, out_size):
    """Render PCA strips per probe slice (pyramid: C4 + residuals + E; legacy: E only)."""
    idxs, imgs = probe
    was_training = model.training
    model.eval()
    keys = model.stage_keys
    n = len(imgs)
    if model.cfg.legacy_jepa:
        ncol = 1 + len(keys)
    else:
        ncol = 1 + 1 + 3 + len(keys)   # orig | C4 | R1-3 | E1-4
    fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 3.2 * n + 0.5), squeeze=False)
    for r, (idx, img) in enumerate(zip(idxs, imgs)):
        fg_px = None
        if model.cfg.foreground_mask:
            fg_px = build_slice_fg_mask(img[0], model.cfg.fg_std_thresh).unsqueeze(0)
        panels = [(img[0, 0].cpu().numpy(), f"slice {idx}", "gray")]
        if model.cfg.legacy_jepa:
            feats = model.extract_features(img, normalize=True, project=False,
                                           use_latent=True, fg_px=fg_px)
            for key in keys:
                grid = feats[key].shape[-1]
                panels.append((_pca_stage_rgb(feats[key][0], out_size),
                               f"{key} ({grid}x{grid})", None))
        else:
            decomp = model.extract_pyramid_probe(img, fg_px=fg_px)
            c4 = decomp["C4"]["s4"][0]
            panels.append((_pca_stage_rgb(c4, out_size),
                           f"C4 ({c4.shape[-1]}x{c4.shape[-1]})", None))
            for key in ("s1", "s2", "s3"):
                feat = decomp["R"][key][0]
                panels.append((_pca_stage_rgb(feat, out_size),
                               f"R{key[1]} ({feat.shape[-1]}x{feat.shape[-1]})", None))
            feats = decomp["E"]
            for key in keys:
                grid = feats[key].shape[-1]
                panels.append((_pca_stage_rgb(feats[key][0], out_size),
                               f"E{key} ({grid}x{grid})", None))
        for c, (im, title, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cmap,
                      interpolation="none" if cmap is None else "nearest")
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    mode = "legacy" if model.cfg.legacy_jepa else "pyramid"
    fig.suptitle(f"multi-scale PCA ({mode}) @ step {step}", fontsize=13)
    os.makedirs(pca_dir, exist_ok=True)
    out = os.path.join(pca_dir, f"pca_step{step:06d}.png")
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    if was_training:
        model.train()
    return out


def main():
    args = parse_args()
    cfg = from_args(args)
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
    log_wandb = args.wandb and D.is_main()
    if log_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config={**vars(args)})

    # --- data: one image per sample (labels unused) ------------------------
    train_ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant=args.augment, img_size=cfg.img_size,
        is_train=True, backend=args.backend, crop_mode=args.crop_mode,
        foreground_mask=cfg.foreground_mask, fg_std_thresh=cfg.fg_std_thresh,
        fg_key=cfg.fg_key or None,
    )
    sampler = (DistributedSampler(train_ds, shuffle=True, drop_last=True)
               if D.is_distributed() else None)
    loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=(sampler is None),
                         drop_last=True, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"), sampler=sampler)
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(train_ds, **loader_kwargs)
    if D.is_main():
        ddp = f" x {ws} GPUs (global batch {cfg.batch_size * ws})" if D.is_distributed() else ""
        print(f"Dataset: {len(train_ds)} slices over {len(train_ds.files)} volume(s) "
              f"-> {len(loader)} steps/epoch on {device}{ddp}", flush=True)

    # --- model -------------------------------------------------------------
    model = SwinMSJEPA(cfg).to(device)
    assert not model.has_ema, "SwinMSJEPA must have no EMA/momentum teacher."
    if D.is_main():
        n_train = sum(p.numel() for p in model.parameters()
                      if p.requires_grad) / 1e6
        cur = (f"stage_curriculum={cfg.stage_curriculum}"
               if cfg.stage_curriculum == "fine_in"
               else f"stage_curriculum=coarse_in ramp={list(cfg.coarse_ramp_stages)}")
        print(f"Backbone {cfg.backbone_name} dims {model.out_chans} -> "
              f"lat_dims {model.lat_chans}; "
              f"{'legacy_jepa' if cfg.legacy_jepa else 'pyramid_residual'}; "
              f"predictor="
              f"{cfg.predictor_enabled} (cross_scale={cfg.predictor_cross_scale}); "
              f"beta_sig={list(cfg.beta_sig)}; sigreg_queue={cfg.sigreg_queue_len}; "
              f"sigreg_tok/slice={cfg.sigreg_tokens_per_slice}; "
              f"sigreg_min_dist={cfg.sigreg_min_token_dist}; "
              f"{cur}; warmup_frac={cfg.warmup_frac}; "
              f"mask_ratio={cfg.mask_ratio} on s4 grid {model.grids[-1]}"
              f"{'; foreground_mask' if cfg.foreground_mask else ''}", flush=True)
        print(f"Trainable params: {n_train:.2f}M; SIGReg-only collapse control "
              f"(no EMA, no decoder)", flush=True)

    opt = torch.optim.AdamW(build_param_groups(model, cfg.weight_decay),
                            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)
    trainable = [p for p in model.parameters() if p.requires_grad]

    steps_per_epoch = len(loader)
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
    use_amp = (amp_dtype != torch.float32) and device.type == "cuda"

    start_epoch, global_step = maybe_resume(args.ckpt_dir, model, opt, sched, device)

    pca_probe = None
    pca_dir = os.path.join(args.out_dir, "pca")
    if args.pca_every > 0 and D.is_main():          # probe/viz on rank 0 only
        pca_probe = build_pca_probe(args, cfg, device)
        print(f"PCA probe: slices {pca_probe[0]} -> {pca_dir} every "
              f"{args.pca_every} steps (all {model.num_stages} stages)", flush=True)

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(total=steps_per_epoch, desc=f"epoch {epoch}") if D.is_main() else None
        for vs, fg_vs in loader:
            x = vs[:, 0].to(device, non_blocking=True)         # [B, C, H, W]
            fg = None
            if cfg.foreground_mask:
                fg = fg_vs[:, 0].to(device, non_blocking=True)  # [B, 1, H, W]
            opt.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = model.compute_loss(x, fg_px=fg, step=global_step,
                                                total_steps=total_steps)
            loss.backward()
            D.average_grads_(trainable)                         # data-parallel mean
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            sched.step()

            if D.is_main():
                # Per-batch loss on every step (the postfix would otherwise look
                # frozen between log_every intervals even though the loss moves).
                pbar.set_postfix(loss=f"{logs['total']:.3f}",
                                 pred=f"{logs['l_pred']:.3f}",
                                 mae=f"{logs.get('l_mae', 0):.3f}",
                                 sig=f"{logs['l_sig']:.2f}",
                                 er4=f"{logs['effrank/s4']:.1f}")
                if global_step % args.log_every == 0:
                    sk = model.stage_keys
                    fmt = lambda pre: "[" + " ".join(f"{logs[f'{pre}/{k}']:.3g}"
                                                     for k in sk if f'{pre}/{k}' in logs) + "]"
                    extra = ""
                    if cfg.foreground_mask and f"fg_cov/{sk[0]}" in logs:
                        extra = f" fg_cov {fmt('fg_cov')}"
                    if cfg.legacy_jepa:
                        sig_str = fmt('sig')
                    else:
                        sig_str = (f"[c4={logs.get('sig/c4', 0):.3g} "
                                   f"r1={logs.get('sig/r1', 0):.3g} "
                                   f"r2={logs.get('sig/r2', 0):.3g} "
                                   f"r3={logs.get('sig/r3', 0):.3g}]")
                    mae_str = f" mae={logs['l_mae']:.3g}" if not cfg.legacy_jepa else ""
                    pbar.write(
                        f"[step {global_step}] pred {fmt('pred')}{mae_str} sig {sig_str} "
                        f"effrank {fmt('effrank')} fstd {fmt('fstd')} "
                        f"lambda {fmt('lambda')}{extra}", file=sys.stderr)
                if log_wandb and global_step % args.log_every == 0:
                    wandb.log({f"train/{k}": v for k, v in logs.items()}
                              | {"train/lr": sched.get_last_lr()[0]}, step=global_step)
            if (D.is_main() and args.save_every and global_step > 0
                    and global_step % args.save_every == 0):
                save_ckpt(args.ckpt_dir, "last", epoch, global_step, model, opt, sched)
            if pca_probe is not None and global_step % args.pca_every == 0:
                try:
                    out = run_pca_probe(model, pca_probe, global_step, pca_dir, cfg.img_size)
                    pbar.write(f"[pca] saved {out}", file=sys.stderr)
                except Exception as e:  # a viz hiccup must never kill training
                    model.train()
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

    if log_wandb:
        wandb.finish()
    if D.is_main():
        print("done.", flush=True)
    D.cleanup()


if __name__ == "__main__":
    main()
