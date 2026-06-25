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
import gc
import os
import re
import sys
import glob
import math
import argparse
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    p.add_argument("--random_rotate_deg", type=float, default=180.0,
                   help="In-plane random rotation on each train load (+/- deg); "
                        "0=off. Overrides the legacy tomo2 0-180 default with "
                        "symmetric +/- range for more diversity.")
    p.add_argument("--resize_jitter", type=float, nargs=2, default=(0.9, 1.1),
                   metavar=("MIN", "MAX"),
                   help="Uniform scale jitter around --img_size after crop/resize "
                        "(center-crop or zero-pad back). Pass 0 0 to disable.")
    p.add_argument("--crop_mode", choices=["resized", "native", "resize", "crop_down"],
                   default="resized",
                   help="'resized' = RandomResizedCrop to --img_size; 'native' = "
                        "RandomCrop a true --img_size window (no rescaling); 'resize' = "
                        "rescale the whole slice to --img_size; 'crop_down' = "
                        "downsample x2, rotate, RandomCrop --img_size")
    p.add_argument("--crop_size", type=int, default=0,
                   help="Deprecated (ignored for crop_down); legacy compat only")
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader workers (keep low for large zarr slices; each "
                        "worker holds decoded slice buffers)")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Batches prefetched per worker when num_workers > 0")
    p.add_argument("--persistent_workers", action="store_true",
                   help="Keep worker processes alive between epochs (uses more RAM)")
    # io / logging
    p.add_argument("--ckpt_dir", default="checkpoints_swinjepa")
    p.add_argument("--out_dir", default="outputs_swinjepa")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=25,
                   help="Save ckpt_last (and ckpt_stepNNNNNN) every N steps (0=off)")
    p.add_argument("--pca_every", type=int, default=25,
                   help="render a multi-scale PCA probe (all 4 stages) every N "
                        "steps (0 = off)")
    p.add_argument("--pca_samples", type=int, default=1,
                   help="Number of fixed slices in the PCA strip (each runs a probe forward)")
    p.add_argument("--pca_mode", choices=["inference", "pyramid", "legacy"],
                   default="inference",
                   help="inference: clean extract_features Es only; pyramid: full "
                        "C4/T4/R/E decomp; legacy: legacy_jepa token maps")
    p.add_argument("--pca_es_every", type=int, default=100,
                   help="render a 4xN PCA grid of target-pass Es (one slice) every "
                        "N steps (0 = off)")
    p.add_argument("--pca_es_terms", type=int, default=9,
                   help="PCA terms per stage row in the Es grid (default 9)")
    p.add_argument("--pca_size", type=int, default=0,
                   help="Deprecated (ignored): PCA panels use native token grids")
    p.add_argument("--pca_es_size", type=int, default=0,
                   help="Deprecated (ignored): Es PCA uses native token grids")
    p.add_argument("--pca_max_tokens", type=int, default=4096,
                   help="Max spatial tokens for PCA fit (subsample; projects full grid)")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="SwinMSJEPA_Tomo")
    p.add_argument("--schedule", default="",
                   help="YAML job config: stage schedule + optional augmentations block")
    p.add_argument("--aug_config", default="",
                   help="Standalone YAML augmentation config (overrides schedule augmentations)")
    p.add_argument("--global_views", type=int, default=1,
                   help="Wide-area train views per slice")
    p.add_argument("--local_views", type=int, default=0,
                   help="Zoomed-in train views per slice")
    p.add_argument("--global_scale", type=float, nargs=2, default=(0.4, 1.0),
                   metavar=("MIN", "MAX"))
    p.add_argument("--local_scale", type=float, nargs=2, default=(0.1, 0.4),
                   metavar=("MIN", "MAX"))
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


def resolve_job_config(args):
    """Load training schedule + augmentation config from YAML, then CLI overrides."""
    from ..core.aug_config import (
        AugmentationConfig, load_augmentation_yaml, merge_augmentation_cli)
    from .schedule import apply_job_run_overrides, load_job_yaml

    schedule = None
    aug_cfg = AugmentationConfig()
    aug_sched = None
    job_meta = None
    if args.schedule:
        schedule, aug_cfg, aug_sched, job_meta = load_job_yaml(args.schedule)
        if job_meta and job_meta.get("run"):
            apply_job_run_overrides(args, job_meta["run"])
    if args.aug_config:
        aug_cfg, aug_sched = load_augmentation_yaml(args.aug_config)
    aug_cfg = merge_augmentation_cli(aug_cfg, args)
    return schedule, aug_cfg, aug_sched, job_meta


def _yaml_safe(value):
    """Make nested structures YAML-serializable."""
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_config_log_dict(args, cfg, schedule, aug_cfg, aug_sched, job_meta):
    """Merged run configuration for parseable train.log header."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_yaml": args.schedule or None,
        "ckpt_dir": args.ckpt_dir,
        "out_dir": args.out_dir,
        "args": _yaml_safe(vars(args)),
        "cfg": _yaml_safe(asdict(cfg)),
        "augmentations": _yaml_safe(asdict(aug_cfg)),
    }
    if job_meta:
        payload["job_name"] = job_meta.get("name")
        payload["run"] = _yaml_safe(job_meta.get("run", {}))
    if schedule is not None:
        payload["schedule"] = {
            "name": schedule.name,
            "progress_scope": schedule.progress_scope,
            "s4_beta_sig_knots": schedule._stage_knots["s4"]["beta_sig"],
        }
    if aug_sched is not None:
        payload["aug_schedule_lines"] = aug_sched.summary_lines()
    return payload


def print_config_log_block(args, cfg, schedule, aug_cfg, aug_sched, job_meta):
    """Emit parseable YAML config block to stdout (train.log)."""
    import yaml

    if os.path.exists(os.path.join(args.ckpt_dir, "ckpt_last.pth")):
        return
    payload = build_config_log_dict(args, cfg, schedule, aug_cfg, aug_sched, job_meta)
    print("__START_CONFIG__", flush=True)
    print(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False), end="")
    print("__END_CONFIG__", flush=True)


def build_pca_probe(args, cfg, aug_cfg):
    """Pick fixed slice indices for PCA strips.

    Probe loads use deterministic eval geometry: intensity windowing, then
    resize/downsample to ``img_size`` only (no jitter, rotation, or random crop).
    """
    crop_size = args.crop_size if args.crop_size > 0 else None
    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        img_size=cfg.img_size, is_train=False, backend=args.backend,
        crop_size=crop_size, probe_geom=True,
        foreground_mask=cfg.foreground_mask, fg_mode=cfg.fg_mode,
        fg_std_thresh=cfg.fg_std_thresh,
        fg_circle_diameter_frac=cfg.fg_circle_diameter_frac,
        fg_key=cfg.fg_key or None,
        aug_config=aug_cfg,
    )
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(ds), size=min(args.pca_samples, len(ds)),
                             replace=False).tolist())
    return idxs, ds


def _load_pca_probe_views(ds, idxs, foreground_mask):
    """One deterministic view per slice index (eval resize/downsample only)."""
    imgs, fgs = [], []
    for i in idxs:
        view, fg = ds[int(i)]
        imgs.append(view.unsqueeze(0).cpu())
        if foreground_mask:
            fgs.append(fg.unsqueeze(0).cpu())
    return imgs, fgs if foreground_mask else None


def _release_viz_memory(device):
    """Drop PCA peak allocations before resuming the training step."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _norm_tokens(feat_cpu):
    """``[C,H,W]`` on CPU -> normalized token matrix ``[N,C]``."""
    c, h, w = feat_cpu.shape
    tok = feat_cpu.reshape(c, h * w).T.float()
    return tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-6), h, w


def _pca_basis(tok, q, max_tokens):
    """Fit ``q`` PCs on CPU; optionally subsample tokens for the fit only."""
    q = min(q, tok.shape[0], tok.shape[1])
    if q <= 0:
        return None
    fit_tok = tok
    if tok.shape[0] > max_tokens:
        idx = torch.randperm(tok.shape[0], device="cpu")[:max_tokens]
        fit_tok = tok[idx]
    _, _, v = torch.pca_lowrank(fit_tok, q=q)
    return v[:, :q]


def _pca_basis_shared(feat_list, q, max_tokens):
    """Fit one PCA basis on tokens pooled from several ``[C, h, w]`` maps."""
    if not feat_list:
        return None
    pooled = torch.cat([_norm_tokens(f)[0] for f in feat_list], dim=0)
    return _pca_basis(pooled, q, max_tokens)


def _pool_img_to_grid(img, grid_h, grid_w):
    """Average-pool a grayscale ``[H,W]`` image to ``(grid_h, grid_w)`` token cells."""
    t = torch.as_tensor(img, dtype=torch.float32)
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    return F.adaptive_avg_pool2d(t, (grid_h, grid_w)).squeeze().numpy()


def _project_pca_rgb(feat_cpu, basis):
    """Project ``feat_cpu`` with a fixed 3-PC basis; return ``[h,w,3]`` on the token grid."""
    tok, h, w = _norm_tokens(feat_cpu)
    coeffs = tok @ basis[:, :3]
    if coeffs.shape[1] < 3:
        coeffs = F.pad(coeffs, (0, 3 - coeffs.shape[1]))
    return coeffs.reshape(h, w, 3).numpy()


def _pca_stage_rgb(feat_cpu, max_tokens=4096, basis=None, ch_scales=None):
    """Top-3 PCA -> RGB for one stage map ``[C, h, w]`` (CPU), native ``h x w`` grid."""
    tok, h, w = _norm_tokens(feat_cpu)
    blank = np.zeros((h, w, 3), dtype=np.float32)
    v = basis if basis is not None else _pca_basis(tok, 3, max_tokens)
    if v is None:
        return blank
    pcs = _project_pca_rgb(feat_cpu, v)
    if ch_scales is None:
        ch_scales = [tuple(np.percentile(pcs[..., ch], [1, 99])) for ch in range(3)]
    for ch in range(3):
        lo, hi = ch_scales[ch]
        pcs[..., ch] = np.clip((pcs[..., ch] - lo) / (hi - lo), 0, 1) if hi > lo else 0.0
    return pcs.astype(np.float32)


def _pca_stage_rgb_shared(feat_list, max_tokens=4096):
    """Shared 3-PC basis + shared channel scaling across ``feat_list``."""
    n = len(feat_list)
    if n == 0:
        return []
    _, h, w = _norm_tokens(feat_list[0])
    blank = np.zeros((h, w, 3), dtype=np.float32)
    basis = _pca_basis_shared(feat_list, 3, max_tokens)
    if basis is None:
        return [blank] * n
    projected = [_project_pca_rgb(f, basis) for f in feat_list]
    ch_scales = []
    for ch in range(3):
        vals = np.concatenate([p[..., ch].ravel() for p in projected])
        ch_scales.append(tuple(np.percentile(vals, [1, 99])))
    out = []
    for pcs in projected:
        for ch in range(3):
            lo, hi = ch_scales[ch]
            pcs[..., ch] = np.clip((pcs[..., ch] - lo) / (hi - lo), 0, 1) if hi > lo else 0.0
        out.append(pcs.astype(np.float32))
    return out


def _pca_term_maps(feat_cpu, n_terms=9, max_tokens=4096):
    """First ``n_terms`` PCA maps for one stage ``[C, h, w]`` -> list of ``[h, w]``."""
    tok, h, w = _norm_tokens(feat_cpu)
    q = min(n_terms, tok.shape[0], tok.shape[1])
    v = _pca_basis(tok, q, max_tokens)
    blank = np.zeros((h, w), dtype=np.float32)
    if v is None:
        return [blank] * n_terms
    pcs = (tok @ v).reshape(h, w, q).permute(2, 0, 1)
    maps = []
    for i in range(q):
        arr = pcs[i].numpy()
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)
        maps.append(arr.astype(np.float32))
    maps.extend([blank] * (n_terms - len(maps)))
    return maps


def _cpu_pyramid_decomp(model, img_cpu, dev, fg_cpu=None):
    """One ``extract_pyramid_probe`` forward; return CPU tensors and drop GPU refs."""
    img = img_cpu.to(dev, non_blocking=True)
    fg_px = None
    if model.cfg.foreground_mask:
        if fg_cpu is None:
            raise ValueError("foreground_mask probe requires carried fg_px from dataset")
        fg_px = fg_cpu.to(dev, non_blocking=True)
    decomp = model.extract_pyramid_probe(img, fg_px=fg_px)
    out = {
        "C4": decomp["C4"]["s4"][0].detach().cpu(),
        "T4": decomp["T4"]["s4"][0].detach().cpu(),
        "R": {k: decomp["R"][k][0].detach().cpu() for k in ("s1", "s2", "s3")},
        "E": {k: decomp["E"][k][0].detach().cpu() for k in model.stage_keys},
    }
    del img, fg_px, decomp
    _release_viz_memory(dev)
    return out


def _cpu_legacy_feats(model, img_cpu, dev, fg_cpu=None):
    img = img_cpu.to(dev, non_blocking=True)
    fg_px = None
    if model.cfg.foreground_mask:
        if fg_cpu is None:
            raise ValueError("foreground_mask probe requires carried fg_px from dataset")
        fg_px = fg_cpu.to(dev, non_blocking=True)
    feats = model.extract_features(
        img, normalize=True, project=False, use_latent=True, fg_px=fg_px)
    out = {k: feats[k][0].detach().cpu() for k in model.stage_keys}
    del img, fg_px, feats
    _release_viz_memory(dev)
    return out


@torch.no_grad()
def run_pca_viz(model, probe, step, pca_dir, pca_es_dir,
                n_terms, max_tokens, do_strip, do_es, device=None, *,
                pca_mode: str = "inference"):
    """Single probe forward; optional strip and/or Es grid (CPU PCA + matplotlib)."""
    idxs, probe_ds = probe
    imgs, fgs = _load_pca_probe_views(probe_ds, idxs, model.cfg.foreground_mask)
    dev = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    keys = model.stage_keys
    if pca_mode == "legacy":
        strip_kind = "legacy"
    elif pca_mode == "pyramid":
        strip_kind = "pyramid"
    else:
        strip_kind = "inference"
    outs = {}
    try:
        idx = idxs[0]
        all_decomps = None
        all_feats = None
        feats_cpu = None
        decomp_cpu = None

        if do_strip:
            n = len(imgs)
            if strip_kind == "legacy":
                ncol = 1 + len(keys)
            elif strip_kind == "inference":
                ncol = 1 + len(keys)
            else:
                ncol = 4 + 3 + len(keys)
            strip_cell_in = 3.2
            strip_ref_rows = 3
            row_h = strip_cell_in * strip_ref_rows / max(n, 1)
            s1_h, s1_w = model.grids[0]

            if strip_kind == "legacy" or strip_kind == "inference":
                all_feats = []
                for r, im_cpu in enumerate(imgs):
                    fg_r = fgs[r] if fgs is not None else None
                    all_feats.append(_cpu_legacy_feats(model, im_cpu, dev, fg_cpu=fg_r))
                shared_rgb = {
                    key: _pca_stage_rgb_shared(
                        [fe[key] for fe in all_feats], max_tokens)
                    for key in keys
                }
                decomp_cpu = None
                feats_cpu = all_feats[0]
            else:
                all_decomps = []
                for r, im_cpu in enumerate(imgs):
                    fg_r = fgs[r] if fgs is not None else None
                    all_decomps.append(_cpu_pyramid_decomp(model, im_cpu, dev, fg_cpu=fg_r))
                shared_rgb = {
                    "C4": _pca_stage_rgb_shared(
                        [d["C4"] for d in all_decomps], max_tokens),
                    "T4": _pca_stage_rgb_shared(
                        [d["T4"] for d in all_decomps], max_tokens),
                }
                diffs = [(d["C4"] - d["T4"]).abs() for d in all_decomps]
                shared_rgb["dCT"] = _pca_stage_rgb_shared(diffs, max_tokens)
                for rk in ("s1", "s2", "s3"):
                    shared_rgb[f"R{rk[1]}"] = _pca_stage_rgb_shared(
                        [d["R"][rk] for d in all_decomps], max_tokens)
                for key in keys:
                    shared_rgb[f"E{key}"] = _pca_stage_rgb_shared(
                        [d["E"][key] for d in all_decomps], max_tokens)
                decomp_cpu = all_decomps[0]
                feats_cpu = all_decomps[0]["E"]

            # Square cells: column width must match row height or aspect="equal"
            # thumbnails shrink to row height and leave huge horizontal gaps.
            fig, axes = plt.subplots(
                n, ncol, figsize=(row_h * ncol, row_h * n + 0.15), squeeze=False)
            fig.subplots_adjust(left=0.002, right=0.998, top=0.94, bottom=0.002,
                                wspace=0.01, hspace=0.01)
            for r, (sl_idx, im_cpu) in enumerate(zip(idxs, imgs)):
                in_h, in_w = int(im_cpu.shape[-2]), int(im_cpu.shape[-1])
                im_s1 = _pool_img_to_grid(im_cpu[0, 0].numpy(), s1_h, s1_w)
                panels = [(im_s1, f"slice {sl_idx} ({in_h}x{in_w})", "gray")]
                if strip_kind == "legacy" or strip_kind == "inference":
                    for key in keys:
                        grid = all_feats[r][key].shape[-1]
                        panels.append((
                            shared_rgb[key][r],
                            f"E{key} ({grid}x{grid})", None))
                else:
                    d = all_decomps[r]
                    c4 = d["C4"]
                    panels.append((
                        shared_rgb["C4"][r],
                        f"C4 ({c4.shape[-1]}x{c4.shape[-1]})", None))
                    t4 = d["T4"]
                    panels.append((
                        shared_rgb["T4"][r],
                        f"T4 ({t4.shape[-1]}x{t4.shape[-1]})", None))
                    panels.append((
                        shared_rgb["dCT"][r],
                        f"|C4-T4| ({c4.shape[-1]}x{c4.shape[-1]})", None))
                    for rk in ("s1", "s2", "s3"):
                        feat = d["R"][rk]
                        panels.append((
                            shared_rgb[f"R{rk[1]}"][r],
                            f"R{rk[1]} ({feat.shape[-1]}x{feat.shape[-1]})", None))
                    for key in keys:
                        grid = d["E"][key].shape[-1]
                        panels.append((
                            shared_rgb[f"E{key}"][r],
                            f"E{key} ({grid}x{grid})", None))
                for c, (im, title, cmap) in enumerate(panels):
                    ax = axes[r][c]
                    ax.imshow(im, cmap=cmap, interpolation="nearest", aspect="equal")
                    ax.set_anchor("C")
                    if r == 0:
                        ax.set_title(title, fontsize=9, pad=2)
                    ax.axis("off")
                    ax.margins(0)
            fig.suptitle(f"multi-scale PCA ({strip_kind}) @ step {step}", fontsize=11, y=0.98)
            os.makedirs(pca_dir, exist_ok=True)
            strip_out = os.path.join(pca_dir, f"pca_step{step:06d}.png")
            fig.savefig(strip_out, dpi=72, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            outs["strip"] = strip_out
            del fig, axes, panels, shared_rgb
            if strip_kind == "legacy" or strip_kind == "inference":
                feats_cpu = all_feats[0]
                del all_feats
            else:
                feats_cpu = all_decomps[0]["E"]
                del all_decomps

        if do_es:
            if feats_cpu is None:
                img_cpu = imgs[0]
                fg_cpu = fgs[0] if fgs is not None else None
                if strip_kind == "legacy" or strip_kind == "inference":
                    feats_cpu = _cpu_legacy_feats(model, img_cpu, dev, fg_cpu=fg_cpu)
                else:
                    decomp_cpu = _cpu_pyramid_decomp(model, img_cpu, dev, fg_cpu=fg_cpu)
                    feats_cpu = decomp_cpu["E"]
            fig, axes = plt.subplots(
                len(keys), n_terms,
                figsize=(2.0 * n_terms, 2.2 * len(keys) + 0.4),
                squeeze=False)
            for r, key in enumerate(keys):
                grid = feats_cpu[key].shape[-1]
                terms = _pca_term_maps(
                    feats_cpu[key], n_terms=n_terms, max_tokens=max_tokens)
                for c, arr in enumerate(terms):
                    ax = axes[r][c]
                    ax.imshow(arr, cmap="gray", interpolation="nearest", aspect="equal")
                    if r == 0:
                        ax.set_title(f"PC{c + 1}", fontsize=9)
                    if c == 0:
                        ax.set_ylabel(f"E{key}\n({grid}x{grid})", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"Es PCA (slice {idx}) @ step {step}", fontsize=12)
            os.makedirs(pca_es_dir, exist_ok=True)
            es_out = os.path.join(pca_es_dir, f"pca_es_step{step:06d}.png")
            fig.savefig(es_out, dpi=72, bbox_inches="tight")
            plt.close(fig)
            outs["es"] = es_out
            del fig, axes

        del feats_cpu, decomp_cpu
        return outs
    finally:
        _release_viz_memory(dev)
        if was_training:
            model.train()


def main():
    args = parse_args()
    schedule, aug_cfg, aug_sched, job_meta = resolve_job_config(args)
    cfg = from_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if cfg.coarse_mim_mode == "integrated" and cfg.dual_view:
        if aug_cfg.global_views < 2:
            raise ValueError(
                f"coarse_mim_mode='integrated' with dual_view requires "
                f"global_views >= 2, got {aug_cfg.global_views}")

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
    log_wandb = args.wandb and D.is_main()
    if log_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config={**vars(args)})

    # --- data: one image per sample (labels unused) ------------------------
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
        aug_config=aug_cfg,
        aug_schedule=aug_sched,
        shared_aug_state=shared_aug,
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
    if D.is_main():
        ddp = f" x {ws} GPUs (global batch {cfg.batch_size * ws})" if D.is_distributed() else ""
        rot = (f"; random_rotate +/-{train_ds.random_rotate_deg}deg"
               if train_ds.random_rotate_deg else "")
        rj = train_ds.resize_jitter_scale
        jitter = f"; resize_jitter [{rj[0]:g},{rj[1]:g}]" if rj else ""
        crop = f"; crop_mode={aug_cfg.crop_mode}"
        print(f"Dataset: {len(train_ds)} slices over {len(train_ds.files)} volume(s) "
              f"-> {len(loader)} steps/epoch on {device}{ddp}; "
              f"augment: {aug_cfg.summary_line()}{rot}{jitter}", flush=True)
        if aug_sched is not None:
            for line in aug_sched.summary_lines():
                print(line, flush=True)

    # --- model -------------------------------------------------------------
    model = SwinMSJEPA(cfg).to(device)
    assert not model.has_ema, "SwinMSJEPA must have no EMA/momentum teacher."
    if args.schedule:
        model.set_schedule(schedule)
    if D.is_main():
        n_train = sum(p.numel() for p in model.parameters()
                      if p.requires_grad) / 1e6
        cur = (f"stage_curriculum={cfg.stage_curriculum}"
               if cfg.stage_curriculum == "fine_in"
               else f"stage_curriculum=coarse_in ramp={list(cfg.coarse_ramp_stages)}")
        embed_note = (f"embed_dim={cfg.backbone_embed_dim} -> "
                      if cfg.backbone_embed_dim is not None else "")
        print(f"Backbone {cfg.backbone_name} {embed_note}"
              f"stage dims {model.out_chans} -> lat_dims {model.lat_chans}; "
              f"mim_mode={cfg.coarse_mim_mode}; "
              f"{'legacy_jepa' if cfg.legacy_jepa else 'pyramid_residual'}; "
              f"predictor="
              f"{cfg.predictor_enabled} (cross_scale={cfg.predictor_cross_scale}"
              f"{', rope' if cfg.use_rope else ''}); "
              f"fusion_depth={cfg.fusion_depth} fusion_heads={cfg.fusion_heads}; "
              f"dual_view={cfg.dual_view}; "
              f"beta_sig={list(cfg.beta_sig)}; sigreg_queue={cfg.sigreg_queue_len}; "
              f"sigreg_tok/slice={cfg.sigreg_tokens_per_slice}; "
              f"sigreg_tok/frac={cfg.sigreg_token_frac}; "
              f"sigreg_min_dist={cfg.sigreg_min_token_dist}; "
              f"{cur}; warmup_frac={cfg.warmup_frac}; "
              f"mask_ratio={cfg.mask_ratio} on s4 grid {model.grids[-1]}"
              f"{('; foreground_mask fg_mode=' + cfg.fg_mode
                 + (f' circle_d={cfg.fg_circle_diameter_frac}*W'
                    if cfg.fg_mode == 'circle' else ''))
                if cfg.foreground_mask else ''}", flush=True)
        print(f"Trainable params: {n_train:.2f}M; AdamW weight_decay={cfg.weight_decay}; "
              f"SIGReg-only collapse control (no EMA, no decoder)", flush=True)
        if any(e > 0 for e in cfg.freeze_after_epoch):
            print(f"Freeze schedule (epoch>=N): "
                  f"{list(zip(model.stage_keys, cfg.freeze_after_epoch))}", flush=True)
        if model.schedule is not None:
            for line in model.schedule.summary_lines():
                print(line, flush=True)

    opt = torch.optim.AdamW(build_param_groups(model, cfg.weight_decay),
                            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)
    trainable = [p for p in model.parameters() if p.requires_grad]

    total_steps = steps_per_epoch * cfg.epochs
    model.set_steps_per_epoch(steps_per_epoch)
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
    if start_epoch > 0 and model.schedule is None:
        model.apply_freeze_schedule(start_epoch)
        trainable = [p for p in model.parameters() if p.requires_grad]
    elif start_epoch > 0 and model.schedule is not None:
        model._sync_freeze_from_schedule(global_step, total_steps)
        trainable = [p for p in model.parameters() if p.requires_grad]

    pca_probe = None
    pca_dir = os.path.join(args.out_dir, "pca")
    pca_es_dir = os.path.join(args.out_dir, "pca_es")
    if D.is_main() and (args.pca_every > 0 or args.pca_es_every > 0):
        pca_probe = build_pca_probe(args, cfg, aug_cfg)
        grids = model.grids
        if args.pca_every > 0:
            print(f"PCA probe: slices {pca_probe[0]} (deterministic resize) -> {pca_dir} every "
                  f"{args.pca_every} steps (native token grids "
                  f"{', '.join(f'{h}x{w}' for h, w in grids)})", flush=True)
        if args.pca_es_every > 0:
            print(f"Es PCA grid: slice {pca_probe[0][0]} -> {pca_es_dir} every "
                  f"{args.pca_es_every} steps ({args.pca_es_terms} terms x "
                  f"{len(grids)} stages, native grids)", flush=True)

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        if model.schedule is not None and model.schedule.progress_scope == "epoch":
            model.reset_schedule_epoch()
            trainable = [p for p in model.parameters() if p.requires_grad]
            if D.is_main():
                print(f"[epoch {epoch}] schedule epoch cycle reset "
                      f"(progress_scope=epoch)", flush=True)
        elif model.schedule is None:
            newly_frozen = model.apply_freeze_schedule(epoch)
            if newly_frozen:
                trainable = [p for p in model.parameters() if p.requires_grad]
                if D.is_main():
                    print(f"[epoch {epoch}] frozen stages: {newly_frozen}", flush=True)
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(total=steps_per_epoch, desc=f"epoch {epoch}") if D.is_main() else None
        for vs, fg_vs in loader:
            train_ds.update_augmentations(global_step, total_steps, steps_per_epoch)
            dual = (cfg.coarse_mim_mode == "integrated" and cfg.dual_view
                    and not cfg.legacy_jepa)
            if dual:
                x = vs.to(device, non_blocking=True)              # [B, V, C, H, W]
                fg = None
                if cfg.foreground_mask:
                    fg = fg_vs.to(device, non_blocking=True)      # [B, V, 1, H, W]
            else:
                x = vs[:, 0].to(device, non_blocking=True)         # [B, C, H, W]
                fg = None
                if cfg.foreground_mask:
                    fg = fg_vs[:, 0].to(device, non_blocking=True)  # [B, 1, H, W]
            opt.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = model.compute_loss(x, fg_px=fg, step=global_step,
                                                total_steps=total_steps, epoch=epoch)
            if not cfg.legacy_jepa and logs.get("mae/cos") is not None:
                model.note_mae_cos(logs["mae/cos"])
            loss.backward()
            if model._last_newly_frozen:
                trainable = [p for p in model.parameters() if p.requires_grad]
                if D.is_main():
                    print(f"[step {global_step}] frozen stages: "
                          f"{model._last_newly_frozen}", flush=True)
                model._last_newly_frozen = []
            D.average_grads_(trainable)                         # data-parallel mean
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            sched.step()

            if D.is_main():
                # Per-batch loss on every step (the postfix would otherwise look
                # frozen between log_every intervals even though the loss moves).
                postfix = dict(loss=f"{logs['total']:.3f}",
                               pred=f"{logs['l_pred']:.3f}",
                               mae=f"{logs.get('l_mae', 0):.3f}",
                               sig=f"{logs['l_sig']:.2f}",
                               er4=f"{logs['effrank/s4']:.1f}")
                if not cfg.legacy_jepa and logs.get("mae/cos") is not None:
                    postfix["cos"] = f"{logs['mae/cos']:.3g}"
                if cfg.s4_cosine_level > 0 and "sigreg/cos_ema" in logs:
                    postfix["cos_ema"] = f"{logs['sigreg/cos_ema']:.3g}"
                    postfix["sig_gate"] = f"{logs['sigreg/cos_gate']:.3g}"
                if logs.get("sig/raw/s4") is not None:
                    postfix["sig_r"] = f"{logs['sig/raw/s4']:.2f}"
                pbar.set_postfix(**postfix)
                if global_step % args.log_every == 0:
                    sk = model.stage_keys
                    fmt = lambda pre: "[" + " ".join(f"{logs[f'{pre}/{k}']:.3g}"
                                                     for k in sk if f'{pre}/{k}' in logs) + "]"
                    extra = ""
                    if cfg.foreground_mask and f"fg_cov/{sk[0]}" in logs:
                        extra = f" fg_cov {fmt('fg_cov')}"
                    sig_str = fmt('sig')
                    if "sig/raw/s4" in logs:
                        sig_str += f" raw {fmt('sig/raw')}"
                    mae_str = f" mae={logs['l_mae']:.3g}" if not cfg.legacy_jepa else ""
                    if not cfg.legacy_jepa and logs.get("mae/cos") is not None:
                        mae_str += f" cos={logs['mae/cos']:.3g}"
                    if cfg.s4_cosine_level > 0 and "sigreg/cos_gate" in logs:
                        mae_str += (f" cos_ema={logs['sigreg/cos_ema']:.3g}"
                                    f" sig_gate={logs['sigreg/cos_gate']:.3g}")
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
                save_ckpt(args.ckpt_dir, f"step_{global_step:06d}", epoch, global_step,
                          model, opt, sched)
                if pbar is not None:
                    pbar.write(f"[ckpt] saved step {global_step}", file=sys.stderr)
            do_strip = (pca_probe is not None and args.pca_every > 0
                        and global_step % args.pca_every == 0)
            do_es = (pca_probe is not None and args.pca_es_every > 0
                     and global_step % args.pca_es_every == 0)
            if do_strip or do_es:
                try:
                    viz = run_pca_viz(
                        model, pca_probe, global_step, pca_dir, pca_es_dir,
                        args.pca_es_terms, args.pca_max_tokens, do_strip, do_es,
                        device=device, pca_mode=args.pca_mode)
                    if pbar is not None:
                        if "strip" in viz:
                            pbar.write(f"[pca] saved {viz['strip']}", file=sys.stderr)
                        if "es" in viz:
                            pbar.write(f"[pca_es] saved {viz['es']}", file=sys.stderr)
                except Exception as e:
                    model.train()
                    if pbar is not None:
                        pbar.write(f"[pca] skipped step {global_step}: {e}",
                                   file=sys.stderr)

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
