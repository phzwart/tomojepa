"""Intrinsic validation for Swin multi-scale latent-JEPA checkpoints (no labels).

For each ``ckpt_epoch_*.pth`` in a run directory, loads the encoder and reports
per-stage metrics on the **regularized lateral latent** (what SIGReg/JEPA shape):

  - token_effrank/s* : effective rank of within-image stage tokens (collapse /
                       spatial diversity; higher = more dimensions in use).
  - emb_effrank/s*   : effective rank of FG-pooled (or mean-pooled) slice
                       embeddings across a fixed slice set.
  - aug_cos/s*       : cosine similarity of pooled features across two
                       independent augmentations of the same slice.
  - fstd/s*          : mean per-channel token std (feature spread).

When ``--eval_pred`` is set (default), also evaluates the full JEPA objective on
a fixed held-out batch with a fixed mask RNG (comparable ``eval_pred/s*`` and
``eval_pred_total`` across epochs).

Results are written to ``run_dir/metrics.json`` (same convention as
``ssl/validate.py``).

Usage:
    python -m tomojepa.swinjepa.validate \\
        --run_dir runs/swin_msjepa_soil_512_fg \\
        --data_dir . --pattern soild_stack.zarr --backend zarr \\
        --img_size 512 --crop_mode resize --foreground_mask \\
        --beta_sig 0.003
"""
import os
import re
import glob
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..core.dataset import TomographyDataset
from ..core.augmentations import build_fg_stages
from .config import add_argparse_args, from_args
from .losses import effective_rank, stage_feature_diagnostics
from .model import SwinMSJEPA, SwinMSEncoder, extract_fg_masks, extract_images


def _pool_stage(feat: torch.Tensor, fg: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-pool stage tokens ``[B, C, h, w]`` -> ``[B, C]`` (FG-only when given)."""
    b, c, h, w = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if fg is None:
        return tok.mean(1)
    m = fg.reshape(b, h * w).float()
    denom = m.sum(1, keepdim=True).clamp_min(1.0)
    return (tok * m.unsqueeze(-1)).sum(1) / denom


def _fg_stages_from_px(fg_px: torch.Tensor, enc: SwinMSEncoder,
                       fg_coverage: float) -> Dict[str, torch.Tensor]:
    grids = [enc.backbone.stage_grid(s) for s in range(enc.backbone._num_stages)]
    return build_fg_stages(fg_px, grids, fg_coverage)


@torch.no_grad()
def eval_encoder_ckpt(ckpt_path, enc: SwinMSEncoder, ds_aug, idxs: List[int],
                      n_token_imgs: int, device, use_fg: bool,
                      fg_coverage: float) -> Dict[str, float]:
    """Per-stage intrinsic metrics on clean backbone features."""
    enc.backbone.load_state_dict({
        k[len("backbone."):]: v for k, v in
        torch.load(ckpt_path, map_location=device)["model"].items()
        if k.startswith("backbone.")
    })
    enc.eval()

    stage_keys = enc.stage_keys
    embs0 = {k: [] for k in stage_keys}
    embs1 = {k: [] for k in stage_keys}
    token_ranks = {k: [] for k in stage_keys}

    for k, i in enumerate(idxs):
        item = ds_aug[i]
        if use_fg:
            views, fg_views = item                          # [2,C,H,W], [2,1,H,W]
        else:
            views, _ = item
            fg_views = None
        views = views.to(device)
        fg_views = fg_views.to(device) if fg_views is not None else None

        f0 = enc.extract_features(views[0:1], normalize=True, project=False,
                                  use_latent=True)
        f1 = enc.extract_features(views[1:2], normalize=True, project=False,
                                  use_latent=True)
        fg0 = _fg_stages_from_px(fg_views[0:1], enc, fg_coverage) if fg_views is not None else None
        fg1 = _fg_stages_from_px(fg_views[1:2], enc, fg_coverage) if fg_views is not None else None

        for key in stage_keys:
            e0 = _pool_stage(f0[key], fg0[key] if fg0 else None)
            e1 = _pool_stage(f1[key], fg1[key] if fg1 else None)
            embs0[key].append(e0[0])
            embs1[key].append(e1[0])
            if k < n_token_imgs:
                tok = f0[key][0].permute(1, 2, 0).reshape(-1, f0[key].shape[1])
                if fg0 is not None:
                    tok = tok[fg0[key][0].reshape(-1)]
                token_ranks[key].append(effective_rank(tok))

    out: Dict[str, float] = {"n_slices": len(idxs)}
    for key in stage_keys:
        e0 = torch.stack(embs0[key])
        e1 = torch.stack(embs1[key])
        cos = F.cosine_similarity(e0, e1, dim=-1)
        out[f"emb_effrank/{key}"] = effective_rank(e0)
        out[f"aug_cos/{key}"] = float(cos.mean())
        out[f"aug_cos_std/{key}"] = float(cos.std())
        out[f"token_effrank/{key}"] = float(np.mean(token_ranks[key]))
    return out


@torch.no_grad()
def eval_pred_ckpt(ckpt_path, model: SwinMSJEPA, loader, device,
                   total_steps: int) -> Dict[str, float]:
    """Fixed-mask held-out JEPA prediction loss (full model)."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    torch.manual_seed(0)
    np.random.seed(0)

    sums: Dict[str, float] = {}
    n = 0
    for batch in loader:
        x = extract_images(batch).to(device)
        fg = extract_fg_masks(batch)
        if fg is not None:
            fg = fg.to(device)
        _, logs = model.compute_loss(x, fg_px=fg, step=total_steps,
                                     total_steps=total_steps)
        for k, v in logs.items():
            if k.startswith("pred/") or k in ("total", "l_pred", "l_sig"):
                sums[k] = sums.get(k, 0.0) + float(v)
        n += 1

    if n == 0:
        return {}
    out = {k: v / n for k, v in sums.items()}
    mapped = {}
    for k, v in out.items():
        if k.startswith("pred/"):
            mapped["eval_" + k] = v
        elif k in ("total", "l_pred", "l_sig"):
            mapped["eval_" + k] = v
    return mapped


def main():
    p = argparse.ArgumentParser(description="Intrinsic validation of SwinMSJEPA runs")
    p.add_argument("--run_dir", required=True, help="dir with ckpt/ subdir")
    p.add_argument("--ckpt_subdir", default="ckpt")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="soild_stack.zarr")
    p.add_argument("--backend", default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--augment", choices=["tomo", "tomo2"], default="tomo2")
    p.add_argument("--crop_mode", choices=["resized", "native", "resize"], default="resize")
    p.add_argument("--n_slices", type=int, default=384,
                   help="slices for emb/aug metrics")
    p.add_argument("--n_token_imgs", type=int, default=16,
                   help="slices averaged for token_effrank")
    p.add_argument("--eval_batches", type=int, default=32,
                   help="held-out batches for eval_pred (0 = skip)")
    p.add_argument("--eval_batch_size", type=int, default=0,
                   help="eval batch size (0 = use config batch_size)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON path (default run_dir/metrics.json)")
    p.add_argument("--steps_per_epoch", type=int, default=0,
                   help="override step count for curriculum at eval (0 = infer from data)")
    add_argparse_args(p)
    args = p.parse_args()
    cfg = from_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_aug = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=2, local_views=0, variant=args.augment, img_size=cfg.img_size,
        is_train=True, backend=args.backend, crop_mode=args.crop_mode,
        foreground_mask=cfg.foreground_mask, fg_mode=cfg.fg_mode,
        fg_std_thresh=cfg.fg_std_thresh,
        fg_circle_diameter_frac=cfg.fg_circle_diameter_frac,
        fg_key=cfg.fg_key or None,
    )
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(ds_aug), size=min(args.n_slices, len(ds_aug)),
                             replace=False).tolist())

    enc = SwinMSEncoder(cfg, with_lateral=False).to(device)
    ckpts = sorted(
        glob.glob(os.path.join(args.run_dir, args.ckpt_subdir, "ckpt_epoch_*.pth")),
        key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)),
    )
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_epoch_*.pth under {args.run_dir}/{args.ckpt_subdir}")

    pred_loader = None
    total_steps = 1
    model = None
    if args.eval_batches > 0:
        ds_eval = TomographyDataset(
            data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
            global_views=1, local_views=0, variant=args.augment, img_size=cfg.img_size,
            is_train=True, backend=args.backend, crop_mode=args.crop_mode,
            foreground_mask=cfg.foreground_mask, fg_mode=cfg.fg_mode,
            fg_std_thresh=cfg.fg_std_thresh,
            fg_circle_diameter_frac=cfg.fg_circle_diameter_frac,
            fg_key=cfg.fg_key or None,
        )
        eval_n = min(len(ds_eval), args.eval_batches * max(1, args.eval_batch_size or cfg.batch_size))
        eval_idxs = sorted(rng.choice(len(ds_eval), size=eval_n, replace=False).tolist())
        bs = args.eval_batch_size or cfg.batch_size
        pred_loader = DataLoader(Subset(ds_eval, eval_idxs), batch_size=bs, shuffle=False)
        steps_per_epoch = args.steps_per_epoch or max(1, len(ds_eval) // bs)
        total_steps = steps_per_epoch * cfg.epochs
        model = SwinMSJEPA(cfg).to(device)

    results = []
    for c in ckpts:
        ep = int(re.search(r"epoch_(\d+)", c).group(1))
        m = eval_encoder_ckpt(c, enc, ds_aug, idxs, args.n_token_imgs, device,
                              cfg.foreground_mask, cfg.fg_coverage)
        # fstd from one deterministic pass on the first eval slice
        sd = torch.load(c, map_location=device)["model"]
        enc.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()
                                      if k.startswith("backbone.")})
        enc.eval()
        item = ds_aug[idxs[0]]
        v = item[0][0:1].to(device) if isinstance(item, (tuple, list)) else item[0:1].to(device)
        feats = enc.extract_features(v, normalize=True, project=False,
                                     use_latent=True)
        m.update(stage_feature_diagnostics(feats))
        if pred_loader is not None and model is not None:
            m.update(eval_pred_ckpt(c, model, pred_loader, device, total_steps))
        m["epoch"] = ep
        m["ckpt"] = os.path.basename(c)
        results.append(m)
        er4 = m.get("token_effrank/s4", float("nan"))
        pr4 = m.get("eval_pred/s4", m.get("pred/s4", float("nan")))
        ac4 = m.get("aug_cos/s4", float("nan"))
        print(f"[{os.path.basename(args.run_dir)}] epoch {ep:>2}  "
              f"tok_er4={er4:6.1f}  eval_pred_s4={pr4:.3f}  aug_cos_s4={ac4:.4f}",
              flush=True)

    out_path = args.out or os.path.join(args.run_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
