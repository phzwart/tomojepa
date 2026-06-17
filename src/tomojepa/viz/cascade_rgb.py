"""Cascading PCA-RGB maps for a checkpoint.

For each image we fit a robust PCA basis on the foreground patch tokens (same
machinery as ``validate.py``'s eigen grid: L2-normalized tokens, foreground-only
fit, directional-outlier rejection) and then render the components in *triplets*
mapped to RGB: (PC1,PC2,PC3) -> RGB, (PC4,PC5,PC6) -> RGB, (PC7,PC8,PC9) -> RGB,
...  Each triplet is a separate column; each row is one image.

The leading triplet is the usual "DINO PCA" view; the later triplets expose the
progressively finer structure that a single top-3 RGB throws away.

Usage:
    python cascade_rgb.py --run_dir runs/soil_residual_fg \
        --pattern soild_stack.zarr --eigen_ckpt 14 \
        --n_images 10 --n_triplets 6 --foreground_mask --fg_std_thresh 0.05
"""
import os
import re
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.dataset import TomographyDataset
from ..core.model import DINOv3ViTEncoder, foreground_tokens
from ..ssl.validate import _resolve_eigen_ckpt


@torch.no_grad()
def extract_tokens(net, view, device, fg_thresh=None):
    """L2-normalized patch tokens + foreground mask for one image.

    Returns ``(tokens[P, D], grid, fg[P] bool)`` on ``device``.
    """
    feat = net.backbone.forward_features(view.to(device))
    tokens = feat[:, net.backbone.num_prefix_tokens:].squeeze(0).float()
    tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    P = tokens.shape[0]
    grid = int(round(P ** 0.5))
    if fg_thresh is not None:
        fg = foreground_tokens(view.to(device), grid, fg_thresh).squeeze(0)
    else:
        fg = torch.ones(P, dtype=torch.bool, device=tokens.device)
    return tokens, grid, fg


def fit_shared_basis(fg_tokens, k, outlier_pct=2.0):
    """Robust PCA basis on pooled foreground tokens from many images.

    Returns ``(mu[1, D], Vh[k, D], explained_var_ratio[k])``. A directional
    outlier reject (same as the per-image eigen grid) keeps a few artifact
    patches from hijacking the shared basis.
    """
    fit = fg_tokens
    med = fit.median(0, keepdim=True).values
    dist = (fit - med).norm(dim=-1)
    if outlier_pct > 0 and fit.shape[0] > 20:
        keep = dist <= torch.quantile(dist, 1.0 - outlier_pct / 100.0)
        fit = fit[keep]
    mu = fit.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(fit - mu, full_matrices=False)
    k = min(k, S.numel())
    ev = (S.square() / S.square().sum()).cpu().numpy()[:k]
    return mu, Vh[:k], ev


def triplet_rgb(maps, c0, fg_grid, norm):
    """Build an RGB image from components ``c0, c0+1, c0+2``.

    ``norm[c] = (lo, hi)`` are *global* per-component scale limits (computed once
    across all images) so a given color means the same thing in every row.
    Background cells -> black.
    """
    h, w, k = maps.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(3):
        c = c0 + ch
        if c >= k:
            continue
        m = maps[..., c]
        lo, hi = norm[c]
        rgb[..., ch] = np.clip((m - lo) / (hi - lo), 0, 1) if hi > lo else 0.0
    if fg_grid is not None:
        rgb[~fg_grid] = 0.0
    return rgb


def main():
    p = argparse.ArgumentParser(description="Cascading PCA-RGB triplet maps")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt_subdir", default="ckpt")
    p.add_argument("--eigen_ckpt", default="last",
                   help="'last', an epoch int, or a checkpoint path")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="soild_stack.zarr")
    p.add_argument("--backend", default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_chans", type=int, default=1)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--n_triplets", type=int, default=6,
                   help="how many RGB triplets (PCs 1..3*n_triplets)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--foreground_mask", action="store_true")
    p.add_argument("--fg_std_thresh", type=float, default=0.05)
    p.add_argument("--outlier_pct", type=float, default=2.0)
    p.add_argument("--out", default=None, help="output png (default run_dir/out/cascade_rgb_*.png)")
    args = p.parse_args()
    fg_thresh = args.fg_std_thresh if args.foreground_mask else None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _resolve_eigen_ckpt(args.run_dir, args.ckpt_subdir, args.eigen_ckpt)
    ep_tag = re.search(r"epoch_(\d+)", os.path.basename(ckpt))
    ep_tag = ep_tag.group(1) if ep_tag else "last"

    net = DINOv3ViTEncoder(proj_dim=args.proj_dim, img_size=args.img_size,
                           in_chans=args.in_chans, pretrained=False).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)["net"])
    net.eval()

    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant="tomo2", img_size=args.img_size,
        is_train=False, backend=args.backend,
    )
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(ds), size=min(args.n_images, len(ds)),
                             replace=False).tolist())

    k = args.n_triplets * 3

    # Pass 1: extract tokens / foreground / slice for every image, and pool the
    # foreground tokens to fit ONE shared PCA basis (unified colors across rows).
    per_img, fg_pool = [], []
    for si in idxs:
        item = ds[si]
        view = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(view, (list, tuple)):
            view = view[0]
        view = view.unsqueeze(0)
        tokens, grid, fg = extract_tokens(net, view, device, fg_thresh=fg_thresh)
        orig = view[0].mean(0).cpu().numpy()
        per_img.append((si, tokens, grid, fg, orig))
        fg_pool.append(tokens[fg])
    mu, Vh, ev = fit_shared_basis(torch.cat(fg_pool, 0), k,
                                  outlier_pct=args.outlier_pct)

    # Project every image onto the shared basis; collect maps + foreground values
    # to derive GLOBAL per-component normalization limits.
    proj_maps, fg_grids, all_vals = [], [], [[] for _ in range(k)]
    for si, tokens, grid, fg, orig in per_img:
        proj = ((tokens - mu) @ Vh.T).reshape(grid, grid, k).cpu().numpy()
        fg_grid = fg.reshape(grid, grid).cpu().numpy() if fg_thresh is not None else None
        proj_maps.append(proj)
        fg_grids.append(fg_grid)
        sel = fg_grid if fg_grid is not None else np.ones(proj.shape[:2], bool)
        for c in range(k):
            all_vals[c].append(proj[..., c][sel])
    norm = [tuple(np.percentile(np.concatenate(all_vals[c]), [2, 98]))
            for c in range(k)]

    # Pass 2: render with the shared basis + global normalization.
    nrow = len(idxs)
    ncol = args.n_triplets + 1                      # slice + one col per triplet
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 2.0))
    axes = np.array(axes).reshape(nrow, ncol)

    for r, ((si, _t, _g, _f, orig), maps, fg_grid) in enumerate(
            zip(per_img, proj_maps, fg_grids)):
        axes[r, 0].imshow(orig, cmap="gray")
        axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("slice", fontsize=9)
        axes[r, 0].set_ylabel(f"#{si}", fontsize=8)
        for t in range(args.n_triplets):
            c0 = t * 3
            rgb = triplet_rgb(maps, c0, fg_grid, norm)
            ax = axes[r, t + 1]
            ax.imshow(rgb, interpolation="nearest" if fg_grid is not None else "bilinear")
            ax.axis("off")
            if r == 0:
                evs = ev[c0:c0 + 3] * 100
                evs = "/".join(f"{x:.1f}" for x in evs)
                ax.set_title(f"PC{c0+1},{c0+2},{c0+3}\n{evs}%", fontsize=8)

    fig.suptitle(
        f"{os.path.basename(args.run_dir)}  ep{ep_tag}  cascading PCA-RGB "
        f"(R,G,B = consecutive PCs; shared basis across images)",
        fontsize=11, y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(args.run_dir, "out",
                                   f"cascade_rgb_ep{ep_tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
