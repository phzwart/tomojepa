"""Intrinsic validation for LeJEPA / residual-MIM checkpoints (no labels needed).

For each ``ckpt_epoch_*.pth`` in a run directory, loads the encoder and reports:

  - emb_effrank   : effective rank of the pooled backbone features over a fixed
                    set of slices (collapse detector; higher = more informative).
  - token_effrank : effective rank of within-image patch tokens, averaged over a
                    few slices (spatial feature diversity).
  - aug_cos       : mean cosine similarity of the pooled features across two
                    independent augmentations of the same slice (invariance;
                    higher = more augmentation-invariant).
  - aug_cos_std   : spread of that similarity across slices.

Effective rank = exp(entropy(normalized singular values)) of the centered
feature matrix -- a smooth, scale-free stand-in for "how many dimensions are
actually used" (Roy & Vetterli, 2007).

Both runs share the same backbone architecture, so these are directly
comparable. Metrics are computed on the backbone (not the proj / residual head)
so the comparison is fair regardless of which objective shaped the features.

Usage:
    python validate.py --run_dir runs/val_residual --data_dir . \
        --pattern upsampled_1024.zarr --backend zarr --img_size 512
"""
import os
import re
import glob
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.dataset import TomographyDataset
from ..core.model import DINOv3ViTEncoder, foreground_tokens, masked_mean


def effective_rank(x):
    """exp(entropy of normalized singular values) of centered ``x`` [N, D]."""
    x = x.float()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float((-(p * p.log()).sum()).exp())


@torch.no_grad()
def eval_ckpt(ckpt_path, net, ds_aug, idxs, n_token_imgs, device,
              fg_thresh=None):
    net.load_state_dict(torch.load(ckpt_path, map_location=device)["net"])
    net.eval()

    embs0, embs1, token_ranks = [], [], []
    for k, i in enumerate(idxs):
        views, _ = ds_aug[i]                       # [2, C, H, W] (two global augs)
        v = views.to(device)
        feat = net.backbone.forward_features(v)    # [2, L, D]
        tok = feat[:, net.backbone.num_prefix_tokens:]
        if fg_thresh is not None:
            grid = int(round(tok.shape[1] ** 0.5))
            fg = foreground_tokens(v, grid, fg_thresh)         # [2, P]
            emb = masked_mean(tok, fg)             # [2, D] foreground pooled
        else:
            fg = None
            emb = tok.mean(1)                      # [2, D] pooled patch tokens
        embs0.append(emb[0])
        embs1.append(emb[1])
        if k < n_token_imgs:
            t0 = tok[0][fg[0]] if fg is not None else tok[0]
            token_ranks.append(effective_rank(t0))

    e0 = torch.stack(embs0)                         # [N, D]
    e1 = torch.stack(embs1)
    cos = torch.nn.functional.cosine_similarity(e0, e1, dim=-1)
    return {
        "emb_effrank": effective_rank(e0),
        "token_effrank": float(np.mean(token_ranks)),
        "aug_cos": float(cos.mean()),
        "aug_cos_std": float(cos.std()),
        "n_slices": len(idxs),
        "feat_dim": e0.size(-1),
    }


@torch.no_grad()
def eigen_map_grid(net, view, k, device, fg_thresh=None, outlier_pct=2.0):
    """Spatial maps of the top-``k`` PCA/SVD components of the patch tokens.

    Returns ``(maps[grid, grid, k], explained_var_ratio[k], fg_grid)``. Unlike a
    top-3 PCA RGB, this exposes the *higher* components. When ``fg_thresh`` is
    set, the PCA basis is fit on foreground (sample-ROI) tokens only, so the
    leading components describe sample structure instead of the background/frame
    boundary; ``fg_grid`` marks which cells are foreground (None otherwise).
    """
    feat = net.backbone.forward_features(view.to(device))     # [1, L, D]
    tokens = feat[:, net.backbone.num_prefix_tokens:].squeeze(0).float()   # [P, D]
    # L2-normalize tokens so a few high-norm outlier ("artifact") tokens don't
    # dominate the SVD spectrum (standard DINO PCA practice).
    tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    P = tokens.shape[0]
    grid = int(round(P ** 0.5))
    if fg_thresh is not None:
        fg = foreground_tokens(view.to(device), grid, fg_thresh).squeeze(0)  # [P]
    else:
        fg = torch.ones(P, dtype=torch.bool, device=tokens.device)
    # Reject directionally-extreme "artifact" tokens from the PCA fit: a handful
    # of outlier patches otherwise capture a single dominant component. Keep them
    # for display (clipped by percentile), but fit the basis on the bulk.
    fit = tokens[fg]
    med = fit.median(0, keepdim=True).values
    dist = (fit - med).norm(dim=-1)
    if outlier_pct > 0 and fit.shape[0] > 20:
        keep = dist <= torch.quantile(dist, 1.0 - outlier_pct / 100.0)
        fit = fit[keep]
    mu = fit.mean(0, keepdim=True)
    # SVD fit on the robust bulk; project ALL tokens onto the basis for display
    _, S, Vh = torch.linalg.svd(fit - mu, full_matrices=False)
    k = min(k, S.numel())
    proj = ((tokens - mu) @ Vh[:k].T).reshape(grid, grid, k).cpu().numpy()
    ev = (S.square() / S.square().sum()).cpu().numpy()[:k]
    fg_grid = fg.reshape(grid, grid).cpu().numpy() if fg_thresh is not None else None
    return proj, ev, fg_grid


def save_eigen_grid(maps, ev, orig, out_path, side, title, fg_grid=None):
    """Save a ``side x side`` grid of per-component spatial maps (+ the slice).

    With ``fg_grid``, background cells are blanked (gray) and the per-map
    contrast is computed from foreground values only.
    """
    k = side * side
    cmap = plt.cm.magma.copy()
    cmap.set_bad("0.6")
    interp = "nearest" if fg_grid is not None else "bilinear"
    fig, axes = plt.subplots(side, side, figsize=(side * 2.1, side * 2.1))
    axes = np.array(axes).reshape(-1)
    # top-left cell = the input slice for reference
    axes[0].imshow(orig, cmap="gray"); axes[0].set_title("slice", fontsize=8)
    axes[0].axis("off")
    for i in range(1, k):
        ax = axes[i]
        c = i - 1                                  # component index (PC1.. in cell 1..)
        if c < maps.shape[-1]:
            m = maps[..., c]
            vals = m[fg_grid] if fg_grid is not None else m
            lo, hi = np.percentile(vals, [2, 98])
            mm = np.clip((m - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(m)
            if fg_grid is not None:
                mm = np.where(fg_grid, mm, np.nan)
            ax.imshow(mm, cmap=cmap, interpolation=interp)
            ax.set_title(f"PC{c + 1}  {ev[c] * 100:.1f}%", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _resolve_eigen_ckpt(run_dir, ckpt_subdir, spec):
    """Resolve --eigen_ckpt: a path, an int epoch, or 'last'."""
    d = os.path.join(run_dir, ckpt_subdir)
    if spec not in (None, "last") and os.path.exists(spec):
        return spec
    if spec in (None, "last"):
        cand = os.path.join(d, "ckpt_last.pth")
        if os.path.exists(cand):
            return cand
        # fall back to highest epoch
        eps = glob.glob(os.path.join(d, "ckpt_epoch_*.pth"))
        return max(eps, key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)))
    return os.path.join(d, f"ckpt_epoch_{int(spec)}.pth")


def main():
    p = argparse.ArgumentParser(description="Intrinsic validation of encoders")
    p.add_argument("--run_dir", required=True, help="dir with ckpt/ and out/")
    p.add_argument("--ckpt_subdir", default="ckpt")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="upsampled_1024.zarr")
    p.add_argument("--backend", default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_chans", type=int, default=1)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--n_slices", type=int, default=384, help="slices for emb metrics")
    p.add_argument("--n_token_imgs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON results path (default run_dir/metrics.json)")
    # eigen/singular-map grid viz
    p.add_argument("--eigen_grid", action="store_true",
                   help="render a grid of per-component singular maps (top components)")
    p.add_argument("--skip_metrics", action="store_true",
                   help="skip the per-epoch metric sweep (e.g. only do --eigen_grid)")
    p.add_argument("--eigen_ckpt", default="last",
                   help="checkpoint for eigen grid: 'last', an epoch int, or a path")
    p.add_argument("--eigen_side", type=int, default=5, help="grid side (side*side maps)")
    p.add_argument("--eigen_samples", type=int, default=3, help="number of slices to render")
    p.add_argument("--foreground_mask", action="store_true",
                   help="restrict pooling / PCA to foreground (sample-ROI) tokens, "
                        "ignoring the flat background/frame")
    p.add_argument("--fg_std_thresh", type=float, default=0.05,
                   help="per-patch intensity-std threshold for the foreground mask")
    p.add_argument("--eigen_outlier_pct", type=float, default=2.0,
                   help="reject this %% of directionally-extreme artifact tokens "
                        "from the eigen-grid PCA fit (0 disables)")
    args = p.parse_args()
    fg_thresh = args.fg_std_thresh if args.foreground_mask else None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two global augmentations per item (no local crops) for the invariance probe.
    ds_aug = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=2, local_views=0, variant="tomo2", img_size=args.img_size,
        is_train=True, backend=args.backend,
    )
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(ds_aug), size=min(args.n_slices, len(ds_aug)),
                             replace=False).tolist())

    net = DINOv3ViTEncoder(proj_dim=args.proj_dim, img_size=args.img_size,
                           in_chans=args.in_chans, pretrained=False).to(device)

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, args.ckpt_subdir, "ckpt_epoch_*.pth")),
                   key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_epoch_*.pth under {args.run_dir}/{args.ckpt_subdir}")

    if not args.skip_metrics:
        results = []
        for c in ckpts:
            ep = int(re.search(r"epoch_(\d+)", c).group(1))
            m = eval_ckpt(c, net, ds_aug, idxs, args.n_token_imgs, device,
                          fg_thresh=fg_thresh)
            m["epoch"] = ep
            m["ckpt"] = os.path.basename(c)
            results.append(m)
            print(f"[{os.path.basename(args.run_dir)}] epoch {ep:>2}  "
                  f"emb_effrank={m['emb_effrank']:6.2f}  "
                  f"token_effrank={m['token_effrank']:6.2f}  "
                  f"aug_cos={m['aug_cos']:.4f}±{m['aug_cos_std']:.4f}", flush=True)

        out = args.out or os.path.join(args.run_dir, "metrics.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out}", flush=True)

    if args.eigen_grid:
        ckpt = _resolve_eigen_ckpt(args.run_dir, args.ckpt_subdir, args.eigen_ckpt)
        sd = torch.load(ckpt, map_location=device)
        net.load_state_dict(sd["net"])
        net.eval()

        # clean (deterministic) single-view slices, not the aug probe views
        ds_clean = TomographyDataset(
            data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
            global_views=1, local_views=0, variant="tomo2", img_size=args.img_size,
            is_train=False, backend=args.backend,
        )
        out_dir = os.path.join(args.run_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        rng2 = np.random.default_rng(args.seed + 1)
        sample_idxs = sorted(rng2.choice(len(ds_clean),
                             size=min(args.eigen_samples, len(ds_clean)),
                             replace=False).tolist())
        ep_tag = re.search(r"epoch_(\d+)", os.path.basename(ckpt))
        ep_tag = ep_tag.group(1) if ep_tag else "last"
        k = args.eigen_side * args.eigen_side
        for si in sample_idxs:
            item = ds_clean[si]
            view = item[0] if isinstance(item, (list, tuple)) else item
            if isinstance(view, (list, tuple)):
                view = view[0]
            view = view.unsqueeze(0)               # [1, C, H, W]
            maps, ev, fg_grid = eigen_map_grid(net, view, k - 1, device,
                                               fg_thresh=fg_thresh,
                                               outlier_pct=args.eigen_outlier_pct)
            orig = view[0].mean(0).cpu().numpy()
            fp = os.path.join(out_dir, f"eigen_grid_ep{ep_tag}_slice{si}.png")
            save_eigen_grid(maps, ev, orig, fp, args.eigen_side,
                            f"{os.path.basename(args.run_dir)}  ep{ep_tag}  slice {si}",
                            fg_grid=fg_grid)
            print(f"wrote {fp}", flush=True)


if __name__ == "__main__":
    main()
