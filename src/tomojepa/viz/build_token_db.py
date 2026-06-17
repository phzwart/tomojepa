"""Build a patch-token feature database on a shared PCA basis.

NOTE: superseded by the ``patchdb`` package (DuckDB + FAISS). Prefer
``python -m patchdb.cli build``. Kept as a standalone npz-based prototype.


Pipeline:
  1. Fit ONE robust PCA basis on foreground patch tokens pooled across a sample
     of images (same machinery as the cascade / eigen grid: L2-normalized tokens,
     foreground-only fit, directional-outlier rejection).
  2. For every image, encode its patch tokens and store the top-``K`` projections
     ("codes") on that shared basis as a ``[G, G, K]`` grid, plus the foreground
     mask. Because the basis is shared, codes are directly comparable *across*
     images -- the requirement for cross-image patch retrieval.

The result is a compact, geometry-aware representation: each image becomes a
``G x G x K`` float16 tensor (e.g. 32 x 32 x 25), and patch (i, j) maps back to
pixels ``[i*ps:(i+1)*ps, j*ps:(j+1)*ps]``. Query any rectangle of patches (any
size) and retrieve similar regions across the whole database -- see
``query_patches.py``.

Output: a single ``.npz`` holding codes, foreground, the basis (mu, Vh, ev),
dataset indices, and geometry metadata.

Usage:
    python build_token_db.py --run_dir runs/soil_residual_fg \
        --pattern soild_stack.zarr --eigen_ckpt 14 --k 25 \
        --foreground_mask --fg_std_thresh 0.05 \
        --out runs/soil_residual_fg/token_db.npz
"""
import os
import re
import argparse

import numpy as np
import torch

from ..core.dataset import TomographyDataset
from ..core.model import DINOv3ViTEncoder
from ..ssl.validate import _resolve_eigen_ckpt
from .cascade_rgb import extract_tokens, fit_shared_basis


def _get_view(ds, i):
    item = ds[i]
    view = item[0] if isinstance(item, (list, tuple)) else item
    if isinstance(view, (list, tuple)):
        view = view[0]
    return view.unsqueeze(0)                         # [1, C, H, W]


def main():
    p = argparse.ArgumentParser(description="Build shared-basis patch-token DB")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt_subdir", default="ckpt")
    p.add_argument("--eigen_ckpt", default="last")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="soild_stack.zarr")
    p.add_argument("--backend", default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_chans", type=int, default=1)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--k", type=int, default=25, help="components stored per token")
    p.add_argument("--n_fit", type=int, default=64,
                   help="images sampled to fit the shared basis")
    p.add_argument("--n_images", type=int, default=0,
                   help="images to encode (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--foreground_mask", action="store_true")
    p.add_argument("--fg_std_thresh", type=float, default=0.05)
    p.add_argument("--outlier_pct", type=float, default=2.0)
    p.add_argument("--out", default=None)
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
    n_total = len(ds)
    n_enc = n_total if args.n_images <= 0 else min(args.n_images, n_total)
    enc_ids = list(range(n_enc))

    # --- 1. fit shared basis on a sample of images ---
    rng = np.random.default_rng(args.seed)
    fit_ids = sorted(rng.choice(n_enc, size=min(args.n_fit, n_enc),
                                replace=False).tolist())
    fg_pool = []
    with torch.no_grad():
        for i in fit_ids:
            tokens, grid, fg = extract_tokens(net, _get_view(ds, i), device,
                                              fg_thresh=fg_thresh)
            fg_pool.append(tokens[fg])
    mu, Vh, ev = fit_shared_basis(torch.cat(fg_pool, 0), args.k,
                                  outlier_pct=args.outlier_pct)
    K = Vh.shape[0]
    print(f"fit shared basis on {len(fit_ids)} imgs  K={K}  "
          f"top-EV={ev[:3]*100}", flush=True)

    # --- 2. encode every image onto the shared basis ---
    G = grid
    ps = args.img_size // G
    codes = np.zeros((n_enc, G, G, K), dtype=np.float16)
    fgs = np.zeros((n_enc, G, G), dtype=bool)
    with torch.no_grad():
        for r, i in enumerate(enc_ids):
            tokens, grid_i, fg = extract_tokens(net, _get_view(ds, i), device,
                                                fg_thresh=fg_thresh)
            proj = ((tokens - mu) @ Vh.T).reshape(grid_i, grid_i, K)
            codes[r] = proj.cpu().numpy().astype(np.float16)
            fgs[r] = fg.reshape(grid_i, grid_i).cpu().numpy()
            if (r + 1) % 100 == 0 or r + 1 == n_enc:
                print(f"  encoded {r + 1}/{n_enc}", flush=True)

    out = args.out or os.path.join(args.run_dir, f"token_db_ep{ep_tag}.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        codes=codes, fg=fgs,
        basis=Vh.cpu().numpy().astype(np.float32),
        mean=mu.cpu().numpy().astype(np.float32),
        ev=ev.astype(np.float32),
        image_ids=np.array(enc_ids, dtype=np.int64),
        grid=G, patch_size=ps, img_size=args.img_size, k=K,
        pattern=args.pattern, backend=args.backend,
        dataset_key=args.dataset_key, data_dir=args.data_dir,
        ckpt=ckpt,
    )
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({n_enc} imgs x {G}x{G}x{K})  {mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
