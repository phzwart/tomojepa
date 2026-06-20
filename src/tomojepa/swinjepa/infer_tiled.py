"""Tiled Swin-MSJEPA inference + single-image PCA on a large slice.

Slides overlapping native ``img_size`` windows over a downsampled full slice,
stitches per-stage token maps with qlty, and renders a PCA grid at native token
resolution (matching :mod:`tomojepa.swinjepa.train` viz).

Example:
    python -m tomojepa.swinjepa.infer_tiled \\
        --ckpt runs/petiole_zoomed2x_cropped224/ckpt/ckpt_last.pth \\
        --data_dir . --pattern 'petiole.zarr' --downsample 2
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from ..core.augmentations import build_fg_mask
from ..core.dataset import TomographyDataset
from ..core.tiling import pixel_quilt, token_quilt, unstitch_tiles, stitch_maps
from .config import SwinMSJEPAConfig
from .model import SwinMSJEPA
from .train import _pca_stage_rgb, _pool_img_to_grid


def parse_args():
    p = argparse.ArgumentParser(description="Tiled Swin-MSJEPA PCA on one slice")
    p.add_argument("--ckpt", default="runs/petiole_zoomed2x_cropped224/ckpt/ckpt_last.pth")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="petiole.zarr")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--slice_idx", type=int, default=None,
                   help="slice index (default: random)")
    p.add_argument("--downsample", type=int, default=2,
                   help="integer factor applied to native slice before tiling")
    p.add_argument("--tile_window", type=int, default=224)
    p.add_argument("--tile_overlap", type=int, default=64,
                   help="overlap between 224px tiles (step = window - overlap)")
    p.add_argument("--tile_border", type=int, default=32,
                   help="qlty border downweight (px); default half of overlap")
    p.add_argument("--border_weight", type=float, default=0.1)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--out", default=None,
                   help="output PNG (default: <run>/out/pca_tiled/pca_tiled_slice<N>.png)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> SwinMSJEPA:
    cfg = SwinMSJEPAConfig(
        img_size=224,
        foreground_mask=True,
        fg_mode="circle",
        fg_circle_diameter_frac=1.0,
        legacy_jepa=False,
    )
    model = SwinMSJEPA(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval(), cfg


@torch.no_grad()
def _tile_decomp(model: SwinMSJEPA, tile: torch.Tensor, fg_tile: torch.Tensor | None):
    """Pyramid probe for one ``[1,1,H,W]`` tile; return CPU stage maps."""
    decomp = model.extract_pyramid_probe(tile, fg_px=fg_tile)
    out = {
        "C4": decomp["C4"]["s4"][0].float().cpu(),
        "R": {k: decomp["R"][k][0].float().cpu() for k in ("s1", "s2", "s3")},
        "E": {k: decomp["E"][k][0].float().cpu() for k in model.stage_keys},
    }
    return out


@torch.no_grad()
def tiled_pyramid_probe(
    model: SwinMSJEPA,
    cfg: SwinMSJEPAConfig,
    img: torch.Tensor,
    fg_full: torch.Tensor | None = None,
    *,
    window: int = 224,
    step: int = 160,
    border: int = 32,
    border_weight: float = 0.1,
    amp: bool = True,
) -> dict:
    """Run ``extract_pyramid_probe`` on overlapping tiles; stitch token grids.

    ``fg_full`` must be built once on the full downsampled slice ``[1,H,W]``
    and is unstitched with the same quilt as ``img`` (never per-tile).
    """
    device = img.device
    _, _, H, W = img.shape
    strides = model.backbone.strides
    keys = model.stage_keys

    quilt = pixel_quilt(H, W, window, step, border, border_weight)
    tiles = unstitch_tiles(quilt, img)
    fg_tiles = unstitch_tiles(quilt, fg_full.unsqueeze(0)) if fg_full is not None else None
    aenabled = amp and device.type == "cuda"

    tile_maps: dict[str, list[torch.Tensor]] = {
        "C4": [], **{f"R{k}": [] for k in ("1", "2", "3")},
        **{f"E{k}": [] for k in keys},
    }

    for i in range(tiles.shape[0]):
        t = tiles[i:i + 1].to(device)
        fg = fg_tiles[i:i + 1].to(device) if fg_tiles is not None else None
        with autocast(device.type, dtype=torch.bfloat16, enabled=aenabled):
            d = _tile_decomp(model, t, fg)
        tile_maps["C4"].append(d["C4"])
        for rk in ("s1", "s2", "s3"):
            tile_maps[f"R{rk[1]}"].append(d["R"][rk])
        for key in keys:
            tile_maps[f"E{key}"].append(d["E"][key])

    def _stitch(name: str, stride: int) -> torch.Tensor:
        lq = token_quilt(H, W, window, step, border, stride, border_weight)
        return stitch_maps(lq, tile_maps[name]).cpu()

    stitched: dict[str, torch.Tensor] = {"C4": _stitch("C4", strides[3])}
    for rk in ("s1", "s2", "s3"):
        stitched[f"R{rk[1]}"] = _stitch(f"R{rk[1]}", strides[int(rk[1]) - 1])
    for si, key in enumerate(keys):
        stitched[f"E{key}"] = _stitch(f"E{key}", strides[si])
    return stitched


def _save_pca_figure(img_np, feats: dict, out_path: str, slice_idx: int,
                     grid_hw: tuple[int, int], max_tokens: int, downsample: int):
    """Single-image PCA strip: slice + C4 + R1..R3 + Es1..Es4."""
    gh, gw = grid_hw
    slice_grid = _pool_img_to_grid(img_np, gh, gw)

    panels = [("slice", slice_grid, f"slice {slice_idx} ({gh}x{gw})")]
    if "C4" in feats:
        panels.append(("C4", _pca_stage_rgb(feats["C4"], max_tokens), "C4"))
    for rk in ("1", "2", "3"):
        k = f"R{rk}"
        if k in feats:
            panels.append((k, _pca_stage_rgb(feats[k], max_tokens), k))
    for sk in ("s1", "s2", "s3", "s4"):
        k = f"E{sk}"
        if k in feats:
            h, w = feats[k].shape[-2:]
            panels.append((k, _pca_stage_rgb(feats[k], max_tokens), f"E{sk} ({h}x{w})"))

    n = len(panels)
    cell = 3.2
    fig, axes = plt.subplots(1, n, figsize=(cell * n + 0.2, cell + 0.3))
    if n == 1:
        axes = [axes]
    for ax, (_, rgb, title) in zip(axes, panels):
        if rgb.ndim == 2:
            ax.imshow(rgb, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"tiled PCA (downsample x{downsample}, stitch)", fontsize=10)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.88, bottom=0.002, wspace=0.02)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tiled_pca] saved {out_path}", flush=True)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.ckpt, device)
    window = args.tile_window if args.tile_window > 0 else cfg.img_size
    overlap = max(0, min(args.tile_overlap, window - 1))
    step = window - overlap
    border = args.tile_border if args.tile_border > 0 else max(1, overlap // 2)

    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant="tomo2",
        img_size=2048, is_train=False, backend=args.backend,
        crop_mode="resized",
    )
    if args.slice_idx is not None:
        idx = args.slice_idx
    else:
        idx = int(np.random.default_rng(args.seed).integers(len(ds)))
    native = ds[int(idx)][0]
    _, H0, W0 = native.shape
    ds_factor = max(1, args.downsample)
    new_h, new_w = H0 // ds_factor, W0 // ds_factor
    img = F.interpolate(
        native.unsqueeze(0), size=(new_h, new_w),
        mode="area" if ds_factor > 1 else "bilinear", antialias=False,
    ).to(device)
    img_np = img[0, 0].float().cpu().numpy()

    fg_full = None
    if cfg.foreground_mask:
        fg_full = build_fg_mask(
            img[0].cpu(), cfg.fg_mode, cfg.fg_std_thresh, cfg.fg_circle_diameter_frac,
        )

    print(
        f"Tiled inference: slice {idx}, native {H0}x{W0} -> {new_h}x{new_w} "
        f"(/{ds_factor}), window={window} overlap={overlap} step={step} border={border}",
        flush=True,
    )

    stitched = tiled_pyramid_probe(
        model, cfg, img, fg_full,
        window=window, step=step, border=border,
        border_weight=args.border_weight, amp=device.type == "cuda",
    )

    s1_h, s1_w = stitched["Es1"].shape[-2:]
    out = args.out
    if out is None:
        run_dir = os.path.dirname(os.path.dirname(args.ckpt))
        out = os.path.join(run_dir, "out", "pca_tiled", f"pca_tiled_slice{idx:04d}.png")

    _save_pca_figure(img_np, stitched, out, idx, (s1_h, s1_w),
                     args.max_tokens, args.downsample)


if __name__ == "__main__":
    main()
