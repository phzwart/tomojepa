"""PCA visualization of ViT-Up upsampled features vs. the low-res backbone.

Loads a trained ViT-Up checkpoint, runs it on a random set of slices, and for
each renders a 3-panel comparison:

  1. the input slice,
  2. the low-resolution backbone (layer-L) feature PCA, bilinearly upsampled,
  3. the ViT-Up dense feature PCA at high resolution.

A single PCA basis (top-3 components, fit on the ViT-Up dense features) and a
single per-channel color normalization are applied to both feature panels, so
the comparison is apples-to-apples: same feature space, low-res vs. ViT-Up.

Example:
    python vitup_pca.py --ckpt runs/vitup_soil_1024/ckpt/ckpt_last.pth \
        --data_dir . --pattern 'soild_stack.zarr' --backend zarr \
        --num_samples 6 --backbone_res 512 --upsample 512
"""
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.dataset import TomographyDataset
from .config import ViTUpConfig
from .backbone_adapter import build_backbone, BackboneAdapter
from .model import ViTUp


def parse_args():
    p = argparse.ArgumentParser(description="ViT-Up PCA visualization")
    p.add_argument("--ckpt", default="runs/vitup_soil_1024/ckpt/ckpt_last.pth")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="soild_stack.zarr")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--slices", type=int, nargs="+", default=None,
                   help="explicit slice indices (overrides --num_samples random draw)")
    p.add_argument("--backbone_res", type=int, default=512,
                   help="resolution fed to the backbone (low-res token grid = res/p)")
    p.add_argument("--upsample", type=int, default=512,
                   help="ViT-Up dense output resolution (h*, w*)")
    p.add_argument("--query_chunk_size", type=int, default=32768)
    p.add_argument("--n_comp", type=int, default=18,
                   help="number of PCA components to visualize (first 3 form RGB)")
    p.add_argument("--comp_cols", type=int, default=0,
                   help="columns in the component grid (0 = auto: 10 if n_comp>24 else 6)")
    # tiled (native-crop) inference: run the model on overlapping native windows
    # and stitch (qlty), instead of feeding the whole resized slice.
    p.add_argument("--tiled", action="store_true",
                   help="tiled inference: slide a native --tile_window over the "
                        "full-resolution slice and stitch (for models trained on "
                        "native crops). Ignores --backbone_res/--upsample.")
    p.add_argument("--full_res", type=int, default=1024,
                   help="full-resolution slice size used in --tiled mode")
    p.add_argument("--tile_window", type=int, default=256, help="tile size (px)")
    p.add_argument("--tile_step", type=int, default=192, help="tile stride (px)")
    p.add_argument("--tile_border", type=int, default=32,
                   help="downweighted border per tile (px)")
    p.add_argument("--out_dir", default="runs/vitup_soil_1024/out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_vitup(ckpt_path, device):
    """Rebuild ViT-Up with default architecture and load its trained weights."""
    cfg = ViTUpConfig()
    bb = build_backbone(cfg.backbone_name, cfg.input_channels, cfg.backbone_img_size)
    adapter = BackboneAdapter(bb)
    adapter.apply_lora(cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    vitup = ViTUp(adapter, cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("engine", ckpt)
    vsd = {k[len("vitup."):]: v for k, v in sd.items() if k.startswith("vitup.")}
    missing, unexpected = vitup.load_state_dict(vsd, strict=False)
    real_missing = [m for m in missing if "lora_" not in m]
    if real_missing:
        print(f"[load] missing: {real_missing[:6]}", flush=True)
    return vitup.to(device).eval(), cfg


def pca_maps(dense, low, n_comp=18):
    """Shared-basis PCA for ViT-Up dense and low-res feature maps.

    Args:
        dense: ``[Hd, Wd, C]`` ViT-Up features.
        low:   ``[Hl, Wl, C]`` backbone features.
        n_comp: number of components to extract (first 3 form the RGB).
    Returns ``(dense_rgb, low_rgb, comps)`` where ``dense_rgb``/``low_rgb`` are
    ``[Hd,Wd,3]`` (common basis + color normalization, ``low`` bilinearly
    upsampled), and ``comps`` is a list of ``n_comp`` single-channel ``[Hd,Wd]``
    maps of the ViT-Up dense components (each normalized to [0,1]).
    """
    Hd, Wd, C = dense.shape
    d = dense.reshape(-1, C).float()
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    _, _, v = torch.pca_lowrank(d, q=n_comp)
    basis = v[:, :n_comp]

    def project(x_hwc, k):
        h, w, c = x_hwc.shape
        x = x_hwc.reshape(-1, c).float()
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x @ basis[:, :k]).reshape(h, w, k)

    dense_p = project(dense, n_comp)                                  # [Hd,Wd,n_comp]
    low_p = project(low, 3)                                           # RGB only
    low_p = F.interpolate(low_p.permute(2, 0, 1)[None], size=(Hd, Wd),
                          mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    dense_np = dense_p.cpu().numpy()
    low_np = low_p.cpu().numpy()

    # RGB: shared per-channel normalization from the dense projection (1-99 pct)
    dn, ln = np.empty(dense_np.shape[:2] + (3,)), np.empty_like(low_np)
    for c in range(3):
        lo, hi = np.percentile(dense_np[..., c], [1, 99])
        rng = (hi - lo) if hi > lo else 1.0
        dn[..., c] = np.clip((dense_np[..., c] - lo) / rng, 0, 1)
        ln[..., c] = np.clip((low_np[..., c] - lo) / rng, 0, 1)

    # individual components: each normalized by its own 1-99 percentile
    comps = []
    for c in range(n_comp):
        lo, hi = np.percentile(dense_np[..., c], [1, 99])
        rng = (hi - lo) if hi > lo else 1.0
        comps.append(np.clip((dense_np[..., c] - lo) / rng, 0, 1))
    return dn, ln, comps


@torch.no_grad()
def _tile_dense_low(vitup, cfg, tile, out_h, out_w, chunk_size, want_low):
    """Dense ViT-Up map (and optional last-layer backbone grid) for one tile."""
    ctx = vitup.encode_image(tile)
    coords = vitup.dense_grid_coords(ctx, out_h, out_w, device=tile.device).unsqueeze(0)
    o_last = vitup.query(ctx, coords, stages="last", chunk_size=chunk_size)[-1]
    dense = o_last.reshape(out_h, out_w, vitup.output_dim)
    low = None
    if want_low:
        low = ctx.hidden[cfg.layer_indices[-1]][0].permute(1, 2, 0)   # [g,g,C]
    return dense, low


@torch.no_grad()
def tiled_upsample(vitup, cfg, img, window=256, step=192, border=32,
                   border_weight=0.1, chunk_size=4096, amp=True, want_low=False):
    """Dense ViT-Up features over a large image via overlapping native tiles.

    The model is run on each ``window``-sized native crop (its training scale),
    and the per-tile dense maps are stitched with qlty's border-downweighted
    weighted average -- avoiding the large position-embedding extrapolation that
    feeding the whole image would require.

    Args:
        img: ``[1, Cin, H, W]`` tensor on the target device.
        window/step/border: tile geometry in pixels (``step < window`` overlaps).
        want_low: also return the stitched last-layer backbone grid (token res).
    Returns ``(dense [H,W,Cf], low [H/p,W/p,C] | None)``.
    """
    from qlty import NCYXQuilt

    device = img.device
    _, _, H, W = img.shape
    p = vitup.adapter.p
    quilt = NCYXQuilt(Y=H, X=W, window=(window, window), step=(step, step),
                      border=(border, border), border_weight=border_weight)
    tiles = quilt.unstitch(img.detach().cpu())                 # [M,Cin,window,window]
    aenabled = amp and device.type == "cuda"

    dense_tiles, low_tiles = [], []
    for i in range(tiles.shape[0]):
        t = tiles[i:i + 1].to(device)
        with autocast(device.type, dtype=torch.bfloat16, enabled=aenabled):
            d, lo = _tile_dense_low(vitup, cfg, t, window, window, chunk_size, want_low)
        dense_tiles.append(d.float().permute(2, 0, 1).cpu())   # [Cf,window,window]
        if want_low:
            low_tiles.append(lo.float().permute(2, 0, 1).cpu())

    dense_full, _ = quilt.stitch(torch.stack(dense_tiles))     # [1,Cf,H,W]
    dense_full = dense_full[0].permute(1, 2, 0).contiguous()   # [H,W,Cf]
    if not want_low:
        return dense_full, None

    # Stitch the low-res backbone grids on the matching token-scale quilt.
    g = window // p
    lquilt = NCYXQuilt(Y=H // p, X=W // p, window=(g, g),
                       step=(step // p, step // p),
                       border=(max(1, border // p), max(1, border // p)),
                       border_weight=border_weight)
    low_full, _ = lquilt.stitch(torch.stack(low_tiles))        # [1,C,H/p,W/p]
    return dense_full, low_full[0].permute(1, 2, 0).contiguous()


def _component_grid(n_comp, comp_cols=0):
    """Return (n_rows, n_cols) for the eigen component panel."""
    cols = comp_cols if comp_cols > 0 else (10 if n_comp > 24 else 6)
    rows = (n_comp + cols - 1) // cols
    return rows, cols


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    vitup, cfg = load_vitup(args.ckpt, device)

    ds_img_size = args.full_res if args.tiled else max(args.backbone_res, args.upsample)
    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant="tomo2",
        img_size=ds_img_size, is_train=False, backend=args.backend,
    )
    if args.tiled:
        print(f"Tiled inference: {args.full_res}px slice, window={args.tile_window} "
              f"step={args.tile_step} border={args.tile_border}", flush=True)
    rng = np.random.default_rng(args.seed)
    if args.slices is not None:
        idxs = np.array(args.slices, dtype=int)
        print(f"Visualizing {len(idxs)} slices: {idxs.tolist()}", flush=True)
    else:
        idxs = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)
        print(f"Visualizing {len(idxs)} random slices: {sorted(idxs.tolist())}", flush=True)

    for i, idx in enumerate(idxs):
        img = ds[int(idx)][0].unsqueeze(0).to(device)              # [1,1,S,S]
        use_amp = device.type == "cuda"
        if args.tiled:
            dense, low = tiled_upsample(
                vitup, cfg, img, window=args.tile_window, step=args.tile_step,
                border=args.tile_border, chunk_size=args.query_chunk_size,
                amp=use_amp, want_low=True)
            orig = img[0, 0].float().cpu().numpy()
        else:
            img_in = F.interpolate(img, size=(args.backbone_res, args.backbone_res),
                                   mode="bilinear", align_corners=False)
            with autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                ctx = vitup.encode_image(img_in)
                low = ctx.hidden[cfg.layer_indices[-1]][0].permute(1, 2, 0)  # [h,w,C]
                dense = vitup.upsample(img_in, args.upsample, args.upsample,
                                       chunk_size=args.query_chunk_size)[0]  # [U,U,C]
            orig = img_in[0, 0].cpu().numpy()
        low, dense = low.float(), dense.float()

        dense_rgb, low_rgb, comps = pca_maps(dense, low, n_comp=args.n_comp)
        h = low.shape[0]

        comp_rows, comp_cols = _component_grid(args.n_comp, args.comp_cols)
        # top row: RGB triptych; below: individual ViT-Up PCA components.
        fig_w = max(18, comp_cols * 2.2)
        fig_h = 3.5 + comp_rows * 2.4
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            1 + comp_rows, comp_cols,
            height_ratios=[1.4] + [1.0] * comp_rows,
            hspace=0.15, wspace=0.06,
        )

        in_res = orig.shape[0]
        out_res = dense_rgb.shape[0]
        in_lbl = (f"{in_res}px, tiled {args.tile_window}/{args.tile_step}"
                  if args.tiled else f"{args.backbone_res}px in")
        t1, t2 = comp_cols // 3, 2 * (comp_cols // 3)
        ax = fig.add_subplot(gs[0, 0:t1]); ax.imshow(orig, cmap="gray")
        ax.set_title(f"slice {int(idx)}  ({in_lbl})"); ax.axis("off")
        ax = fig.add_subplot(gs[0, t1:t2]); ax.imshow(low_rgb)
        ax.set_title(f"backbone layer {cfg.layer_indices[-1]} PCA "
                     f"({h}x{h} -> bilinear)"); ax.axis("off")
        ax = fig.add_subplot(gs[0, t2:comp_cols]); ax.imshow(dense_rgb)
        ax.set_title(f"ViT-Up PCA RGB ({out_res}x{out_res})"); ax.axis("off")

        for c, comp in enumerate(comps):
            ax = fig.add_subplot(gs[1 + c // comp_cols, c % comp_cols])
            ax.imshow(comp, cmap="viridis")
            ax.set_title(f"PC {c + 1}", fontsize=8); ax.axis("off")

        out = os.path.join(args.out_dir,
                           f"vitup_pca_{i}_slice{int(idx)}_pc{args.n_comp}.png")
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
