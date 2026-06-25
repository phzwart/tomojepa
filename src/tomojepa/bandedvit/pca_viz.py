"""PCA strip visualization for BandedViT patch-token features."""
from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tomojepa.core.augmentations import pool_fg_to_stage
from tomojepa.core.dataset import TomographyDataset


def build_probe(
    data_dir: str,
    img_size: int,
    pattern: str,
    backend: str,
    dataset_key: str,
    aug_cfg=None,
    seed: int = 0,
    n_samples: int = 1,
    foreground_mask: bool = False,
    fg_mode: str = "std",
    fg_std_thresh: float = 0.05,
    fg_circle_diameter_frac: float = 1.0,
    fg_key: str = "",
) -> Tuple[List[int], TomographyDataset]:
    ds = TomographyDataset(
        data_dir=data_dir,
        dataset_key=dataset_key,
        pattern=pattern,
        img_size=img_size,
        is_train=False,
        backend=backend,
        aug_config=aug_cfg,
        probe_geom=True,
        foreground_mask=foreground_mask,
        fg_mode=fg_mode,
        fg_std_thresh=fg_std_thresh,
        fg_circle_diameter_frac=fg_circle_diameter_frac,
        fg_key=fg_key or None,
    )
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False).tolist())
    return idxs, ds


def _grid_side(n_tokens: int, fallback: int) -> int:
    side = int(round(math.sqrt(n_tokens)))
    return side if side * side == n_tokens else fallback


def _norm_tokens(flat: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """``[N, C]`` -> L2-normalized tokens and ``(h, w)`` grid side."""
    n, _ = flat.shape
    side = _grid_side(n, int(math.sqrt(n)))
    if side * side != n:
        side = int(math.ceil(math.sqrt(n)))
        # pad to square if needed (should not happen for ViT patches)
        pad = side * side - n
        if pad > 0:
            flat = torch.cat([flat, flat[-1:].expand(pad, -1)], dim=0)
    tok = flat.float()
    tok = tok / tok.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return tok, side, side


def _fg_patch_flat(
    fg_px: Optional[torch.Tensor], grid_hw: Tuple[int, int], coverage: float,
) -> Optional[torch.Tensor]:
    """Pixel FG mask -> flat bool ``[N]`` on the patch grid."""
    if fg_px is None:
        return None
    if fg_px.dim() == 2:
        fg_px = fg_px.unsqueeze(0)
    if fg_px.dim() == 3:
        fg_px = fg_px.unsqueeze(0)
    fg = pool_fg_to_stage(fg_px.float(), grid_hw, coverage)
    return fg.reshape(-1).bool()


def _mask_bg_maps(
    maps: Sequence[np.ndarray], fg_flat: Optional[torch.Tensor], h: int, w: int,
) -> List[np.ndarray]:
    if fg_flat is None:
        return list(maps)
    bg = ~fg_flat.reshape(h, w).numpy()
    return [np.where(bg, 0.0, m).astype(np.float32) for m in maps]


def _norm_fg_percentile(arr: np.ndarray, fg_hw: Optional[np.ndarray]) -> np.ndarray:
    vals = arr[fg_hw] if fg_hw is not None else arr.ravel()
    if vals.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(vals, [1, 99])
    out = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1) if hi > lo else np.zeros_like(arr)
    if fg_hw is not None:
        out = np.where(fg_hw, out, 0.0)
    return out.astype(np.float32)


def _pca_basis(
    tok: torch.Tensor,
    n_terms: int,
    max_tokens: int,
    fg_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    fit = tok[fg_mask] if fg_mask is not None else tok
    if fit.shape[0] == 0:
        return None
    q = min(n_terms, fit.shape[0], fit.shape[1])
    if q <= 0:
        return None
    if fit.shape[0] > max_tokens:
        fit = fit[torch.randperm(fit.shape[0])[:max_tokens]]
    _, _, v = torch.pca_lowrank(fit, q=q)
    return v[:, :q]


def _pca_term_maps(
    flat: torch.Tensor,
    n_terms: int = 9,
    max_tokens: int = 4096,
    basis: Optional[torch.Tensor] = None,
    fg_flat: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], int, int]:
    """Grayscale PCA maps from flat patch tokens ``[N, C]``."""
    tok, h, w = _norm_tokens(flat)
    q = min(n_terms, tok.shape[0], tok.shape[1])
    blank = np.zeros((h, w), dtype=np.float32)
    if basis is None:
        basis = _pca_basis(tok, q, max_tokens, fg_mask=fg_flat)
    if basis is None:
        return [blank] * n_terms, h, w
    q_use = min(n_terms, basis.shape[1])
    pcs = (tok @ basis[:, :q_use]).reshape(h, w, q_use).permute(2, 0, 1)
    fg_hw = fg_flat.reshape(h, w).numpy() if fg_flat is not None else None
    maps: List[np.ndarray] = []
    for i in range(q):
        maps.append(_norm_fg_percentile(pcs[i].numpy(), fg_hw))
    maps.extend([blank] * (n_terms - len(maps)))
    return _mask_bg_maps(maps, fg_flat, h, w), h, w


def _pca_rgb(
    flat: torch.Tensor,
    basis: Optional[torch.Tensor] = None,
    max_tokens: int = 4096,
    fg_flat: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, int, int]:
    tok, h, w = _norm_tokens(flat)
    q = min(3, tok.shape[0], tok.shape[1])
    if q <= 0:
        return np.zeros((h, w, 3), dtype=np.float32), h, w
    if basis is None:
        basis = _pca_basis(tok, q, max_tokens, fg_mask=fg_flat)
    if basis is None:
        return np.zeros((h, w, 3), dtype=np.float32), h, w
    pcs = (tok @ basis[:, :3]).reshape(h, w, 3).numpy()
    fg_hw = fg_flat.reshape(h, w).numpy() if fg_flat is not None else None
    for ch in range(3):
        pcs[..., ch] = _norm_fg_percentile(pcs[..., ch], fg_hw)
    return pcs.astype(np.float32), h, w


def _pool_input_gray(
    img: np.ndarray,
    gh: int,
    gw: int,
    fg_px: Optional[torch.Tensor] = None,
    fg_coverage: float = 0.01,
) -> np.ndarray:
    t = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(t, (gh, gw)).squeeze().numpy()
    fg_flat = _fg_patch_flat(fg_px, (gh, gw), fg_coverage)
    if fg_flat is not None:
        pooled = np.where(fg_flat.reshape(gh, gw).numpy(), pooled, 0.0)
    return pooled


@torch.no_grad()
def run_pca_strip(
    model,
    probe: Tuple[Sequence[int], TomographyDataset],
    step: int,
    out_dir: Path,
    device: torch.device,
    n_terms: int = 9,
    max_tokens: int = 4096,
    fg_coverage: float = 0.01,
) -> Path:
    """Render input | RGB | PC1..PCn for each probe slice."""
    idxs, ds = probe
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    use_fg = bool(getattr(ds, "foreground_mask", False))

    imgs: List[torch.Tensor] = []
    fg_masks: List[Optional[torch.Tensor]] = []
    token_rows: List[torch.Tensor] = []
    for i in idxs:
        view, fg = ds[int(i)]
        imgs.append(view)
        fg_px = fg if use_fg and isinstance(fg, torch.Tensor) else None
        fg_masks.append(fg_px)
        x = view.unsqueeze(0).to(device)
        tok = model.extract_patch_tokens(x).float().cpu()[0]  # [N, C]
        token_rows.append(tok)

    grid_side = int(round(math.sqrt(token_rows[0].shape[0])))
    if grid_side * grid_side != token_rows[0].shape[0]:
        raise ValueError(
            f"expected square patch grid, got {token_rows[0].shape[0]} tokens"
        )
    grid_hw = (grid_side, grid_side)
    fg_flats: List[Optional[torch.Tensor]] = [
        _fg_patch_flat(fg_px, grid_hw, fg_coverage) for fg_px in fg_masks
    ]

    # Shared PCA basis across probe slices for stable comparison (FG tokens only)
    pooled = torch.cat(token_rows, dim=0)
    pooled, _, _ = _norm_tokens(pooled)
    pooled_fg = torch.cat(fg_flats, dim=0) if use_fg else None
    q_fit = min(n_terms, pooled.shape[0], pooled.shape[1])
    shared_basis = _pca_basis(pooled, q_fit, max_tokens, fg_mask=pooled_fg)

    ncol = 2 + n_terms  # input + RGB + PC bands
    n = len(imgs)
    fig, axes = plt.subplots(n, ncol, figsize=(2.2 * ncol, 2.4 * n), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05, hspace=0.05)
    title = f"BandedViT PCA @ step {step}"
    if use_fg:
        title += " (FG only)"
    fig.suptitle(title, fontsize=11)

    for r, (sl_idx, im, tok, fg_px, fg_flat) in enumerate(
        zip(idxs, imgs, token_rows, fg_masks, fg_flats)
    ):
        terms, gh, gw = _pca_term_maps(
            tok, n_terms=n_terms, max_tokens=max_tokens, basis=shared_basis, fg_flat=fg_flat,
        )
        rgb, _, _ = _pca_rgb(tok, basis=shared_basis, max_tokens=max_tokens, fg_flat=fg_flat)

        gray = im[0].numpy()
        axes[r, 0].imshow(
            _pool_input_gray(gray, gh, gw, fg_px=fg_px, fg_coverage=fg_coverage),
            cmap="gray", aspect="equal",
        )
        axes[r, 0].set_title(f"input\nslice {sl_idx}", fontsize=8)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(rgb, aspect="equal")
        axes[r, 1].set_title("RGB\nPC1-3", fontsize=8)
        axes[r, 1].axis("off")

        for c in range(n_terms):
            ax = axes[r, c + 2]
            ax.imshow(terms[c], cmap="gray", aspect="equal", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"PC{c + 1}", fontsize=8)
            ax.axis("off")

    path = out_dir / f"pca_step{step:06d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if was_training:
        model.train()
    return path
