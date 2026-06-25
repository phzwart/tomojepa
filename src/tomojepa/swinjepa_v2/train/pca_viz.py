"""PCA strip visualization for HL-JEPA encode() band pyramid."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tomojepa.core.dataset import TomographyDataset


def build_probe(data_dir: str, img_size: int, pattern: str, backend: str,
                dataset_key: str, seed: int = 0, n_samples: int = 4) -> Tuple[List[int], TomographyDataset]:
    ds = TomographyDataset(
        data_dir=data_dir,
        dataset_key=dataset_key,
        pattern=pattern,
        img_size=img_size,
        is_train=False,
        backend=backend,
        crop_mode="resize",
        probe_geom=True,
        global_views=1,
        local_views=0,
    )
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False).tolist())
    return idxs, ds


def _norm_tokens(feat: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    c, h, w = feat.shape
    tok = feat.reshape(c, h * w).T.float()
    return tok / tok.norm(dim=-1, keepdim=True).clamp(min=1e-6), h, w


def _pca_rgb(feat: torch.Tensor, basis: Optional[torch.Tensor] = None,
             max_tokens: int = 4096) -> np.ndarray:
    tok, h, w = _norm_tokens(feat)
    q = min(3, tok.shape[0], tok.shape[1])
    if q <= 0:
        return np.zeros((h, w, 3), dtype=np.float32)
    if basis is None:
        fit = tok
        if tok.shape[0] > max_tokens:
            fit = tok[torch.randperm(tok.shape[0])[:max_tokens]]
        _, _, v = torch.pca_lowrank(fit, q=q)
        basis = v[:, :3]
    pcs = (tok @ basis[:, :3]).reshape(h, w, 3).numpy()
    for ch in range(3):
        lo, hi = np.percentile(pcs[..., ch], [1, 99])
        pcs[..., ch] = np.clip((pcs[..., ch] - lo) / (hi - lo + 1e-8), 0, 1)
    return pcs.astype(np.float32)


def _pool_gray(img: np.ndarray, gh: int, gw: int) -> np.ndarray:
    t = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return F.adaptive_avg_pool2d(t, (gh, gw)).squeeze().numpy()


@torch.no_grad()
def run_pca_strip(model, probe, step: int, out_dir: Path, device: torch.device,
                  max_tokens: int = 4096) -> Path:
    idxs, ds = probe
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()

    imgs = []
    for i in idxs:
        view, _ = ds[int(i)]
        imgs.append(view)

    bands_per_row: List[List[torch.Tensor]] = []
    for im in imgs:
        x = im.unsqueeze(0).to(device)
        bands = model.encode(x)
        bands_per_row.append([b[0].detach().cpu() for b in bands])

    n_bands = len(bands_per_row[0])
    shared_basis = []
    for bi in range(n_bands):
        pooled = torch.cat([_norm_tokens(row[bi])[0] for row in bands_per_row], dim=0)
        q = min(3, pooled.shape[0], pooled.shape[1])
        if q <= 0:
            shared_basis.append(None)
        else:
            fit = pooled
            if pooled.shape[0] > max_tokens:
                fit = pooled[torch.randperm(pooled.shape[0])[:max_tokens]]
            _, _, v = torch.pca_lowrank(fit, q=q)
            shared_basis.append(v[:, :3])

    ncol = 1 + n_bands
    n = len(imgs)
    fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 3.2 * n), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04, hspace=0.04)
    fig.suptitle(f"HL-JEPA bands @ step {step}", fontsize=11)

    for r, (sl_idx, im) in enumerate(zip(idxs, imgs)):
        gray = im[0].numpy()
        gh, gw = bands_per_row[r][0].shape[-2:]
        axes[r, 0].imshow(_pool_gray(gray, gh, gw), cmap="gray", aspect="equal")
        axes[r, 0].set_title(f"slice {sl_idx}", fontsize=8)
        axes[r, 0].axis("off")
        for bi in range(n_bands):
            feat = bands_per_row[r][bi]
            rgb = _pca_rgb(feat, basis=shared_basis[bi], max_tokens=max_tokens)
            axes[r, bi + 1].imshow(rgb, aspect="equal")
            axes[r, bi + 1].set_title(f"band{bi} {feat.shape[-1]}x{feat.shape[-2]}", fontsize=8)
            axes[r, bi + 1].axis("off")

    path = out_dir / f"pca_step{step:06d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if was_training:
        model.train()
    return path
