"""Per-band training diagnostics (spec §10)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..data.augment import GeomParams, apply_latent_geom
from ..nd import flatten_tokens
from ..models.model import SwinJEPA


def effective_rank(z: torch.Tensor, max_tokens: int = 4096) -> float:
    z = z.float()
    if z.shape[0] > max_tokens:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_tokens]
        z = z[idx]
    z = z - z.mean(0, keepdim=True)
    if z.shape[0] < 2:
        return 0.0
    sv = torch.linalg.svdvals(z)
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return 0.0
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.log()).sum()).item())


def masked_r2(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    b, c = pred.shape[:2]
    pred_t = pred.reshape(b, c, -1).permute(0, 2, 1)
    tgt_t = tgt.reshape(b, c, -1).permute(0, 2, 1)
    m = mask.reshape(b, -1)
    vals = []
    for bi in range(b):
        sel = m[bi]
        if int(sel.sum()) < 2:
            continue
        p = pred_t[bi, sel]
        t = tgt_t[bi, sel]
        resid = t - p
        var_t = t.var()
        if var_t < 1e-8:
            continue
        vals.append(float((1.0 - resid.var() / var_t).item()))
    return sum(vals) / max(1, len(vals))


def equivariance_error(
    model: SwinJEPA,
    x: torch.Tensor,
    geom: GeomParams,
) -> Dict[str, float]:
    """Relative error between encode(g(x)) and g_latent(encode(x)) per band."""
    from ..data.augment import _apply_geom_2d

    with torch.no_grad():
        enc = model.encode(x.unsqueeze(0) if x.ndim == 3 else x)
        img = x.unsqueeze(0) if x.ndim == 3 else x
        c, h, w = img.shape[1], img.shape[2], img.shape[3]
        geom2 = GeomParams()
        geom2._scale = (1.0, 1.0)  # type: ignore[attr-defined]
        geom2.crop_y0, geom2.crop_x0 = 0, 0
        geom2.crop_h, geom2.crop_w = h, w
        geom2.rot_k, geom2.hflip, geom2.vflip = geom.rot_k, geom.hflip, geom.vflip
        warped = _apply_geom_2d(img[0], geom2, h)
        enc_warp = model.encode(warped.unsqueeze(0))
        out = {}
        for i, (a, b) in enumerate(zip(enc, enc_warp)):
            b_lat = apply_latent_geom(b, geom)
            num = (a - b_lat).pow(2).mean().sqrt()
            den = a.pow(2).mean().sqrt().clamp(min=1e-6)
            out[f"equiv/band{i}"] = float((num / den).item())
        return out


@torch.no_grad()
def collect_metrics(
    model: SwinJEPA,
    pred_bands: List[torch.Tensor],
    tgt_bands: List[torch.Tensor],
    band_masks: List[torch.Tensor],
    sigreg_modules,
    step: int = 0,
    x: Optional[torch.Tensor] = None,
    geom: Optional[GeomParams] = None,
) -> Dict[str, float]:
    logs: Dict[str, float] = {}
    for k, (pred, tgt, mask) in enumerate(zip(pred_bands, tgt_bands, band_masks)):
        z = flatten_tokens(tgt)
        logs[f"effrank/band{k}"] = effective_rank(z)
        logs[f"r2/band{k}"] = masked_r2(pred, tgt, mask)
        logs[f"sigstat/band{k}"] = float(sigreg_modules[k](z, step).item())
    if x is not None and geom is not None:
        logs.update(equivariance_error(model, x, geom))
    return logs
