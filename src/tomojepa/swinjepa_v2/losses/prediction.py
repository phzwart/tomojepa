"""Per-band masked prediction loss (spec §7.1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import LossCfg


def prediction_loss(
    pred_band: torch.Tensor,
    tgt_band: torch.Tensor,
    mask_band: torch.Tensor,
    cfg: LossCfg,
) -> torch.Tensor:
    """Scalar loss at masked positions only."""
    tgt = tgt_band.detach() if cfg.stop_grad_target else tgt_band
    b, c = pred_band.shape[:2]
    pred_t = pred_band.reshape(b, c, -1).permute(0, 2, 1)
    tgt_t = tgt.reshape(b, c, -1).permute(0, 2, 1)
    m = mask_band.reshape(b, -1)
    parts = []
    for bi in range(b):
        sel = m[bi]
        if int(sel.sum()) == 0:
            continue
        pred_sel = pred_t[bi, sel]
        tgt_sel = tgt_t[bi, sel]
        if cfg.pred_l2norm:
            pred_sel = F.normalize(pred_sel, dim=-1)
            tgt_sel = F.normalize(tgt_sel, dim=-1)
        if cfg.pred_loss == "cosine":
            parts.append((1.0 - F.cosine_similarity(pred_sel, tgt_sel, dim=-1)).mean())
        else:
            parts.append(F.smooth_l1_loss(pred_sel, tgt_sel))
    if not parts:
        return pred_band.sum() * 0.0
    return torch.stack(parts).mean()
