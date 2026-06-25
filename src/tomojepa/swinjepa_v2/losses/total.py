"""Weighted total loss across bands (spec §7.3)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..config import HLJEPAConfig
from ..nd import flatten_tokens, pool_image
from .prediction import prediction_loss
from .sigreg import SIGReg


def total_loss(
    pred_bands: List[torch.Tensor],
    tgt_bands: List[torch.Tensor],
    band_masks: List[torch.Tensor],
    cfg: HLJEPAConfig,
    sigreg_modules: nn.ModuleList,
    step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.tensor(0.0, device=pred_bands[0].device)
    logs: Dict[str, float] = {}
    num = len(pred_bands)
    for k in range(num):
        lp = prediction_loss(pred_bands[k], tgt_bands[k], band_masks[k], cfg.loss)
        z = flatten_tokens(tgt_bands[k]) if cfg.sigreg.per_token else pool_image(tgt_bands[k])
        ls = sigreg_modules[k](z, step)
        total = total + cfg.loss.w_pred[k] * lp + cfg.loss.lambda_sig[k] * ls
        logs[f"pred/band{k}"] = float(lp.detach())
        logs[f"sig/band{k}"] = float(ls.detach())
    logs["total"] = float(total.detach())
    return total, logs
