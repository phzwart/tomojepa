"""Optional per-band projection heads before SIGReg (spec §5)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..config import BandCfg
from ..nd import conv_nd


class BandHeads(nn.Module):
    def __init__(self, cfg: BandCfg, num_bands: int, ndim: int = 2):
        super().__init__()
        self.enabled = cfg.use_proj_head
        if self.enabled:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    conv_nd(ndim, cfg.band_dim, cfg.band_dim),
                    nn.GELU(),
                    conv_nd(ndim, cfg.band_dim, cfg.band_dim),
                )
                for _ in range(num_bands)
            ])
        else:
            self.heads = nn.ModuleList()

    def forward(self, bands: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.enabled:
            return bands
        return [self.heads[i](bands[i]) for i in range(len(bands))]
