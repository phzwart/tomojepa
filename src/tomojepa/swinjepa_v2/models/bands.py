"""Laplacian latent pyramid bands (spec §5)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..config import BandCfg
from ..nd import conv_nd, upsample_to_nd


class BandFormer(nn.Module):
    """Project stages to common band_dim and form laplacian or raw bands."""

    def __init__(self, stage_dims: List[int], cfg: BandCfg, ndim: int = 2):
        super().__init__()
        self.cfg = cfg
        self.ndim = ndim
        self.projs = nn.ModuleList([conv_nd(ndim, d, cfg.band_dim) for d in stage_dims])

    def forward(self, stage_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        proj = [self.projs[i](stage_feats[i]) for i in range(len(stage_feats))]
        if self.cfg.mode == "raw_stage":
            return proj
        if self.cfg.mode != "laplacian":
            raise ValueError(f"unknown band mode: {self.cfg.mode!r}")
        bands: List[torch.Tensor] = []
        for i in range(len(proj) - 1, -1, -1):
            if i == len(proj) - 1:
                bands.insert(0, proj[i])
            else:
                up = upsample_to_nd(proj[i + 1], proj[i].shape[2:], self.ndim)
                bands.insert(0, proj[i] - up)
        return bands


def make_bands(stage_feats: List[torch.Tensor], former: BandFormer) -> List[torch.Tensor]:
    return former(stage_feats)
