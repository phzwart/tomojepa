"""SwinJEPA assembly: encode() vs forward_train() (spec §8)."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ..config import HLJEPAConfig
from ..losses.sigreg import SIGReg
from ..losses.total import total_loss
from .backbone import HierarchicalSwin
from .bands import BandFormer, make_bands
from .heads import BandHeads
from .predictor import TopDownPredictor


class SwinJEPA(nn.Module):
    def __init__(self, cfg: HLJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = HierarchicalSwin(cfg.backbone, img_size=cfg.img_size)
        stage_dims = self.backbone.backbone.out_chans
        self.band_former = BandFormer(stage_dims, cfg.band, ndim=cfg.data.ndim)
        self.band_heads = BandHeads(cfg.band, cfg.num_bands, ndim=cfg.data.ndim)
        self.predictor = TopDownPredictor(cfg.band.band_dim, cfg.predictor, ndim=cfg.data.ndim)
        self.sigreg_modules = nn.ModuleList([
            SIGReg(cfg.sigreg, cfg.band.band_dim) for _ in range(cfg.num_bands)
        ])

    def _to_bands(self, stage_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        bands = make_bands(stage_feats, self.band_former)
        return self.band_heads(bands)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Inference / downstream: unmasked clean band pyramid."""
        feats = self.backbone(x)
        return self._to_bands(feats)

    def forward_train(
        self,
        view_ctx: torch.Tensor,
        view_tgt: torch.Tensor,
        band_masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        tgt_feats = self.backbone(view_tgt)
        tgt_bands = self._to_bands(tgt_feats)
        ctx_feats = self.backbone(view_ctx, mask_tokens=band_masks)
        ctx_bands = self._to_bands(ctx_feats)
        pred_bands = self.predictor(ctx_bands, band_masks)
        return pred_bands, tgt_bands, ctx_bands

    def compute_loss(
        self,
        view_ctx: torch.Tensor,
        view_tgt: torch.Tensor,
        band_masks: List[torch.Tensor],
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        pred_bands, tgt_bands, ctx_bands = self.forward_train(view_ctx, view_tgt, band_masks)
        return total_loss(pred_bands, tgt_bands, band_masks, self.cfg, self.sigreg_modules, step)
