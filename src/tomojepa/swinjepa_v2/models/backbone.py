"""HierarchicalSwin adapter over existing SwinMultiScaleBackbone."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from tomojepa.swinjepa.backbone import SwinMultiScaleBackbone

from ..config import BackboneCfg


class HierarchicalSwin(nn.Module):
    """Expose stage pyramid + finest-grid mask token injection."""

    def __init__(self, cfg: BackboneCfg, img_size: int):
        super().__init__()
        self.cfg = cfg
        self.num_stages = len(cfg.depths)
        self.backbone = SwinMultiScaleBackbone(
            model_name=cfg.model_name,
            img_size=img_size,
            in_chans=cfg.in_chans,
            pretrained=cfg.pretrained,
            drop_path_rate=cfg.drop_path_rate,
            use_rope=cfg.use_rope,
            rope_theta=cfg.rope_theta,
            embed_dim=cfg.embed_dim if cfg.embed_dim != 96 else None,
        )
        self.stage_keys = self.backbone.stage_keys

    def stage_grid(self, stage_idx: int):
        return self.backbone.stage_grid(stage_idx)

    def forward(
        self,
        x: torch.Tensor,
        mask_tokens: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        mask1 = None
        if mask_tokens is not None and len(mask_tokens) > 0:
            mask1 = mask_tokens[0]
        feats = self.backbone(x, mask1=mask1)
        return [feats[k] for k in self.stage_keys]
