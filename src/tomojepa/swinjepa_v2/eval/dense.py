"""Light FPN decoder + orbit averaging for dense tasks (spec §12)."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.augment import GeomParams, _apply_geom_2d, apply_latent_geom


class FPNHead(nn.Module):
    """Minimal top-down fusion head over band pyramid."""

    def __init__(self, band_dim: int, out_ch: int = 1):
        super().__init__()
        self.out = nn.Conv2d(band_dim, out_ch, 1)

    def forward(self, bands: List[torch.Tensor]) -> torch.Tensor:
        x = bands[0]
        for b in bands[1:]:
            up = F.interpolate(b, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x + up
        return self.out(x)


def orbit_average_encode(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Average encode() over 8 D4 transforms."""
    c, h, w = x.shape
    acc: Optional[List[torch.Tensor]] = None
    count = 0
    for k in range(4):
        for hf in (False, True):
            for vf in (False, True):
                geom = GeomParams(rot_k=k, hflip=hf, vflip=vf, crop_h=h, crop_w=w)
                geom._scale = (1.0, 1.0)  # type: ignore[attr-defined]
                img = _apply_geom_2d(x, geom, h)
                bands = model.encode(img.unsqueeze(0))
                if acc is None:
                    acc = [b.clone() for b in bands]
                else:
                    for i, b in enumerate(bands):
                        aligned = apply_latent_geom(b, geom)
                        acc[i] = acc[i] + aligned
                count += 1
    return [a / count for a in acc]  # type: ignore[operator]
