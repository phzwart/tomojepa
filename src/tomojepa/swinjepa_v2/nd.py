"""Dimension-parameterized spatial helpers (ndim in {2, 3})."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(ndim: int, in_ch: int, out_ch: int, kernel_size: int = 1) -> nn.Module:
    if ndim == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size)
    if ndim == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size)
    raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def avg_pool_nd(ndim: int, kernel_size: int, stride: int) -> nn.Module:
    if ndim == 2:
        return nn.AvgPool2d(kernel_size, stride)
    if ndim == 3:
        return nn.AvgPool3d(kernel_size, stride)
    raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def upsample_to_nd(x: torch.Tensor, size: Sequence[int], ndim: int) -> torch.Tensor:
    if ndim == 2:
        return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    if ndim == 3:
        return F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)
    raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def spatial_dims(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(x.shape[2:])


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, *spatial] -> [B*spatial, C]."""
    b, c = x.shape[:2]
    return x.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c)


def pool_image(x: torch.Tensor) -> torch.Tensor:
    """[B, C, *spatial] -> [B, C] spatial mean."""
    dims = list(range(2, x.ndim))
    return x.mean(dim=dims)


def stage_grids(img_size: int, patch_size: int, num_stages: int = 4,
                ndim: int = 2) -> List[Tuple[int, ...]]:
    """Token grid sizes per stage (finest first)."""
    finest = img_size // patch_size
    grids = []
    for s in range(num_stages):
        f = 2 ** s
        if ndim == 2:
            grids.append((finest // f, finest // f))
        else:
            grids.append((finest // f, finest // f, finest // f))
    return grids
