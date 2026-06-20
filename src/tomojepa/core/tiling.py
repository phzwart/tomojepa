"""qlty NCYXQuilt helpers for overlapping native-crop inference."""
from __future__ import annotations

import torch


def pixel_quilt(H: int, W: int, window: int, step: int, border: int,
                border_weight: float = 0.1):
    """Quilt for unstitching / stitching at input pixel resolution."""
    from qlty import NCYXQuilt

    return NCYXQuilt(
        Y=H, X=W, window=(window, window), step=(step, step),
        border=(border, border), border_weight=border_weight,
    )


def token_quilt(H: int, W: int, window: int, step: int, border: int, stride: int,
                border_weight: float = 0.1):
    """Quilt matching ``pixel_quilt`` but on a ``stride``-downsampled token grid."""
    from qlty import NCYXQuilt

    g = window // stride
    return NCYXQuilt(
        Y=H // stride, X=W // stride, window=(g, g),
        step=(step // stride, step // stride),
        border=(max(1, border // stride), max(1, border // stride)),
        border_weight=border_weight,
    )


def unstitch_tiles(quilt, img: torch.Tensor) -> torch.Tensor:
    """``[1,C,H,W]`` on device -> ``[M,C,window,window]`` on CPU."""
    return quilt.unstitch(img.detach().cpu())


def stitch_maps(quilt, maps: list[torch.Tensor]) -> torch.Tensor:
    """Stitch per-tile ``[C,h,w]`` maps -> ``[C,H',W']`` (channel-first)."""
    full, _ = quilt.stitch(torch.stack(maps))
    return full[0]
