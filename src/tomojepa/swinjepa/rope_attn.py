"""RoPE window attention for Swin blocks.

timm's :class:`WindowAttention` adds a learned relative-position bias table.
This module applies continuous 2D RoPE to Q/K instead, using absolute ``(y, x)``
token coordinates within each stage's feature map (aligned through shift/pad/
window partition the same way as the token tensor).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.models.swin_transformer import window_partition, window_reverse

from tomojepa.vitup.rope2d import RoPE2D


class WindowAttentionRoPE(nn.Module):
    """Window MSA with 2D RoPE on Q/K (no relative-position bias table)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (7, 7),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_theta: float = 100.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"Swin head_dim must be divisible by 4 for RoPE, got {head_dim}")
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rope = RoPE2D(head_dim, theta=rope_theta)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: ``[num_windows*B, N, C]`` window tokens.
            coords: ``[num_windows*B, N, 2]`` absolute ``(y, x)`` per token.
            mask: optional shifted-window attention mask.
        """
        b_, n, _ = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = self.rope.cos_sin(coords)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = RoPE2D.apply(q, cos, sin)
        k = RoPE2D.apply(k, cos, sin)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(b_, n, -1)
        x = self.proj(x)
        return self.proj_drop(x)


def _absolute_coords(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[B, H, W, 2]`` float grid of ``(y, x)`` indices."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([yy, xx], dim=-1)
    return coords.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def _attn_with_rope(block, x: torch.Tensor) -> torch.Tensor:
    """RoPE-aware drop-in for timm :meth:`SwinTransformerBlock._attn`."""
    b, h, w, c = x.shape
    has_shift = any(block.shift_size)
    if has_shift:
        shifted_x = torch.roll(
            x, shifts=(-block.shift_size[0], -block.shift_size[1]), dims=(1, 2))
    else:
        shifted_x = x

    pad_h = (block.window_size[0] - h % block.window_size[0]) % block.window_size[0]
    pad_w = (block.window_size[1] - w % block.window_size[1]) % block.window_size[1]

    coords = _absolute_coords(b, h, w, x.device, x.dtype)
    if has_shift:
        coords = torch.roll(
            coords, shifts=(-block.shift_size[0], -block.shift_size[1]), dims=(1, 2))
    coords = F.pad(coords, (0, 0, 0, pad_w, 0, pad_h))
    shifted_x = F.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    _, hp, wp, _ = shifted_x.shape

    x_windows = window_partition(shifted_x, block.window_size)
    x_windows = x_windows.view(-1, block.window_area, c)
    coord_windows = window_partition(coords, block.window_size)
    coord_windows = coord_windows.view(-1, block.window_area, 2)

    if getattr(block, "dynamic_mask", False):
        attn_mask = block.get_attn_mask(shifted_x)
    else:
        attn_mask = block.attn_mask
    attn_windows = block.attn(x_windows, coord_windows, mask=attn_mask)

    attn_windows = attn_windows.view(-1, block.window_size[0], block.window_size[1], c)
    shifted_x = window_reverse(attn_windows, block.window_size, hp, wp)
    shifted_x = shifted_x[:, :h, :w, :].contiguous()

    if has_shift:
        x = torch.roll(shifted_x, shifts=block.shift_size, dims=(1, 2))
    else:
        x = shifted_x
    return x


def enable_rope_on_swin(stages: nn.ModuleList, rope_theta: float = 100.0) -> None:
    """Replace timm relative-bias window attention with RoPE in place."""
    import timm.models.swin_transformer as st

    for stage in stages:
        for block in stage.blocks:
            old = block.attn
            if not isinstance(old, st.WindowAttention):
                raise TypeError(
                    f"Expected timm WindowAttention, got {type(old).__name__}")
            block.attn = WindowAttentionRoPE(
                dim=old.dim,
                num_heads=old.num_heads,
                window_size=old.window_size,
                qkv_bias=old.qkv.bias is not None,
                attn_drop=old.attn_drop.p,
                proj_drop=old.proj_drop.p,
                rope_theta=rope_theta,
            )
            block._attn = _attn_with_rope.__get__(block, type(block))
