"""Continuous 2D rotary position embedding (RoPE) for ViT-Up attention.

Standard RoPE assumes integer token positions; ViT-Up needs rotations at
arbitrary continuous coordinates: the query rotation ``R_q`` is evaluated at the
continuous query coordinate ``x_q``, and the key rotations ``R_X`` at the
patch-token center coordinates. Because RoPE encodes relative position via a dot
product, this makes attention a function of the continuous query-to-token
offset.

The head dimension is split in half between the two spatial axes ``(y, x)``;
within each axis a GPT-NeoX-style rotation is applied, so any real-valued
coordinate is supported. ``head_dim`` must be divisible by 4.
"""
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE2D(nn.Module):
    """Continuous axial 2D RoPE.

    Args:
        head_dim: per-head channel dimension (must be divisible by 4).
        theta: frequency base.
    """

    def __init__(self, head_dim: int, theta: float = 100.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta
        dim_per_axis = head_dim // 2
        # frequencies for one axis: dim_per_axis/2 distinct values
        idx = torch.arange(0, dim_per_axis, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (idx / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, coords: torch.Tensor):
        """Rotation tensors for coordinates ``[..., 2]`` (order ``(y, x)``).

        Returns ``(cos, sin)`` each of shape ``[..., head_dim]``.
        """
        inv = self.inv_freq.to(coords.dtype)
        y = coords[..., 0:1] * inv                    # [..., dim_per_axis/2]
        x = coords[..., 1:2] * inv
        angles = torch.cat([y, x], dim=-1)            # [..., head_dim/2]
        angles = torch.cat([angles, angles], dim=-1)  # [..., head_dim]
        return angles.cos(), angles.sin()

    @staticmethod
    def apply(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate ``t`` (``[..., head_dim]``) by ``(cos, sin)`` (broadcastable)."""
        return t * cos + rotate_half(t) * sin
