"""ViT-Up refinement block U_t (paper section 2.2, Fig. 2C).

    q_t = U_t(q_{t-1}, x_q, H_{l[t]})

Four additively-fused parts:
  (a) transition MLP (residual) -- realigns the query to the current layer's
      feature space (layers may be skipped):
          x = q_{t-1} + MLP_transition(LN(q_{t-1}))
  (b) cross-window attention with continuous 2D RoPE:
          x_attn = W_O * CW-MHA(R_q W_Q LN_Q(x), R_X W_K LN_KV(H), W_V LN_KV(H))
  (c) FeatX local sub-token extractor:
          x_subtoken = FeatX(h_nn, x_nn, x_q)
  (d) fusion:
          x_fused = x + x_attn + x_subtoken
          q_t     = x_fused + MLP_fusion(LN(x_fused))
"""
import torch
import torch.nn as nn

from .attention import CrossWindowAttention
from .featx import FeatX


def _mlp(dim: int, ratio: float) -> nn.Sequential:
    hidden = int(dim * ratio)
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class ViTUpBlock(nn.Module):
    def __init__(self, internal_dim: int, backbone_dim: int, num_heads: int,
                 window: int, posenc_dim: int = 64, mlp_ratio: float = 4.0,
                 rope_theta: float = 100.0):
        super().__init__()
        D, C = internal_dim, backbone_dim
        # (a) transition
        self.ln_trans = nn.LayerNorm(D)
        self.mlp_trans = _mlp(D, mlp_ratio)
        # (b) cross-window attention
        self.ln_q = nn.LayerNorm(D)
        self.ln_kv = nn.LayerNorm(C)
        self.attn = CrossWindowAttention(
            dim=D, num_heads=num_heads, window=window, q_dim=D, kv_dim=C,
            rope_theta=rope_theta)
        # (c) FeatX
        self.featx = FeatX(backbone_dim=C, internal_dim=D, posenc_dim=posenc_dim)
        # (d) fusion
        self.ln_fuse = nn.LayerNorm(D)
        self.mlp_fuse = _mlp(D, mlp_ratio)
        self.patch_size = None  # set by model (for the documented Delta x / p form)

    def prepare_kv(self, h_grid: torch.Tensor) -> dict:
        """Precompute the per-image (query-independent) key/value tensors.

        Applies ``LN_KV`` then projects/rotates keys and values on the backbone
        grid once, so all query chunks reuse them.
        """
        kv = self.ln_kv(h_grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LN over C
        return self.attn.prepare_kv(kv)

    def forward(self, q_prev: torch.Tensor, h_grid: torch.Tensor,
                q_coords: torch.Tensor, patch_size: int,
                kv: dict = None) -> torch.Tensor:
        # (a) transition (residual)
        x = q_prev + self.mlp_trans(self.ln_trans(q_prev))            # [B,Q,D]
        # (b) cross-window attention with continuous 2D RoPE
        if kv is None:
            kv = self.prepare_kv(h_grid)
        x_attn = self.attn.forward_prepared(self.ln_q(x), kv, q_coords)  # [B,Q,D]
        # (c) FeatX sub-token (operates on the raw nearest-token feature)
        x_sub = self.featx(h_grid, q_coords, patch_size)              # [B,Q,D]
        # (d) fusion
        x_fused = x + x_attn + x_sub
        return x_fused + self.mlp_fuse(self.ln_fuse(x_fused))
