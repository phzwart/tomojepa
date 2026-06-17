"""Per-stage decoders D_t (paper section 2.4 / 3.3).

Each refinement stage ``t`` has its own output projection ``D_t`` mapping the
latent query ``q_t`` back to ViT feature space, producing ``o_t = D_t(q_t)`` for
``t = 0..T`` (``o_0 = D_0(q_0)`` is the embedding stage ``l[0] = 0``). All stages
are supervised in training, so the per-stage decoders must not be collapsed.

Stages ``0..T-1`` are single linear projections; the final decoder ``D_T`` is a
single-layer MLP with LayerNorm followed by a linear projection. The latent
dimension equals the output feature dimension.
"""
from typing import List

import torch
import torch.nn as nn


class StageDecoders(nn.Module):
    def __init__(self, num_blocks: int, internal_dim: int, output_dim: int):
        super().__init__()
        self.num_blocks = num_blocks  # T
        decoders: List[nn.Module] = []
        for t in range(num_blocks + 1):                 # t = 0..T
            if t == num_blocks:                          # final D_T: LN + linear
                decoders.append(nn.Sequential(
                    nn.LayerNorm(internal_dim),
                    nn.Linear(internal_dim, output_dim),
                ))
            else:                                        # D_t: linear
                decoders.append(nn.Linear(internal_dim, output_dim))
        self.decoders = nn.ModuleList(decoders)

    def forward(self, t: int, q_t: torch.Tensor) -> torch.Tensor:
        return self.decoders[t](q_t)
