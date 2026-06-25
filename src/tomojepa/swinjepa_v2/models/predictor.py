"""Per-band top-down predictor (spec §6)."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import PredictorCfg
from ..nd import upsample_to_nd


class BandTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        k = self.norm1(mem)
        v = mem
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TopDownPredictor(nn.Module):
    """Shallow per-band transformer; optional coarse-to-fine conditioning."""

    def __init__(self, band_dim: int, cfg: PredictorCfg, ndim: int = 2):
        super().__init__()
        self.cfg = cfg
        self.ndim = ndim
        self.band_dim = band_dim
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                BandTransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.depth_per_band)
            ])
            for _ in range(4)
        ])
        self.in_proj = nn.ModuleList([nn.Linear(band_dim, cfg.embed_dim) for _ in range(4)])
        self.out_proj = nn.ModuleList([nn.Linear(cfg.embed_dim, band_dim) for _ in range(4)])
        self.mask_query = nn.ParameterList([nn.Parameter(torch.zeros(cfg.embed_dim)) for _ in range(4)])
        self.stage_embed = nn.ParameterList([nn.Parameter(torch.zeros(cfg.embed_dim)) for _ in range(4)])
        self.cond_proj = nn.ModuleList([nn.Linear(band_dim, cfg.embed_dim) for _ in range(3)])

    def _band_forward(
        self,
        band_idx: int,
        ctx: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, *spatial = ctx.shape
        n = int(torch.tensor(spatial).prod().item())
        flat_ctx = ctx.reshape(b, c, n).permute(0, 2, 1)
        flat_mask = mask.reshape(b, n)
        pred = ctx.clone()
        for bi in range(b):
            vis = ~flat_mask[bi]
            msk = flat_mask[bi]
            if msk.sum() == 0:
                continue
            mem = self.in_proj[band_idx](flat_ctx[bi, vis])
            queries = self.mask_query[band_idx].unsqueeze(0).expand(int(msk.sum()), -1)
            queries = queries + self.stage_embed[band_idx]
            if cond is not None:
                cond_flat = cond[bi].reshape(c, n).permute(1, 0)
                queries = queries + self.cond_proj[band_idx](cond_flat[msk])
            for blk in self.blocks[band_idx]:
                queries = blk(queries, mem)
            out = self.out_proj[band_idx](queries)
            flat_pred = pred[bi].reshape(c, n)
            flat_pred[:, msk] = out.t().to(flat_pred.dtype)
            pred[bi] = flat_pred.reshape(c, *spatial)
        return pred

    def forward(
        self,
        ctx_bands: List[torch.Tensor],
        band_masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        num = len(ctx_bands)
        out: List[torch.Tensor | None] = [None] * num
        cond: Optional[torch.Tensor] = None
        order = list(range(num - 1, -1, -1)) if self.cfg.top_down else list(range(num))
        for i in order:
            cond_up = None
            if self.cfg.top_down and cond is not None:
                cond_up = upsample_to_nd(cond, ctx_bands[i].shape[2:], self.ndim)
            out[i] = self._band_forward(i, ctx_bands[i], band_masks[i], cond_up)
            if self.cfg.top_down:
                cond = out[i]
        return [out[i] for i in range(num)]  # type: ignore[misc]
