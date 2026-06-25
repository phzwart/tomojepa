"""Lightweight in-pyramid band fusion for integrated MIM.

Replaces the external :class:`CrossScalePredictor` with stage-chunked
cross-scale RoPE attention operating directly in the JEPA latent space
(``lat_dims``). Each stage's masked queries attend to visible tokens from
that stage and all coarser stages, capping peak memory versus concatenating
the full pyramid into one attention memory.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from .predictor import (
    CrossScaleDecoderBlock,
    build_2d_sincos_pos_embed,
    stage_token_coords,
)


class PyramidBandFusion(nn.Module):
    """Predict masked latent tokens via stage-chunked cross-scale fusion.

    Args:
        out_chans: per-stage latent dims ``C_s`` (``s1..s4``).
        grids: per-stage token grids ``(h_s, w_s)``.
        depth: fusion decoder blocks per stage (default 1).
        heads: attention heads (must satisfy RoPE head_dim constraints).
        mlp_ratio: MLP expansion in decoder blocks.
        cross_scale: attend to coarser stages; if False, same-stage only.
        use_rope: 2D RoPE on Q/K (matches backbone).
        rope_theta: RoPE frequency base.
    """

    def __init__(self, out_chans: List[int], grids: List[Tuple[int, int]],
                 depth: int = 1, heads: int = 4, mlp_ratio: float = 4.0,
                 cross_scale: bool = True, use_rope: bool = True,
                 rope_theta: float = 100.0):
        super().__init__()
        self.out_chans = list(out_chans)
        self.grids = [tuple(g) for g in grids]
        self.num_stages = len(out_chans)
        self.cross_scale = cross_scale
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self._uniform_dim = len(set(out_chans)) == 1

        for c in out_chans:
            if use_rope and (c // heads) % 4 != 0:
                raise ValueError(
                    f"lat dim {c} / fusion_heads {heads} must yield head_dim "
                    f"divisible by 4 for RoPE, got {c // heads}")

        self.mask_query = nn.ParameterList(
            [nn.Parameter(torch.zeros(c)) for c in out_chans])
        self.stage_embed = nn.ParameterList(
            [nn.Parameter(torch.zeros(c)) for c in out_chans])
        for p in self.mask_query:
            trunc_normal_(p, std=0.02)
        for p in self.stage_embed:
            trunc_normal_(p, std=0.02)

        if not use_rope:
            for s, g in enumerate(self.grids):
                self.register_buffer(
                    f"pos_embed_{s}",
                    build_2d_sincos_pos_embed(out_chans[s], g),
                    persistent=False)

        block_fn = lambda dim: nn.ModuleList([
            CrossScaleDecoderBlock(
                dim, heads, mlp_ratio, use_rope=use_rope, rope_theta=rope_theta)
            for _ in range(depth)])
        self.blocks = nn.ModuleList([block_fn(c) for c in out_chans])
        self.norm_out = nn.ModuleList([nn.LayerNorm(c) for c in out_chans])

    def _pos_embed(self, s: int, device, dtype) -> torch.Tensor:
        return getattr(self, f"pos_embed_{s}").to(device=device, dtype=dtype)

    @staticmethod
    def _gather_visible(ctx: torch.Tensor, vis: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, d = ctx.shape
        lens = [int(v.sum()) for v in vis]
        max_n = max(lens) if lens else 0
        out = ctx.new_zeros(b, max_n, d)
        pad = torch.ones(b, max_n, dtype=torch.bool, device=ctx.device)
        for bi, n in enumerate(lens):
            if n:
                out[bi, :n] = ctx[bi, vis[bi]]
                pad[bi, :n] = False
        return out, pad

    def _stage_context(self, E_ctx: Dict[str, torch.Tensor],
                       mask: Dict[str, torch.Tensor],
                       fg_stages: Optional[Dict[str, torch.Tensor]],
                       query_stage: int, device, dtype
                       ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
                                  List[Optional[torch.Tensor]]]:
        """Visible context tokens from stages ``query_stage..S-1``."""
        b = next(iter(E_ctx.values())).shape[0]
        ctx_parts, coord_parts, pad_parts = [], [], []
        mem_stage_ids: List[torch.Tensor] = []
        for s in range(query_stage, self.num_stages):
            key = f"s{s + 1}"
            feat = E_ctx[key]
            c, h, w = feat.shape[1:]
            tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            mflat = mask[key].reshape(b, h * w)
            vis = ~mflat
            if fg_stages is not None:
                vis = vis & fg_stages[key].reshape(b, h * w)
            coords = stage_token_coords(s, self.grids, device, dtype)

            if self.use_rope:
                ctx = tok.to(dtype) + self.stage_embed[s].unsqueeze(0).unsqueeze(0)
            else:
                pos = self._pos_embed(s, device, dtype) + self.stage_embed[s]
                ctx = tok.to(dtype) + pos.unsqueeze(0)

            if fg_stages is not None:
                mem, mem_pad = self._gather_visible(ctx, vis)
                mem_coord, _ = self._gather_visible(
                    coords.unsqueeze(0).expand(b, -1, -1), vis)
                ctx_parts.append(mem)
                coord_parts.append(mem_coord)
                pad_parts.append(mem_pad)
                nvis = mem.shape[1]
            else:
                nvis = int(vis[0].sum())
                ctx_parts.append(ctx[vis].view(b, nvis, c))
                coord_parts.append(
                    coords.unsqueeze(0).expand(b, -1, -1)[vis].view(b, nvis, 2))
                pad_parts.append(None)
            mem_stage_ids.append(
                torch.full((nvis,), s, device=device, dtype=torch.long))
        return ctx_parts, coord_parts, pad_parts

    def forward(self, E_ctx: Dict[str, torch.Tensor],
                mask: Dict[str, torch.Tensor],
                fg_stages: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        """Predict masked latents per stage (stage-chunked cross-scale).

        Returns:
            ``{s: [B, N_mask_s, C_s]}`` predictions at masked positions only.
        """
        b = next(iter(E_ctx.values())).shape[0]
        device = next(iter(E_ctx.values())).device
        dtype = self.blocks[0][0].norm_q1.weight.dtype

        pred: Dict[str, torch.Tensor] = {}
        for s in range(self.num_stages):
            key = f"s{s + 1}"
            feat = E_ctx[key]
            c, h, w = feat.shape[1:]
            tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            mflat = mask[key].reshape(b, h * w)
            mquery = mflat
            if fg_stages is not None:
                mquery = mflat & fg_stages[key].reshape(b, h * w)
            k = int(mquery[0].sum())
            if fg_stages is not None and not (mquery.sum(1) == k).all():
                raise ValueError(
                    "masked FG query counts differ across batch; "
                    "mask must not select background tokens.")
            if k == 0:
                pred[key] = tok.new_zeros(b, 0, c)
                continue

            coords = stage_token_coords(s, self.grids, device, dtype)
            if self.use_rope:
                q_full = (self.mask_query[s] + self.stage_embed[s]).unsqueeze(0)
                q_full = q_full.expand(b, h * w, -1)
            else:
                pos = self._pos_embed(s, device, dtype) + self.stage_embed[s]
                q_full = self.mask_query[s] + pos
                q_full = q_full.unsqueeze(0).expand(b, -1, -1)

            queries = q_full[mquery].view(b, k, c)
            q_coords = coords.unsqueeze(0).expand(b, -1, -1)[mquery].view(b, k, 2)

            mem_start = s if self.cross_scale else s
            ctx_parts, coord_parts, pad_parts = self._stage_context(
                E_ctx, mask, fg_stages, mem_start, device, dtype)
            if not ctx_parts:
                pred[key] = tok.new_zeros(b, k, c)
                continue

            memory = torch.cat(ctx_parts, dim=1)
            mem_coords = torch.cat(coord_parts, dim=1)
            memory_key_padding_mask = None
            if fg_stages is not None and any(p is not None for p in pad_parts):
                memory_key_padding_mask = torch.cat(
                    [p for p in pad_parts if p is not None], dim=1)

            self_mask = cross_mask = None
            if not self.cross_scale:
                qid = torch.full((k,), s, device=device, dtype=torch.long)
                mid = torch.cat([
                    torch.full((ctx_parts[i].shape[1],), s + i, device=device,
                               dtype=torch.long)
                    for i in range(len(ctx_parts))])
                self_mask = qid[:, None] != qid[None, :]
                cross_mask = qid[:, None] != mid[None, :]

            h = queries
            for blk in self.blocks[s]:
                h = blk(h, memory, q_coords, mem_coords,
                        self_mask, cross_mask, memory_key_padding_mask)
            h = self.norm_out[s](h)
            pred[key] = h
        return pred
