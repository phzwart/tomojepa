"""Cross-scale predictor for masked multi-scale latent JEPA.

A small transformer-decoder that predicts the masked target latents at **every**
pyramid stage, conditioned on the visible student context tokens from **all**
stages. A masked fine token is therefore predicted with coarse global context in
scope -- which is the reason the whole pyramid is in the loop rather than four
independent single-scale predictors.

Topology (SimMIM "keep-all"): the student backbone already produced features at
*every* position (mask tokens injected at stage 1), so ``E_ctx`` is dense. We
build the cross-attention *memory* from the **visible** positions and one
*query* per **masked** position per stage. Because the masking budget is a fixed
count per sample (see :mod:`.mask`), visible/masked counts are identical across
the batch and everything stays rectangular ``[B, N, D]`` -- no per-sample padding.

Spatial encoding follows the backbone: with ``use_rope=True`` (default), Q/K in
both self- and cross-attention receive continuous 2D RoPE at token centers mapped
into the finest (s1) grid coordinate system. ``use_rope=False`` falls back to
fixed sin-cos positional embeddings.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from tomojepa.vitup.rope2d import RoPE2D


def build_2d_sincos_pos_embed(dim: int, grid: Tuple[int, int],
                              temperature: float = 10000.0) -> torch.Tensor:
    """Fixed 2-D sin-cos positional embedding ``[h*w, dim]`` (row-major y, x).

    ``dim`` must be divisible by 4 (a quarter each for sin/cos of the two axes).
    """
    if dim % 4 != 0:
        raise ValueError(f"pos-embed dim must be divisible by 4, got {dim}")
    h, w = grid
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (temperature ** omega)                       # [quarter]
    oy = gy.flatten()[:, None] * omega[None, :]                # [L, quarter]
    ox = gx.flatten()[:, None] * omega[None, :]
    pe = torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], dim=1)   # [L, dim]
    return pe


def stage_token_coords(stage_idx: int, grids: List[Tuple[int, int]],
                       device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
    """Token centers in the finest-grid coordinate system ``[L, 2]`` (y, x)."""
    h, w = grids[stage_idx]
    h1, w1 = grids[0]
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype), indexing="ij")
    sy, sx = h1 / h, w1 / w
    y = (gy + 0.5) * sy
    x = (gx + 0.5) * sx
    return torch.stack([y, x], dim=-1).reshape(h * w, 2)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with 2D RoPE on Q/K (same :class:`RoPE2D` as backbone)."""

    def __init__(self, dim: int, num_heads: int, rope_theta: float = 100.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for RoPE, got {self.head_dim}")
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = RoPE2D(self.head_dim, theta=rope_theta)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                q_coords: torch.Tensor, k_coords: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, nq, _ = q.shape
        nk = k.shape[1]
        q = self.q_proj(q).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        cos_q, sin_q = self.rope.cos_sin(q_coords)
        cos_k, sin_k = self.rope.cos_sin(k_coords)
        q = RoPE2D.apply(q, cos_q.unsqueeze(1), sin_q.unsqueeze(1))
        k = RoPE2D.apply(k, cos_k.unsqueeze(1), sin_k.unsqueeze(1))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0),
                                        torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),
                                        torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, nq, -1)
        return self.out_proj(out)


class CrossScaleDecoderBlock(nn.Module):
    """Pre-norm decoder block: masked-self-attn over queries + cross-attn to memory."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, *,
                 use_rope: bool = True, rope_theta: float = 100.0):
        super().__init__()
        self.use_rope = use_rope
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)
        self.norm_q3 = nn.LayerNorm(dim)
        if use_rope:
            self.self_attn = RoPEMultiheadAttention(dim, heads, rope_theta=rope_theta)
            self.cross_attn = RoPEMultiheadAttention(dim, heads, rope_theta=rope_theta)
        else:
            self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, memory: torch.Tensor,
                q_coords: Optional[torch.Tensor],
                mem_coords: Optional[torch.Tensor],
                self_mask: Optional[torch.Tensor],
                cross_mask: Optional[torch.Tensor],
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.norm_q1(q)
        if self.use_rope:
            q = q + self.self_attn(qn, qn, qn, q_coords, q_coords, attn_mask=self_mask)
        else:
            q = q + self.self_attn(qn, qn, qn, attn_mask=self_mask, need_weights=False)[0]
        qn = self.norm_q2(q)
        mem = self.norm_mem(memory)
        if self.use_rope:
            q = q + self.cross_attn(
                qn, mem, mem, q_coords, mem_coords,
                attn_mask=cross_mask, key_padding_mask=memory_key_padding_mask)
        else:
            q = q + self.cross_attn(
                qn, mem, mem, attn_mask=cross_mask,
                key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        q = q + self.mlp(self.norm_q3(q))
        return q


class CrossScalePredictor(nn.Module):
    """Predict masked stage latents from visible all-stage context.

    Args:
        out_chans: per-stage channel dims ``C_s`` (``s1..s4``).
        grids: per-stage token grids ``(h_s, w_s)``.
        dim: predictor width ``D_pred``.
        depth: number of decoder blocks ``N_pred``.
        heads: attention heads.
        mlp_ratio: MLP expansion.
        cross_scale: if False, restrict each stage's queries to attend only to
            same-stage context (and same-stage queries) -- a cheaper ablation.
        use_rope: apply 2D RoPE to Q/K (matches backbone); else sin-cos pos embed.
        rope_theta: RoPE frequency base (shared with backbone when enabled).
    """

    def __init__(self, out_chans: List[int], grids: List[Tuple[int, int]],
                 dim: int = 384, depth: int = 4, heads: int = 6,
                 mlp_ratio: float = 4.0, cross_scale: bool = True,
                 use_rope: bool = True, rope_theta: float = 100.0):
        super().__init__()
        self.out_chans = list(out_chans)
        self.grids = [tuple(g) for g in grids]
        self.num_stages = len(out_chans)
        self.dim = dim
        self.cross_scale = cross_scale
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        if use_rope and (dim // heads) % 4 != 0:
            raise ValueError(
                f"pred_dim {dim} / pred_heads {heads} must yield head_dim divisible "
                f"by 4 for RoPE, got {dim // heads}")

        self.linear_in = nn.ModuleList([nn.Linear(c, dim) for c in out_chans])
        self.linear_out = nn.ModuleList([nn.Linear(dim, c) for c in out_chans])
        self.mask_query = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in out_chans])
        self.stage_embed = nn.Parameter(torch.zeros(self.num_stages, dim))
        for p in self.mask_query:
            trunc_normal_(p, std=0.02)
        trunc_normal_(self.stage_embed, std=0.02)

        if not use_rope:
            for s, g in enumerate(self.grids):
                self.register_buffer(
                    f"pos_embed_{s}", build_2d_sincos_pos_embed(dim, g),
                    persistent=False)

        self.blocks = nn.ModuleList([
            CrossScaleDecoderBlock(
                dim, heads, mlp_ratio, use_rope=use_rope, rope_theta=rope_theta)
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)

    def _pos_embed(self, s: int, device, dtype) -> torch.Tensor:
        return getattr(self, f"pos_embed_{s}").to(device=device, dtype=dtype)

    @staticmethod
    def _gather_visible(ctx: torch.Tensor, vis: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select visible tokens per sample; pad to a rectangular batch.

        Returns ``(padded [B, Nmax, D], pad_mask [B, Nmax])`` where pad_mask is
        True on padded slots (ignored by cross-attention).
        """
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

    def forward(self, E_ctx: Dict[str, torch.Tensor],
                mask: Dict[str, torch.Tensor],
                fg_stages: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        """Predict masked latents per stage.

        Args:
            E_ctx: ``{s: [B, C_s, h_s, w_s]}`` student features (all positions).
            mask:  ``{s: [B, h_s, w_s] bool}`` True = masked.
            fg_stages: optional strict per-stage FG grids; when set, only FG
                visible tokens enter the cross-attention memory (BG stays fixed
                and must not condition predictions).

        Returns:
            ``{s: [B, N_mask_s, C_s]}`` predictions at masked positions only.
        """
        b = next(iter(E_ctx.values())).shape[0]
        device = next(iter(E_ctx.values())).device
        dtype = self.linear_in[0].weight.dtype

        ctx_vis, q_masked, q_coords_parts, mem_coords_parts = [], [], [], []
        q_sizes, q_stage_ids, m_stage_ids = [], [], []
        mem_pads: List[torch.Tensor] = []
        for s in range(self.num_stages):
            key = f"s{s + 1}"
            feat = E_ctx[key]                                  # [B, C, h, w]
            c, h, w = feat.shape[1:]
            tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)    # [B, L, C]
            mflat = mask[key].reshape(b, h * w)                    # [B, L] True=masked
            vis = ~mflat
            if fg_stages is not None:
                vis = vis & fg_stages[key].reshape(b, h * w)

            coords = stage_token_coords(s, self.grids, device, dtype)   # [L, 2]

            if self.use_rope:
                ctx = self.linear_in[s](tok.to(dtype)) + self.stage_embed[s]
                q_full = (self.mask_query[s] + self.stage_embed[s]).unsqueeze(0)
                q_full = q_full.expand(b, h * w, -1)
            else:
                pos = self._pos_embed(s, device, dtype) + self.stage_embed[s]
                ctx = self.linear_in[s](tok.to(dtype)) + pos.unsqueeze(0)
                q_full = self.mask_query[s] + pos
                q_full = q_full.unsqueeze(0).expand(b, -1, -1)

            if fg_stages is not None:
                mem, mem_pad = self._gather_visible(ctx, vis)
                mem_coord, _ = self._gather_visible(
                    coords.unsqueeze(0).expand(b, -1, -1), vis)
                ctx_vis.append(mem)
                mem_coords_parts.append(mem_coord)
                mem_pads.append(mem_pad)
                nvis = mem.shape[1]
            else:
                nvis = int(vis[0].sum())
                ctx_vis.append(ctx[vis].view(b, nvis, self.dim))
                mem_coords_parts.append(
                    coords.unsqueeze(0).expand(b, -1, -1)[vis].view(b, nvis, 2))
                mem_pads.append(None)

            mquery = mflat
            if fg_stages is not None:
                mquery = mflat & fg_stages[key].reshape(b, h * w)
            k = int(mquery[0].sum())
            if fg_stages is not None and not (mquery.sum(1) == k).all():
                raise ValueError(
                    "masked FG query counts differ across batch; "
                    "mask must not select background tokens.")
            q_masked.append(q_full[mquery].view(b, k, self.dim))
            q_coords_parts.append(
                coords.unsqueeze(0).expand(b, -1, -1)[mquery].view(b, k, 2))
            q_sizes.append(k)
            q_stage_ids.append(torch.full((k,), s, device=device, dtype=torch.long))
            m_stage_ids.append(torch.full((nvis,), s, device=device, dtype=torch.long))

        memory = torch.cat(ctx_vis, dim=1)                    # [B, Nvis_total, D]
        query_coords = torch.cat(q_coords_parts, dim=1)       # [B, Nmask_total, 2]
        mem_coords = torch.cat(mem_coords_parts, dim=1)       # [B, Nvis_total, 2]
        memory_key_padding_mask = None
        if fg_stages is not None:
            memory_key_padding_mask = torch.cat(mem_pads, dim=1)
        queries = torch.cat(q_masked, dim=1)                  # [B, Nmask_total, D]

        self_mask = cross_mask = None
        if not self.cross_scale:
            qid = torch.cat(q_stage_ids)                      # [Nmask_total]
            mid = torch.cat(m_stage_ids)                      # [Nvis_total]
            self_mask = qid[:, None] != qid[None, :]          # True = disallowed
            cross_mask = qid[:, None] != mid[None, :]

        h = queries
        for blk in self.blocks:
            h = blk(h, memory, query_coords, mem_coords,
                    self_mask, cross_mask, memory_key_padding_mask)
        h = self.norm_out(h)

        pred: Dict[str, torch.Tensor] = {}
        offset = 0
        for s in range(self.num_stages):
            k = q_sizes[s]
            hs = h[:, offset:offset + k, :]
            pred[f"s{s + 1}"] = self.linear_out[s](hs)        # [B, k, C_s]
            offset += k
        return pred
