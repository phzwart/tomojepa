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

Forward-compat: the query interface is ``(E_ctx dict, mask dict)`` and the pos
embeds are built from per-stage ``(h_s, w_s)`` grids, so a later sparse
group-masking context pass (encoder sees only visible tokens) or a 3-D swap does
not change this module.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.layers import trunc_normal_


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


class CrossScaleDecoderBlock(nn.Module):
    """Pre-norm decoder block: masked-self-attn over queries + cross-attn to memory."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, memory: torch.Tensor,
                self_mask: Optional[torch.Tensor],
                cross_mask: Optional[torch.Tensor],
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.norm_q1(q)
        q = q + self.self_attn(qn, qn, qn, attn_mask=self_mask, need_weights=False)[0]
        qn = self.norm_q2(q)
        mem = self.norm_mem(memory)
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
    """

    def __init__(self, out_chans: List[int], grids: List[Tuple[int, int]],
                 dim: int = 384, depth: int = 4, heads: int = 6,
                 mlp_ratio: float = 4.0, cross_scale: bool = True):
        super().__init__()
        self.out_chans = list(out_chans)
        self.grids = [tuple(g) for g in grids]
        self.num_stages = len(out_chans)
        self.dim = dim
        self.cross_scale = cross_scale

        self.linear_in = nn.ModuleList([nn.Linear(c, dim) for c in out_chans])
        self.linear_out = nn.ModuleList([nn.Linear(dim, c) for c in out_chans])
        self.mask_query = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in out_chans])
        self.stage_embed = nn.Parameter(torch.zeros(self.num_stages, dim))
        for p in self.mask_query:
            trunc_normal_(p, std=0.02)
        trunc_normal_(self.stage_embed, std=0.02)

        for s, g in enumerate(self.grids):
            self.register_buffer(f"pos_embed_{s}", build_2d_sincos_pos_embed(dim, g),
                                 persistent=False)

        self.blocks = nn.ModuleList(
            [CrossScaleDecoderBlock(dim, heads, mlp_ratio) for _ in range(depth)])
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

        ctx_vis, q_masked, q_sizes, q_stage_ids, m_stage_ids = [], [], [], [], []
        mem_pads: List[torch.Tensor] = []
        for s in range(self.num_stages):
            key = f"s{s + 1}"
            feat = E_ctx[key]                                  # [B, C, h, w]
            c, h, w = feat.shape[1:]
            tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)    # [B, L, C], row-major
            mflat = mask[key].reshape(b, h * w)                    # [B, L] True=masked
            vis = ~mflat
            if fg_stages is not None:
                vis = vis & fg_stages[key].reshape(b, h * w)

            pos = self._pos_embed(s, device, dtype) + self.stage_embed[s]   # [L, D]
            ctx = self.linear_in[s](tok.to(dtype)) + pos.unsqueeze(0)       # [B, L, D]
            if fg_stages is not None:
                mem, mem_pad = self._gather_visible(ctx, vis)
                ctx_vis.append(mem)
                mem_pads.append(mem_pad)
                nvis = mem.shape[1]
            else:
                nvis = int(vis[0].sum())
                ctx_vis.append(ctx[vis].view(b, nvis, self.dim))
                mem_pads.append(None)

            q_full = self.mask_query[s] + pos                              # [L, D]
            q_full = q_full.unsqueeze(0).expand(b, -1, -1)
            k = int(mflat[0].sum())
            q_masked.append(q_full[mflat].view(b, k, self.dim))
            q_sizes.append(k)
            q_stage_ids.append(torch.full((k,), s, device=device, dtype=torch.long))
            m_stage_ids.append(torch.full((nvis,), s, device=device, dtype=torch.long))

        memory = torch.cat(ctx_vis, dim=1)                    # [B, Nvis_total, D]
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
            h = blk(h, memory, self_mask, cross_mask, memory_key_padding_mask)
        h = self.norm_out(h)

        pred: Dict[str, torch.Tensor] = {}
        offset = 0
        for s in range(self.num_stages):
            k = q_sizes[s]
            hs = h[:, offset:offset + k, :]
            pred[f"s{s + 1}"] = self.linear_out[s](hs)        # [B, k, C_s]
            offset += k
        return pred
