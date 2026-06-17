"""Cross-window multi-head attention with continuous 2D RoPE (paper section 2.2b).

Each continuous query attends only to the backbone tokens inside a
``window x window`` neighborhood centered on the query's nearest patch token.
Queries are rotated (RoPE) at their continuous coordinate ``x_q``; keys are
rotated at the corresponding patch-token center coordinates.

Two equivalent code paths:
  * ``forward`` -- the default *gather* path: for each query, gather its window
    of tokens and run masked softmax attention. O(Q * window^2), the path used in
    training/inference.
  * ``forward_dense`` -- a brute-force reference (full Q-by-token attention with a
    window mask) used only to verify the gather path in tests.

NATTEN (neighborhood attention) can accelerate the grid-aligned case; a guarded
hook is provided and falls back to the gather path when NATTEN is unavailable or
the queries are not grid-aligned. The mathematical definition is identical
across paths.
"""
from typing import Optional

import torch
import torch.nn as nn

from .rope2d import RoPE2D


def natten_available() -> bool:
    try:
        import natten  # noqa: F401
        return True
    except Exception:
        return False


class CrossWindowAttention(nn.Module):
    """Windowed cross-attention from continuous queries to backbone tokens.

    The internal attention width is ``dim`` (= ``num_heads * head_dim``). Queries
    enter at ``q_dim`` and keys/values at ``kv_dim`` (the backbone width); the
    output is projected back to ``q_dim``.
    """

    def __init__(self, dim: int, num_heads: int, window: int,
                 q_dim: Optional[int] = None, kv_dim: Optional[int] = None,
                 rope_theta: float = 100.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        q_dim = q_dim or dim
        kv_dim = kv_dim or dim
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.scale = self.head_dim ** -0.5
        self.W_Q = nn.Linear(q_dim, dim, bias=True)
        self.W_K = nn.Linear(kv_dim, dim, bias=True)
        self.W_V = nn.Linear(kv_dim, dim, bias=True)
        self.W_O = nn.Linear(dim, q_dim, bias=True)
        self.rope = RoPE2D(self.head_dim, theta=rope_theta)

    # -- window bookkeeping --------------------------------------------------
    def _window_indices(self, q_coords: torch.Tensor, h: int, w: int):
        """Per-query window token indices, centers, and validity.

        Returns ``(flat_idx [B,Q,Wn], centers [B,Q,Wn,2], valid [B,Q,Wn])``.
        """
        device = q_coords.device
        win = self.window
        half = win // 2
        # nearest token = floor(coord) clamped to grid
        qi = torch.clamp(torch.floor(q_coords[..., 0]).long(), 0, h - 1)  # [B,Q]
        qj = torch.clamp(torch.floor(q_coords[..., 1]).long(), 0, w - 1)
        off = torch.arange(-half, -half + win, device=device)             # [win]
        ni = qi[..., None, None] + off[None, None, :, None]               # [B,Q,win,1]
        nj = qj[..., None, None] + off[None, None, None, :]               # [B,Q,1,win]
        ni = ni.expand(*qi.shape, win, win).reshape(*qi.shape, win * win)
        nj = nj.expand(*qj.shape, win, win).reshape(*qj.shape, win * win)
        valid = (ni >= 0) & (ni < h) & (nj >= 0) & (nj < w)               # [B,Q,Wn]
        ci = ni.clamp(0, h - 1)
        cj = nj.clamp(0, w - 1)
        flat_idx = ci * w + cj                                            # [B,Q,Wn]
        centers = torch.stack([ni.to(q_coords.dtype) + 0.5,
                               nj.to(q_coords.dtype) + 0.5], dim=-1)      # [B,Q,Wn,2]
        return flat_idx, centers, valid

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # [..., dim] -> [..., heads, head_dim]
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def prepare_kv(self, kv_grid: torch.Tensor) -> dict:
        """Project + RoPE-rotate the keys/values on the full backbone grid once.

        Key/value projections and the key RoPE depend only on the (fixed) token
        grid, not on the queries, so they are computed a single time per image
        and reused across all query chunks. Returns rotated keys, values, and the
        grid size.
        """
        B, C, h, w = kv_grid.shape
        kv_flat = kv_grid.permute(0, 2, 3, 1).reshape(B, h * w, C)        # [B,hw,C]
        k = self._heads(self.W_K(kv_flat))                              # [B,hw,H,d]
        v = self._heads(self.W_V(kv_flat))                              # [B,hw,H,d]
        ys = torch.arange(h, device=kv_grid.device, dtype=torch.float32) + 0.5
        xs = torch.arange(w, device=kv_grid.device, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([gy, gx], dim=-1).reshape(h * w, 2)        # [hw,2]
        ck, sk = self.rope.cos_sin(centers.to(k.dtype))                  # [hw,head_dim]
        k = self.rope.apply(k, ck.unsqueeze(-2), sk.unsqueeze(-2))      # [B,hw,H,d]
        return {"k": k, "v": v, "h": h, "w": w}

    def forward_prepared(self, q_in: torch.Tensor, kv: dict,
                         q_coords: torch.Tensor) -> torch.Tensor:
        """Cross-window attention against pre-projected keys/values (``kv``)."""
        B, Q, _ = q_in.shape
        h, w, H, d = kv["h"], kv["w"], self.num_heads, self.head_dim
        k_grid, v_grid = kv["k"], kv["v"]                               # [B,hw,H,d]

        flat_idx, _, valid = self._window_indices(q_coords, h, w)
        Wn = flat_idx.shape[-1]
        idx = flat_idx.view(B, Q, Wn, 1, 1).expand(B, Q, Wn, H, d)
        kg = torch.gather(k_grid.unsqueeze(1).expand(B, Q, h * w, H, d), 2, idx)
        vg = torch.gather(v_grid.unsqueeze(1).expand(B, Q, h * w, H, d), 2, idx)

        q = self._heads(self.W_Q(q_in))                                # [B,Q,H,d]
        cq, sq = self.rope.cos_sin(q_coords)                           # [B,Q,head_dim]
        q = self.rope.apply(q, cq.unsqueeze(-2), sq.unsqueeze(-2))     # [B,Q,H,d]

        scores = torch.einsum("bqhd,bqkhd->bqhk", q, kg) * self.scale  # [B,Q,H,Wn]
        scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bqhk,bqkhd->bqhd", attn, vg)               # [B,Q,H,d]
        return self.W_O(out.reshape(B, Q, self.dim))

    def forward(self, q_in: torch.Tensor, kv_grid: torch.Tensor,
                q_coords: torch.Tensor) -> torch.Tensor:
        """Gather-path cross-window attention (prepare keys/values, then attend).

        Args:
            q_in: ``[B, Q, dim]`` normalized query features (``LN_Q(x)``).
            kv_grid: ``[B, dim, h, w]`` normalized backbone tokens (``LN_KV(H)``).
            q_coords: ``[B, Q, 2]`` continuous query coordinates (token units).
        Returns:
            ``[B, Q, dim]`` attention output (after ``W_O``).
        """
        return self.forward_prepared(q_in, self.prepare_kv(kv_grid), q_coords)

    @torch.no_grad()
    def forward_dense(self, q_in: torch.Tensor, kv_grid: torch.Tensor,
                      q_coords: torch.Tensor) -> torch.Tensor:
        """Brute-force reference: full attention over all tokens, window-masked.

        Mathematically identical to :meth:`forward`; used only for testing.
        """
        B, Q, _ = q_in.shape
        _, C, h, w = kv_grid.shape
        kv_flat = kv_grid.permute(0, 2, 3, 1).reshape(B, h * w, C)       # [B,hw,C]
        centers = torch.stack(torch.meshgrid(
            torch.arange(h, device=q_in.device, dtype=q_in.dtype) + 0.5,
            torch.arange(w, device=q_in.device, dtype=q_in.dtype) + 0.5,
            indexing="ij"), dim=-1).reshape(h * w, 2)                    # [hw,2]

        win, half = self.window, self.window // 2
        qi = torch.clamp(torch.floor(q_coords[..., 0]).long(), 0, h - 1)  # [B,Q]
        qj = torch.clamp(torch.floor(q_coords[..., 1]).long(), 0, w - 1)
        ti = torch.arange(h * w, device=q_in.device) // w                 # [hw]
        tj = torch.arange(h * w, device=q_in.device) % w
        di = ti[None, None, :] - qi[..., None]                            # [B,Q,hw]
        dj = tj[None, None, :] - qj[..., None]
        in_win = ((di >= -half) & (di < -half + win) &
                  (dj >= -half) & (dj < -half + win))                     # [B,Q,hw]

        q = self._heads(self.W_Q(q_in))                                  # [B,Q,H,d]
        k = self._heads(self.W_K(kv_flat))                              # [B,hw,H,d]
        v = self._heads(self.W_V(kv_flat))
        cq, sq = self.rope.cos_sin(q_coords)
        q = self.rope.apply(q, cq.unsqueeze(-2), sq.unsqueeze(-2))
        ck, sk = self.rope.cos_sin(centers)                            # [hw,head_dim]
        k = self.rope.apply(k, ck.unsqueeze(-2), sk.unsqueeze(-2))      # [B,hw,H,d]

        scores = torch.einsum("bqhd,bkhd->bqhk", q, k) * self.scale     # [B,Q,H,hw]
        scores = scores.masked_fill(~in_win[:, :, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bqhk,bkhd->bqhd", attn, v).reshape(B, Q, self.dim)
        return self.W_O(out)
