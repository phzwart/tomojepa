"""ViTUp -- query embedding -> T refinement blocks -> per-stage decoders.

Assembles the full model and implements dense querying with memory-bounded
*query chunking* (paper section IV.H): output queries are conditionally
independent given the backbone features, so the image context (backbone hidden
states + high-res patch-embed cache) is computed once and queries are evaluated
in chunks of at most ``query_chunk_size``. Concatenating chunk results yields the
exact same map as processing all queries at once (chunk-invariant).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import ViTUpConfig
from .backbone_adapter import BackboneAdapter
from .query_embedding import QueryEmbedding
from .block import ViTUpBlock
from .decoder import StageDecoders


@dataclass
class ImageContext:
    """Per-image, query-independent state reused across all query chunks."""
    hidden: Dict[int, torch.Tensor]   # {layer: [B,C,h,w]}
    cache: torch.Tensor               # high-res patch-embed grid [B,C,G,G]
    h: int
    w: int
    kv: List[dict]                    # per-block pre-projected/rotated keys+values


class ViTUp(nn.Module):
    def __init__(self, adapter: BackboneAdapter, cfg: ViTUpConfig):
        super().__init__()
        self.adapter = adapter
        self.cfg = cfg
        self.T = cfg.num_blocks
        self.layer_indices = list(cfg.layer_indices)
        if len(self.layer_indices) != self.T:
            raise ValueError(
                f"layer_indices ({len(self.layer_indices)}) must have length "
                f"num_blocks ({self.T})")
        C = adapter.C
        D = cfg.resolved_internal_dim(C)
        self.internal_dim = D
        self.output_dim = C

        self.query_embedding = QueryEmbedding(adapter, cfg.query_embed_grid)
        self.blocks = nn.ModuleList([
            ViTUpBlock(internal_dim=D, backbone_dim=C, num_heads=cfg.num_heads,
                       window=cfg.attention_window, posenc_dim=cfg.featx_posenc_dim,
                       mlp_ratio=cfg.mlp_ratio, rope_theta=cfg.rope_theta)
            for _ in range(self.T)
        ])
        self.decoders = StageDecoders(self.T, D, C)

    # -- context (computed once per image) ----------------------------------
    def encode_image(self, img: torch.Tensor) -> ImageContext:
        """Backbone hidden states + high-res patch-embed cache for ``img``.

        Also precomputes each block's key/value projections + RoPE once (they are
        query-independent), so chunked querying does not redo them per chunk.
        """
        hidden = self.adapter.hidden_states(img, self.layer_indices)
        cache = self.query_embedding.compute_cache(img)
        h = img.shape[-2] // self.adapter.p
        w = img.shape[-1] // self.adapter.p
        kv = [self.blocks[t].prepare_kv(hidden[self.layer_indices[t]])
              for t in range(self.T)]
        return ImageContext(hidden=hidden, cache=cache, h=h, w=w, kv=kv)

    # -- query a (chunk of) coordinates -------------------------------------
    def query_stages(self, ctx: ImageContext, q_coords: torch.Tensor,
                     stages: str = "all") -> List[torch.Tensor]:
        """Run the refinement stages on ``q_coords`` ``[B, Q, 2]``.

        Returns a list of decoded outputs ``o_t`` ``[B, Q, C]``. If
        ``stages == 'all'`` the list is ``[o_0, ..., o_T]``; if ``'last'`` only
        ``[o_T]``.
        """
        p = self.adapter.p
        q = self.query_embedding.sample(ctx.cache, q_coords, ctx.h, ctx.w)  # q0
        outs: List[torch.Tensor] = []
        if stages == "all":
            outs.append(self.decoders(0, q))
        for t in range(1, self.T + 1):
            h_grid = ctx.hidden[self.layer_indices[t - 1]]
            q = self.blocks[t - 1](q, h_grid, q_coords, p, kv=ctx.kv[t - 1])
            if stages == "all" or t == self.T:
                outs.append(self.decoders(t, q))
        return outs

    # -- chunked querying ----------------------------------------------------
    def query(self, ctx: ImageContext, q_coords: torch.Tensor,
              stages: str = "last", chunk_size: Optional[int] = None
              ) -> List[torch.Tensor]:
        """Chunk-invariant querying over arbitrary coordinates.

        Splits ``q_coords`` into chunks of at most ``chunk_size`` queries,
        evaluates each, and concatenates -- producing the exact same result as
        an unchunked pass.
        """
        chunk = chunk_size or self.cfg.query_chunk_size
        Q = q_coords.shape[1]
        if chunk is None or chunk <= 0 or Q <= chunk:
            return self.query_stages(ctx, q_coords, stages)
        partials: List[List[torch.Tensor]] = []
        for s in range(0, Q, chunk):
            partials.append(self.query_stages(ctx, q_coords[:, s:s + chunk], stages))
        n_stages = len(partials[0])
        return [torch.cat([p[i] for p in partials], dim=1) for i in range(n_stages)]

    # -- dense upsampling ----------------------------------------------------
    def dense_grid_coords(self, ctx: ImageContext, out_h: int, out_w: int,
                          device=None, dtype=torch.float32) -> torch.Tensor:
        """Output-grid coordinates (token units) covering the image extent.

        Row ``r`` maps to ``y = (r + 0.5) / out_h * h`` (and likewise for ``x``),
        so the dense grid spans the same extent as the low-res token grid.
        """
        device = device or ctx.cache.device
        ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) / out_h * ctx.h
        xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) / out_w * ctx.w
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx], dim=-1).reshape(out_h * out_w, 2)

    def upsample(self, img: torch.Tensor, out_h: int, out_w: int,
                 chunk_size: Optional[int] = None) -> torch.Tensor:
        """Dense high-resolution feature map ``H_L^up`` ``[B, out_h, out_w, C]``."""
        ctx = self.encode_image(img)
        B = img.shape[0]
        coords = self.dense_grid_coords(ctx, out_h, out_w, device=img.device)
        coords = coords.unsqueeze(0).expand(B, -1, 2)
        o_last = self.query(ctx, coords, stages="last", chunk_size=chunk_size)[-1]
        return o_last.reshape(B, out_h, out_w, self.output_dim)

    def forward(self, img: torch.Tensor, q_coords: torch.Tensor,
                stages: str = "all", chunk_size: Optional[int] = None
                ) -> List[torch.Tensor]:
        ctx = self.encode_image(img)
        return self.query(ctx, q_coords, stages=stages, chunk_size=chunk_size)

    # -- parameter helpers ---------------------------------------------------
    def vitup_parameters(self):
        """ViT-Up's own parameters (blocks + decoders), excluding the backbone."""
        for name, p in self.named_parameters():
            if not name.startswith("adapter.backbone"):
                yield p
