"""Initial query embedding q0 (paper section 2.1, Fig. 2B).

Reuses the backbone patch-embedding conv (so q0 lives in the backbone
patch-embed space) but applies it at a higher input resolution -- the input is
resized so the resulting patch-token grid is ``query_embed_grid x
query_embed_grid`` tokens. That high-res grid is cached per image and bilinearly
sampled at each continuous query coordinate ``x_q`` to produce ``q0 in R^C``.
Because patch embedding is a single conv, the high-res pass is cheap and the
cache is reused across every query (and every chunk) for the same image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEmbedding(nn.Module):
    def __init__(self, adapter, query_embed_grid: int = 224):
        super().__init__()
        self.adapter = adapter
        self.grid = query_embed_grid

    @torch.no_grad()
    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        side = self.grid * self.adapter.p
        if img.shape[-2] == side and img.shape[-1] == side:
            return img
        return F.interpolate(img, size=(side, side), mode="bilinear",
                             align_corners=False)

    def compute_cache(self, img: torch.Tensor) -> torch.Tensor:
        """High-res patch-token grid ``[B, C, G, G]`` for ``img``.

        Runs only the patch-embed module (not the backbone blocks). The LoRA on
        the student patch-embed conv (if present) participates here so q0 is
        aligned with the adapted embedding space.
        """
        hi = self._resize(img)
        feat = self.adapter.patch_embed(hi)
        return self.adapter._to_nchw(feat, self.grid, self.grid)

    def sample(self, cache: torch.Tensor, q_coords: torch.Tensor,
               h: int, w: int) -> torch.Tensor:
        """Bilinearly sample ``cache`` at ``q_coords`` (low-res token units).

        ``q_coords`` are in the low-resolution token grid units (extent
        ``[0, h] x [0, w]``); they are mapped to image fractions and sampled
        from the high-res cache via ``grid_sample`` (align_corners=False, so a
        coordinate at a cell center returns that cell's value).

        Returns ``q0`` of shape ``[B, Q, C]``.
        """
        B, Q, _ = q_coords.shape
        fy = q_coords[..., 0] / h
        fx = q_coords[..., 1] / w
        grid = torch.stack([fx * 2.0 - 1.0, fy * 2.0 - 1.0], dim=-1)  # (x,y) order
        grid = grid.view(B, Q, 1, 2).to(cache.dtype)
        out = F.grid_sample(cache, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)  # [B,C,Q,1]
        return out[..., 0].transpose(1, 2).contiguous()                  # [B,Q,C]
