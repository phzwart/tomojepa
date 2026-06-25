"""Single-grid patch masking for BandedViT SimMIM training."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class PatchBlockMask:
    """Sample patch-grid masks with a fixed masked-cell budget per sample."""

    def __init__(
        self,
        grid: int,
        mask_ratio: float = 0.45,
        mask_mode: str = "random_cell",
        num_blocks: int = 4,
        block_scale_range: Tuple[float, float] = (0.1, 0.4),
    ):
        if mask_mode not in ("random_cell", "block"):
            raise ValueError(f"unknown mask_mode: {mask_mode!r}")
        self.grid = int(grid)
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = mask_mode
        self.num_blocks = int(num_blocks)
        self.block_scale_range = tuple(block_scale_range)
        self.n_cells = self.grid * self.grid
        self.k = max(1, min(self.n_cells - 1, int(round(self.mask_ratio * self.n_cells))))

    def _sample_random_cell(
        self, b: int, device: torch.device, fg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = torch.rand(b, self.n_cells, device=device)
        if fg is not None:
            fg_flat = fg.reshape(b, -1)
            scores = scores.masked_fill(~fg_flat, -1.0)
            for s in range(b):
                n_fg = int(fg_flat[s].sum())
                if n_fg <= 0:
                    raise ValueError("foreground mask is empty for a sample")
                if self.k >= n_fg:
                    scores[s].masked_fill_(fg_flat[s], 1.0)
        idx = scores.topk(self.k, dim=1).indices
        flat = torch.zeros(b, self.n_cells, dtype=torch.bool, device=device)
        flat.scatter_(1, idx, True)
        return flat.view(b, self.grid, self.grid)

    def _sample_block(
        self, b: int, device: torch.device, fg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lo, hi = self.block_scale_range
        mask = torch.zeros(b, self.grid, self.grid, dtype=torch.bool, device=device)
        for s in range(b):
            guard = 0
            while int(mask[s].sum()) < self.k and guard < 10 * max(1, self.num_blocks):
                guard += 1
                scale = torch.empty(1, device=device).uniform_(lo, hi).item()
                area = max(1, int(round(scale * self.n_cells)))
                log_ar = torch.empty(1, device=device).uniform_(
                    math.log(0.5), math.log(2.0)
                ).item()
                ar = math.exp(log_ar)
                bh = min(max(int(round(math.sqrt(area * ar))), 1), self.grid)
                bw = min(max(int(round(math.sqrt(area / ar))), 1), self.grid)
                top = int(torch.randint(0, self.grid - bh + 1, (1,), device=device))
                left = int(torch.randint(0, self.grid - bw + 1, (1,), device=device))
                block = torch.zeros(self.grid, self.grid, dtype=torch.bool, device=device)
                block[top : top + bh, left : left + bw] = True
                mask[s] |= block
            if fg is not None:
                mask[s] &= fg[s]
            flat = mask[s].view(-1)
            cur = int(flat.sum())
            if cur > self.k:
                on = flat.nonzero(as_tuple=False).squeeze(1)
                drop = on[torch.randperm(on.numel(), device=device)[: cur - self.k]]
                flat[drop] = False
            elif cur < self.k:
                off = (~flat).nonzero(as_tuple=False).squeeze(1)
                need = self.k - cur
                if off.numel() >= need:
                    add = off[torch.randperm(off.numel(), device=device)[:need]]
                    flat[add] = True
        return mask

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        fg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[B, grid, grid]`` bool mask (True = masked position)."""
        if self.mask_mode == "random_cell":
            return self._sample_random_cell(batch_size, device, fg=fg)
        return self._sample_block(batch_size, device, fg=fg)

    def flat(self, mask: torch.Tensor) -> torch.Tensor:
        """``[B, grid, grid]`` -> ``[B, num_patches]``."""
        return mask.reshape(mask.shape[0], -1)
