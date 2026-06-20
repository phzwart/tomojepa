"""Multi-scale block mask generator.

Masking is defined on the **coarsest (stage-4) grid** and expanded by
nearest-neighbour to the finer grids, so every coarser-stage token is cleanly
fully-masked or fully-visible: a stage-4 cell's footprint is exactly
``2^(S-1) = 8`` stage-1 tokens on a side (4x4 at s2, 2x2 at s3, 1x1 at s4). The
stage-1 expansion (``mask["s1"]``) is what the backbone injects mask tokens at;
``mask["s_i"]`` selects the loss positions at stage ``i``.

A **fixed** number of stage-4 cells ``k = round(mask_ratio * cells)`` is masked
per sample (constant across the batch). That keeps the masked/visible token
counts identical for every sample, so the predictor can assemble rectangular
``[B, N, D]`` memory/query tensors without per-sample padding. Both
``random_cell`` and ``block`` modes honour the exact-``k`` budget.

Forward-compatibility: the grid is factored ``(h, w)`` (not a square scalar) and
expansion is a generic nearest upsample, so a later 3-D swap (``(h, w, d)``
sub-volume blocks) only changes this module and the backbone -- not the
predictor, loss, or training interface.
"""
from typing import Dict, Optional

import torch


class MultiScaleBlockMask:
    """Sample a stage-4 mask and expand it to every pyramid stage.

    Args:
        grid4: coarsest grid ``(h4, w4)`` (stage-4 token grid).
        num_stages: pyramid depth ``S`` (4 for Swin-T). Finer grids are
            ``grid4 * 2^(S-1-s)``.
        mask_ratio: fraction of stage-4 cells masked (exact ``k`` per sample).
        mask_mode: ``"random_cell"`` (Bernoulli cells) or ``"block"``
            (contiguous rectangular blocks, I-JEPA-style).
        num_blocks: number of rectangular blocks for ``"block"`` mode.
        block_scale_range: per-block area fraction range (of the s4 grid) for
            ``"block"`` mode.
        device: tensor device for the generated masks.
    """

    def __init__(self, grid4, num_stages: int = 4, mask_ratio: float = 0.55,
                 mask_mode: str = "random_cell", num_blocks: int = 4,
                 block_scale_range=(0.1, 0.4), device=None):
        if mask_mode not in ("random_cell", "block"):
            raise ValueError(f"unknown mask_mode: {mask_mode!r}")
        self.grid4 = (int(grid4[0]), int(grid4[1]))
        self.num_stages = int(num_stages)
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = mask_mode
        self.num_blocks = int(num_blocks)
        self.block_scale_range = tuple(block_scale_range)
        self.device = device
        h4, w4 = self.grid4
        self.n_cells = h4 * w4
        # At least one masked and one visible cell.
        self.k = max(1, min(self.n_cells - 1, int(round(self.mask_ratio * self.n_cells))))

    def stage_grid(self, stage_idx: int):
        """Token grid ``(h_s, w_s)`` at stage ``stage_idx`` (0-based)."""
        f = 2 ** (self.num_stages - 1 - stage_idx)
        return (self.grid4[0] * f, self.grid4[1] * f)

    def _eligible_s4_from_s1(self, fg_s1: torch.Tensor) -> torch.Tensor:
        """s4 cells whose entire stage-1 expansion lies inside ``fg_s1``.

        Sampling masked cells from this grid guarantees the expanded ``mask["s1"]``
        never selects out-of-FOV tokens (avg-pool FG at s4 alone can disagree with
        the finer s1 footprint near the FOV boundary).
        """
        b, h1, w1 = fg_s1.shape
        h4, w4 = self.grid4
        if h1 % h4 != 0 or w1 % w4 != 0:
            raise ValueError(
                f"fg_s1 grid {(h1, w1)} not a clean multiple of s4 {self.grid4}.")
        fh, fw = h1 // h4, w1 // w4
        fg = fg_s1.view(b, h4, fh, w4, fw)
        return fg.all(dim=(2, 4))

    def _effective_k(self, fg_s4: torch.Tensor) -> int:
        """Largest ``k`` that keeps one visible FG cell per sample (rectangular batch)."""
        b = fg_s4.shape[0]
        k_eff = self.k
        for s in range(b):
            n_fg = int(fg_s4[s].sum().item())
            k_eff = min(k_eff, max(0, n_fg - 1))
        k_eff = max(1, k_eff)
        for s in range(b):
            n_fg = int(fg_s4[s].sum().item())
            if n_fg < k_eff + 1:
                raise ValueError(
                    f"FG sample {s} has {n_fg} eligible s4 cells but k_eff={k_eff} "
                    f"requires at least {k_eff + 1}")
        return k_eff

    @staticmethod
    def _top_up_fg_to_k(flat: torch.Tensor, fg_flat: torch.Tensor,
                        k_eff: int, device) -> None:
        """Enable random unmasked FG cells until ``flat`` has exactly ``k_eff`` True."""
        cur = int(flat.sum())
        if cur >= k_eff:
            return
        off = (~flat & fg_flat).nonzero(as_tuple=False).squeeze(1)
        need = k_eff - cur
        if off.numel() >= need:
            add = off[torch.randperm(off.numel(), device=device)[:need]]
            flat[add] = True
        elif off.numel() > 0:
            flat[off] = True

    def _finalize_fg_mask4(self, mask4: torch.Tensor, fg_s4: torch.Tensor,
                           device) -> torch.Tensor:
        """Drop any BG-masked s4 cells and rebalance to batch-constant ``k_eff``."""
        mask4 = mask4 & fg_s4
        k_eff = self._effective_k(fg_s4)
        for s in range(mask4.shape[0]):
            flat = mask4[s].view(-1)
            fg_flat = fg_s4[s].reshape(-1)
            cur = int(flat.sum())
            if cur > k_eff:
                on = flat.nonzero(as_tuple=False).squeeze(1)
                drop = on[torch.randperm(on.numel(), device=device)[: cur - k_eff]]
                flat[drop] = False
            elif cur < k_eff:
                self._top_up_fg_to_k(flat, fg_flat, k_eff, device)
        return mask4

    def _sample_random_cell_fg(self, b: int, device, fg_s4: torch.Tensor) -> torch.Tensor:
        """``[b, h4, w4]`` bool: exactly ``k_eff`` True cells per sample, FG only."""
        k_eff = self._effective_k(fg_s4)
        mask4 = torch.zeros(b, *self.grid4, dtype=torch.bool, device=device)
        for s in range(b):
            fg_idx = fg_s4[s].reshape(-1).nonzero(as_tuple=True)[0]
            n_fg = fg_idx.numel()
            k_s = min(k_eff, max(0, n_fg - 1))
            if k_s > 0:
                pick = fg_idx[torch.randperm(n_fg, device=device)[:k_s]]
                mask4[s].view(-1)[pick] = True
            flat = mask4[s].view(-1)
            fg_flat = fg_s4[s].reshape(-1)
            self._top_up_fg_to_k(flat, fg_flat, k_eff, device)
        return mask4

    def _sample_block_fg(self, b: int, device, fg_s4: torch.Tensor) -> torch.Tensor:
        """Block mode restricted to the FOV; trimmed/topped within FG cells."""
        import math
        h4, w4 = self.grid4
        lo, hi = self.block_scale_range
        k_eff = self._effective_k(fg_s4)
        mask = torch.zeros(b, h4, w4, dtype=torch.bool, device=device)
        for s in range(b):
            fg = fg_s4[s]
            guard = 0
            while int(mask[s].sum()) < k_eff and guard < 10 * max(1, self.num_blocks):
                guard += 1
                scale = torch.empty(1, device=device).uniform_(lo, hi).item()
                area = max(1, int(round(scale * self.n_cells)))
                log_ar = torch.empty(1, device=device).uniform_(math.log(0.5),
                                                                math.log(2.0)).item()
                ar = math.exp(log_ar)
                bh = min(max(int(round(math.sqrt(area * ar))), 1), h4)
                bw = min(max(int(round(math.sqrt(area / ar))), 1), w4)
                top = int(torch.randint(0, h4 - bh + 1, (1,), device=device))
                left = int(torch.randint(0, w4 - bw + 1, (1,), device=device))
                block = torch.zeros(h4, w4, dtype=torch.bool, device=device)
                block[top:top + bh, left:left + bw] = True
                mask[s] |= block & fg
            flat = mask[s].view(-1)
            cur = int(flat.sum())
            fg_flat = fg.reshape(-1)
            if cur > k_eff:
                on = flat.nonzero(as_tuple=False).squeeze(1)
                drop = on[torch.randperm(on.numel(), device=device)[: cur - k_eff]]
                flat[drop] = False
            elif cur < k_eff:
                off = (~flat & fg_flat).nonzero(as_tuple=False).squeeze(1)
                self._top_up_fg_to_k(flat, fg_flat, k_eff, device)
        return mask

    def _sample_random_cell(self, b: int, device) -> torch.Tensor:
        """``[b, h4, w4]`` bool with exactly ``k`` True cells per sample."""
        scores = torch.rand(b, self.n_cells, device=device)
        idx = scores.topk(self.k, dim=1).indices                 # k smallest-rank cells
        flat = torch.zeros(b, self.n_cells, dtype=torch.bool, device=device)
        flat.scatter_(1, idx, True)
        return flat.view(b, *self.grid4)

    def _sample_block(self, b: int, device) -> torch.Tensor:
        """``[b, h4, w4]`` bool: contiguous blocks, then trimmed/topped to ``k``."""
        import math
        h4, w4 = self.grid4
        lo, hi = self.block_scale_range
        mask = torch.zeros(b, h4, w4, dtype=torch.bool, device=device)
        for s in range(b):
            guard = 0
            while int(mask[s].sum()) < self.k and guard < 10 * max(1, self.num_blocks):
                guard += 1
                scale = torch.empty(1, device=device).uniform_(lo, hi).item()
                area = max(1, int(round(scale * self.n_cells)))
                log_ar = torch.empty(1, device=device).uniform_(math.log(0.5),
                                                                math.log(2.0)).item()
                ar = math.exp(log_ar)
                bh = min(max(int(round(math.sqrt(area * ar))), 1), h4)
                bw = min(max(int(round(math.sqrt(area / ar))), 1), w4)
                top = int(torch.randint(0, h4 - bh + 1, (1,), device=device))
                left = int(torch.randint(0, w4 - bw + 1, (1,), device=device))
                mask[s, top:top + bh, left:left + bw] = True
            # Enforce the exact-k budget so batch tensors stay rectangular.
            flat = mask[s].view(-1)
            cur = int(flat.sum())
            if cur > self.k:                                     # trim random extras
                on = flat.nonzero(as_tuple=False).squeeze(1)
                drop = on[torch.randperm(on.numel(), device=device)[: cur - self.k]]
                flat[drop] = False
            elif cur < self.k:                                   # top up random cells
                off = (~flat).nonzero(as_tuple=False).squeeze(1)
                add = off[torch.randperm(off.numel(), device=device)[: self.k - cur]]
                flat[add] = True
        return mask

    @staticmethod
    def expand(mask4: torch.Tensor, grid_s) -> torch.Tensor:
        """Nearest-expand a stage-4 mask ``[B, h4, w4]`` to ``[B, h_s, w_s]``."""
        h4, w4 = mask4.shape[-2:]
        hs, ws = grid_s
        if hs % h4 != 0 or ws % w4 != 0:
            raise ValueError(f"grid {grid_s} not a clean multiple of s4 {(h4, w4)}.")
        fh, fw = hs // h4, ws // w4
        return mask4.repeat_interleave(fh, dim=-2).repeat_interleave(fw, dim=-1)

    def generate(self, b: int, device=None,
                 fg_s4: Optional[torch.Tensor] = None,
                 fg_s1: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return ``{"s1".."sS": [b, h_s, w_s] bool}`` (True = masked).

        The stage-4 mask is sampled once and expanded to every grid, so all
        stages select exactly the same coarse footprint. When ``fg_s1`` or ``fg_s4``
        is given, masked cells are sampled only from foreground positions; ``k`` is
        capped per batch so at least one visible FG cell remains in every sample.
        Prefer ``fg_s1`` so the expanded stage-1 mask stays inside the FOV.
        Background tokens are never masked: the stage-4 mask is scrubbed against
        ``fg_s4`` and rebalanced before expansion.
        """
        device = device or self.device or torch.device("cpu")
        if fg_s1 is not None:
            if fg_s1.shape[0] != b:
                raise ValueError(f"fg_s1 batch {fg_s1.shape[0]} != b={b}")
            fg_s4 = self._eligible_s4_from_s1(fg_s1)
        if fg_s4 is not None:
            if fg_s4.shape[0] != b:
                raise ValueError(f"fg_s4 batch {fg_s4.shape[0]} != b={b}")
            if tuple(fg_s4.shape[-2:]) != self.grid4:
                raise ValueError(
                    f"fg_s4 grid {tuple(fg_s4.shape[-2:])} != s4 {self.grid4}")
            if self.mask_mode == "random_cell":
                mask4 = self._sample_random_cell_fg(b, device, fg_s4)
            else:
                mask4 = self._sample_block_fg(b, device, fg_s4)
            mask4 = self._finalize_fg_mask4(mask4, fg_s4, device)
        elif self.mask_mode == "random_cell":
            mask4 = self._sample_random_cell(b, device)
        else:
            mask4 = self._sample_block(b, device)
        counts = mask4.flatten(1).sum(1)
        if counts.min() != counts.max():
            raise AssertionError(
                f"FG/block mask path: per-sample masked counts differ "
                f"{counts.tolist()}; expected batch-constant k={int(counts.max())}")
        out: Dict[str, torch.Tensor] = {}
        for s in range(self.num_stages):
            out[f"s{s + 1}"] = self.expand(mask4, self.stage_grid(s))
        return out


def assert_mask_consistency(mask: Dict[str, torch.Tensor], num_stages: int = 4) -> None:
    """Verify every stage mask is the exact nearest-expansion of stage-4.

    Re-derives each finer stage from ``mask["s{num_stages}"]`` and asserts
    equality. Cheap enough to call once per step in debug / tests.
    """
    coarse_key = f"s{num_stages}"
    mask4 = mask[coarse_key]
    for s in range(num_stages):
        grid_s = mask[f"s{s + 1}"].shape[-2:]
        expanded = MultiScaleBlockMask.expand(mask4, grid_s)
        if not torch.equal(expanded, mask[f"s{s + 1}"]):
            raise AssertionError(
                f"mask['s{s + 1}'] is not the nearest expansion of {coarse_key}.")
