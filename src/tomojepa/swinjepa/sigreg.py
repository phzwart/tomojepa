"""Per-stage SIGReg -- the sole anti-collapse term for Swin multi-scale JEPA.

The design (§4.7) is explicit: *if the repo already has a reference SIGReg
implementation, match it exactly instead of the BHEP formula.* It does --
:class:`tomojepa.core.model.SIGReg`, the LeJEPA collapse-prevention regularizer.
It matches the empirical characteristic function of random 1-D projections to
that of a standard isotropic Gaussian (knot-grid trapezoidal quadrature), is
``>= 0``, is minimized by ``N(0, I)``, and is fully differentiable. We reuse it
verbatim as the per-direction Gaussianity goodness-of-fit and wrap it with the
per-stage concerns the design calls for:

- **per-stage direction count** ``n_dirs`` (resampled every step inside the core
  module),
- a **token cap** so coarse-vs-fine token counts are bounded per step,
- an optional **FIFO feature queue** per stage so statistics-starved coarse
  stages (49 tokens/image at s4) can borrow detached history,
- a light explicit **mean penalty** ``w_mean * ||mu||^2`` for stability.

Because SIGReg is the only mechanism that controls collapse, the input must keep
its scale -- do **not** standardize variance away before calling this.

Scale note: the core SIGReg multiplies its CF goodness-of-fit by the sample
count (calibrated for LeJEPA where the sample dim is the batch, ~16). Applied
per-stage the sample dim is the *token* count (thousands), which would inflate
the term ~1000x and let it dominate the loss. We divide it back out so the
returned statistic is intensive (per-token) -- matching the design's O(1) BHEP
scale and making ``beta_sig`` robust to token count and input resolution.
"""
from typing import Optional

import torch
import torch.nn as nn

from ..core.model import SIGReg
from .losses import effective_rank
from .pyramid import gather_stage_tokens


class StageSIGReg(nn.Module):
    """SIGReg for one pyramid stage's token distribution (legacy ``legacy_jepa`` path).

    Returns an **intensive** (per-token) statistic: ``sig(...) / n_tokens``. The
    matching ``beta_sig`` entry is a per-token weight and is **not** comparable to
    the per-slice-scaled :class:`ImageGroupedStageSIGReg` used in the pyramid path.

    Args:
        dim: stage channel dim ``C_s`` (queue width).
        n_dirs: random projection directions (``n_sketches`` of the core SIGReg).
        knots, t_max: characteristic-function quadrature grid (core SIGReg).
        w_mean: weight on the explicit ``||mu||^2`` mean penalty.
        n_tokens_cap: cap on tokens used per step (0 = use all).
        queue_len: FIFO detached-feature queue length (0 = off).
    """

    def __init__(self, dim: int, n_dirs: int = 256, knots: int = 17,
                 t_max: float = 3.0, w_mean: float = 0.1,
                 n_tokens_cap: int = 4096, queue_len: int = 0):
        super().__init__()
        self.dim = dim
        self.w_mean = w_mean
        self.n_tokens_cap = n_tokens_cap
        self.queue_len = queue_len
        self.sig = SIGReg(knots=knots, t_max=t_max, n_sketches=n_dirs)
        if queue_len > 0:
            self.register_buffer("queue", torch.zeros(queue_len, dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_size", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, z: torch.Tensor) -> None:
        """Push detached features ``z`` [M, C] into the ring buffer."""
        m = z.shape[0]
        if m == 0:
            return
        if m >= self.queue_len:                       # keep the freshest queue_len
            self.queue.copy_(z[-self.queue_len:])
            self.queue_ptr[0] = 0
            self.queue_size[0] = self.queue_len
            return
        ptr = int(self.queue_ptr[0])
        end = ptr + m
        if end <= self.queue_len:
            self.queue[ptr:end] = z
        else:                                         # wrap around
            first = self.queue_len - ptr
            self.queue[ptr:] = z[:first]
            self.queue[: end - self.queue_len] = z[first:]
        self.queue_ptr[0] = end % self.queue_len
        self.queue_size[0] = min(self.queue_len, int(self.queue_size[0]) + m)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """SIGReg statistic for stage tokens ``z`` [N, C] (grad flows through z).

        Pairwise/CF sums run in fp32 for numerical stability of the exponentials,
        regardless of the surrounding autocast dtype.
        """
        z = z.float()
        if self.n_tokens_cap and z.shape[0] > self.n_tokens_cap:
            idx = torch.randperm(z.shape[0], device=z.device)[: self.n_tokens_cap]
            z = z[idx]
        mu = z.mean(0)

        est = z
        if self.queue_len > 0:
            qn = int(self.queue_size[0])
            if qn > 0:
                # Detached history only sharpens the estimate; gradient stays on z.
                est = torch.cat([z, self.queue[:qn].to(z.dtype)], dim=0)

        # The repo's core SIGReg multiplies the CF goodness-of-fit by the sample
        # count (calibrated for LeJEPA where N = batch ~ 16). Per stage here N is
        # the *token* count (up to thousands), which inflated the term ~1000x and
        # let SIGReg dominate the loss. Divide it back out so the statistic is
        # intensive (per-token), matching the design's O(1) BHEP scale -- this is
        # what makes ``beta_sig`` token-count- and resolution-robust.
        stat = self.sig(est.unsqueeze(0)) / est.shape[0]   # per-token CF GoF
        loss = stat + self.w_mean * mu.square().sum()

        if self.queue_len > 0:
            self._enqueue(z.detach())
        # Intensive per-token scale; see class docstring re beta_sig vs pyramid path.
        return loss


class PooledStageSIGReg(nn.Module):
    """Deprecated: FG-pooled ``[B,C]`` SIGReg (collapses within-slice structure).

    Kept for checkpoint compatibility; pyramid training uses
    :class:`ImageGroupedStageSIGReg` instead.
    """

    def __init__(self, dim: int, n_dirs: int = 256, knots: int = 17,
                 t_max: float = 3.0, w_mean: float = 0.1, queue_len: int = 512):
        super().__init__()
        self.dim = dim
        self.w_mean = w_mean
        self.queue_len = queue_len
        self.sig = SIGReg(knots=knots, t_max=t_max, n_sketches=n_dirs)
        if queue_len > 0:
            self.register_buffer("queue", torch.zeros(queue_len, dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_size", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, z: torch.Tensor) -> None:
        m = z.shape[0]
        if m == 0 or self.queue_len <= 0:
            return
        if m >= self.queue_len:
            self.queue.copy_(z[-self.queue_len:])
            self.queue_ptr[0] = 0
            self.queue_size[0] = self.queue_len
            return
        ptr = int(self.queue_ptr[0])
        end = ptr + m
        if end <= self.queue_len:
            self.queue[ptr:end] = z
        else:
            first = self.queue_len - ptr
            self.queue[ptr:] = z[:first]
            self.queue[: end - self.queue_len] = z[first:]
        self.queue_ptr[0] = end % self.queue_len
        self.queue_size[0] = min(self.queue_len, int(self.queue_size[0]) + m)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` ``[B, C]`` slice embeddings; grad flows through ``z``."""
        z = z.float()
        mu = z.mean(0)
        est = z
        if self.queue_len > 0:
            qn = int(self.queue_size[0])
            if qn > 0:
                est = torch.cat([z, self.queue[:qn].to(z.dtype)], dim=0)
        loss = self.sig(est.unsqueeze(0)) + self.w_mean * mu.square().sum()
        if self.queue_len > 0:
            self._enqueue(z.detach())
        return loss


class ImageGroupedStageSIGReg(nn.Module):
    """SIGReg on FG token subsamples with **slice-balanced** calibration (pyramid path).

    Returns a **per-batch-slice** statistic: ``stat * n_batch_slices / n_tokens``.
    The matching ``beta_sig`` entry weights that scale and is **not** comparable
    to the intensive :class:`StageSIGReg` used when ``legacy_jepa=True``.

    Each image contributes ``n_tokens_per_slice`` tokens (not one pooled vector).
    The core :class:`SIGReg` multiplies its statistic by the token count ``N``;
    correlated tokens from the same slice must not inflate that count. We
    re-scale by ``n_batch_slices / N`` so the loss magnitude tracks the current
    mini-batch (SSL / LeJEPA batch scale). Detached FIFO queue slices are
    included in the CF estimate for sharper statistics but do **not** multiply
    the returned value -- ``beta_sig`` stays stable as the queue fills.

    Projection directions are capped by ``n_dirs`` (bounded by latent dim) and
    ``n_slices``; optional rank-based capping is off by default because
    low-variance (collapsed) directions are the signal SIGReg must test.
    """

    def __init__(self, dim: int, n_dirs: int = 256, knots: int = 17,
                 t_max: float = 3.0, w_mean: float = 0.1,
                 n_tokens_per_slice: int = 32, min_grid_dist: int = 2,
                 queue_len: int = 512, cap_dirs_by_rank: bool = False,
                 min_dirs: int = 16, queue_token_cap: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.n_dirs = n_dirs
        self.w_mean = w_mean
        self.n_tokens_per_slice = n_tokens_per_slice
        self.min_grid_dist = min_grid_dist
        self.queue_len = queue_len
        self.cap_dirs_by_rank = cap_dirs_by_rank
        self.min_dirs = min_dirs
        self.queue_token_cap = (
            queue_token_cap if queue_token_cap is not None
            else max(n_tokens_per_slice, 1))
        self.sig = SIGReg(knots=knots, t_max=t_max, n_sketches=n_dirs)
        if queue_len > 0:
            self.register_buffer(
                "queue",
                torch.zeros(queue_len, self.queue_token_cap, dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_size", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, slices: torch.Tensor) -> None:
        """Push detached ``[B, M, C]`` slice token blocks into the ring buffer."""
        b = slices.shape[0]
        if b == 0 or self.queue_len <= 0:
            return
        if b >= self.queue_len:
            self.queue.copy_(slices[-self.queue_len:])
            self.queue_ptr[0] = 0
            self.queue_size[0] = self.queue_len
            return
        ptr = int(self.queue_ptr[0])
        end = ptr + b
        if end <= self.queue_len:
            self.queue[ptr:end] = slices
        else:
            first = self.queue_len - ptr
            self.queue[ptr:] = slices[:first]
            self.queue[: end - self.queue_len] = slices[first:]
        self.queue_ptr[0] = end % self.queue_len
        self.queue_size[0] = min(self.queue_len, int(self.queue_size[0]) + b)

    def _effective_dirs(self, n_slices: int, batch_tokens: torch.Tensor) -> int:
        """Cap random projections by slice count; rank cap optional.

        Low-variance directions are the collapse signal (variance != 1 shows up
        in the CF test) -- do not drop them unless ``cap_dirs_by_rank`` is on.
        """
        floor = max(self.min_dirs, n_slices)
        n_eff = min(self.n_dirs, floor)
        if self.cap_dirs_by_rank:
            r = effective_rank(batch_tokens.detach(),
                               max_tokens=batch_tokens.shape[0])
            n_eff = min(n_eff, max(self.min_dirs, int(round(r))))
        return n_eff

    def forward(self, feat: torch.Tensor,
                fg_stage: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``feat`` ``[B,C,h,w]`` stage map; grad flows through current-batch tokens."""
        if fg_stage is not None:
            slices, valid = gather_stage_tokens(
                feat, fg_stage, self.n_tokens_per_slice, self.min_grid_dist,
                token_cap=(self.queue_token_cap
                           if self.n_tokens_per_slice <= 0 else None),
                return_valid=True)
            if not valid.any():
                return feat.new_zeros(())
            slices = slices[valid]
        else:
            slices = gather_stage_tokens(
                feat, fg_stage, self.n_tokens_per_slice, self.min_grid_dist,
                token_cap=(self.queue_token_cap
                           if self.n_tokens_per_slice <= 0 else None))
        b, m, _ = slices.shape
        z = slices.reshape(b * m, -1).float()
        mu = z.mean(0)

        n_batch_slices = b
        est = z
        n_queue_slices = 0
        if self.queue_len > 0:
            qn = int(self.queue_size[0])
            if qn > 0:
                q_tok = self.queue[:qn].reshape(qn * m, -1).to(z.dtype)
                est = torch.cat([z, q_tok], dim=0)
                n_queue_slices = qn

        n_slices = n_batch_slices + n_queue_slices
        n_tokens = est.shape[0]

        n_eff_dirs = self._effective_dirs(n_slices, z)
        old_n = self.sig.n_sketches
        self.sig.n_sketches = n_eff_dirs
        stat = self.sig(est.unsqueeze(0))
        self.sig.n_sketches = old_n

        # Core SIGReg ~ O(n_tokens); rescale to O(n_batch_slices) (queue-free).
        stat = stat * (float(n_batch_slices) / max(1, n_tokens))
        loss = stat + self.w_mean * mu.square().sum()

        if self.queue_len > 0:
            self._enqueue(slices.detach())
        # Per-slice scale; see class docstring re beta_sig vs legacy StageSIGReg.
        return loss
