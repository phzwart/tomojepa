"""Per-block SIGReg for BandedViT transformer taps."""
from __future__ import annotations

import torch
import torch.nn as nn

from tomojepa.core.model import SIGReg


class BlockSIGReg(nn.Module):
    """Intensive (per-token) SIGReg on one block's patch tokens."""

    def __init__(
        self,
        dim: int,
        n_dirs: int = 256,
        knots: int = 17,
        t_max: float = 3.0,
        w_mean: float = 0.1,
        n_tokens_cap: int = 4096,
        queue_len: int = 0,
    ):
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
        m = z.shape[0]
        if m == 0:
            return
        if m >= self.queue_len:
            self.queue.copy_(z[-self.queue_len :])
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
        """``z`` [N, C] or [B, N, C] -> scalar intensive SIGReg statistic."""
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])
        z = z.float()
        if self.n_tokens_cap and z.shape[0] > self.n_tokens_cap:
            idx = torch.randperm(z.shape[0], device=z.device)[: self.n_tokens_cap]
            z = z[idx]
        mu = z.mean(0)
        est = z
        if self.queue_len > 0:
            qn = int(self.queue_size[0])
            if qn > 0:
                est = torch.cat([z, self.queue[:qn].to(z.dtype)], dim=0)
        stat = self.sig(est.unsqueeze(0)) / est.shape[0]
        loss = stat + self.w_mean * mu.square().sum()
        if self.queue_len > 0:
            self._enqueue(z.detach())
        return loss
