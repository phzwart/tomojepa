"""Epps-Pulley SIGReg — exact LeJEPA formulation (spec §7.2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from tomojepa.core import dist as D

from ..config import SigRegCfg


class SIGReg(nn.Module):
    """Epps-Pulley characteristic-function Gaussianity statistic."""

    def __init__(self, cfg: SigRegCfg, feat_dim: int):
        super().__init__()
        t = torch.linspace(-cfg.t_max, cfg.t_max, cfg.n_knots)
        dt = t[1] - t[0]
        trap = torch.full_like(t, dt)
        trap[0] *= 0.5
        trap[-1] *= 0.5
        w = torch.exp(-t ** 2 / cfg.sigma ** 2)
        self.register_buffer("t", t)
        self.register_buffer("quad_w", trap * w)
        self.register_buffer("target_cf", torch.exp(-t ** 2 / 2.0))
        self.cfg = cfg
        self.feat_dim = feat_dim

    def sample_dirs(self, device: torch.device, step: int) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(int(step) % (2 ** 31))
        a = torch.randn(self.cfg.num_slices, self.feat_dim, generator=g, device=device)
        return a / a.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def _ecf_terms(self, proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """proj: [n, M] -> cos_mean, sin_mean each [K, M]."""
        import torch.distributed as dist

        n = proj.shape[0]
        t = self.t.view(-1, 1, 1)
        ang = t * proj.unsqueeze(0)
        cos_sum = ang.cos().sum(dim=1)
        sin_sum = ang.sin().sum(dim=1)
        if D.is_distributed():
            dist.all_reduce(cos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sin_sum, op=dist.ReduceOp.SUM)
            n_total = torch.tensor(float(n), device=proj.device)
            dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
            denom = n_total.item()
        else:
            denom = float(n)
        return cos_sum / denom, sin_sum / denom

    def forward(self, z: torch.Tensor, step: int = 0) -> torch.Tensor:
        z = z.float()
        a = self.sample_dirs(z.device, step)
        proj = z @ a.t()
        cos_mean, sin_mean = self._ecf_terms(proj)
        diff2 = (cos_mean - self.target_cf.view(-1, 1)) ** 2 + sin_mean ** 2
        ep_per_dir = (self.quad_w.view(-1, 1) * diff2).sum(dim=0)
        loss = ep_per_dir.mean()
        if not self.cfg.fold_N_into_lambda:
            loss = loss * z.shape[0]
        return loss
