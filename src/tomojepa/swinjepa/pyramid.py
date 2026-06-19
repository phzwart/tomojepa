"""Hierarchical residual pyramid for Swin multi-scale JEPA.

Coarse stage s4 carries a smooth envelope ``C4`` from an SSL-style MIM head on
masked student features. Finer stages hold residuals ``R_s = E_s - sg(up(parent))``
where the parent of s3 is ``C4`` and finer parents are the full target-pass
maps ``E_{s+1}``.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomojepa.core.model import masked_mean


def upsample_stage(feat: torch.Tensor, grid_fine: Tuple[int, int]) -> torch.Tensor:
    """Nearest 2× upsample ``[B,C,h,w]`` to ``grid_fine`` (matches mask.expand)."""
    h_c, w_c = feat.shape[-2:]
    h_f, w_f = grid_fine
    if (h_f, w_f) == (h_c, w_c):
        return feat
    if h_f % h_c or w_f % w_c:
        raise ValueError(
            f"fine grid {(h_f, w_f)} not a clean multiple of coarse {(h_c, w_c)}")
    fh, fw = h_f // h_c, w_f // w_c
    return feat.repeat_interleave(fh, dim=-2).repeat_interleave(fw, dim=-1)


def fg_gate(feat: torch.Tensor, fg: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero maps outside strict FG when ``fg`` is given."""
    if fg is None:
        return feat
    return feat * fg.unsqueeze(1).to(feat.dtype)


def pool_stage_embeddings(feat: torch.Tensor,
                          fg_stage: Optional[torch.Tensor] = None) -> torch.Tensor:
    """FG mean-pool ``[B,C,h,w]`` -> ``[B,C]`` (diagnostics / probes only)."""
    b, c, h, w = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if fg_stage is None:
        return tok.mean(dim=1)
    fg = fg_stage.reshape(b, h * w)
    return masked_mean(tok, fg)


def _greedy_min_dist_pick(coords: torch.Tensor, n_pick: int,
                          min_dist: int) -> torch.Tensor:
    """Pick ``n_pick`` indices from ``coords`` ``[N,2]`` (y, x) with min L∞ separation."""
    n = coords.shape[0]
    if n_pick <= 0:
        raise ValueError(f"n_pick must be positive, got {n_pick}")
    if n_pick >= n or min_dist <= 0:
        return torch.randperm(n, device=coords.device)[:n_pick]
    perm = torch.randperm(n, device=coords.device)
    picked: list[int] = []
    for idx in perm.tolist():
        if len(picked) >= n_pick:
            break
        if not picked:
            picked.append(idx)
            continue
        sel = coords[picked]
        sep = (coords[idx] - sel).abs().max(dim=-1).values.min()
        if sep >= min_dist:
            picked.append(idx)
    if len(picked) < n_pick:
        for idx in perm.tolist():
            if len(picked) >= n_pick:
                break
            if idx not in picked:
                picked.append(idx)
    idx = torch.tensor(picked[:n_pick], device=coords.device, dtype=torch.long)
    return idx


def gather_stage_tokens(feat: torch.Tensor,
                        fg_stage: Optional[torch.Tensor] = None,
                        n_per_slice: int = 32,
                        min_grid_dist: int = 0) -> torch.Tensor:
    """FG token subsample per slice: ``[B,C,h,w]`` -> ``[B, M, C]`` with fixed ``M``.

    Used by image-grouped SIGReg so within-slice correlation is not treated as
    independent cross-image samples. When ``min_grid_dist > 0``, tokens are
    chosen with minimum Chebyshev grid separation (decorrelates nearby patches).
    """
    if n_per_slice <= 0:
        raise ValueError(f"n_per_slice must be positive, got {n_per_slice}")
    b, c, h, w = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=feat.device),
        torch.arange(w, device=feat.device),
        indexing="ij")
    coords_flat = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    out = []
    for i in range(b):
        if fg_stage is not None:
            fg = fg_stage[i].reshape(-1)
            idx_pool = fg.nonzero(as_tuple=False).squeeze(-1)
            if idx_pool.numel() == 0:
                idx_pool = torch.arange(h * w, device=feat.device)
        else:
            idx_pool = torch.arange(h * w, device=feat.device)
        coords = coords_flat[idx_pool]
        pick = _greedy_min_dist_pick(coords, n_per_slice, min_grid_dist)
        out.append(tok[i][idx_pool[pick]])
    return torch.stack(out, dim=0)


class CoarseMIMHead(nn.Module):
    """SSL-style smooth field ``C4`` from masked student s4 latents.

    Per-token MLP implemented as 1×1 convs on the stage map ``[B,C,h4,w4]``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


def hierarchical_residuals(
        E_full: Dict[str, torch.Tensor],
        C4: torch.Tensor,
        grids: List[Tuple[int, int]],
        fg_stages: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """``R3,R2,R1`` from target-pass maps and stop-grad coarse parents."""
    c4_up = upsample_stage(C4.detach(), grids[2])          # -> s3 grid
    r3 = E_full["s3"] - c4_up
    e3_up = upsample_stage(E_full["s3"].detach(), grids[1])
    r2 = E_full["s2"] - e3_up
    e2_up = upsample_stage(E_full["s2"].detach(), grids[0])
    r1 = E_full["s1"] - e2_up
    out = {"s3": r3, "s2": r2, "s1": r1}
    if fg_stages is not None:
        out = {k: fg_gate(v, fg_stages[k]) for k, v in out.items()}
    return out


def masked_coarse_mae(C4: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      beta: float = 1.0) -> torch.Tensor:
    """Smooth-L1 between ``C4`` and detached target at masked s4 positions."""
    from .losses import gather_masked
    pred = gather_masked(C4, mask)
    tgt = gather_masked(target, mask).to(pred.dtype)
    return F.smooth_l1_loss(pred, tgt, beta=beta)


def reconstruct_from_residuals(
        C4: torch.Tensor,
        residuals: Dict[str, torch.Tensor],
        E_full: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rebuild ``E_s ≈ parent_up + R_s`` for probe / sanity checks."""
    return {
        "s4": C4,
        "s3": upsample_stage(C4, E_full["s3"].shape[-2:]) + residuals["s3"],
        "s2": upsample_stage(E_full["s3"], E_full["s2"].shape[-2:]) + residuals["s2"],
        "s1": upsample_stage(E_full["s2"], E_full["s1"].shape[-2:]) + residuals["s1"],
    }
