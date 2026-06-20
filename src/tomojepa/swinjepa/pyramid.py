"""Hierarchical residual pyramid for Swin multi-scale JEPA.

Coarse stage s4 carries a smooth envelope ``C4`` from an SSL-style MIM head on
masked student features. Finer stages hold residuals ``R_s = E_s - sg(up(parent))``.

Default parent choice (``strict_laplacian=False``):
  - ``R3`` parent = student MIM field ``C4`` (detached, upsampled to s3)
  - ``R2`` parent = raw target ``E_full["s3"]`` (not ``pool(E3)``)
  - ``R1`` parent = raw target ``E_full["s2"]``

These residuals are low-redundancy but **not** strictly scale-orthogonal /
Laplacian. With ``strict_laplacian=True``, each parent is
``up(avg_pool(E_s))`` so ``R_s`` is the within-cell high-frequency complement.
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
                        min_grid_dist: int = 0,
                        token_cap: Optional[int] = None,
                        return_valid: bool = False
                        ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """FG tokens per slice: ``[B,C,h,w]`` -> ``[B, M, C]``.

    Used by image-grouped SIGReg so within-slice correlation is not treated as
    independent cross-image samples. ``n_per_slice > 0``: subsample ``M`` tokens
    (``min_grid_dist > 0`` enforces minimum Chebyshev separation). ``n_per_slice
    == 0``: use **all** FG tokens (padded to the batch max, no subsampling).

    When ``fg_stage`` is set, only strict-FG cells are eligible; slices with no
    FG cells yield zero placeholders and ``valid[i]=False`` (never BG tokens).
    """
    b, c, h, w = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=feat.device),
        torch.arange(w, device=feat.device),
        indexing="ij")
    coords_flat = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    out = []
    valid: list[bool] = []
    target_m = n_per_slice if n_per_slice > 0 else None
    for i in range(b):
        if fg_stage is not None:
            fg = fg_stage[i].reshape(-1).bool()
            idx_pool = fg.nonzero(as_tuple=False).squeeze(-1)
            has_fg = idx_pool.numel() > 0
            valid.append(has_fg)
            if not has_fg:
                m = target_m if target_m is not None else (token_cap or 1)
                out.append(tok[i].new_zeros(m, c))
                continue
        else:
            idx_pool = torch.arange(h * w, device=feat.device)
            valid.append(True)
        if n_per_slice <= 0:
            out.append(tok[i][idx_pool])
        else:
            coords = coords_flat[idx_pool]
            pick = _greedy_min_dist_pick(coords, n_per_slice, min_grid_dist)
            sel = tok[i][idx_pool[pick]]
            if sel.shape[0] < n_per_slice:
                pad = torch.randint(0, sel.shape[0], (n_per_slice - sel.shape[0],),
                                    device=feat.device)
                sel = torch.cat([sel, sel[pad]], dim=0)
            out.append(sel)
    valid_t = torch.tensor(valid, device=feat.device, dtype=torch.bool)
    if n_per_slice <= 0:
        max_m = token_cap if token_cap is not None else max(max(s.shape[0] for s in out), 1)
        padded = []
        for sel in out:
            if sel.shape[0] >= max_m:
                padded.append(sel[:max_m])
            elif sel.shape[0] == 0:
                padded.append(tok.new_zeros(max_m, c))
            else:
                pad = torch.randint(0, sel.shape[0], (max_m - sel.shape[0],),
                                    device=feat.device)
                padded.append(torch.cat([sel, sel[pad]], dim=0))
        stacked = torch.stack(padded, dim=0)
    else:
        stacked = torch.stack(out, dim=0)
    if return_valid:
        return stacked, valid_t
    return stacked


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


def _laplacian_parent(child: torch.Tensor, grid_child: Tuple[int, int],
                      grid_parent: Tuple[int, int]) -> torch.Tensor:
    """``up(avg_pool(child))`` from ``grid_child`` to ``grid_parent`` and back."""
    pooled = F.adaptive_avg_pool2d(child, grid_parent)
    return upsample_stage(pooled, grid_child)


def hierarchical_residuals(
        E_full: Dict[str, torch.Tensor],
        C4: torch.Tensor,
        grids: List[Tuple[int, int]],
        fg_stages: Optional[Dict[str, torch.Tensor]] = None,
        strict_laplacian: bool = False,
) -> Dict[str, torch.Tensor]:
    """``R3,R2,R1`` from target-pass maps and stop-grad coarse parents."""
    out = _band_residuals_core(
        E_full, C4, grids, strict_laplacian=strict_laplacian)
    out = {k: out[k] for k in ("s3", "s2", "s1")}
    if fg_stages is not None:
        out = {k: fg_gate(v, fg_stages[k]) for k, v in out.items()}
    return out


def pyramid_band_residuals(
        E_full: Dict[str, torch.Tensor],
        C4: torch.Tensor,
        grids: List[Tuple[int, int]],
        fg_stages: Optional[Dict[str, torch.Tensor]] = None,
        strict_laplacian: bool = False,
) -> Dict[str, torch.Tensor]:
    """Inter-scale band residuals for per-stage SIGReg (keys ``s1..s4``).

    ``s1`` S2-S1, ``s2`` S3-S2, ``s3`` S4-S3 (``E3 - up(C4)``),
    ``s4`` S4-S3 at the coarse grid (``E4 - pool(E3)``).
    """
    out = _band_residuals_core(
        E_full, C4, grids, strict_laplacian=strict_laplacian)
    if fg_stages is not None:
        out = {k: fg_gate(v, fg_stages[k]) for k, v in out.items()}
    return out


def _band_residuals_core(
        E_full: Dict[str, torch.Tensor],
        C4: torch.Tensor,
        grids: List[Tuple[int, int]],
        strict_laplacian: bool = False,
) -> Dict[str, torch.Tensor]:
    if strict_laplacian:
        e3_at_s4 = F.adaptive_avg_pool2d(E_full["s3"].detach(), grids[3])
        r4 = E_full["s4"] - e3_at_s4
        r3 = E_full["s3"] - _laplacian_parent(
            E_full["s3"].detach(), grids[2], grids[3])
        r2 = E_full["s2"] - _laplacian_parent(
            E_full["s2"].detach(), grids[1], grids[2])
        r1 = E_full["s1"] - _laplacian_parent(
            E_full["s1"].detach(), grids[0], grids[1])
    else:
        e3_at_s4 = F.adaptive_avg_pool2d(E_full["s3"].detach(), grids[3])
        r4 = E_full["s4"] - e3_at_s4
        c4_up = upsample_stage(C4.detach(), grids[2])
        r3 = E_full["s3"] - c4_up
        e3_up = upsample_stage(E_full["s3"].detach(), grids[1])
        r2 = E_full["s2"] - e3_up
        e2_up = upsample_stage(E_full["s2"].detach(), grids[0])
        r1 = E_full["s1"] - e2_up
    return {"s4": r4, "s3": r3, "s2": r2, "s1": r1}


def masked_coarse_mae(C4: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      beta: float = 1.0,
                      fg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Smooth-L1 between ``C4`` and detached target at masked s4 positions."""
    from .losses import gather_masked
    if fg_mask is not None:
        mask = mask & fg_mask
    pred = gather_masked(C4, mask)
    tgt = gather_masked(target, mask).to(pred.dtype)
    return F.smooth_l1_loss(pred, tgt, beta=beta)


def reconstruct_from_residuals(
        C4: torch.Tensor,
        residuals: Dict[str, torch.Tensor],
        E_full: Dict[str, torch.Tensor],
        grids: List[Tuple[int, int]],
        strict_laplacian: bool = False,
) -> Dict[str, torch.Tensor]:
    """Rebuild ``E_s ≈ parent_up + R_s`` for probe / sanity checks."""
    if strict_laplacian:
        return {
            "s4": C4,
            "s3": _laplacian_parent(E_full["s3"], grids[2], grids[3]) + residuals["s3"],
            "s2": _laplacian_parent(E_full["s2"], grids[1], grids[2]) + residuals["s2"],
            "s1": _laplacian_parent(E_full["s1"], grids[0], grids[1]) + residuals["s1"],
        }
    return {
        "s4": C4,
        "s3": upsample_stage(C4, E_full["s3"].shape[-2:]) + residuals["s3"],
        "s2": upsample_stage(E_full["s3"], E_full["s2"].shape[-2:]) + residuals["s2"],
        "s1": upsample_stage(E_full["s2"], E_full["s1"].shape[-2:]) + residuals["s1"],
    }
