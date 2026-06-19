"""Masked multi-scale prediction loss, curriculum schedule, and diagnostics.

The prediction loss is a masked-position regression of the predictor outputs
onto the detached, per-token-normalized target latents, summed over stages with
a curriculum weighting that ramps the fine stages in while coarse stages train
from the start. SIGReg lives in :mod:`.sigreg`; this module is the prediction
side plus the collapse-monitoring diagnostics from the design (§8).
"""
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def stage_target_norm(feat: torch.Tensor, mode: str = "ln") -> torch.Tensor:
    """Normalize a stage map ``[B, C, h, w]`` for use as a (detached) target.

    - ``"ln"``: per-token LayerNorm over channels (data2vec-style; stops coarse
      stages sliding to trivial low-variance solutions). No affine.
    - ``"whiten"``: batch-level ZCA whitening of the stage tokens (more
      aggressive; optional ablation).
    - ``"none"``: identity.
    """
    if mode == "none":
        return feat
    b, c, h, w = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(-1, c)             # [B*h*w, C]
    if mode == "ln":
        tok = F.layer_norm(tok, (c,))
    elif mode == "whiten":
        mu = tok.mean(0, keepdim=True)
        tc = tok - mu
        cov = (tc.T @ tc) / max(1, tc.shape[0] - 1)
        cov = cov + 1e-5 * torch.eye(c, device=feat.device, dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(cov.float())
        whiten = (evecs @ torch.diag(evals.clamp_min(1e-8).rsqrt()) @ evecs.T)
        tok = tc @ whiten.to(tc.dtype)
    else:
        raise ValueError(f"unknown target_norm: {mode!r}")
    return tok.reshape(b, h, w, c).permute(0, 3, 1, 2)


def gather_masked(feat: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
    """Gather masked-position tokens from ``[B, C, h, w]`` -> ``[B, k, C]``.

    Row-major over ``(h, w)`` and a fixed per-sample masked count ``k`` (the mask
    generator guarantees this), so the ordering matches the predictor's masked
    queries exactly.
    """
    b, c, h, w = feat.shape
    tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    mflat = mask_s.reshape(b, h * w)
    k = int(mflat[0].sum())
    return tok[mflat].view(b, k, c)


def _stage_active_fine_in(step: int, total_steps: int, warmup_frac: float,
                          stage_min_w: float, fine_stages, num_stages: int) -> List[float]:
    denom = max(1.0, warmup_frac * total_steps)
    ramp = min(1.0, max(0.0, step / denom))
    fine = set(int(s) for s in fine_stages)
    out = []
    for s in range(num_stages):
        if (s + 1) in fine:
            out.append(stage_min_w + (1.0 - stage_min_w) * ramp)
        else:
            out.append(1.0)
    return out


def _stage_active_coarse_in(step: int, total_steps: int, warmup_frac: float,
                            stage_min_w: float, ramp_stages, num_stages: int) -> List[float]:
    warmup_total = max(1.0, warmup_frac * total_steps)
    ramp = [int(s) for s in ramp_stages]
    seg = warmup_total / max(1, len(ramp))
    coarsest = num_stages
    out = []
    for s in range(num_stages):
        sid = s + 1
        if sid == coarsest:
            out.append(1.0)
            continue
        if sid not in ramp:
            out.append(stage_min_w)
            continue
        idx = ramp.index(sid)
        seg_start = idx * seg
        if step <= seg_start:
            out.append(stage_min_w)
        else:
            prog = min(1.0, (step - seg_start) / max(1.0, seg))
            out.append(stage_min_w + (1.0 - stage_min_w) * prog)
    return out


def stage_active_schedule(step: int, total_steps: int, warmup_frac: float,
                          stage_min_w: float, fine_stages, num_stages: int = 4, *,
                          stage_curriculum: str = "fine_in",
                          coarse_ramp_stages=(3, 2, 1)) -> List[float]:
    """Per-stage curriculum factor in ``[stage_min_w, 1.0]`` (s1..s4)."""
    if stage_curriculum == "fine_in":
        return _stage_active_fine_in(
            step, total_steps, warmup_frac, stage_min_w, fine_stages, num_stages)
    if stage_curriculum == "coarse_in":
        return _stage_active_coarse_in(
            step, total_steps, warmup_frac, stage_min_w, coarse_ramp_stages, num_stages)
    raise ValueError(f"unknown stage_curriculum: {stage_curriculum!r}")


def lambda_schedule_fine_in(step: int, total_steps: int, warmup_frac: float,
                            fine_min_w: float, stage_base_weights, fine_stages,
                            num_stages: int = 4) -> List[float]:
    """Fine-in curriculum: coarse stages (default s3,s4) train from step 0;
    fine stages ramp from ``base * fine_min_w`` to ``base``."""
    active = _stage_active_fine_in(
        step, total_steps, warmup_frac, fine_min_w, fine_stages, num_stages)
    return [float(stage_base_weights[s]) * active[s] for s in range(num_stages)]


def lambda_schedule_coarse_in(step: int, total_steps: int, warmup_frac: float,
                              stage_min_w: float, stage_base_weights,
                              ramp_stages, num_stages: int = 4) -> List[float]:
    """Coarse-in curriculum: lowest-res stage s4 at full weight from step 0;
    higher-res stages in ``ramp_stages`` (default s3,s2,s1) each ramp in over
    an equal slice of ``warmup_frac * total_steps``."""
    active = _stage_active_coarse_in(
        step, total_steps, warmup_frac, stage_min_w, ramp_stages, num_stages)
    return [float(stage_base_weights[s]) * active[s] for s in range(num_stages)]


def lambda_schedule(step: int, total_steps: int, warmup_frac: float,
                    fine_min_w: float, stage_base_weights, fine_stages,
                    num_stages: int = 4, *, stage_curriculum: str = "fine_in",
                    coarse_ramp_stages=(3, 2, 1)) -> List[float]:
    """Per-stage loss weights ``lambda_s(step)`` (list ordered ``s1..s4``).

    ``stage_curriculum``:
      - ``fine_in``: legacy — s3/s4 from step 0, s1/s2 ramp in.
      - ``coarse_in``: s4 (lowest res) from step 0, then s3→s2→s1 stir in.
    """
    if stage_curriculum == "fine_in":
        return lambda_schedule_fine_in(
            step, total_steps, warmup_frac, fine_min_w, stage_base_weights,
            fine_stages, num_stages)
    if stage_curriculum == "coarse_in":
        return lambda_schedule_coarse_in(
            step, total_steps, warmup_frac, fine_min_w, stage_base_weights,
            coarse_ramp_stages, num_stages)
    raise ValueError(f"unknown stage_curriculum: {stage_curriculum!r}")


def masked_prediction_loss(pred: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           mask: Dict[str, torch.Tensor],
                           lambdas: List[float], loss_type: str = "smooth_l1",
                           beta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted masked-position regression over stages.

    Args:
        pred: ``{s: [B, k_s, C_s]}`` predictions at masked positions.
        targets: ``{s: [B, C_s, h_s, w_s]}`` detached normalized target maps.
        mask: ``{s: [B, h_s, w_s] bool}``.
        lambdas: per-stage weights (``s1..s4``).
        loss_type: ``"smooth_l1"`` (Huber) or ``"mse"``.
        beta: Huber beta for smooth-l1.

    Returns ``(total, per_stage_values)`` where ``per_stage_values`` are the
    *unweighted* per-stage losses for logging.
    """
    total = pred[next(iter(pred))].new_zeros(())
    per_stage: Dict[str, float] = {}
    for i, key in enumerate(sorted(pred.keys())):
        tgt = gather_masked(targets[key], mask[key]).to(pred[key].dtype)
        if loss_type == "mse":
            ls = F.mse_loss(pred[key], tgt)
        else:
            ls = F.smooth_l1_loss(pred[key], tgt, beta=beta)
        total = total + lambdas[i] * ls
        per_stage[key] = float(ls.detach())
    return total, per_stage


# --- diagnostics (design §8) ------------------------------------------------
@torch.no_grad()
def effective_rank(z: torch.Tensor, max_tokens: int = 4096) -> float:
    """Effective rank ``exp(entropy(normalized singular values))`` of centered ``z``.

    ``z`` is ``[N, C]`` (a token sample). Collapse shows up first as a low s4
    effective rank -- the early-warning signal. (Roy & Vetterli, 2007.)
    """
    z = z.float()
    if z.shape[0] > max_tokens:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_tokens]
        z = z[idx]
    z = z - z.mean(0, keepdim=True)
    if z.shape[0] < 2:
        return 0.0
    sv = torch.linalg.svdvals(z)
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


@torch.no_grad()
def stage_feature_diagnostics(feats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Per-stage effective rank and mean-over-channels feature std, flattened to
    ``{"effrank/s1": ..., "fstd/s1": ...}`` for logging."""
    logs: Dict[str, float] = {}
    for key, feat in feats.items():
        b, c, h, w = feat.shape
        tok = feat.permute(0, 2, 3, 1).reshape(-1, c)
        logs[f"effrank/{key}"] = effective_rank(tok)
        logs[f"fstd/{key}"] = float(tok.float().std(dim=0).mean())
    return logs
