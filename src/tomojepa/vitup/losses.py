"""Distillation losses (paper section III.C.b).

Three terms compare a predicted feature map against the frozen teacher:
  (1) target-normalized L2 -- both prediction and target normalized by the
      *teacher's* per-vector channel mean/std;
  (2) cosine -- angular alignment;
  (3) relational KL -- preserves the pairwise similarity structure across the N
      spatial features of an image (diagonal masked), with temperature ``tau``.

All operate on ``[B, N, C]`` tensors (an image's N spatial features). The total
objective sums the weighted terms over all supervised layers and token grids;
:func:`feature_loss` computes the weighted sum for a single (layer, grid) pair
and :func:`aggregate` averages a list of such terms.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F


def normalized_l2(pred: torch.Tensor, target: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """Target-normalized squared L2, averaged over vectors.

    ``mu``, ``sigma`` are the teacher's channel-wise mean/std; both prediction
    and target are normalized by them, so the loss reduces to
    ``|| (pred - target) / sigma ||_2^2`` per vector.
    """
    var = target.var(dim=-1, unbiased=False, keepdim=True)
    sigma = torch.sqrt(var + eps)
    diff = (pred - target) / sigma
    return diff.square().sum(dim=-1).mean()


def cosine_loss(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    """``1 - cosine_similarity`` averaged over vectors."""
    cos = F.cosine_similarity(pred, target, dim=-1, eps=eps)
    return (1.0 - cos).mean()


def relational_kl(pred: torch.Tensor, target: torch.Tensor, tau: float,
                  eps: float = 1e-6) -> torch.Tensor:
    """Relational KL over the per-image similarity matrices (diagonal masked).

    ``S = (f_i . f_j) / tau`` (teacher), ``S_hat`` (student) on L2-normalized
    features; ``KL(softmax(S) || softmax(S_hat))`` with the self-similarity
    diagonal excluded from the softmax.
    """
    B, N, _ = pred.shape
    pn = F.normalize(pred, dim=-1, eps=eps)
    tn = F.normalize(target, dim=-1, eps=eps)
    s_t = torch.bmm(tn, tn.transpose(1, 2)) / tau     # [B,N,N] teacher
    s_s = torch.bmm(pn, pn.transpose(1, 2)) / tau     # [B,N,N] student
    diag = torch.eye(N, device=pred.device, dtype=torch.bool)
    neg_inf = torch.finfo(s_t.dtype).min
    s_t = s_t.masked_fill(diag, neg_inf)
    s_s = s_s.masked_fill(diag, neg_inf)
    log_p = F.log_softmax(s_t, dim=-1)                 # teacher (target dist)
    log_q = F.log_softmax(s_s, dim=-1)                 # student
    p = log_p.exp()
    # KL(P||Q) summed over j (diagonal contributes 0 since p=0 there), mean rows
    kl = (p * (log_p - log_q))
    kl = kl.masked_fill(diag, 0.0).sum(dim=-1)         # [B,N]
    return kl.mean()


def feature_loss(pred: torch.Tensor, target: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    """Weighted three-term loss for one (layer, grid) prediction/target pair.

    ``pred``, ``target``: ``[B, N, C]``. Returns the individual terms and the
    weighted total.
    """
    l2 = normalized_l2(pred, target, cfg.eps)
    cos = cosine_loss(pred, target, cfg.eps)
    rel = relational_kl(pred, target, cfg.rel_temperature, cfg.eps)
    total = cfg.lambda_l2 * l2 + cfg.lambda_cos * cos + cfg.lambda_rel * rel
    return {"l2": l2, "cos": cos, "rel": rel, "total": total}


def aggregate(terms: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average a list of per-(layer, grid) loss dicts (the paper's mean)."""
    keys = terms[0].keys()
    n = len(terms)
    return {k: sum(t[k] for t in terms) / n for k in keys}
