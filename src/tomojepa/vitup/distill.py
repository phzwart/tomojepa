"""Multi-scale feature distillation training (paper section III.C).

A frozen teacher ViT supervises the LoRA-adapted student backbone + ViT-Up:

  * The teacher is run on the training image at several square resolutions
    ``S`` (token grids ``N``), giving multi-scale targets ``H_l^n``.
  * The student input is the same image downscaled by ``s ~ U(s_min, s_max)`` and
    pasted at a random position into a black ``student_canvas`` canvas.
  * A regular ``query_grid x query_grid`` grid of coordinates over the pasted
    region is queried; ViT-Up predicts ``o_t`` at every stage ``t = 0..T``.
  * Each prediction is average-pooled from the finest query grid down to every
    teacher grid ``n`` and compared to the teacher target with the three losses.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import feature_loss, aggregate
from .model import ViTUp
from .backbone_adapter import BackboneAdapter


def build_student_batch(img: torch.Tensor, canvas: int, scale: Tuple[float, float],
                        patch_size: int, query_grid: int, generator=None):
    """Downscale + random-paste each image into a black canvas; sample queries.

    Returns ``(canvas_batch [B,C,canvas,canvas], q_coords [B,Q,2])`` where the
    query coordinates are in the canvas token grid units (``canvas / p`` tokens),
    placed at cell centers over each sample's pasted region.
    """
    B, C, _, _ = img.shape
    device = img.device
    s_min, s_max = scale
    out = torch.zeros(B, C, canvas, canvas, device=device, dtype=img.dtype)
    coords = torch.empty(B, query_grid * query_grid, 2, device=device, dtype=torch.float32)
    rs = torch.arange(query_grid, device=device, dtype=torch.float32) + 0.5
    for b in range(B):
        s = float(torch.empty(1, device=device).uniform_(s_min, s_max, generator=generator))
        size = max(patch_size, int(round(s * canvas)))
        size = min(size, canvas)
        resized = F.interpolate(img[b:b + 1], size=(size, size), mode="bilinear",
                                align_corners=False)[0]
        max_off = canvas - size
        top = int(torch.randint(0, max_off + 1, (1,), device=device, generator=generator)) if max_off > 0 else 0
        left = int(torch.randint(0, max_off + 1, (1,), device=device, generator=generator)) if max_off > 0 else 0
        out[b, :, top:top + size, left:left + size] = resized
        # query coordinates over the pasted region, in token units
        top_t, left_t, size_t = top / patch_size, left / patch_size, size / patch_size
        ys = top_t + rs / query_grid * size_t
        xs = left_t + rs / query_grid * size_t
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords[b] = torch.stack([gy, gx], dim=-1).reshape(-1, 2)
    return out, coords


class MultiScaleTeacher(nn.Module):
    """Frozen teacher producing ``H_l^n`` targets at multiple resolutions."""

    def __init__(self, adapter: BackboneAdapter, resolutions: List[int],
                 layers: List[int]):
        super().__init__()
        self.adapter = adapter
        self.resolutions = list(resolutions)
        self.layers = sorted(set(layers))
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        self.adapter.eval()

    @torch.no_grad()
    def targets(self, img: torch.Tensor) -> Dict[Tuple[int, int], torch.Tensor]:
        """``{(layer, n): [B, n, n, C]}`` for every layer and teacher grid."""
        p = self.adapter.p
        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for r in self.resolutions:
            n = r // p
            x = F.interpolate(img, size=(r, r), mode="bilinear", align_corners=False)
            hid = self.adapter.hidden_states(x, self.layers)  # {layer:[B,C,n,n]}
            for l, h in hid.items():
                out[(l, n)] = h.permute(0, 2, 3, 1).contiguous()  # [B,n,n,C]
        return out


class DistillEngine(nn.Module):
    """Couples the frozen teacher with the student ViT-Up and the objective."""

    def __init__(self, teacher: MultiScaleTeacher, vitup: ViTUp, cfg):
        super().__init__()
        self.teacher = teacher
        self.vitup = vitup
        self.cfg = cfg
        # stage -> backbone layer: stage 0 = embedding (l[0]=0); stage t = l[t]
        self.stage_layers = [0] + list(cfg.layer_indices)
        self.token_grids = cfg.teacher_token_grids(vitup.adapter.p)

    def _pool(self, pred_grid: torch.Tensor, n: int) -> torch.Tensor:
        # pred_grid: [B,qg,qg,C] -> [B,n,n,C] -> [B,n*n,C]
        B = pred_grid.shape[0]
        x = pred_grid.permute(0, 3, 1, 2)                       # [B,C,qg,qg]
        x = F.adaptive_avg_pool2d(x, (n, n))                    # [B,C,n,n]
        return x.permute(0, 2, 3, 1).reshape(B, n * n, -1)

    def compute_loss(self, img: torch.Tensor, chunk_size=None):
        """Full multi-scale objective for one image batch.

        Returns ``(total_loss, logs)`` with per-term scalar logs.
        """
        cfg = self.cfg
        qg = cfg.query_grid
        targets = self.teacher.targets(img)                    # {(layer,n):[B,n,n,C]}

        canvas, coords = build_student_batch(
            img, cfg.student_canvas, cfg.student_scale, self.vitup.adapter.p, qg)
        o_stages = self.vitup(canvas, coords, stages="all", chunk_size=chunk_size)
        # o_stages[t]: [B, qg*qg, C]
        B, _, Cdim = o_stages[0].shape

        terms = []
        for t, o in enumerate(o_stages):                        # t = 0..T
            layer = self.stage_layers[t]
            pred_grid = o.reshape(B, qg, qg, Cdim)
            for n in self.token_grids:
                pred = self._pool(pred_grid, n)                 # [B,n*n,C]
                tgt = targets[(layer, n)].reshape(B, n * n, Cdim)
                terms.append(feature_loss(pred, tgt, cfg))
        agg = aggregate(terms)
        logs = {k: float(v.detach()) for k, v in agg.items()}
        return agg["total"], logs
