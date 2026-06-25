"""Swin multi-scale latent-JEPA model + lightweight inference encoder.

:class:`SwinMSJEPA` ties the two-pass forward together: one shared backbone is
run twice per step -- a **target** pass on the full image (features carry grad
for SIGReg, are detached for the prediction targets) and a **student** pass on
the masked image (mask tokens injected at stage 1). Collapse is prevented by
per-stage SIGReg only; there is **no EMA teacher and no pixel decoder**. All
JEPA/SIGReg machinery is training-only.

:class:`SwinMSEncoder` is the inference-time counterpart: it wraps a pretrained
backbone (and optional FPN-style lateral projections) and exposes
``extract_features`` with none of the training-only modules present.
"""
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomojepa.core.augmentations import build_fg_stages
from .config import SwinMSJEPAConfig
from .backbone import SwinMultiScaleBackbone
from .mask import MultiScaleBlockMask, assert_mask_consistency
from .predictor import CrossScalePredictor
from .pyramid_fusion import PyramidBandFusion
from .sigreg import StageSIGReg, ImageGroupedStageSIGReg
from .pyramid import (CoarseMIMHead, hierarchical_residuals, masked_coarse_mae,
                      fg_gate, reconstruct_from_residuals, pyramid_band_residuals,
                      pyramid_sigreg_features, assemble_coarse_field,
                      build_residual_aligns, scatter_masked)
from .losses import (stage_target_norm, gather_masked, lambda_schedule,
                     stage_active_schedule, masked_prediction_loss,
                     stage_feature_diagnostics)
from .schedule import TrainingSchedule

_RESIDUAL_KEYS = ("s1", "s2", "s3")


def _identity(x):
    return x


def _flatten_tokens(feat: torch.Tensor) -> torch.Tensor:
    """``[B, C, h, w]`` -> ``[B*h*w, C]`` (row-major)."""
    b, c, h, w = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, c)


def _project_latent(feats: Dict[str, torch.Tensor],
                    lateral: nn.ModuleDict) -> Dict[str, torch.Tensor]:
    """1x1 conv each stage map to its configured JEPA latent width."""
    return {k: lateral[k](v) for k, v in feats.items()}


def _fg_gate_feats(feats: Dict[str, torch.Tensor],
                   fg_stages: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Zero feature maps outside the strict per-stage FG (BG never becomes a target)."""
    return {
        k: v * fg_stages[k].unsqueeze(1).to(v.dtype)
        for k, v in feats.items()
    }


class SwinMSJEPA(nn.Module):
    """Multi-scale latent-JEPA training module (the "engine").

    Args:
        cfg: :class:`SwinMSJEPAConfig`.
        augment: single swappable on-tensor augmentation applied to ``x`` before
            both passes (JEPA needs the *same* augmented view for target and
            student; masking is the only asymmetry). Defaults to identity.
        check_masks: assert mask expansion consistency each step (cheap; on by
            default to catch any stage/mask drift early).
    """

    has_ema = False  # explicit: no momentum teacher exists (collapse control = SIGReg)

    def __init__(self, cfg: SwinMSJEPAConfig,
                 augment: Optional[Callable] = None, check_masks: bool = True):
        super().__init__()
        self.cfg = cfg
        self.augment = augment if augment is not None else _identity
        self.check_masks = check_masks
        self.schedule: Optional[TrainingSchedule] = None
        self._steps_per_epoch: int = 1
        self._last_newly_frozen: List[str] = []
        self._mae_cos_ema: Optional[float] = None
        self._sigreg_cos_latched: bool = False
        self._sigreg_gate_progress: Optional[float] = None
        self._last_sigreg_cos_gate: float = 1.0
        self._last_schedule_betas: List[float] = []

        self.backbone = SwinMultiScaleBackbone(
            model_name=cfg.backbone_name, img_size=cfg.img_size,
            in_chans=cfg.in_chans, pretrained=False,
            drop_path_rate=cfg.drop_path_rate, use_rope=cfg.use_rope,
            rope_theta=cfg.rope_theta, embed_dim=cfg.backbone_embed_dim)
        self.out_chans: List[int] = self.backbone.out_chans
        self.num_stages = self.backbone._num_stages
        self.grids = [self.backbone.stage_grid(s) for s in range(self.num_stages)]

        self.lat_chans: List[int] = list(cfg.lat_dims)

        # Per-stage 1x1 projections: backbone C_s -> lat_dims[s].
        self.lateral = nn.ModuleDict({
            f"s{s + 1}": nn.Conv2d(self.out_chans[s], self.lat_chans[s], kernel_size=1)
            for s in range(self.num_stages)})

        self.residual_align = build_residual_aligns(self.lat_chans)

        grid4 = self.backbone.stage_grid(self.num_stages - 1)
        self.mask_gen = MultiScaleBlockMask(
            grid4=grid4, num_stages=self.num_stages, mask_ratio=cfg.mask_ratio,
            mask_mode=cfg.mask_mode, num_blocks=cfg.mask_num_blocks,
            block_scale_range=cfg.block_scale_range)

        self.predictor = None
        self.fusion = None
        if cfg.coarse_mim_mode == "integrated":
            self.fusion = PyramidBandFusion(
                out_chans=self.lat_chans, grids=self.grids,
                depth=cfg.fusion_depth, heads=cfg.fusion_heads,
                mlp_ratio=cfg.fusion_mlp_ratio,
                cross_scale=cfg.fusion_cross_scale,
                use_rope=cfg.use_rope, rope_theta=cfg.rope_theta)
        elif cfg.predictor_enabled:
            self.predictor = CrossScalePredictor(
                out_chans=self.lat_chans, grids=self.grids, dim=cfg.pred_dim,
                depth=cfg.pred_depth, heads=cfg.pred_heads,
                mlp_ratio=cfg.pred_mlp_ratio, cross_scale=cfg.predictor_cross_scale,
                use_rope=cfg.use_rope, rope_theta=cfg.rope_theta)

        self.coarse_head = None
        self.sigreg_r = None
        self.sigreg = None
        if cfg.legacy_jepa:
            self.sigreg = nn.ModuleList([
                StageSIGReg(dim=self.lat_chans[s], n_dirs=cfg.sigreg_n_dirs[s],
                            knots=cfg.sigreg_knots, t_max=cfg.sigreg_t_max,
                            w_mean=cfg.sigreg_w_mean,
                            n_tokens_cap=cfg.sigreg_n_tokens_cap,
                            queue_len=cfg.sigreg_queue_len)
                for s in range(self.num_stages)])
        else:
            s4 = self.num_stages - 1
            self.coarse_head = CoarseMIMHead(self.lat_chans[s4])
            q_len = cfg.sigreg_queue_len if cfg.sigreg_pooled else 0
            mdist = cfg.sigreg_min_token_dist

            def _make_sigreg_r(stage_idx: int) -> ImageGroupedStageSIGReg:
                gh, gw = self.grids[stage_idx]
                cap = gh * gw
                if stage_idx == s4:
                    if cfg.sigreg_token_frac > 0.0:
                        tok = max(1, int(round(cfg.sigreg_token_frac * cap)))
                        qcap = tok
                    elif cfg.sigreg_tokens_per_slice == 0:
                        tok, qcap = 0, cap
                    else:
                        tok = min(cfg.sigreg_tokens_per_slice, cap)
                        qcap = tok
                else:
                    tok = cfg.sigreg_tokens_per_slice if cfg.sigreg_tokens_per_slice > 0 else 32
                    tok = min(tok, cap)
                    qcap = tok
                return ImageGroupedStageSIGReg(
                    dim=self.lat_chans[stage_idx],
                    n_dirs=cfg.sigreg_n_dirs[stage_idx],
                    knots=cfg.sigreg_knots, t_max=cfg.sigreg_t_max,
                    w_mean=cfg.sigreg_w_mean, queue_len=q_len,
                    n_tokens_per_slice=tok, min_grid_dist=mdist,
                    cap_dirs_by_rank=cfg.sigreg_cap_dirs_by_rank,
                    min_dirs=cfg.sigreg_min_dirs,
                    queue_token_cap=qcap)

            self.sigreg_r = nn.ModuleList([
                _make_sigreg_r(s) for s in range(self.num_stages)])

        self.register_buffer(
            "_frozen_stages",
            torch.zeros(self.num_stages, dtype=torch.bool),
            persistent=True)

    def _stage_key(self, stage_idx: int) -> str:
        return f"s{stage_idx + 1}"

    def _residual_align_for_pyramid(self) -> Optional[nn.ModuleDict]:
        if len(self.residual_align) == 0:
            return None
        return self.residual_align

    def stage_frozen(self, stage_idx: int) -> bool:
        """True when stage ``s{stage_idx+1}`` is frozen (no loss, detached latents)."""
        return bool(self._frozen_stages[stage_idx].item())

    @property
    def frozen_stage_keys(self) -> List[str]:
        return [self._stage_key(s) for s in range(self.num_stages)
                if self.stage_frozen(s)]

    def apply_freeze_schedule(self, epoch: int) -> List[str]:
        """Freeze stages whose ``freeze_after_epoch[s] <= epoch`` (0-based, train loop).

        Returns stage keys newly frozen on this call (idempotent across repeats).
        """
        newly: List[str] = []
        for s, start in enumerate(self.cfg.freeze_after_epoch):
            if start <= 0 or epoch < start:
                continue
            if self.stage_frozen(s):
                continue
            self._frozen_stages[s] = True
            key = self._stage_key(s)
            newly.append(key)
            self._freeze_stage_modules(s)
        self._last_newly_frozen = newly
        return newly

    def _freeze_stage_modules(self, stage_idx: int) -> None:
        key = self._stage_key(stage_idx)
        for p in self.lateral[key].parameters():
            p.requires_grad = False
        if (not self.cfg.legacy_jepa
                and stage_idx == self.num_stages - 1
                and self.coarse_head is not None
                and self.cfg.coarse_mim_mode == "conv"):
            for p in self.coarse_head.parameters():
                p.requires_grad = False
        if (not self.cfg.legacy_jepa
                and stage_idx == self.num_stages - 1
                and self.fusion is not None
                and self.cfg.coarse_mim_mode == "integrated"):
            for p in self.fusion.parameters():
                p.requires_grad = False
        if (not self.cfg.legacy_jepa
                and stage_idx == self.num_stages - 1
                and self.predictor is not None
                and self.cfg.coarse_mim_mode == "cross_attn"):
            for p in self.predictor.parameters():
                p.requires_grad = False

    def _stage_loss_weights(
            self, active: List[float], betas: List[float],
            bases: List[float], pred_active: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Per-stage prediction (or s4 MAE) and SIGReg multipliers."""
        pa = pred_active if pred_active is not None else active
        pred = [float(bases[s]) * pa[s] for s in range(self.num_stages)]
        sig = [betas[s] * active[s] for s in range(self.num_stages)]
        return pred, sig

    def _stage_grad_flags(
            self, pred_w: List[float], sig_w: List[float]) -> List[bool]:
        """True when this stage's latents may carry gradients (loss or SIGReg)."""
        flags: List[bool] = []
        for s in range(self.num_stages):
            if self.stage_frozen(s):
                flags.append(False)
            else:
                flags.append(pred_w[s] > 0.0 or sig_w[s] > 0.0)
        return flags

    def _detach_inactive_latents(
            self, feats: Dict[str, torch.Tensor],
            grad_active: List[bool]) -> Dict[str, torch.Tensor]:
        """Stop-gradient on stages with zero loss weight (lambda/beta) or frozen."""
        out = dict(feats)
        for s in range(self.num_stages):
            if not grad_active[s]:
                key = self._stage_key(s)
                out[key] = out[key].detach()
        return out

    def _freeze_logs(self) -> Dict[str, float]:
        return {f"frozen/{self._stage_key(s)}": float(self.stage_frozen(s))
                for s in range(self.num_stages)}

    def _zero_frozen_lambdas(self, lambdas: List[float],
                             residual: bool = False) -> List[float]:
        """Zero loss weights for frozen stages (residual=True -> s1..s3 only)."""
        out = list(lambdas)
        n = self.num_stages - 1 if residual else self.num_stages
        for s in range(n):
            if self.stage_frozen(s):
                out[s] = 0.0
        return out

    def set_schedule(self, schedule: Optional[TrainingSchedule]) -> None:
        """Attach a YAML :class:`TrainingSchedule` (overrides curriculum + epoch freeze)."""
        self.schedule = schedule

    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        self._steps_per_epoch = max(1, int(steps_per_epoch))

    def reset_schedule_epoch(self) -> None:
        """Clear YAML-schedule freezes at epoch boundary (``progress_scope=epoch``)."""
        if self.schedule is None or self.schedule.progress_scope != "epoch":
            return
        self._frozen_stages.zero_()
        for key in self.stage_keys:
            for p in self.lateral[key].parameters():
                p.requires_grad = True
        if self.coarse_head is not None:
            for p in self.coarse_head.parameters():
                p.requires_grad = True
        if self.predictor is not None:
            for p in self.predictor.parameters():
                p.requires_grad = True
        if self.fusion is not None:
            for p in self.fusion.parameters():
                p.requires_grad = True

    def _sync_freeze_from_schedule(self, step: int, total_steps: int) -> List[str]:
        if self.schedule is None:
            return []
        state = self.schedule.at(step, total_steps, self._steps_per_epoch)
        newly: List[str] = []
        for s in range(self.num_stages):
            key = self._stage_key(s)
            if not state.stages[key].frozen or self.stage_frozen(s):
                continue
            self._frozen_stages[s] = True
            newly.append(key)
            self._freeze_stage_modules(s)
        self._last_newly_frozen = newly
        return newly

    def _update_sigreg_cos_gate(self, step: int, total_steps: int) -> float:
        """One-way cos/fallback latch; ramp s4 SIGReg beta multiplier in [0, 1]."""
        if self.cfg.s4_cosine_level <= 0.0:
            return 1.0
        progress = float(step) / max(1, total_steps)
        cos_ema = self._mae_cos_ema if self._mae_cos_ema is not None else 0.0
        if not self._sigreg_cos_latched:
            if cos_ema >= self.cfg.s4_cosine_level:
                self._sigreg_cos_latched = True
                self._sigreg_gate_progress = progress
            elif progress >= self.cfg.s4_sigreg_fallback_progress:
                self._sigreg_cos_latched = True
                self._sigreg_gate_progress = progress
        if not self._sigreg_cos_latched:
            return 0.0
        gate_p = self._sigreg_gate_progress if self._sigreg_gate_progress is not None else 0.0
        ramp = self.cfg.s4_sigreg_ramp_progress
        if ramp <= 0.0:
            return 1.0
        return min(1.0, max(0.0, (progress - gate_p) / ramp))

    def note_mae_cos(self, mae_cos: float) -> None:
        """Update mae/cos EMA after a training step (feeds next step's gate)."""
        if self.cfg.s4_cosine_level <= 0.0:
            return
        d = self.cfg.s4_cosine_ema_decay
        val = float(mae_cos)
        if self._mae_cos_ema is None:
            self._mae_cos_ema = val
        else:
            self._mae_cos_ema = d * self._mae_cos_ema + (1.0 - d) * val

    def _apply_sigreg_cos_gate(
            self, betas: List[float], step: int, total_steps: int) -> List[float]:
        self._last_schedule_betas = list(betas)
        gate = self._update_sigreg_cos_gate(step, total_steps)
        self._last_sigreg_cos_gate = gate
        if gate == 1.0 or self.cfg.s4_cosine_level <= 0.0:
            return betas
        s4 = self.num_stages - 1
        out = list(betas)
        out[s4] = out[s4] * gate
        return out

    def _resolve_training_knobs(
            self, step: int, total_steps: int,
            epoch: Optional[int] = None) -> Tuple[List[float], List[float], float]:
        """Return ``(active, beta_sig, pred_active, progress)`` for this step."""
        if self.schedule is not None:
            self._sync_freeze_from_schedule(step, total_steps)
            state = self.schedule.at(step, total_steps, self._steps_per_epoch)
            active = [state.stages[self._stage_key(s)].active
                      for s in range(self.num_stages)]
            pred_active = [state.stages[self._stage_key(s)].pred_active
                           for s in range(self.num_stages)]
            betas = [state.stages[self._stage_key(s)].beta_sig
                     for s in range(self.num_stages)]
            betas = self._apply_sigreg_cos_gate(betas, step, total_steps)
            return active, betas, pred_active, state.progress
        if epoch is not None:
            self.apply_freeze_schedule(epoch)
        active = stage_active_schedule(
            step, total_steps, self.cfg.warmup_frac, self.cfg.fine_min_w,
            self.cfg.fine_stages, self.num_stages,
            stage_curriculum=self.cfg.stage_curriculum,
            coarse_ramp_stages=self.cfg.coarse_ramp_stages)
        progress = float(step) / max(1, total_steps)
        betas = self._apply_sigreg_cos_gate(list(self.cfg.beta_sig), step, total_steps)
        return active, betas, list(active), progress

    @property
    def stage_keys(self) -> List[str]:
        return self.backbone.stage_keys

    @property
    def strides(self) -> List[int]:
        return self.backbone.strides

    def set_augment(self, fn: Callable) -> None:
        self.augment = fn

    @staticmethod
    def _split_dual_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 5:
            if x.shape[1] < 2:
                raise ValueError(
                    f"dual_view requires >= 2 views in [B, V, C, H, W], got V={x.shape[1]}")
            return x[:, 0], x[:, 1]
        raise ValueError(
            "dual_view integrated mode expects [B, V, C, H, W] with V >= 2")

    @staticmethod
    def _split_dual_fg(fg_px: Optional[torch.Tensor]
                       ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if fg_px is None:
            return None, None
        if fg_px.dim() == 5:
            if fg_px.shape[1] < 2:
                raise ValueError(
                    f"dual_view FG expects [B, V, 1, H, W] with V >= 2, got V={fg_px.shape[1]}")
            return fg_px[:, 0], fg_px[:, 1]
        return fg_px, fg_px

    def _fg_from_px(self, fg_px: Optional[torch.Tensor]
                    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        if not self.cfg.foreground_mask or fg_px is None:
            return None, None
        if fg_px.dim() == 5:
            fg_px = fg_px[:, 0]
        fg_stages = build_fg_stages(fg_px, self.grids, self.cfg.fg_coverage)
        return fg_stages, ~fg_stages["s1"]

    # -- core training objective --------------------------------------------
    def compute_loss(self, x: torch.Tensor, fg_px: Optional[torch.Tensor] = None,
                     step: int = 0, total_steps: int = 1, epoch: Optional[int] = None):
        """Two-pass forward + pyramid residual or legacy full-latent JEPA."""
        if self.cfg.legacy_jepa:
            return self._compute_loss_legacy(x, fg_px, step, total_steps, epoch=epoch)
        return self._compute_loss_pyramid(x, fg_px, step, total_steps, epoch=epoch)

    def _shared_forward(self, x: torch.Tensor, fg_px: Optional[torch.Tensor],
                        grad_active: List[bool]):
        """Augment, masks, FG stages, target + student latents (single-view path)."""
        if x.dim() == 5:
            x = x[:, 0]
        x_aug = self.augment(x)
        b = x_aug.shape[0]

        fg_stages, bg1 = self._fg_from_px(fg_px)

        mask = self.mask_gen.generate(b, device=x_aug.device,
                                      fg_s1=fg_stages["s1"] if fg_stages else None)
        if self.check_masks:
            assert_mask_consistency(mask, self.num_stages)
        if fg_stages is not None:
            for s in range(self.num_stages):
                key = f"s{s + 1}"
                if (mask[key] & ~fg_stages[key]).any():
                    raise AssertionError(
                        f"masked positions at {key} must lie inside the FOV.")

        E_full = _project_latent(
            self.backbone(x_aug, mask1=None, bg1=bg1), self.lateral)
        E_ctx = _project_latent(
            self.backbone(x_aug, mask1=mask["s1"], bg1=bg1), self.lateral)
        E_full = self._detach_inactive_latents(E_full, grad_active)
        E_ctx = self._detach_inactive_latents(E_ctx, grad_active)
        return x_aug, mask, fg_stages, E_full, E_ctx

    def _shared_forward_integrated(
            self, x_student: torch.Tensor, x_teacher: torch.Tensor,
            fg_student: Optional[torch.Tensor],
            fg_teacher: Optional[torch.Tensor],
            grad_active: List[bool],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]],
               Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor],
               Dict[str, torch.Tensor]]:
        """Dual-view forward: masked student + stop-grad teacher latents."""
        b = x_student.shape[0]
        fg_stages, bg1 = self._fg_from_px(fg_student)
        fg_teacher_stages, bg1_teacher = self._fg_from_px(fg_teacher)

        mask = self.mask_gen.generate(b, device=x_student.device,
                                      fg_s1=fg_stages["s1"] if fg_stages else None)
        if self.check_masks:
            assert_mask_consistency(mask, self.num_stages)
        if fg_stages is not None:
            for s in range(self.num_stages):
                key = f"s{s + 1}"
                if (mask[key] & ~fg_stages[key]).any():
                    raise AssertionError(
                        f"masked positions at {key} must lie inside the FOV.")

        with torch.no_grad():
            E_teacher = _project_latent(
                self.backbone(x_teacher, mask1=None, bg1=bg1_teacher), self.lateral)

        E_ctx = _project_latent(
            self.backbone(x_student, mask1=mask["s1"], bg1=bg1), self.lateral)
        E_ctx = self._detach_inactive_latents(E_ctx, grad_active)
        return mask, fg_stages, fg_teacher_stages, E_teacher, E_ctx

    def _compute_loss_legacy(self, x: torch.Tensor, fg_px: Optional[torch.Tensor],
                             step: int, total_steps: int, *,
                             epoch: Optional[int] = None):
        active, betas, pred_active, progress = self._resolve_training_knobs(
            step, total_steps, epoch=epoch)
        bases = [float(self.cfg.stage_base_weights[s]) for s in range(self.num_stages)]
        pred_w, sig_w = self._stage_loss_weights(
            active, betas, bases, pred_active=pred_active)
        grad_active = self._stage_grad_flags(pred_w, sig_w)
        x_aug, mask, fg_stages, E_full, E_ctx = self._shared_forward(
            x, fg_px, grad_active)

        target_feats = E_full
        if fg_stages is not None:
            target_feats = _fg_gate_feats(E_full, fg_stages)
        targets = {k: stage_target_norm(v, self.cfg.target_norm).detach()
                   for k, v in target_feats.items()}

        lambdas = self._zero_frozen_lambdas(list(pred_w))
        need_pred = any(w > 0.0 for w in lambdas)
        if need_pred:
            if self.predictor is not None:
                pred = self.predictor(E_ctx, mask, fg_stages=fg_stages)
            else:
                pred = {k: gather_masked(E_ctx[k], mask[k]) for k in E_ctx}
            l_pred, per_stage_pred = masked_prediction_loss(
                pred, targets, mask, lambdas, self.cfg.pred_loss,
                self.cfg.smooth_l1_beta)
        else:
            l_pred = x_aug.new_zeros(())
            per_stage_pred = {self._stage_key(s): 0.0 for s in range(self.num_stages)}

        scales = self.cfg.stage_scale(self.lat_chans)
        l_sig = x_aug.new_zeros(())
        per_stage_sig: Dict[str, float] = {}
        for s in range(self.num_stages):
            key = f"s{s + 1}"
            if sig_w[s] <= 0.0:
                per_stage_sig[key] = 0.0
                continue
            flat = _flatten_tokens(E_full[key])
            if fg_stages is not None:
                flat = flat[fg_stages[key].reshape(-1)]
            sig_s = self.sigreg[s](flat)
            per_stage_sig[key] = float(sig_s.detach())
            l_sig = l_sig + sig_w[s] * scales[s] * sig_s

        loss = l_pred + l_sig
        logs = {"total": float(loss.detach()),
                "l_pred": float(l_pred.detach()),
                "l_sig": float(l_sig.detach()),
                "l_mae": 0.0,
                "schedule_progress": progress}
        for s in range(self.num_stages):
            key = f"s{s + 1}"
            logs[f"pred/{key}"] = per_stage_pred[key]
            logs[f"sig/{key}"] = per_stage_sig[key]
            logs[f"lambda/{key}"] = lambdas[s]
            logs[f"active/{key}"] = active[s]
            if fg_stages is not None:
                logs[f"fg_cov/{key}"] = float(fg_stages[key].float().mean())
        logs.update(self._freeze_logs())
        logs.update(stage_feature_diagnostics(E_full))
        return loss, logs

    def _s4_mae_mask(self, mask: Dict[str, torch.Tensor],
                     fg_stages: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        m = mask["s4"]
        if fg_stages is not None:
            m = m & fg_stages["s4"]
        return m

    def _build_s4_coarse_field(
            self, E_ctx: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor],
            fg_stages: Optional[Dict[str, torch.Tensor]], grad_active_s4: bool,
            pred_all: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Assemble full-grid C4 for pyramid MAE / SIGReg / probe."""
        if self.cfg.coarse_mim_mode in ("cross_attn", "integrated"):
            if pred_all is None:
                if self.cfg.coarse_mim_mode == "integrated":
                    pred_all = self.fusion(E_ctx, mask, fg_stages=fg_stages)
                else:
                    pred_all = self.predictor(E_ctx, mask, fg_stages=fg_stages)
            pred_s4 = pred_all["s4"]
            ctx = E_ctx["s4"] if grad_active_s4 else E_ctx["s4"].detach()
            if not grad_active_s4:
                pred_s4 = pred_s4.detach()
            C4 = assemble_coarse_field(ctx, pred_s4, mask["s4"])
        else:
            if grad_active_s4:
                C4 = self.coarse_head(E_ctx["s4"])
            else:
                C4 = self.coarse_head(E_ctx["s4"]).detach()
        if fg_stages is not None:
            C4 = fg_gate(C4, fg_stages["s4"])
        return C4, pred_all

    def _compute_loss_pyramid(self, x: torch.Tensor, fg_px: Optional[torch.Tensor],
                              step: int, total_steps: int, *,
                              epoch: Optional[int] = None):
        if self.cfg.coarse_mim_mode == "integrated":
            return self._compute_loss_pyramid_integrated(
                x, fg_px, step, total_steps, epoch=epoch)
        return self._compute_loss_pyramid_legacy(
            x, fg_px, step, total_steps, epoch=epoch)

    def _compute_loss_pyramid_integrated(
            self, x: torch.Tensor, fg_px: Optional[torch.Tensor],
            step: int, total_steps: int, *, epoch: Optional[int] = None):
        active, betas, pred_active, progress = self._resolve_training_knobs(
            step, total_steps, epoch=epoch)
        bases = [float(self.cfg.stage_base_weights[s]) for s in range(self.num_stages)]
        pred_w, sig_w = self._stage_loss_weights(
            active, betas, bases, pred_active=pred_active)
        grad_active = self._stage_grad_flags(pred_w, sig_w)

        x_student, x_teacher = self._split_dual_views(x)
        fg_student, fg_teacher = self._split_dual_fg(fg_px)
        mask, fg_stages, fg_teacher_stages, E_teacher, E_ctx = (
            self._shared_forward_integrated(
                x_student, x_teacher, fg_student, fg_teacher, grad_active))

        need_pred = any(w > 0.0 for w in pred_w)
        pred_all: Optional[Dict[str, torch.Tensor]] = None
        if need_pred:
            pred_all = self.fusion(E_ctx, mask, fg_stages=fg_stages)

        E_hat: Dict[str, torch.Tensor] = {}
        if pred_all is not None:
            for s in range(self.num_stages):
                key = self._stage_key(s)
                if pred_w[s] > 0.0 and not self.stage_frozen(s):
                    E_hat[key] = scatter_masked(E_ctx[key], pred_all[key], mask[key])
                else:
                    E_hat[key] = E_ctx[key]
        else:
            E_hat = {k: E_ctx[k] for k in E_ctx}

        s4 = self.num_stages - 1
        if pred_all is not None and pred_w[s4] > 0.0 and not self.stage_frozen(s4):
            C4_hat = assemble_coarse_field(E_ctx["s4"], pred_all["s4"], mask["s4"])
        elif grad_active[s4]:
            C4_hat = E_ctx["s4"]
        else:
            C4_hat = E_ctx["s4"].detach()
        if fg_stages is not None:
            C4_hat = fg_gate(C4_hat, fg_stages["s4"])

        C4_teacher = E_teacher["s4"]
        if fg_teacher_stages is not None:
            C4_teacher = fg_gate(C4_teacher, fg_teacher_stages["s4"])

        align = self._residual_align_for_pyramid()
        bands_S = pyramid_sigreg_features(
            E_hat, C4_hat, self.grids, fg_stages,
            strict_laplacian=self.cfg.strict_laplacian,
            s4_on=self.cfg.sigreg_s4_on,
            residual_align=align)
        bands_T = pyramid_sigreg_features(
            E_teacher, C4_teacher, self.grids, fg_teacher_stages,
            strict_laplacian=self.cfg.strict_laplacian,
            s4_on=self.cfg.sigreg_s4_on,
            residual_align=align)
        bands_T_norm = {
            k: stage_target_norm(v, self.cfg.target_norm).detach()
            for k, v in bands_T.items()}

        lambdas = self._zero_frozen_lambdas(list(pred_w))
        mae_cos = 0.0
        if need_pred and any(w > 0.0 for w in lambdas):
            pred_bands = {
                k: gather_masked(bands_S[k], mask[k]) for k in bands_S}
            l_pred, per_stage_pred = masked_prediction_loss(
                pred_bands, bands_T_norm, mask, lambdas, self.cfg.pred_loss,
                self.cfg.smooth_l1_beta)
            if pred_bands["s4"].numel() > 0:
                tgt_s4 = gather_masked(bands_T_norm["s4"], mask["s4"])
                mae_cos = float(
                    F.cosine_similarity(
                        pred_bands["s4"], tgt_s4.to(pred_bands["s4"].dtype), dim=-1
                    ).mean().detach())
        else:
            l_pred = x_student.new_zeros(())
            per_stage_pred = {self._stage_key(s): 0.0 for s in range(self.num_stages)}

        sched_betas = getattr(self, "_last_schedule_betas", betas)
        need_sig_probe = (
            self.cfg.s4_cosine_level > 0.0 and sched_betas[s4] > 0.0)
        need_sig_bands = (
            any(sig_w[s] > 0.0 for s in range(self.num_stages)) or need_sig_probe)

        scales = self.cfg.stage_scale(self.lat_chans)
        l_sig = x_student.new_zeros(())
        per_stage_sig: Dict[str, float] = {}
        per_stage_sig_raw: Dict[str, float] = {}
        if need_sig_bands:
            sig_feats = bands_S
            for s in range(self.num_stages):
                key = self._stage_key(s)
                probe_s = (
                    need_sig_probe and s == s4 and sched_betas[s] > 0.0)
                if (sig_w[s] <= 0.0 and not probe_s) or not sig_feats:
                    per_stage_sig[key] = 0.0
                    continue
                fg_s = fg_stages[key] if fg_stages is not None else None
                if sig_w[s] > 0.0:
                    sig_s = self.sigreg_r[s](sig_feats[key], fg_s)
                    per_stage_sig[key] = float(sig_s.detach())
                    per_stage_sig_raw[key] = per_stage_sig[key]
                    l_sig = l_sig + sig_w[s] * scales[s] * sig_s
                else:
                    with torch.no_grad():
                        sig_s = self.sigreg_r[s](sig_feats[key], fg_s)
                    per_stage_sig[key] = 0.0
                    per_stage_sig_raw[key] = float(sig_s.detach())

        loss = l_pred + l_sig

        logs = {"total": float(loss.detach()),
                "l_pred": float(l_pred.detach()),
                "l_sig": float(l_sig.detach()),
                "l_mae": float(l_pred.detach()),
                "mae/cos": mae_cos,
                "schedule_progress": progress}
        for s in range(self.num_stages):
            key = self._stage_key(s)
            logs[f"pred/{key}"] = per_stage_pred[key]
            logs[f"mae/{key}"] = per_stage_pred[key]
            logs[f"lambda/{key}"] = lambdas[s]
            logs[f"sig/{key}"] = per_stage_sig.get(key, 0.0)
            if key in per_stage_sig_raw:
                logs[f"sig/raw/{key}"] = per_stage_sig_raw[key]
        for s in range(self.num_stages):
            key = self._stage_key(s)
            logs[f"active/{key}"] = active[s]
            if fg_stages is not None:
                logs[f"fg_cov/{key}"] = float(fg_stages[key].float().mean())
        logs.update(self._freeze_logs())
        logs.update(stage_feature_diagnostics(E_ctx))
        logs.update(stage_feature_diagnostics({"C4": C4_hat}))
        if self.cfg.s4_cosine_level > 0.0:
            logs["sigreg/cos_gate"] = self._last_sigreg_cos_gate
            logs["sigreg/cos_ema"] = float(self._mae_cos_ema or 0.0)
            logs["sigreg/cos_latched"] = float(self._sigreg_cos_latched)
        return loss, logs

    def _compute_loss_pyramid_legacy(self, x: torch.Tensor, fg_px: Optional[torch.Tensor],
                              step: int, total_steps: int, *,
                              epoch: Optional[int] = None):
        active, betas, pred_active, progress = self._resolve_training_knobs(
            step, total_steps, epoch=epoch)
        bases = [float(self.cfg.stage_base_weights[s]) for s in range(self.num_stages)]
        pred_w, sig_w = self._stage_loss_weights(
            active, betas, bases, pred_active=pred_active)
        grad_active = self._stage_grad_flags(pred_w, sig_w)
        x_aug, mask, fg_stages, E_full, E_ctx = self._shared_forward(
            x, fg_px, grad_active)

        s4 = self.num_stages - 1
        mae_w = 0.0 if self.stage_frozen(s4) else pred_w[s4]
        lambdas_res = self._zero_frozen_lambdas(list(pred_w[:s4]), residual=True)
        need_s4_cross = (
            mae_w > 0.0 and self.cfg.coarse_mim_mode == "cross_attn"
            and self.predictor is not None)
        need_pred = need_s4_cross or any(w > 0.0 for w in lambdas_res)
        pred_all: Optional[Dict[str, torch.Tensor]] = None
        if need_pred:
            if self.predictor is not None:
                pred_all = self.predictor(E_ctx, mask, fg_stages=fg_stages)
            elif need_s4_cross:
                raise RuntimeError(
                    "coarse_mim_mode='cross_attn' requires a predictor module")

        C4, _ = self._build_s4_coarse_field(
            E_ctx, mask, fg_stages, grad_active[s4], pred_all=pred_all)

        mae_cos = 0.0
        if mae_w > 0.0:
            t4 = E_full["s4"]
            if fg_stages is not None:
                t4 = _fg_gate_feats({"s4": t4}, fg_stages)["s4"]
            T4 = stage_target_norm(t4, self.cfg.target_norm).detach()
            if self.cfg.coarse_mim_mode == "cross_attn":
                mae_mask = self._s4_mae_mask(mask, fg_stages)
                pred_s4 = pred_all["s4"]
                tgt_m = gather_masked(T4, mae_mask)
                l_mae = F.smooth_l1_loss(
                    pred_s4, tgt_m.to(pred_s4.dtype),
                    beta=self.cfg.smooth_l1_beta)
                if pred_s4.numel() > 0:
                    mae_cos = float(
                        F.cosine_similarity(pred_s4, tgt_m, dim=-1).mean().detach())
            else:
                l_mae = masked_coarse_mae(
                    C4, T4, mask["s4"], self.cfg.smooth_l1_beta,
                    fg_mask=fg_stages["s4"] if fg_stages is not None else None)
        else:
            l_mae = x_aug.new_zeros(())

        need_residuals = any(
            pred_w[s] > 0.0 and not self.stage_frozen(s)
            for s in range(self.num_stages - 1))
        sched_betas = getattr(self, "_last_schedule_betas", betas)
        need_sig_probe = (
            not self.cfg.legacy_jepa
            and self.cfg.s4_cosine_level > 0.0
            and sched_betas[s4] > 0.0)
        need_sig_bands = (
            any(sig_w[s] > 0.0 for s in range(self.num_stages))
            or need_sig_probe)
        band_residuals: Dict[str, torch.Tensor] = {}
        sig_feats: Dict[str, torch.Tensor] = {}
        if need_residuals:
            band_residuals = pyramid_band_residuals(
                E_full, C4, self.grids, fg_stages,
                strict_laplacian=self.cfg.strict_laplacian,
                residual_align=self._residual_align_for_pyramid())
        if need_sig_bands:
            sig_feats = pyramid_sigreg_features(
                E_full, C4, self.grids, fg_stages,
                strict_laplacian=self.cfg.strict_laplacian,
                s4_on=self.cfg.sigreg_s4_on,
                residual_align=self._residual_align_for_pyramid())
        if need_residuals:
            residuals = {k: band_residuals[k] for k in _RESIDUAL_KEYS}
            R_tgt = {k: stage_target_norm(v, self.cfg.target_norm).detach()
                     for k, v in residuals.items()}
        else:
            residuals = {}
            R_tgt = {}

        if any(w > 0.0 for w in lambdas_res):
            if pred_all is not None:
                pred = {k: pred_all[k] for k in _RESIDUAL_KEYS}
            else:
                pred = {k: gather_masked(E_ctx[k], mask[k]) for k in _RESIDUAL_KEYS}
            mask_r = {k: mask[k] for k in _RESIDUAL_KEYS}
            l_pred, per_stage_pred = masked_prediction_loss(
                pred, R_tgt, mask_r, lambdas_res, self.cfg.pred_loss,
                self.cfg.smooth_l1_beta)
        else:
            l_pred = x_aug.new_zeros(())
            per_stage_pred = {k: 0.0 for k in _RESIDUAL_KEYS}

        scales = self.cfg.stage_scale(self.lat_chans)
        l_sig = x_aug.new_zeros(())
        per_stage_sig: Dict[str, float] = {}
        per_stage_sig_raw: Dict[str, float] = {}
        for s in range(self.num_stages):
            key = self._stage_key(s)
            probe_s = (
                need_sig_probe and s == s4 and sched_betas[s] > 0.0 and sig_feats)
            if (sig_w[s] <= 0.0 and not probe_s) or not sig_feats:
                per_stage_sig[key] = 0.0
                continue
            fg_s = fg_stages[key] if fg_stages is not None else None
            if sig_w[s] > 0.0:
                sig_s = self.sigreg_r[s](sig_feats[key], fg_s)
                per_stage_sig[key] = float(sig_s.detach())
                per_stage_sig_raw[key] = per_stage_sig[key]
                l_sig = l_sig + sig_w[s] * scales[s] * sig_s
            else:
                with torch.no_grad():
                    sig_s = self.sigreg_r[s](sig_feats[key], fg_s)
                per_stage_sig[key] = 0.0
                per_stage_sig_raw[key] = float(sig_s.detach())

        loss = mae_w * l_mae + l_pred + l_sig

        logs = {"total": float(loss.detach()),
                "l_pred": float(l_pred.detach()),
                "l_sig": float(l_sig.detach()),
                "l_mae": float(l_mae.detach()),
                "mae/s4": float(l_mae.detach()),
                "mae/cos": mae_cos,
                "schedule_progress": progress}
        for key in _RESIDUAL_KEYS:
            logs[f"pred/{key}"] = per_stage_pred[key]
            i = int(key[1]) - 1
            logs[f"lambda/{key}"] = lambdas_res[i]
        for s in range(self.num_stages):
            key = self._stage_key(s)
            logs[f"sig/{key}"] = per_stage_sig[key]
            if key in per_stage_sig_raw:
                logs[f"sig/raw/{key}"] = per_stage_sig_raw[key]
        logs["lambda/s4"] = mae_w
        for s in range(self.num_stages):
            key = f"s{s + 1}"
            logs[f"active/{key}"] = active[s]
            if fg_stages is not None:
                logs[f"fg_cov/{key}"] = float(fg_stages[key].float().mean())
        logs.update(self._freeze_logs())
        logs.update(stage_feature_diagnostics(E_full))
        logs.update(stage_feature_diagnostics({"C4": C4}))
        if self.cfg.s4_cosine_level > 0.0:
            logs["sigreg/cos_gate"] = self._last_sigreg_cos_gate
            logs["sigreg/cos_ema"] = float(self._mae_cos_ema or 0.0)
            logs["sigreg/cos_latched"] = float(self._sigreg_cos_latched)
        return loss, logs

    @torch.no_grad()
    def extract_pyramid_probe(self, x: torch.Tensor,
                              fg_px: Optional[torch.Tensor] = None
                              ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Decomposition for PCA: ``C4``, residuals ``R``, and reconstructed ``E``."""
        was_training = self.training
        self.eval()
        b = x.shape[0]
        fg_stages = None
        bg1 = None
        if self.cfg.foreground_mask and fg_px is not None:
            if fg_px.dim() == 5:
                fg_px = fg_px[:, 0]
            fg_stages = build_fg_stages(fg_px, self.grids, self.cfg.fg_coverage)
            bg1 = ~fg_stages["s1"]
        mask = self.mask_gen.generate(b, device=x.device,
                                      fg_s1=fg_stages["s1"] if fg_stages else None)
        E_full = _project_latent(self.backbone(x, mask1=None, bg1=bg1), self.lateral)
        E_ctx = _project_latent(
            self.backbone(x, mask1=mask["s1"], bg1=bg1), self.lateral)
        pred_all = None
        if self.cfg.coarse_mim_mode == "integrated" and self.fusion is not None:
            pred_all = self.fusion(E_ctx, mask, fg_stages=fg_stages)
        elif self.cfg.coarse_mim_mode == "cross_attn" and self.predictor is not None:
            pred_all = self.predictor(E_ctx, mask, fg_stages=fg_stages)
        C4, _ = self._build_s4_coarse_field(
            E_ctx, mask, fg_stages, grad_active_s4=True, pred_all=pred_all)
        t4 = E_full["s4"]
        if fg_stages is not None:
            t4 = _fg_gate_feats({"s4": t4}, fg_stages)["s4"]
        T4 = stage_target_norm(t4, self.cfg.target_norm)
        R = hierarchical_residuals(
            E_full, C4, self.grids, fg_stages,
            strict_laplacian=self.cfg.strict_laplacian,
            residual_align=self._residual_align_for_pyramid())
        E_hat = reconstruct_from_residuals(
            C4, R, E_full, self.grids,
            strict_laplacian=self.cfg.strict_laplacian,
            residual_align=self._residual_align_for_pyramid())
        if was_training:
            self.train()
        return {"C4": {"s4": C4}, "T4": {"s4": T4}, "R": R, "E": E_full,
                "E_hat": E_hat}

    def training_step(self, batch, step: int = 0, total_steps: int = 1):
        """Design §5 entry point: ``training_step(batch) -> dict(loss, logs)``."""
        dual = (self.cfg.dual_view and self.cfg.coarse_mim_mode == "integrated"
                and not self.cfg.legacy_jepa)
        x = extract_images(batch, dual_view=dual)
        fg = extract_fg_masks(batch)
        loss, logs = self.compute_loss(x, fg_px=fg, step=step, total_steps=total_steps)
        return {"loss": loss, "logs": logs}

    # -- inference API (mirrors SwinMSEncoder.extract_features) ---------------
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, normalize: bool = True,
                         project: bool = False, use_latent: bool = False,
                         fg_px: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """One clean backbone pass; no mask, predictor, or SIGReg touched.

        ``use_latent=True`` returns the lateral-projected latent that SIGReg/JEPA
        regularize (carries the isotropy guarantee). ``use_latent=False`` (default)
        returns raw backbone maps (LayerNorm-conditioned only, not SIGReg-regularized).
        """
        return _extract_features(self.backbone, self.lateral, x,
                                 self.cfg.target_norm, normalize, project,
                                 use_latent=use_latent, fg_px=fg_px,
                                 grids=self.grids, fg_coverage=self.cfg.fg_coverage,
                                 foreground_mask=self.cfg.foreground_mask)

    @torch.no_grad()
    def regularized_features(self, x: torch.Tensor, normalize: bool = True,
                             fg_px: Optional[torch.Tensor] = None
                             ) -> Dict[str, torch.Tensor]:
        """Projected latent stack that carries the SIGReg isotropy guarantee."""
        return self.extract_features(x, normalize=normalize, project=False,
                                     use_latent=True, fg_px=fg_px)


def extract_fg_masks(batch) -> Optional[torch.Tensor]:
    """Pull the foreground mask out of the repo's batch format when present.

    ``TomographyDataset`` with ``foreground_mask=True`` yields
    ``(views, fg_views)`` where ``fg_views`` is ``[B, V, 1, H, W]``. Returns
    ``None`` when labels are absent or foreground masking is off.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        return None
    fg = batch[1]
    if not isinstance(fg, torch.Tensor) or fg.dim() < 4:
        return None
    if fg.dim() == 5:              # [B, V, 1, H, W] -> first view at compute_loss
        return fg
    return fg


def extract_images(batch, dual_view: bool = False) -> torch.Tensor:
    """Pull the image tensor out of the repo's batch format.

    ``TomographyDataset`` yields ``(views, label)`` where ``views`` is
    ``[B, V, C, H, W]`` (or ``[B, C, H, W]`` for a single view). When
    ``dual_view=False``, only the first view is returned.
    """
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    if x.dim() == 5 and not dual_view:
        x = x[:, 0]
    return x


def _extract_features(backbone: SwinMultiScaleBackbone, lateral: nn.ModuleDict,
                      x: torch.Tensor, target_norm: str, normalize: bool,
                      project: bool, use_latent: bool = False,
                      fg_px: Optional[torch.Tensor] = None,
                      grids: Optional[List] = None,
                      fg_coverage: float = 0.01,
                      foreground_mask: bool = False) -> Dict[str, torch.Tensor]:
    """Backbone forward with optional lateral projection.

    Only ``use_latent=True`` outputs carry the SIGReg isotropy guarantee; raw
    backbone maps (``use_latent=False``) are LayerNorm-conditioned only.
    """
    bg1 = None
    if foreground_mask and fg_px is not None and grids is not None:
        if fg_px.dim() == 5:
            fg_px = fg_px[:, 0]
        fg_stages = build_fg_stages(fg_px, grids, fg_coverage)
        bg1 = ~fg_stages["s1"]
    feats = backbone(x, mask1=None, bg1=bg1)
    if use_latent and lateral is not None:
        feats = _project_latent(feats, lateral)
    if normalize:
        feats = {k: stage_target_norm(v, "ln" if target_norm == "none" else target_norm)
                 for k, v in feats.items()}
    if project:
        if lateral is None:
            raise ValueError("project=True requires lateral projections.")
        if use_latent:
            proj = {f"{k}_proj": v for k, v in feats.items()}
        else:
            proj = {f"{k}_proj": lateral[k](v) for k, v in feats.items()}
        feats = {**feats, **proj}
    return feats


class SwinMSEncoder(nn.Module):
    """Inference-only multi-scale encoder: a pretrained backbone (+ optional
    lateral projections), with every training-only module dropped.

    Hands a clean multi-scale stack to a downstream upsampler (ViT-Up family).
    """

    def __init__(self, cfg: SwinMSJEPAConfig, with_lateral: bool = True):
        super().__init__()
        self.cfg = cfg
        self.backbone = SwinMultiScaleBackbone(
            model_name=cfg.backbone_name, img_size=cfg.img_size,
            in_chans=cfg.in_chans, pretrained=False, drop_path_rate=0.0,
            use_rope=cfg.use_rope, rope_theta=cfg.rope_theta,
            embed_dim=cfg.backbone_embed_dim)
        self.lat_chans: List[int] = list(cfg.lat_dims)
        self.lateral = None
        if with_lateral:
            self.lateral = nn.ModuleDict({
                f"s{s + 1}": nn.Conv2d(self.backbone.out_chans[s], self.lat_chans[s], 1)
                for s in range(self.backbone._num_stages)})

    @property
    def out_chans(self) -> List[int]:
        """Backbone channel widths (before lateral projection)."""
        return self.backbone.out_chans

    @property
    def stage_keys(self) -> List[str]:
        return self.backbone.stage_keys

    @property
    def strides(self) -> List[int]:
        return self.backbone.strides

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, normalize: bool = True,
                         project: bool = False, use_latent: bool = False,
                         fg_px: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Backbone (+ optional lateral) features without training-only modules.

        ``use_latent=True`` returns the lateral-projected latent that SIGReg/JEPA
        regularize (carries the isotropy guarantee). ``use_latent=False`` (default)
        returns raw backbone maps (LayerNorm-conditioned only, not SIGReg-regularized).
        """
        if project and self.lateral is None:
            raise ValueError("project=True requires the encoder built with_lateral=True.")
        return _extract_features(self.backbone, self.lateral, x,
                                 self.cfg.target_norm, normalize, project,
                                 use_latent=use_latent, fg_px=fg_px,
                                 grids=[self.backbone.stage_grid(s)
                                        for s in range(self.backbone._num_stages)],
                                 fg_coverage=self.cfg.fg_coverage,
                                 foreground_mask=self.cfg.foreground_mask)

    @torch.no_grad()
    def regularized_features(self, x: torch.Tensor, normalize: bool = True,
                             fg_px: Optional[torch.Tensor] = None
                             ) -> Dict[str, torch.Tensor]:
        """Projected latent stack that carries the SIGReg isotropy guarantee."""
        return self.extract_features(x, normalize=normalize, project=False,
                                     use_latent=True, fg_px=fg_px)

    @classmethod
    def from_pretrained(cls, ckpt_path: str, cfg: SwinMSJEPAConfig,
                        map_location="cpu", with_lateral: bool = True) -> "SwinMSEncoder":
        """Load backbone (+ lateral) weights from a :class:`SwinMSJEPA` checkpoint.

        The ``predictor`` / ``sigreg`` / ``mask`` state is ignored entirely.
        """
        enc = cls(cfg, with_lateral=with_lateral)
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state = ckpt.get("model", ckpt)
        bb = {k[len("backbone."):]: v for k, v in state.items()
              if k.startswith("backbone.")}
        enc.backbone.load_state_dict(bb)
        if with_lateral and enc.lateral is not None:
            lat = {k[len("lateral."):]: v for k, v in state.items()
                   if k.startswith("lateral.")}
            if lat:
                enc.lateral.load_state_dict(lat)
        return enc
