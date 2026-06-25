"""SwinSimMIM — dual-aug masked latent SimMIM on raw E_s maps (no pyramid residuals)."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomojepa.core.augmentations import build_fg_stages
from .backbone import SwinMultiScaleBackbone
from .mask import MultiScaleBlockMask, assert_mask_consistency
from .sigreg import ImageGroupedStageSIGReg
from .losses import (
    stage_target_norm, gather_masked, masked_prediction_loss,
    stage_feature_diagnostics,
)
from .schedule import TrainingSchedule
from .simmim_config import SwinSimMIMConfig


def _project_latent(feats: Dict[str, torch.Tensor],
                    lateral: nn.ModuleDict) -> Dict[str, torch.Tensor]:
    return {k: lateral[k](v) for k, v in feats.items()}


class SwinSimMIM(nn.Module):
    """Single-model dual-aug SimMIM: masked ctx view vs stop-grad target view."""

    has_ema = False

    def __init__(self, cfg: SwinSimMIMConfig, check_masks: bool = True):
        super().__init__()
        self.cfg = cfg
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

        self.lateral = nn.ModuleDict({
            f"s{s + 1}": nn.Conv2d(self.out_chans[s], self.lat_chans[s], kernel_size=1)
            for s in range(self.num_stages)})

        grid4 = self.backbone.stage_grid(self.num_stages - 1)
        self.mask_gen = MultiScaleBlockMask(
            grid4=grid4, num_stages=self.num_stages, mask_ratio=cfg.mask_ratio,
            mask_mode=cfg.mask_mode, num_blocks=cfg.mask_num_blocks,
            block_scale_range=cfg.block_scale_range)

        s4 = self.num_stages - 1
        q_len = cfg.sigreg_queue_len if cfg.sigreg_pooled else 0
        mdist = cfg.sigreg_min_token_dist

        def _make_sigreg(stage_idx: int) -> ImageGroupedStageSIGReg:
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

        self.sigreg_r = nn.ModuleList([_make_sigreg(s) for s in range(self.num_stages)])

        self.register_buffer(
            "_frozen_stages",
            torch.zeros(self.num_stages, dtype=torch.bool),
            persistent=True)

    def _stage_key(self, stage_idx: int) -> str:
        return f"s{stage_idx + 1}"

    @property
    def stage_keys(self) -> List[str]:
        return self.backbone.stage_keys

    @property
    def strides(self) -> List[int]:
        return self.backbone.strides

    def stage_frozen(self, stage_idx: int) -> bool:
        return bool(self._frozen_stages[stage_idx].item())

    def set_schedule(self, schedule: Optional[TrainingSchedule]) -> None:
        self.schedule = schedule

    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        self._steps_per_epoch = max(1, int(steps_per_epoch))

    def reset_schedule_epoch(self) -> None:
        if self.schedule is None or self.schedule.progress_scope != "epoch":
            return
        self._frozen_stages.zero_()
        for key in self.stage_keys:
            for p in self.lateral[key].parameters():
                p.requires_grad = True

    def apply_freeze_schedule(self, epoch: int) -> List[str]:
        newly: List[str] = []
        for s, start in enumerate(self.cfg.freeze_after_epoch):
            if start <= 0 or epoch < start or self.stage_frozen(s):
                continue
            self._frozen_stages[s] = True
            newly.append(self._stage_key(s))
            for p in self.lateral[self._stage_key(s)].parameters():
                p.requires_grad = False
        self._last_newly_frozen = newly
        return newly

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
            for p in self.lateral[key].parameters():
                p.requires_grad = False
        self._last_newly_frozen = newly
        return newly

    def note_mae_cos(self, mae_cos: float) -> None:
        if self.cfg.s4_cosine_level <= 0.0:
            return
        d = self.cfg.s4_cosine_ema_decay
        val = float(mae_cos)
        if self._mae_cos_ema is None:
            self._mae_cos_ema = val
        else:
            self._mae_cos_ema = d * self._mae_cos_ema + (1.0 - d) * val

    def _update_sigreg_cos_gate(self, step: int, total_steps: int) -> float:
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

    def _apply_sigreg_cos_gate(self, betas: List[float], step: int, total_steps: int) -> List[float]:
        self._last_schedule_betas = list(betas)
        gate = self._update_sigreg_cos_gate(step, total_steps)
        self._last_sigreg_cos_gate = gate
        if gate == 1.0 or self.cfg.s4_cosine_level <= 0.0:
            return betas
        if self.cfg.sigreg_cos_gate_all_stages:
            return [b * gate for b in betas]
        s4 = self.num_stages - 1
        out = list(betas)
        out[s4] = out[s4] * gate
        return out

    def _resolve_training_knobs(
            self, step: int, total_steps: int,
            epoch: Optional[int] = None) -> Tuple[List[float], List[float], List[float], float]:
        if self.schedule is not None:
            self._sync_freeze_from_schedule(step, total_steps)
            state = self.schedule.at(step, total_steps, self._steps_per_epoch)
            active = [state.stages[self._stage_key(s)].active for s in range(self.num_stages)]
            pred_active = [state.stages[self._stage_key(s)].pred_active
                           for s in range(self.num_stages)]
            betas = [state.stages[self._stage_key(s)].beta_sig for s in range(self.num_stages)]
            betas = self._apply_sigreg_cos_gate(betas, step, total_steps)
            return active, betas, pred_active, state.progress
        if epoch is not None:
            self.apply_freeze_schedule(epoch)
        active = [1.0] * self.num_stages
        betas = self._apply_sigreg_cos_gate(list(self.cfg.beta_sig), step, total_steps)
        return active, betas, list(active), float(step) / max(1, total_steps)

    def _stage_loss_weights(
            self, active: List[float], betas: List[float],
            bases: List[float], pred_active: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        pa = pred_active if pred_active is not None else active
        pred = [float(bases[s]) * pa[s] for s in range(self.num_stages)]
        sig = [betas[s] * active[s] for s in range(self.num_stages)]
        return pred, sig

    def _zero_frozen_lambdas(self, lambdas: List[float]) -> List[float]:
        out = list(lambdas)
        for s in range(self.num_stages):
            if self.stage_frozen(s):
                out[s] = 0.0
        return out

    @staticmethod
    def _split_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 5 or x.shape[1] < 2:
            raise ValueError(f"expected [B, V>=2, C, H, W], got {tuple(x.shape)}")
        return x[:, 0], x[:, 1]

    @staticmethod
    def _split_fg(fg_px: Optional[torch.Tensor]
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if fg_px is None:
            return None, None
        if fg_px.dim() == 5:
            if fg_px.shape[1] < 2:
                raise ValueError(f"FG expects [B, V>=2, 1, H, W], got V={fg_px.shape[1]}")
            return fg_px[:, 0], fg_px[:, 1]
        return fg_px, fg_px

    def _fg_from_px(self, fg_px: Optional[torch.Tensor]
                    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        if not self.cfg.foreground_mask or fg_px is None:
            return None, None
        fg_stages = build_fg_stages(fg_px, self.grids, self.cfg.fg_coverage)
        return fg_stages, ~fg_stages["s1"]

    def compute_loss(self, x: torch.Tensor, fg_px: Optional[torch.Tensor] = None,
                     step: int = 0, total_steps: int = 1, epoch: Optional[int] = None):
        active, betas, pred_active, progress = self._resolve_training_knobs(
            step, total_steps, epoch=epoch)
        bases = [float(self.cfg.stage_base_weights[s]) for s in range(self.num_stages)]
        pred_w, sig_w = self._stage_loss_weights(active, betas, bases, pred_active=pred_active)

        x_ctx, x_tgt = self._split_views(x)
        fg_ctx, fg_tgt = self._split_fg(fg_px)
        fg_stages, bg1 = self._fg_from_px(fg_ctx)

        mask = self.mask_gen.generate(x_ctx.shape[0], device=x_ctx.device,
                                      fg_s1=fg_stages["s1"] if fg_stages else None)
        if self.check_masks:
            assert_mask_consistency(mask, self.num_stages)
        if fg_stages is not None:
            for s in range(self.num_stages):
                key = self._stage_key(s)
                if (mask[key] & ~fg_stages[key]).any():
                    raise AssertionError(f"masked positions at {key} must lie inside FOV")

        with torch.no_grad():
            fg_tgt_stages, bg1_tgt = self._fg_from_px(fg_tgt)
            E_tgt_raw = _project_latent(
                self.backbone(x_tgt, mask1=None, bg1=bg1_tgt), self.lateral)
            E_tgt = {
                k: stage_target_norm(v, self.cfg.target_norm).detach()
                for k, v in E_tgt_raw.items()}

        mask1_ctx = mask["s1"] if self.cfg.use_mask_token else None
        E_ctx = _project_latent(
            self.backbone(x_ctx, mask1=mask1_ctx, bg1=bg1), self.lateral)

        lambdas = self._zero_frozen_lambdas(list(pred_w))
        mae_cos = 0.0
        if any(w > 0.0 for w in lambdas):
            pred_tok = {k: gather_masked(E_ctx[k], mask[k]) for k in E_ctx}
            l_mae, per_stage_mae = masked_prediction_loss(
                pred_tok, E_tgt, mask, lambdas, self.cfg.pred_loss,
                self.cfg.smooth_l1_beta)
            s4_key = self._stage_key(self.num_stages - 1)
            if lambdas[self.num_stages - 1] > 0.0 and pred_tok[s4_key].numel() > 0:
                tgt_s4 = gather_masked(E_tgt[s4_key], mask[s4_key])
                mae_cos = float(
                    F.cosine_similarity(
                        pred_tok[s4_key], tgt_s4.to(pred_tok[s4_key].dtype), dim=-1
                    ).mean().detach())
        else:
            l_mae = x_ctx.new_zeros(())
            per_stage_mae = {self._stage_key(s): 0.0 for s in range(self.num_stages)}

        s4 = self.num_stages - 1
        sched_betas = getattr(self, "_last_schedule_betas", betas)
        need_sig_probe = self.cfg.s4_cosine_level > 0.0 and sched_betas[s4] > 0.0
        scales = self.cfg.stage_scale(self.lat_chans)
        l_sig = x_ctx.new_zeros(())
        per_stage_sig: Dict[str, float] = {}
        per_stage_sig_raw: Dict[str, float] = {}
        for s in range(self.num_stages):
            key = self._stage_key(s)
            probe_s = need_sig_probe and s == s4 and sched_betas[s] > 0.0
            if (sig_w[s] <= 0.0 and not probe_s) or self.stage_frozen(s):
                per_stage_sig[key] = 0.0
                continue
            fg_s = fg_stages[key] if fg_stages is not None else None
            if sig_w[s] > 0.0:
                sig_s = self.sigreg_r[s](E_ctx[key], fg_s)
                per_stage_sig[key] = float(sig_s.detach())
                per_stage_sig_raw[key] = per_stage_sig[key]
                l_sig = l_sig + sig_w[s] * scales[s] * sig_s
            else:
                with torch.no_grad():
                    sig_s = self.sigreg_r[s](E_ctx[key], fg_s)
                per_stage_sig[key] = 0.0
                per_stage_sig_raw[key] = float(sig_s.detach())

        loss = l_mae + l_sig
        logs = {"total": float(loss.detach()),
                "l_pred": float(l_mae.detach()),
                "l_sig": float(l_sig.detach()),
                "l_mae": float(l_mae.detach()),
                "mae/cos": mae_cos,
                "schedule_progress": progress}
        for s in range(self.num_stages):
            key = self._stage_key(s)
            logs[f"pred/{key}"] = per_stage_mae.get(key, 0.0)
            logs[f"mae/{key}"] = per_stage_mae.get(key, 0.0)
            logs[f"lambda/{key}"] = lambdas[s]
            logs[f"sig/{key}"] = per_stage_sig.get(key, 0.0)
            if key in per_stage_sig_raw:
                logs[f"sig/raw/{key}"] = per_stage_sig_raw[key]
            logs[f"active/{key}"] = active[s]
            if fg_stages is not None:
                logs[f"fg_cov/{key}"] = float(fg_stages[key].float().mean())
        logs.update({f"frozen/{self._stage_key(s)}": float(self.stage_frozen(s))
                     for s in range(self.num_stages)})
        logs.update(stage_feature_diagnostics(E_ctx))
        if self.cfg.s4_cosine_level > 0.0:
            logs["sigreg/cos_gate"] = self._last_sigreg_cos_gate
            logs["sigreg/cos_ema"] = float(self._mae_cos_ema or 0.0)
            logs["sigreg/cos_latched"] = float(self._sigreg_cos_latched)
        return loss, logs

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, normalize: bool = True,
                         project: bool = False, use_latent: bool = True,
                         fg_px: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.dim() == 5:
            x = x[:, 0]
        bg1 = None
        if self.cfg.foreground_mask and fg_px is not None:
            if fg_px.dim() == 5:
                fg_px = fg_px[:, 0]
            fg_stages = build_fg_stages(fg_px, self.grids, self.cfg.fg_coverage)
            bg1 = ~fg_stages["s1"]
        feats = self.backbone(x, mask1=None, bg1=bg1)
        if use_latent:
            feats = _project_latent(feats, self.lateral)
        if normalize:
            feats = {k: stage_target_norm(v, self.cfg.target_norm) for k, v in feats.items()}
        if project:
            proj = {f"{k}_proj": v for k, v in feats.items()} if use_latent else {
                f"{k}_proj": self.lateral[k](v) for k, v in feats.items()}
            feats = {**feats, **proj}
        return feats
