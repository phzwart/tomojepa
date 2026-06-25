"""BandedViT training model: encoder + distance bands + SIGReg / SimMIM loss."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomojepa.core.augmentations import pool_fg_to_stage
from tomojepa.swinjepa.pyramid import gather_stage_tokens

from .bvit import BandConfig, BandManager, BandedViT, ViTConfig
from .config import BandedJEPAConfig
from .mask import PatchBlockMask
from .sigreg import BlockSIGReg


def _gather_masked_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``tokens`` [B, N, C], ``mask`` [B, N] bool -> [M, C]."""
    return tokens[mask]


def _gather_fg_tokens(tokens: torch.Tensor, fg: Optional[torch.Tensor]) -> torch.Tensor:
    """``tokens`` [B, N, C], optional ``fg`` [B, N] -> [M, C] foreground-only."""
    if fg is None:
        return tokens.reshape(-1, tokens.shape[-1])
    parts = [tokens[b][fg[b]] for b in range(tokens.shape[0]) if fg[b].any()]
    if not parts:
        return tokens.new_zeros(0, tokens.shape[-1])
    return torch.cat(parts, dim=0)


def _gather_sigreg_tokens(
    tokens: torch.Tensor,
    fg: Optional[torch.Tensor],
    grid: int,
    n_per_slice: int,
    min_grid_dist: int,
) -> torch.Tensor:
    """FG patch tokens for SIGReg, optionally min-distance subsampled per slice."""
    b, _, c = tokens.shape
    if min_grid_dist <= 0:
        fg_flat = fg.reshape(b, -1) if fg is not None else None
        return _gather_fg_tokens(tokens, fg_flat)
    feat = tokens.reshape(b, grid, grid, c).permute(0, 3, 1, 2)
    fg_stage = fg.reshape(b, grid, grid) if fg is not None else None
    grouped, valid = gather_stage_tokens(
        feat, fg_stage, n_per_slice=n_per_slice, min_grid_dist=min_grid_dist,
        return_valid=True,
    )
    rows = [grouped[i] for i in range(b) if valid[i]]
    if not rows:
        return tokens.new_zeros(0, c)
    return torch.cat(rows, dim=0)


def extract_fg_masks(batch) -> Optional[torch.Tensor]:
    """Unpack FG masks from a ``TomographyDataset`` batch."""
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        _, fg = batch
        if isinstance(fg, torch.Tensor) and fg.numel() > 0:
            return fg
    return None


class BandedJEPA(nn.Module):
    """Training wrapper around :class:`BandedViT` with band injection and losses."""

    has_ema = False

    def __init__(self, cfg: BandedJEPAConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.fg_mode not in ("std", "circle"):
            raise ValueError(f"fg_mode must be 'std' or 'circle', got {cfg.fg_mode!r}")
        vit_cfg = ViTConfig(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            num_register_tokens=cfg.num_register_tokens,
            use_cls_token=cfg.use_cls_token,
            rope_theta=cfg.rope_theta,
        )
        self.encoder = BandedViT(vit_cfg)
        self.band_mgr = BandManager(
            self.encoder,
            K=cfg.band_K,
            band_cfg=BandConfig(
                keep_self=cfg.band_keep_self,
                sample_mode=cfg.band_sample_mode,
                weights=cfg.band_weights,
            ),
            off_steps=cfg.band_m0,
            on_steps=cfg.band_m1,
        )
        self.sigreg_blocks: List[int] = list(cfg.resolved_sigreg_blocks())
        self.sigreg = nn.ModuleDict({
            str(b): BlockSIGReg(
                dim=cfg.embed_dim,
                n_dirs=cfg.sigreg_n_dirs,
                knots=cfg.sigreg_knots,
                t_max=cfg.sigreg_t_max,
                w_mean=cfg.sigreg_w_mean,
                n_tokens_cap=cfg.sigreg_token_cap,
                queue_len=cfg.sigreg_queue_len,
            )
            for b in self.sigreg_blocks
        })
        depth = cfg.depth
        self.pred_block_idx = (cfg.pred_block if cfg.pred_block >= 0 else depth + cfg.pred_block) % depth

        if cfg.foreground_mask:
            self.bg_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.bg_token, std=0.02)
        else:
            self.bg_token = None

        if cfg.pred_enabled:
            grid = cfg.img_size // cfg.patch_size
            self.mask_gen = PatchBlockMask(
                grid=grid,
                mask_ratio=cfg.mask_ratio,
                mask_mode=cfg.mask_mode,
                num_blocks=cfg.mask_num_blocks,
                block_scale_range=cfg.block_scale_range,
            )
            self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)

    @property
    def num_prefix(self) -> int:
        return self.encoder.num_prefix

    @property
    def patch_grid(self) -> int:
        return self.cfg.img_size // self.cfg.patch_size

    def _patch_taps(self, taps: List[torch.Tensor]) -> List[torch.Tensor]:
        p = self.num_prefix
        return [t[:, p:, :] for t in taps]

    def _split_dual_fg(
        self, fg_px: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if fg_px is None:
            return None, None
        if fg_px.dim() == 5:
            if fg_px.shape[1] < 2:
                raise ValueError(
                    f"dual_view FG expects [B, V, 1, H, W] with V >= 2, got V={fg_px.shape[1]}"
                )
            return fg_px[:, 0], fg_px[:, 1]
        return fg_px, fg_px

    def _fg_patch_grid(self, fg_px: torch.Tensor) -> torch.Tensor:
        """Pixel FG mask ``[B,1,H,W]`` -> patch FG ``[B, grid, grid]`` bool."""
        grid = self.patch_grid
        return pool_fg_to_stage(fg_px.float(), (grid, grid), self.cfg.fg_coverage)

    def _fg_from_px(
        self, fg_px: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.foreground_mask or fg_px is None:
            return None, None
        if fg_px.dim() == 5:
            fg_px = fg_px[:, 0]
        fg = self._fg_patch_grid(fg_px)
        return fg, ~fg

    def _encode(
        self,
        x: torch.Tensor,
        *,
        mask_flat: Optional[torch.Tensor] = None,
        bg_flat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bg_token = self.bg_token if bg_flat is not None else None
        return self.encoder(
            x,
            mask=mask_flat,
            mask_token=self.mask_token if mask_flat is not None else None,
            bg=bg_flat,
            bg_token=bg_token,
        )

    def _sigreg_loss(
        self,
        taps: List[torch.Tensor],
        fg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = taps[0].new_zeros(())
        logs: Dict[str, float] = {}
        fg_flat = fg.reshape(fg.shape[0], -1) if fg is not None else None
        grid = self.patch_grid
        for b in self.sigreg_blocks:
            beta = self.cfg.beta_for_block(b)
            if beta <= 0.0:
                continue
            patch_tok = self._patch_taps(taps)[b]
            tok = _gather_sigreg_tokens(
                patch_tok, fg, grid,
                n_per_slice=self.cfg.sigreg_n_tokens_per_slice,
                min_grid_dist=self.cfg.sigreg_min_token_dist,
            )
            stat = self.sigreg[str(b)](tok)
            total = total + beta * stat
            logs[f"sig/b{b}"] = float(stat.detach())
        logs["l_sig"] = float(total.detach())
        if fg is not None:
            logs["fg_cov"] = float(fg.float().mean())
        return total, logs

    def _pred_loss(
        self,
        ctx_tap: torch.Tensor,
        tgt_tap: torch.Tensor,
        mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        pred = _gather_masked_tokens(ctx_tap, mask_flat)
        tgt = _gather_masked_tokens(tgt_tap, mask_flat)
        if self.cfg.pred_loss == "mse":
            return F.mse_loss(pred, tgt)
        return F.smooth_l1_loss(pred, tgt, beta=self.cfg.smooth_l1_beta)

    def compute_loss(
        self,
        x: torch.Tensor,
        step: int = 0,
        total_steps: int = 1,
        epoch: int = 0,
        fg_px: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del epoch, total_steps
        self.band_mgr.maybe_resample(step)
        logs: Dict[str, float] = {"band_on": float(self.band_mgr.use_bands(step))}

        if self.cfg.pred_enabled:
            if x.ndim != 5 or x.shape[1] < 2:
                raise ValueError(
                    "pred_enabled requires input [B, V>=2, C, H, W] (dual views)"
                )
            ctx = x[:, 0]
            tgt = x[:, 1]
            fg_ctx, fg_tgt = self._split_dual_fg(fg_px)
            fg, bg = self._fg_from_px(fg_ctx)
            _, bg_tgt = self._fg_from_px(fg_tgt)

            mask = self.mask_gen.sample(ctx.shape[0], ctx.device, fg=fg)
            mask_flat = self.mask_gen.flat(mask)
            if fg is not None and (mask_flat & bg.reshape(mask_flat.shape[0], -1)).any():
                raise AssertionError("masked positions must lie inside the FOV")

            with torch.no_grad():
                _, taps_t = self._encode(tgt, bg_flat=bg_tgt.reshape(bg_tgt.shape[0], -1) if bg_tgt is not None else None)
                tgt_tap = self._patch_taps(taps_t)[self.pred_block_idx]

            _, taps_c = self._encode(
                ctx,
                mask_flat=mask_flat,
                bg_flat=bg.reshape(bg.shape[0], -1) if bg is not None else None,
            )
            ctx_tap = self._patch_taps(taps_c)[self.pred_block_idx]

            l_pred = self._pred_loss(ctx_tap, tgt_tap, mask_flat)
            l_sig, sig_logs = self._sigreg_loss(taps_c, fg=fg)
            loss = l_pred + l_sig
            logs.update(sig_logs)
            logs["l_pred"] = float(l_pred.detach())
            logs["total"] = float(loss.detach())
            return loss, logs

        if x.ndim == 5:
            x = x[:, 0]
        fg, bg = self._fg_from_px(fg_px)
        _, taps = self._encode(
            x,
            bg_flat=bg.reshape(bg.shape[0], -1) if bg is not None else None,
        )
        loss, logs = self._sigreg_loss(taps, fg=fg)
        logs["l_pred"] = 0.0
        logs["total"] = float(loss.detach())
        return loss, logs

    @torch.no_grad()
    def extract_patch_tokens(self, x: torch.Tensor, block_idx: int = -1) -> torch.Tensor:
        """Last (or chosen) block patch tokens for PCA probes. ``x``: [B, C, H, W]."""
        if x.ndim == 5:
            x = x[:, 0]
        self.band_mgr.clear()
        _, taps = self.encoder(x)
        idx = (block_idx if block_idx >= 0 else self.cfg.depth + block_idx) % self.cfg.depth
        return self._patch_taps(taps)[idx]
