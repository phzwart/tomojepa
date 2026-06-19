"""Swin multi-scale backbone wrapper with stage exposure + mask-token injection.

Wraps a ``timm`` Swin Transformer (default ``swin_tiny_patch4_window7_224``) and
exposes two things the multi-scale latent-JEPA needs that ``features_only`` cannot
give:

1. **The four pyramid stage feature maps**, channels-first ``[B, C_s, h_s, w_s]``
   (the public contract for downstream ViT-Up consumers).
2. **Mask-token injection at the stage-1 token grid** -- a learnable ``[C1]``
   token is broadcast into the patch-embedding grid *before* the stages run, the
   SimMIM "keep-all" masking topology used by the student pass.

timm version footgun (handled here)
-----------------------------------
This repo pins ``timm>=1.0`` (installed 1.0.27). In that version Swin carries
tokens as **NHWC** (``output_fmt='NHWC'``) end to end, and ``PatchMerging`` lives
at the *start* of each stage (stage 0 is identity-downsample, stages 1-3 merge
2x2). Both facts are asserted at construction / per stage so a future timm bump
that flips the layout fails loudly instead of silently producing garbage. The
wrapper converts to channels-first at its boundary regardless.

timm's plain Swin has only a single final ``norm``; it does not carry a norm per
stage. We therefore own a ``LayerNorm`` per stage (applied channels-last on the
NHWC tokens) so every exposed feature is normalized-ish, matching the design's
"apply each stage's norm before exposing the feature". This is distinct from the
per-token target normalization in the loss (see :mod:`.losses`).
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

from .rope_attn import enable_rope_on_swin


def to_channels_first(x: torch.Tensor) -> torch.Tensor:
    """``[B, h, w, C]`` (NHWC) -> ``[B, C, h, w]`` (channels-first)."""
    return x.permute(0, 3, 1, 2).contiguous()


class SwinMultiScaleBackbone(nn.Module):
    """A Swin-T backbone exposing its four stage maps + stage-1 mask injection.

    Args:
        model_name: timm Swin model name (geometry must be Swin-style: patch 4,
            window 7, four stages).
        img_size: square input side. Set from the repo's tile size, not a
            hardcoded 224.
        in_chans: input channels (1 for grayscale tomography).
        pretrained: load timm pretrained weights (off for from-scratch SSL).
        drop_path_rate: stochastic depth for the Swin blocks.
        use_rope: replace timm relative-position bias with 2D RoPE on Q/K.
        rope_theta: RoPE frequency base (see :mod:`tomojepa.vitup.rope2d`).

    The public forward returns ``{"s1".."s4": [B, C_s, h_s, w_s]}``.
    """

    def __init__(self, model_name: str = "swin_tiny_patch4_window7_224",
                 img_size: int = 224, in_chans: int = 1, pretrained: bool = False,
                 drop_path_rate: float = 0.1, use_rope: bool = True,
                 rope_theta: float = 100.0):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.in_chans = in_chans
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        if use_rope and pretrained:
            raise ValueError(
                "use_rope=True is incompatible with pretrained=True: timm weights "
                "include a relative-position bias table, not RoPE parameters.")
        swin = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            img_size=img_size, in_chans=in_chans, drop_path_rate=drop_path_rate,
        )
        if getattr(swin, "output_fmt", "NHWC") != "NHWC":
            raise RuntimeError(
                f"Expected an NHWC Swin (timm>=1.0); got output_fmt="
                f"{getattr(swin, 'output_fmt', None)!r}. The mask-injection and "
                f"channels-first conversion in this wrapper assume NHWC tokens.")

        # Hold the timm submodules directly so we can inject the mask token
        # between patch_embed and the stages (features_only would not allow it).
        self.patch_embed = swin.patch_embed
        # Plain Swin has no positional dropout; keep the hook for layout parity
        # with the design pseudocode (Identity when absent).
        self.pos_drop = getattr(swin, "pos_drop", nn.Identity())
        self.stages = swin.layers
        if use_rope:
            enable_rope_on_swin(self.stages, rope_theta=rope_theta)

        self._num_stages = len(self.stages)
        if self._num_stages != 4:
            raise ValueError(f"Expected a 4-stage Swin pyramid, got {self._num_stages}.")

        # Per-stage channel dims, inferred from the built stages.
        self._out_chans: List[int] = [int(st.blocks[-1].dim) for st in self.stages]
        c1 = self._out_chans[0]

        # Stage-1 grid (h1, w1) from the patch-embed; kept as a tuple (not a
        # square scalar) so a later 3-D swap only touches geometry here.
        self.patch_grid = tuple(int(g) for g in self.patch_embed.grid_size)

        # Learnable [C1] mask token injected at the stage-1 token grid.
        self.mask_token = nn.Parameter(torch.zeros(c1))
        trunc_normal_(self.mask_token, std=0.02)
        # Per-stage learnable BG tokens: injected at s1 and re-applied after each
        # stage so holder positions stay fixed (attention cannot drift them).
        self.bg_stage_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(c)) for c in self._out_chans])
        for p in self.bg_stage_tokens:
            trunc_normal_(p, std=0.02)

        # One LayerNorm per stage (channels-last on the NHWC tokens).
        self.stage_norms = nn.ModuleList(
            [nn.LayerNorm(c) for c in self._out_chans])

    # -- discoverable geometry (so downstream wiring needs no hardcoding) -----
    @property
    def out_chans(self) -> List[int]:
        """Per-stage channel dims ``C = [96, 192, 384, 768]`` for Swin-T."""
        return list(self._out_chans)

    @property
    def stage_keys(self) -> List[str]:
        return [f"s{i + 1}" for i in range(self._num_stages)]

    @property
    def strides(self) -> List[int]:
        """Per-stage input strides ``[4, 8, 16, 32]`` for patch-4 Swin."""
        p = int(self.patch_embed.patch_size[0])
        return [p * (2 ** i) for i in range(self._num_stages)]

    def stage_grid(self, stage_idx: int) -> tuple:
        """Token grid ``(h_s, w_s)`` at stage ``stage_idx`` (0-based)."""
        h, w = self.patch_grid
        f = 2 ** stage_idx
        return (h // f, w // f)

    @property
    def bg_token(self) -> torch.Tensor:
        """Stage-1 BG token (alias for ``bg_stage_tokens[0]``; legacy ckpt key)."""
        return self.bg_stage_tokens[0]

    def _bg_stages_from_s1(self, bg1: torch.Tensor) -> Dict[str, torch.Tensor]:
        from tomojepa.core.augmentations import strict_bg_stages_from_s1
        grids = [self.stage_grid(s) for s in range(self._num_stages)]
        return strict_bg_stages_from_s1(bg1, grids)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Map legacy ``bg_token`` weights onto ``bg_stage_tokens.0``."""
        if ("bg_token" in state_dict
                and "bg_stage_tokens.0" not in state_dict):
            tok = state_dict.pop("bg_token")
            state_dict["bg_stage_tokens.0"] = tok
            for i in range(1, self._num_stages):
                key = f"bg_stage_tokens.{i}"
                if key not in state_dict:
                    state_dict[key] = tok.clone()
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor,
                mask1: Optional[torch.Tensor] = None,
                bg1: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Run the pyramid, optionally injecting bg/mask tokens at stage 1.

        Args:
            x: ``[B, C_in, H, W]`` input image.
            mask1: optional ``[B, h1, w1]`` bool, True where the stage-1 token is
                masked (replaced by ``mask_token``). Restricted to foreground
                when foreground masking is enabled.
            bg1: optional ``[B, h1, w1]`` bool, True where the stage-1 token is
                background (replaced by the stage-1 ``bg_stage_tokens`` entry).
                After each stage, BG positions are overwritten again so they stay
                fixed through the pyramid. The two masks must not overlap.

        Returns:
            ``{"s1".."s4": [B, C_s, h_s, w_s]}`` channels-first stage maps.
        """
        t = self.patch_embed(x)                              # [B, h1, w1, C1] (NHWC)
        if t.dim() != 4:
            raise RuntimeError(
                f"Expected NHWC patch-embed output [B,h,w,C], got shape {tuple(t.shape)}.")
        b, h1, w1, c1 = t.shape
        bg_stages = self._bg_stages_from_s1(bg1) if bg1 is not None else None
        if bg1 is not None:
            if bg1.shape != (b, h1, w1):
                raise ValueError(
                    f"bg1 shape {tuple(bg1.shape)} != stage-1 grid {(b, h1, w1)}.")
            tok = self.bg_stage_tokens[0].to(t.dtype)
            t = torch.where(bg1.unsqueeze(-1), tok, t)
        if mask1 is not None:
            if mask1.shape != (b, h1, w1):
                raise ValueError(
                    f"mask1 shape {tuple(mask1.shape)} != stage-1 grid {(b, h1, w1)}.")
            if bg1 is not None and (mask1 & bg1).any():
                raise ValueError("mask1 and bg1 must not overlap.")
            # NHWC layout makes the injection a clean boolean scatter over [C1].
            t = torch.where(mask1.unsqueeze(-1), self.mask_token.to(t.dtype), t)
        t = self.pos_drop(t)

        feats: Dict[str, torch.Tensor] = {}
        for s, stage in enumerate(self.stages):
            t = stage(t)                                     # downsamples between stages
            exp_h, exp_w = self.stage_grid(s)
            if t.shape[1:3] != (exp_h, exp_w) or t.shape[-1] != self._out_chans[s]:
                raise RuntimeError(
                    f"stage s{s + 1}: expected NHWC [B,{exp_h},{exp_w},"
                    f"{self._out_chans[s]}], got {tuple(t.shape)} -- timm Swin "
                    f"layout may have changed.")
            normed = self.stage_norms[s](t)
            if bg_stages is not None:
                bg = bg_stages[f"s{s + 1}"]
                tok = self.bg_stage_tokens[s].to(normed.dtype)
                normed = torch.where(bg.unsqueeze(-1), tok, normed)
            feats[f"s{s + 1}"] = to_channels_first(normed)
        return feats
