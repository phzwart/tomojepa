"""Shared fixtures for the Swin multi-scale latent-JEPA tests."""
import torch

from tomojepa.swinjepa.config import SwinMSJEPAConfig


def small_cfg(**overrides) -> SwinMSJEPAConfig:
    base = dict(
        img_size=64,
        in_chans=1,
        drop_path_rate=0.0,
        pred_dim=64,
        pred_depth=2,
        pred_heads=4,
        coarse_mim_mode="cross_attn",
        predictor_enabled=True,
        dual_view=False,
        fusion_depth=1,
        fusion_heads=4,
        sigreg_n_dirs=(32, 32, 32, 32),
        sigreg_n_tokens_cap=512,
        sigreg_queue_len=0,
        lat_dims=(32, 32, 32, 32),
    )
    base.update(overrides)
    return SwinMSJEPAConfig(**base)


def integrated_cfg(**overrides) -> SwinMSJEPAConfig:
    base = dict(
        img_size=64,
        in_chans=1,
        drop_path_rate=0.0,
        coarse_mim_mode="integrated",
        dual_view=True,
        fusion_depth=1,
        fusion_heads=4,
        sigreg_n_dirs=(32, 32, 32, 32),
        sigreg_n_tokens_cap=512,
        sigreg_queue_len=0,
        lat_dims=(32, 32, 32, 32),
    )
    base.update(overrides)
    return SwinMSJEPAConfig(**base)


def dual_views(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, 2, C, H, W]`` for integrated-mode tests."""
    return torch.stack([x, x], dim=1)
