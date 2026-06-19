"""Shared fixtures for the Swin multi-scale latent-JEPA tests.

Everything runs on CPU with random weights at a small ``img_size`` so the suite
needs no GPU and no downloads. ``img_size=64`` gives Swin token grids
``16/8/4/2`` -- the stage-4 grid is ``2x2`` (4 cells), enough for a non-trivial
masked/visible split (the math under test is independent of the absolute size).
"""
from tomojepa.swinjepa.config import SwinMSJEPAConfig


def small_cfg(**overrides) -> SwinMSJEPAConfig:
    base = dict(
        img_size=64,
        in_chans=1,
        drop_path_rate=0.0,
        pred_dim=64,
        pred_depth=2,
        pred_heads=4,
        sigreg_n_dirs=(32, 32, 32, 32),
        sigreg_n_tokens_cap=512,
        sigreg_queue_len=0,
        lat_dims=(32, 32, 32, 32),
    )
    base.update(overrides)
    return SwinMSJEPAConfig(**base)
