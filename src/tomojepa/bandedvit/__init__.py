"""Distance-banded ViT encoder with injectable per-block attention masks."""

from .bvit import (
    BandConfig,
    BandManager,
    BandedViT,
    ViTConfig,
    sanity_check,
)
from .config import BandedJEPAConfig
from .model import BandedJEPA

__all__ = [
    "BandConfig",
    "BandManager",
    "BandedJEPA",
    "BandedJEPAConfig",
    "BandedViT",
    "ViTConfig",
    "sanity_check",
]
