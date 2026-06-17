"""Shared library: the ViT backbone, dataset, and augmentations.

These modules carry no training-loop or CLI logic so every subsystem
(``ssl``, ``vitup``, ``patchdb``, ``viz``) can depend on them without pulling in
heavy optional dependencies.
"""

from .model import (
    DINOv3ViTEncoder,
    SIGReg,
    MaskedLatentPredictor,
    encode_masked,
    foreground_tokens,
    lejepa_projections,
    masked_mean,
)
from .dataset import TomographyDataset
from .augmentations import get_augmentations

__all__ = [
    "DINOv3ViTEncoder",
    "SIGReg",
    "MaskedLatentPredictor",
    "encode_masked",
    "foreground_tokens",
    "lejepa_projections",
    "masked_mean",
    "TomographyDataset",
    "get_augmentations",
]
