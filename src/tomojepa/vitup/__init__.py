"""ViT-Up: faithful, coordinate-conditioned implicit feature upsampling for ViTs.

Implements Wandel, Wang & Wang, *ViT-Up: Faithful Feature Upsampling for Vision
Transformers* (arXiv:2606.14024) on top of this repo's timm DINOv3 backbone.

Given a ViT backbone and a continuous image coordinate ``x_q in R^2``, ViT-Up
predicts the backbone feature at that coordinate from the backbone's
intermediate hidden states. Querying over a dense grid yields a high-resolution
feature map at arbitrary output resolution.
"""

from .config import ViTUpConfig
from .backbone_adapter import BackboneAdapter
from .model import ViTUp

__all__ = ["ViTUpConfig", "BackboneAdapter", "ViTUp"]
