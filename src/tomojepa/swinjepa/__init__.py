"""Swin multi-scale latent-JEPA self-supervised pre-training.

A Swin Transformer backbone is pretrained with multi-scale latent JEPA: a masked
"student" pass predicts the latent stage features that the *same weights* produce
from the full image under stop-gradient, at every pyramid stage. There is no EMA
teacher and no pixel decoder -- collapse is prevented by a per-stage SIGReg term
(reusing :class:`tomojepa.core.model.SIGReg`). Structurally this is data2vec-style
multi-scale prediction with SIGReg substituted for the momentum teacher.

After pretraining the backbone exposes its four stage feature maps for a
downstream multi-scale upsampler (ViT-Up family); all JEPA/SIGReg machinery is
training-only and absent at inference (:class:`SwinMSEncoder`).
"""
from .config import SwinMSJEPAConfig
from .backbone import SwinMultiScaleBackbone
from .mask import MultiScaleBlockMask
from .predictor import CrossScalePredictor
from .sigreg import StageSIGReg
from .model import SwinMSJEPA, SwinMSEncoder

__all__ = [
    "SwinMSJEPAConfig", "SwinMultiScaleBackbone", "MultiScaleBlockMask",
    "CrossScalePredictor", "StageSIGReg", "SwinMSJEPA", "SwinMSEncoder",
]
