"""Shared fixtures/helpers for ViT-Up tests.

Backbones are built with random weights (``pretrained=False``) at small input
sizes so the suite runs on CPU without any downloads. Architecture defaults are
shrunk (fewer blocks, tiny high-res cache, small window) for speed -- the math
under test is independent of these sizes.
"""
import pytest
import torch

from tomojepa.vitup.config import ViTUpConfig
from tomojepa.vitup.backbone_adapter import build_backbone, BackboneAdapter


def small_cfg(**overrides) -> ViTUpConfig:
    base = dict(
        input_channels=1,
        backbone_img_size=64,
        num_blocks=2,
        layer_indices=(2, 4),
        internal_dim=0,
        query_embed_grid=8,
        attention_window=3,
        num_heads=6,
        featx_posenc_dim=64,
        query_chunk_size=16,
        teacher_resolutions=(32, 64),
        student_canvas=64,
        student_scale=(0.5, 1.0),
        query_grid=4,
    )
    base.update(overrides)
    return ViTUpConfig(**base)


@pytest.fixture(scope="session")
def adapter():
    torch.manual_seed(0)
    bb = build_backbone("vit_small_patch16_dinov3", in_chans=1, img_size=64)
    return BackboneAdapter(bb).eval()
