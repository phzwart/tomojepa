"""Band and predictor shape tests for 2D and 3D (spec §11)."""
import torch
import torch.nn as nn

from tomojepa.swinjepa_v2.config import BandCfg, HLJEPAConfig, PredictorCfg, default
from tomojepa.swinjepa_v2.models.bands import BandFormer
from tomojepa.swinjepa_v2.models.model import SwinJEPA
from tomojepa.swinjepa_v2.models.predictor import TopDownPredictor
from tomojepa.swinjepa_v2.nd import conv_nd


def test_band_shapes_2d():
    cfg = default()
    cfg.data.image_size = (224, 224)
    former = BandFormer([96, 192, 384, 768], cfg.band, ndim=2)
    stages = [
        torch.randn(2, 96, 56, 56),
        torch.randn(2, 192, 28, 28),
        torch.randn(2, 384, 14, 14),
        torch.randn(2, 768, 7, 7),
    ]
    bands = former(stages)
    assert len(bands) == 4
    for b, s in zip(bands, stages):
        assert b.shape[0] == 2
        assert b.shape[2:] == s.shape[2:]
        assert b.shape[1] == cfg.band.band_dim


def test_predictor_shapes_2d():
    cfg = default()
    pred = TopDownPredictor(cfg.band.band_dim, cfg.predictor, ndim=2)
    bands = [torch.randn(2, cfg.band.band_dim, 56 // (2 ** i), 56 // (2 ** i)) for i in range(4)]
    masks = [torch.rand(b.shape[0], *b.shape[2:]) > 0.5 for b in bands]
    out = pred(bands, masks)
    assert len(out) == 4
    for p, b in zip(out, bands):
        assert p.shape == b.shape


def test_swinjepa_forward_train_shapes():
    cfg = default()
    cfg.data.image_size = (224, 224)
    cfg.backbone.model_name = "swin_tiny_patch4_window7_224"
    model = SwinJEPA(cfg)
    x = torch.randn(2, 1, 224, 224)
    from tomojepa.swinjepa_v2.data.masking import masks_for_stages
    from tomojepa.swinjepa_v2.config import MaskCfg
    g = torch.Generator().manual_seed(1)
    masks = masks_for_stages(224, 4, MaskCfg(), g)
    masks_b = [m.unsqueeze(0).expand(2, -1, -1) for m in masks]
    pred, tgt, ctx = model.forward_train(x, x, masks_b)
    assert len(pred) == len(tgt) == len(ctx) == 4
    for p, t in zip(pred, tgt):
        assert p.shape == t.shape


def test_band_shapes_3d_stub():
    former = BandFormer([32, 64], BandCfg(band_dim=16), ndim=3)
    stages = [torch.randn(1, 32, 4, 4, 4), torch.randn(1, 64, 2, 2, 2)]
    bands = former(stages)
    assert bands[0].shape[2:] == (4, 4, 4)
    assert bands[1].shape[2:] == (2, 2, 2)
