"""Latent equivariance checks (spec §11)."""
import torch

from tomojepa.swinjepa_v2.config import default
from tomojepa.swinjepa_v2.data.augment import GeomParams, apply_latent_geom, make_two_views
from tomojepa.swinjepa_v2.models.model import SwinJEPA
from tomojepa.swinjepa_v2.train.instrument import equivariance_error


def test_g_latent_identity_rot90():
    x = torch.randn(1, 32, 32)
    geom = GeomParams(rot_k=1, crop_h=32, crop_w=32)
    feat = torch.randn(1, 8, 32, 32)
    out = apply_latent_geom(feat, geom)
    expected = torch.rot90(feat, 1, dims=(2, 3))
    assert torch.allclose(out, expected)


def test_equivariance_metric_finite():
    cfg = default()
    cfg.data.image_size = (224, 224)
    model = SwinJEPA(cfg)
    img = torch.randn(1, 224, 224)
    _, _, geom = make_two_views(img, cfg.data, torch.Generator().manual_seed(0))
    errs = equivariance_error(model, img, geom)
    assert all(v >= 0 for v in errs.values())
    assert len(errs) == 4
