import pytest
import torch

from tomojepa.swinjepa_v2.config import default


@pytest.fixture
def small_cfg():
    cfg = default()
    cfg.data.image_size = (128, 128)
    cfg.band.band_dim = 64
    cfg.predictor.embed_dim = 64
    cfg.predictor.depth_per_band = 1
    cfg.sigreg.num_slices = 32
    return cfg
