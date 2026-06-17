"""Model tests: chunk-invariance (criterion 6) and resolution sweep (criterion 8)."""
import torch

from tomojepa.vitup.model import ViTUp
from .conftest import small_cfg


def test_chunk_invariance(adapter):
    torch.manual_seed(0)
    cfg = small_cfg(query_chunk_size=7)
    model = ViTUp(adapter, cfg).eval()
    img = torch.randn(1, 1, 64, 64)
    ctx = model.encode_image(img)
    Q = 50
    coords = torch.rand(1, Q, 2) * 4.0
    with torch.no_grad():
        full = model.query(ctx, coords, stages="all", chunk_size=Q + 1)
        chunked = model.query(ctx, coords, stages="all", chunk_size=7)
    assert len(full) == len(chunked) == cfg.num_blocks + 1
    for a, b in zip(full, chunked):
        assert torch.allclose(a, b, atol=1e-5)


def test_resolution_sweep(adapter):
    torch.manual_seed(0)
    cfg = small_cfg()
    model = ViTUp(adapter, cfg).eval()
    img = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        for out in (4, 8, 16):
            up = model.upsample(img, out, out, chunk_size=32)
            assert up.shape == (1, out, out, adapter.C)
            assert torch.isfinite(up).all()
