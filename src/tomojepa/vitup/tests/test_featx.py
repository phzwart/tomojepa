"""FeatX test (acceptance criterion 3)."""
import torch
import torch.nn as nn

from tomojepa.vitup.featx import FeatX, SinusoidalPosEnc


def test_posenc_shape_and_zero():
    enc = SinusoidalPosEnc(dim=64, in_dim=2)
    out = enc(torch.zeros(3, 2))
    assert out.shape == (3, 64)
    # at Delta x = 0: sin -> 0, cos -> 1
    assert torch.allclose(out[:, :32], torch.zeros(3, 32), atol=1e-6)
    assert torch.allclose(out[:, 32:], torch.ones(3, 32), atol=1e-6)


def test_nearest_neighbor_selection():
    # C == D and identity LN / sub-token MLP + zero FiLM => output == nearest token
    C = D = 8
    fx = FeatX(backbone_dim=C, internal_dim=D)
    fx.ln = nn.Identity()
    fx.mlp_subtoken = nn.Identity()
    h, w = 4, 4
    # distinct feature per token: token (i,j) -> value (i*w + j)
    h_grid = torch.zeros(1, C, h, w)
    for i in range(h):
        for j in range(w):
            h_grid[0, :, i, j] = float(i * w + j)
    # queries near specific token centers
    coords = torch.tensor([[[0.6, 0.6], [3.4, 2.1], [1.5, 3.9]]])  # (y,x)
    out = fx(h_grid, coords, patch_size=16)
    expected_idx = [0 * w + 0, 3 * w + 2, 1 * w + 3]
    for q, idx in enumerate(expected_idx):
        assert torch.allclose(out[0, q], torch.full((D,), float(idx)), atol=1e-5)


def test_film_shapes():
    C, D = 384, 192
    fx = FeatX(backbone_dim=C, internal_dim=D)
    out = fx(torch.randn(2, C, 4, 4), torch.rand(2, 7, 2) * 4, patch_size=16)
    assert out.shape == (2, 7, D)
