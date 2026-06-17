"""Cross-window attention + block tests (acceptance criteria 4, plus attn parity)."""
import torch

from tomojepa.vitup.attention import CrossWindowAttention
from tomojepa.vitup.block import ViTUpBlock


def test_gather_matches_dense_reference():
    torch.manual_seed(0)
    B, Q, D, C, h, w = 2, 9, 32, 16, 6, 6   # head_dim = 32/4 = 8 (divisible by 4)
    attn = CrossWindowAttention(dim=D, num_heads=4, window=3, q_dim=D, kv_dim=C).eval()
    q_in = torch.randn(B, Q, D)
    kv = torch.randn(B, C, h, w)
    coords = torch.rand(B, Q, 2) * torch.tensor([float(h), float(w)])
    with torch.no_grad():
        fast = attn(q_in, kv, coords)
        ref = attn.forward_dense(q_in, kv, coords)
    assert fast.shape == (B, Q, D)
    assert torch.allclose(fast, ref, atol=1e-5)


def test_block_forward_shape_and_finite():
    torch.manual_seed(0)
    B, Q, D, C, h, w = 2, 12, 384, 384, 4, 4
    blk = ViTUpBlock(internal_dim=D, backbone_dim=C, num_heads=6, window=3)
    q_prev = torch.randn(B, Q, D)
    h_grid = torch.randn(B, C, h, w)
    coords = torch.rand(B, Q, 2) * 4.0
    out = blk(q_prev, h_grid, coords, patch_size=16)
    assert out.shape == (B, Q, D)
    assert torch.isfinite(out).all()
