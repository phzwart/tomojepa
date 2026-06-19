"""Acceptance checks 1 (shapes) and mask expansion exactness."""
import pytest
import torch

from tomojepa.swinjepa.backbone import SwinMultiScaleBackbone
from tomojepa.swinjepa.mask import MultiScaleBlockMask, assert_mask_consistency


def test_backbone_rope_forward():
    """RoPE backbone matches default stage geometry."""
    torch.manual_seed(0)
    bb = SwinMultiScaleBackbone(img_size=64, in_chans=1, use_rope=True).eval()
    assert bb.use_rope
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        feats = bb(x, mask1=None)
    for s, key in enumerate(bb.stage_keys):
        h, w = bb.stage_grid(s)
        assert feats[key].shape == (2, bb.out_chans[s], h, w)


def test_backbone_stage_shapes():
    torch.manual_seed(0)
    for img in (64, 96):                       # both multiples of 32 (clean s4 grid)
        bb = SwinMultiScaleBackbone(img_size=img, in_chans=1).eval()
        assert bb.out_chans == [96, 192, 384, 768]
        assert bb.strides == [4, 8, 16, 32]
        x = torch.randn(2, 1, img, img)
        with torch.no_grad():
            feats = bb(x, mask1=None)
        for s, key in enumerate(bb.stage_keys):
            h, w = bb.stage_grid(s)
            assert feats[key].shape == (2, bb.out_chans[s], h, w)


def test_backbone_mask_injection_shapes():
    torch.manual_seed(0)
    bb = SwinMultiScaleBackbone(img_size=64, in_chans=1).eval()
    h1, w1 = bb.stage_grid(0)
    mask1 = torch.zeros(2, h1, w1, dtype=torch.bool)
    mask1[:, :8, :8] = True
    with torch.no_grad():
        feats = bb(torch.randn(2, 1, 64, 64), mask1=mask1)
    assert feats["s4"].shape[-2:] == bb.stage_grid(3)


def test_mask_expansion_exact():
    """A stage-4 cell maps to an 8x8 stage-1 block (2^(S-1) on a side)."""
    torch.manual_seed(0)
    gen = MultiScaleBlockMask(grid4=(2, 2), num_stages=4, mask_ratio=0.5)
    mask = gen.generate(4)
    assert_mask_consistency(mask, num_stages=4)
    # grids double each finer stage; s1 is 8x the s4 grid.
    assert mask["s1"].shape[-2:] == (16, 16)
    assert mask["s4"].shape[-2:] == (2, 2)
    # every s4 cell expands to an exact 8x8 s1 block of identical value.
    m4 = mask["s4"]
    m1 = mask["s1"]
    for i in range(2):
        for j in range(2):
            block = m1[:, i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            assert (block == m4[:, i:i + 1, j:j + 1]).all()


def test_mask_fixed_count_and_modes():
    """Both modes mask exactly k s4 cells per sample (rectangular batching)."""
    for mode in ("random_cell", "block"):
        gen = MultiScaleBlockMask(grid4=(4, 4), num_stages=4, mask_ratio=0.55,
                                  mask_mode=mode)
        mask = gen.generate(5)
        k = gen.k
        counts = mask["s4"].flatten(1).sum(1)
        assert (counts == k).all()
        assert 1 <= k <= gen.n_cells - 1     # at least one masked and one visible


def test_backbone_bg_token_injection():
    """bg_token changes the forward when bg1 is set; must not overlap mask1."""
    torch.manual_seed(0)
    bb = SwinMultiScaleBackbone(img_size=64, in_chans=1).eval()
    h1, w1 = bb.stage_grid(0)
    x = torch.randn(2, 1, 64, 64)
    bg1 = torch.zeros(2, h1, w1, dtype=torch.bool)
    bg1[:, :4, :] = True
    mask1 = torch.zeros(2, h1, w1, dtype=torch.bool)
    mask1[:, 8:, 8:] = True
    with torch.no_grad():
        base = bb(x, mask1=None, bg1=None)
        inj = bb(x, mask1=None, bg1=bg1)
    assert not torch.allclose(base["s1"], inj["s1"])
    with pytest.raises(ValueError):
        bb(x, mask1=bg1.clone(), bg1=bg1)


def test_bg_tokens_fixed_at_every_stage():
    """BG positions equal the stage bg token after the full pyramid."""
    torch.manual_seed(0)
    bb = SwinMultiScaleBackbone(img_size=64, in_chans=1).eval()
    h1, w1 = bb.stage_grid(0)
    bg1 = torch.zeros(1, h1, w1, dtype=torch.bool)
    bg1[:, :6, :] = True
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        feats = bb(x, bg1=bg1)
    bg_stages = bb._bg_stages_from_s1(bg1)
    for s, key in enumerate(bb.stage_keys):
        bg = bg_stages[key]
        if not bg.any():
            continue
        tok = bb.bg_stage_tokens[s]
        got = feats[key].permute(0, 2, 3, 1)[bg]
        assert torch.allclose(got, tok.expand_as(got))


def test_mask_fov_restricted():
    """Masked s4 cells lie inside fg_s4; counts match k_eff across the batch."""
    torch.manual_seed(0)
    gen = MultiScaleBlockMask(grid4=(4, 4), num_stages=4, mask_ratio=0.55)
    fg_s4 = torch.zeros(5, 4, 4, dtype=torch.bool)
    fg_s4[:, 1:3, 1:3] = True                          # 2x2 FG patch per sample
    mask = gen.generate(5, fg_s4=fg_s4)
    assert_mask_consistency(mask, num_stages=4)
    k_eff = gen._effective_k(fg_s4)
    counts = mask["s4"].flatten(1).sum(1)
    assert (counts == k_eff).all()
    assert not (mask["s4"] & ~fg_s4).any()
