"""BandManager distance-band assignment strategies."""
from __future__ import annotations

import pytest
import torch

from tomojepa.bandedvit.bvit import BandConfig, BandManager, BandedViT, ViTConfig, weighted_band_counts


def _manager(
    depth: int = 12,
    sample_mode: str = "balanced",
    seed: int = 0,
    weights: tuple[float, ...] = (),
) -> BandManager:
    cfg = ViTConfig(
        img_size=64, patch_size=16, in_chans=1, embed_dim=384, depth=depth,
        num_heads=6, use_cls_token=True, num_register_tokens=0,
    )
    model = BandedViT(cfg).eval()
    return BandManager(
        model, K=1,
        band_cfg=BandConfig(sample_mode=sample_mode, seed=seed, weights=weights),
    )


def _indices(mgr: BandManager, n: int = 64) -> list[list[int]]:
    return [mgr._assign_band_indices(len(mgr.attns)) for _ in range(n)]


@pytest.mark.parametrize("mode", ["balanced", "cyclic", "balanced_no_adjacent"])
def test_diverse_modes_include_all_bands(mode: str):
    mgr = _manager(depth=12, sample_mode=mode, seed=1)
    for idx in _indices(mgr):
        assert set(idx) == {0, 1, 2}


@pytest.mark.parametrize("mode", ["balanced", "cyclic", "balanced_no_adjacent"])
def test_diverse_modes_never_all_near(mode: str):
    mgr = _manager(depth=12, sample_mode=mode, seed=2)
    for idx in _indices(mgr):
        assert not all(j == 0 for j in idx)


def test_independent_allows_unequal_band_counts():
    mgr = _manager(depth=12, sample_mode="independent", seed=0)
    counts = [sorted(idx).count(0) for idx in _indices(mgr, n=256)]
    assert any(c != 4 for c in counts)


def test_cyclic_adjacent_blocks_differ():
    mgr = _manager(depth=12, sample_mode="cyclic", seed=3)
    for idx in _indices(mgr):
        assert all(idx[i] != idx[i + 1] for i in range(len(idx) - 1))


def test_balanced_no_adjacent_adjacent_blocks_differ():
    mgr = _manager(depth=12, sample_mode="balanced_no_adjacent", seed=4)
    for idx in _indices(mgr):
        assert all(idx[i] != idx[i + 1] for i in range(len(idx) - 1))


def test_balanced_equal_counts():
    mgr = _manager(depth=12, sample_mode="balanced", seed=5)
    for idx in _indices(mgr):
        assert sorted(idx).count(0) == 4
        assert sorted(idx).count(1) == 4
        assert sorted(idx).count(2) == 4


def test_balanced_weighted_counts():
    weights = (0.05, 0.50, 0.45)
    assert weighted_band_counts(12, weights) == [1, 6, 5]
    mgr = _manager(depth=12, sample_mode="balanced", seed=9, weights=weights)
    for idx in _indices(mgr):
        assert sorted(idx).count(0) == 1
        assert sorted(idx).count(1) == 6
        assert sorted(idx).count(2) == 5


def test_small_depth_uses_distinct_bands():
    mgr = _manager(depth=2, sample_mode="balanced", seed=6)
    for idx in _indices(mgr):
        assert len(set(idx)) == 2


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="sample_mode"):
        _manager(sample_mode="bogus")


def test_maybe_resample_produces_finite_masks():
    cfg = ViTConfig(
        img_size=64, patch_size=16, in_chans=1, embed_dim=384, depth=4,
        num_heads=6, use_cls_token=True,
    )
    vit = BandedViT(cfg).eval()
    mgr = BandManager(
        vit, K=1, band_cfg=BandConfig(sample_mode="balanced", seed=8),
    )
    mgr.maybe_resample(0)
    x = torch.randn(1, 1, 64, 64)
    rep, _ = vit(x)
    assert torch.isfinite(rep).all()
    assert len(mgr._biases) == 4
    t = vit.num_prefix + vit.num_patches
    for b in mgr._biases:
        assert b is not None
        assert b.shape == (t, t)


def test_band_schedule_off_then_on():
    mgr = _manager(depth=4)
    mgr.off_steps = 5
    mgr.on_steps = 3
    for step in range(5):
        assert not mgr.use_bands(step)
        mgr.maybe_resample(step)
        assert all(b is None for b in mgr._biases)
    for step in range(5, 8):
        assert mgr.use_bands(step)
        mgr.maybe_resample(step)
        assert all(b is not None for b in mgr._biases)
    # next cycle
    assert not mgr.use_bands(8)
    mgr.maybe_resample(8)
    assert all(b is None for b in mgr._biases)
    assert mgr.use_bands(13)
    mgr.maybe_resample(13)
    assert all(b is not None for b in mgr._biases)


def test_band_schedule_disabled_when_m0_or_m1_zero():
    mgr = _manager(depth=4)
    mgr.off_steps = 10
    mgr.on_steps = 0
    assert mgr.use_bands(0)
    mgr.maybe_resample(0)
    assert all(b is not None for b in mgr._biases)
