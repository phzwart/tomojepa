"""Mask pooling correctness (spec §11)."""
import torch

from tomojepa.swinjepa_v2.config import MaskCfg
from tomojepa.swinjepa_v2.data.masking import masks_for_stages, pool_mask, sample_mask


def test_pool_fraction_threshold():
    fine = torch.zeros(8, 8, dtype=torch.bool)
    fine[:4, :] = True
    cfg = MaskCfg(pool_thresh=0.5)
    masks = pool_mask(fine, 2, cfg, ndim=2)
    assert masks[0].shape == (8, 8)
    assert masks[1].shape == (4, 4)
    assert masks[1][:2].all()
    assert not masks[1][2:].any()


def test_pool_any_threshold():
    fine = torch.zeros(4, 4, dtype=torch.bool)
    fine[0, 0] = True
    cfg = MaskCfg(pool_thresh=0.0)
    masks = pool_mask(fine, 2, cfg, ndim=2)
    assert masks[1][0, 0]


def test_pool_all_threshold():
    fine = torch.zeros(4, 4, dtype=torch.bool)
    fine[0, :2] = True
    cfg = MaskCfg(pool_thresh=1.0)
    masks = pool_mask(fine, 2, cfg, ndim=2)
    assert not masks[1][0, 0]


def test_masks_for_stages_consistent():
    cfg = MaskCfg(mask_ratio=0.5, num_blocks=2)
    g = torch.Generator().manual_seed(0)
    masks = masks_for_stages(224, 4, cfg, g, ndim=2, num_stages=4)
    assert len(masks) == 4
    assert masks[0].shape == (56, 56)
    assert masks[-1].shape == (7, 7)


def test_hand_constructed_roundtrip():
    fine = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=torch.bool)
    cfg = MaskCfg(pool_thresh=0.5)
    coarse = pool_mask(fine, 2, cfg, ndim=2)[1]
    assert coarse.shape == (2, 2)
    assert coarse[0, 0] and coarse[1, 1]
