"""Augmentation YAML config and schedule."""
from pathlib import Path

import pytest

from tomojepa.core.aug_config import (
    AugmentationConfig,
    AugmentationSchedule,
    augmentation_config_from_dict,
    merge_augmentation_cli,
)
from tomojepa.swinjepa.schedule import load_job_yaml

_REPO = Path(__file__).resolve().parents[4]
_B1 = _REPO / "configs/swinjepa/schedules/isolate_c4_sigreg_b1.yaml"


def test_augmentation_config_from_dict():
    cfg = augmentation_config_from_dict({
        "variant": "tomo2",
        "crop_mode": "resize",
        "resize_jitter": [0.9, 1.1],
        "random_rotate_deg": 0,
        "blur": {"p": 0.3, "kernel_size": 5, "sigma": [0.2, 0.8]},
    })
    assert cfg.crop_mode == "resize"
    assert cfg.resize_jitter_scale() == (0.9, 1.1)
    assert cfg.blur_p == 0.3
    assert cfg.blur_kernel_size == 5


def test_rotate_block_yaml():
    cfg = augmentation_config_from_dict({
        "rotate": {"p": 0.5, "deg": 90},
        "hflip_p": 0.3,
    })
    assert cfg.rotate_p == 0.5
    assert cfg.random_rotate_deg == 90.0
    assert cfg.hflip_p == 0.3


def test_load_job_yaml_includes_augmentations():
    sched, aug_cfg, aug_sched, meta = load_job_yaml(_B1)
    assert sched is not None
    assert aug_cfg.variant == "tomo2"
    assert aug_cfg.random_rotate_deg == 0
    assert aug_cfg.resize_jitter_scale() == (0.9, 1.1)
    assert aug_sched is None
    assert meta["name"] == "isolate_c4_sigreg_b1"


def test_augmentation_schedule_interp():
    sched = AugmentationSchedule.from_dict({
        "schedule": {
            "blur_p": [
                {"progress": 0.0, "value": 0.0},
                {"progress": 1.0, "value": 1.0},
            ],
            "resize_jitter": [
                {"progress": 0.0, "value": [1.0, 1.0]},
                {"progress": 1.0, "value": [0.8, 1.2]},
            ],
        },
    })
    base = AugmentationConfig()
    mid = sched.at(base, 50, 100)
    assert mid.blur_p == pytest.approx(0.5)
    assert mid.resize_jitter == pytest.approx((0.9, 1.1))


def test_cli_override_merge():
    cfg = AugmentationConfig(random_rotate_deg=0, resize_jitter=(0.9, 1.1))
    ns = type("NS", (), {
        "augment": "tomo",
        "crop_mode": "resize",
        "global_views": 1,
        "local_views": 0,
        "global_scale": (0.4, 1.0),
        "local_scale": (0.1, 0.4),
        "random_rotate_deg": 45.0,
        "resize_jitter": (0.0, 0.0),
    })()
    merged = merge_augmentation_cli(cfg, ns, argv=[
        "train.py", "--random_rotate_deg", "45", "--resize_jitter", "0", "0"])
    assert merged.variant == "tomo2"
    assert merged.random_rotate_deg == 45.0
    assert merged.resize_jitter_scale() is None
