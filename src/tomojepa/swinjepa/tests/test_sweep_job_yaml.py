"""Sweep job YAML, run overrides, and generator smoke tests."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from tomojepa.core.aug_config import augmentation_config_from_dict
from tomojepa.core.augmentations import _geom_ops
from tomojepa.swinjepa.schedule import (
    apply_job_run_overrides,
    load_job_yaml,
    parse_run_block,
)

_REPO = Path(__file__).resolve().parents[4]
_B1 = _REPO / "configs/swinjepa/schedules/isolate_c4_sigreg_b1.yaml"
_SWEEP_GEN = _REPO / "configs/swinjepa/sweeps/c4_random/generate.py"


def test_flip_config_parsed():
    cfg = augmentation_config_from_dict({"hflip_p": 0.5, "vflip_p": 0.25})
    assert cfg.hflip_p == 0.5
    assert cfg.vflip_p == 0.25
    ops = _geom_ops(cfg, img_size=64, scale_band="global")
    op_names = {type(o).__name__ for o in ops}
    assert "RandomHorizontalFlip" in op_names
    assert "RandomVerticalFlip" in op_names


def test_flip_off_by_default():
    cfg = augmentation_config_from_dict({})
    ops = _geom_ops(cfg, img_size=64, scale_band="global")
    op_names = {type(o).__name__ for o in ops}
    assert "RandomHorizontalFlip" not in op_names
    assert "RandomVerticalFlip" not in op_names


def test_load_job_yaml_run_block(tmp_path):
    job = {
        "name": "test_run",
        "run": {"epochs": 10, "mask_ratio": 0.62, "seed": 42},
        "stages": {
            "s1": {"active": [{"progress": 0.0, "value": 0.0}],
                   "beta_sig": [{"progress": 0.0, "value": 0.0}],
                   "freeze": [{"progress": 0.0, "value": False}]},
            "s2": {"active": [{"progress": 0.0, "value": 0.0}],
                   "beta_sig": [{"progress": 0.0, "value": 0.0}],
                   "freeze": [{"progress": 0.0, "value": False}]},
            "s3": {"active": [{"progress": 0.0, "value": 0.0}],
                   "beta_sig": [{"progress": 0.0, "value": 0.0}],
                   "freeze": [{"progress": 0.0, "value": False}]},
            "s4": {"active": [{"progress": 0.0, "value": 1.0}],
                   "beta_sig": [{"progress": 0.0, "value": 3.7}],
                   "freeze": [{"progress": 0.0, "value": False}]},
        },
        "augmentations": {"random_rotate_deg": 0, "hflip_p": 0.5},
    }
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(job), encoding="utf-8")
    sched, aug_cfg, aug_sched, meta = load_job_yaml(path)
    assert sched is not None
    assert meta["run"]["mask_ratio"] == pytest.approx(0.62)
    assert meta["run"]["epochs"] == 10
    assert aug_cfg.hflip_p == 0.5

    ns = type("NS", (), {"epochs": 100, "mask_ratio": 0.75, "seed": 0, "batch_size": 16})()
    apply_job_run_overrides(ns, parse_run_block(job))
    assert ns.epochs == 10
    assert ns.mask_ratio == pytest.approx(0.62)
    assert ns.seed == 42


def test_load_job_yaml_b1_still_works():
    sched, aug_cfg, aug_sched, meta = load_job_yaml(_B1)
    assert sched is not None
    assert meta["name"] == "isolate_c4_sigreg_b1"
    assert meta["run"] == {}


def test_generator_produces_unique_beta_sig(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("c4_gen", _SWEEP_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    jobs_dir = tmp_path / "jobs"
    manifest = tmp_path / "manifest.yaml"
    entries = mod.generate(100, seed=123, jobs_dir=jobs_dir, manifest_path=manifest)
    assert len(entries) == 100
    betas = [e["beta_sig"] for e in entries]
    assert len(set(betas)) > 50
    assert all(0.1 <= b <= 100.0 for b in betas)
    assert all((jobs_dir / f"run_{i:03d}.yaml").is_file() for i in range(1, 101))
    with manifest.open("r", encoding="utf-8") as f:
        man = yaml.safe_load(f)
    assert man["n_runs"] == 100
    assert len(man["runs"]) == 100
