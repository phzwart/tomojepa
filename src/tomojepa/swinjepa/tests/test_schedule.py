"""YAML training schedule loader and interpolation."""
from pathlib import Path

import pytest
import torch

from tomojepa.swinjepa.schedule import (
    _interp_bool_sticky,
    _interp_numeric,
    load_training_schedule,
)
from tomojepa.swinjepa.model import SwinMSJEPA
from .conftest import small_cfg

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PETIOLE_SCHED = _REPO_ROOT / "configs/swinjepa/schedules/petiole_448_coarse_first.yaml"


def test_numeric_interp():
    knots = [(0.0, 0.1), (0.5, 1.0)]
    assert _interp_numeric(knots, 0.0) == pytest.approx(0.1)
    assert _interp_numeric(knots, 0.25) == pytest.approx(0.55)
    assert _interp_numeric(knots, 1.0) == pytest.approx(1.0)


def test_bool_sticky():
    knots = [(0.0, False), (0.2, True), (0.8, False)]
    assert not _interp_bool_sticky(knots, 0.1)
    assert _interp_bool_sticky(knots, 0.2)
    assert _interp_bool_sticky(knots, 0.9)
    assert _interp_bool_sticky(knots, 1.0)


def test_load_petiole_schedule():
    sched = load_training_schedule(_PETIOLE_SCHED)
    assert sched.name == "petiole_448_coarse_first_25pct"
    assert sched.progress_scope == "epoch"
    s0 = sched.at(0, 1000, steps_per_epoch=172)
    assert s0.stages["s4"].active == 1.0
    assert not s0.stages["s4"].frozen
    s_freeze = sched.at(43, 8600, steps_per_epoch=172)
    assert s_freeze.stages["s4"].frozen
    s_mid = sched.at(64, 8600, steps_per_epoch=172)
    assert s_mid.stages["s3"].active == pytest.approx(0.55, abs=0.02)


def test_schedule_epoch_progress():
    sched = load_training_schedule(_PETIOLE_SCHED)
    spe = 172
    assert sched.at(0, 8600, spe).progress == pytest.approx(0.0)
    assert sched.at(43, 8600, spe).progress == pytest.approx(0.25)
    assert sched.at(172, 8600, spe).progress == pytest.approx(0.0)


def test_schedule_overrides_curriculum_and_freezes():
    sched = load_training_schedule(_PETIOLE_SCHED)
    model = SwinMSJEPA(small_cfg())
    model.set_schedule(sched)
    model.set_steps_per_epoch(172)
    model._sync_freeze_from_schedule(43, 8600)
    assert model.frozen_stage_keys == ["s4"]

    x = torch.randn(2, 1, 64, 64)
    _, logs = model.compute_loss(x, step=43, total_steps=8600)
    assert logs["lambda/s4"] == 0.0
    assert logs["schedule_progress"] == pytest.approx(0.25)
