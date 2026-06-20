"""Per-stage freeze schedule (``freeze_after_epoch``)."""
from pathlib import Path

import torch

from tomojepa.swinjepa.config import SwinMSJEPAConfig
from tomojepa.swinjepa.model import SwinMSJEPA
from tomojepa.swinjepa.schedule import load_training_schedule
from .conftest import small_cfg

_REPO = Path(__file__).resolve().parents[4]


def test_freeze_after_epoch_validation():
    SwinMSJEPAConfig(freeze_after_epoch=(0, 0, 0, 5))
    SwinMSJEPAConfig(freeze_after_epoch=(5,))  # broadcast
    try:
        SwinMSJEPAConfig(freeze_after_epoch=(0, 0, 0))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        SwinMSJEPAConfig(freeze_after_epoch=(0, 0, 0, -1))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_freeze_schedule_idempotent():
    model = SwinMSJEPA(small_cfg(freeze_after_epoch=(0, 0, 0, 1)))
    assert model.apply_freeze_schedule(0) == []
    assert model.apply_freeze_schedule(1) == ["s4"]
    assert model.apply_freeze_schedule(1) == []
    assert model.frozen_stage_keys == ["s4"]


def test_freeze_s4_stops_coarse_gradients():
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(freeze_after_epoch=(0, 0, 0, 1)))
    x = torch.randn(2, 1, 64, 64)
    model.apply_freeze_schedule(1)
    assert model.frozen_stage_keys == ["s4"]

    loss, logs = model.compute_loss(x, step=0, total_steps=100)
    assert logs["lambda/s4"] == 0.0
    assert logs["frozen/s4"] == 1.0
    model.zero_grad(set_to_none=True)
    loss.backward()

    head_w = model.coarse_head.head[0].weight
    lat_w = model.lateral["s4"].weight
    assert not head_w.requires_grad and head_w.grad is None
    assert not lat_w.requires_grad and lat_w.grad is None
    g = model.lateral["s3"].weight.grad
    assert g is not None and float(g.norm()) > 0


def test_freeze_s3_zeros_residual_losses():
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(freeze_after_epoch=(0, 0, 3, 0)))
    model.apply_freeze_schedule(3)
    assert "s3" in model.frozen_stage_keys

    x = torch.randn(2, 1, 64, 64)
    _, logs = model.compute_loss(x, step=50, total_steps=100)
    assert logs["lambda/s3"] == 0.0
    assert logs["frozen/s3"] == 1.0
    assert logs["frozen/s4"] == 0.0


def test_zero_lambda_stages_carry_no_grad():
    """Inactive stages (lambda=0, beta=0) detach latents and skip predictor."""
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(
        stage_base_weights=(0.0, 0.0, 0.0, 1.0),
        sigreg_tokens_per_slice=8,
    ))
    model.set_schedule(load_training_schedule(
        _REPO / "configs/swinjepa/schedules/isolate_c4_sigreg.yaml"))
    x = torch.randn(2, 1, 64, 64)
    loss, logs = model.compute_loss(x, step=0, total_steps=100)
    assert logs["lambda/s1"] == 0.0
    assert logs["lambda/s4"] == 1.0
    assert logs["pred/s1"] == 0.0

    model.zero_grad(set_to_none=True)
    loss.backward()

    for key in ("s1", "s2", "s3"):
        g = model.lateral[key].weight.grad
        assert g is None or float(g.norm()) == 0.0
    for p in model.predictor.parameters():
        assert p.grad is None or float(p.grad.norm()) == 0.0
    g4 = model.lateral["s4"].weight.grad
    assert g4 is not None and float(g4.norm()) > 0
    g_head = model.coarse_head.head[0].weight.grad
    assert g_head is not None and float(g_head.norm()) > 0
