"""Tests for SwinSimMIM (standalone model, no pyramid residuals)."""
import torch

from tomojepa.swinjepa.simmim_config import SwinSimMIMConfig
from tomojepa.swinjepa.simmim_model import SwinSimMIM


def _cfg(**kwargs) -> SwinSimMIMConfig:
    defaults = dict(
        img_size=64,
        batch_size=2,
        backbone_embed_dim=36,
        lat_dims=(36, 36, 36, 36),
        sigreg_queue_len=0,
        sigreg_n_dirs=(36, 36, 36, 36),
        stage_base_weights=(0.0, 0.0, 0.0, 1.0),
        beta_sig=(0.0, 0.0, 0.0, 0.5),
        mask_ratio=0.45,
    )
    defaults.update(kwargs)
    return SwinSimMIMConfig(**defaults)


def _dual(batch=2, ch=1, h=64, w=64):
    return torch.randn(batch, 2, ch, h, w)


def test_simmim_forward_backward_s4():
    torch.manual_seed(0)
    model = SwinSimMIM(_cfg()).train()
    x = _dual()
    loss, logs = model.compute_loss(x, step=0, total_steps=10)
    assert torch.isfinite(loss)
    assert logs["l_mae"] > 0
    loss.backward()
    w = model.lateral["s4"].weight.grad
    assert w is not None and float(w.norm()) > 0


def test_simmim_sigreg_backprops_ctx_only():
    torch.manual_seed(0)
    cfg = _cfg(beta_sig=(0.0, 0.0, 0.0, 1.0), s4_cosine_level=0.0,
               stage_base_weights=(0.0, 0.0, 0.0, 0.0))
    model = SwinSimMIM(cfg).train()
    x = _dual()
    loss, logs = model.compute_loss(x, step=0, total_steps=10)
    assert logs["l_mae"] == 0.0
    assert logs["l_sig"] > 0
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.lateral["s4"].weight.grad is not None
    assert float(model.lateral["s4"].weight.grad.norm()) > 0


def test_simmim_no_mask_token_still_mae():
    torch.manual_seed(0)
    cfg = _cfg(use_mask_token=False)
    model = SwinSimMIM(cfg).train()
    loss, logs = model.compute_loss(_dual(), step=0, total_steps=10)
    assert torch.isfinite(loss)
    assert logs["l_mae"] > 0
    loss.backward()
    assert model.lateral["s4"].weight.grad is not None


def test_sigreg_cos_gate_all_stages():
    torch.manual_seed(0)
    cfg = _cfg(beta_sig=(0.1, 0.1, 0.1, 0.5), s4_cosine_level=0.8,
               sigreg_cos_gate_all_stages=True, s4_sigreg_ramp_progress=0.05)
    model = SwinSimMIM(cfg).train()
    _, logs = model.compute_loss(_dual(), step=0, total_steps=100)
    assert logs["sigreg/cos_gate"] == 0.0
    assert logs["sig/s1"] == 0.0
    for step in range(1, 40):
        _, logs = model.compute_loss(_dual(), step=step, total_steps=100)
        model.note_mae_cos(0.85)
    _, logs2 = model.compute_loss(_dual(), step=55, total_steps=100)
    assert logs2["sigreg/cos_latched"] == 1.0
    assert logs2["sig/s4"] > 0.0


def test_simmim_no_pyramid_imports():
    import tomojepa.swinjepa.simmim_model as m
    src = open(m.__file__, encoding="utf-8").read()
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        assert "pyramid" not in stripped.lower()
        assert "predictor" not in stripped.lower()
