"""Tests for integrated PyramidBandFusion and dual-view band MAE."""
import torch

from tomojepa.swinjepa.model import SwinMSJEPA
from tomojepa.swinjepa.pyramid_fusion import PyramidBandFusion
from tomojepa.swinjepa.predictor import RoPEMultiheadAttention
from .conftest import integrated_cfg, dual_views


def test_fusion_output_shapes():
    torch.manual_seed(0)
    cfg = integrated_cfg()
    model = SwinMSJEPA(cfg)
    assert model.fusion is not None
    assert model.predictor is None
    b = 2
    mask = model.mask_gen.generate(b)
    E_ctx = model.backbone(torch.randn(b, 1, 64, 64), mask1=mask["s1"])
    E_ctx = {k: model.lateral[k](v) for k, v in E_ctx.items()}
    pred = model.fusion(E_ctx, mask)
    for s in range(model.num_stages):
        key = f"s{s + 1}"
        k = int(mask[key][0].sum())
        assert pred[key].shape == (b, k, model.lat_chans[s])


def test_fusion_uses_rope_by_default():
    model = SwinMSJEPA(integrated_cfg())
    assert model.fusion.use_rope
    assert isinstance(model.fusion.blocks[0][0].self_attn, RoPEMultiheadAttention)


def test_integrated_dual_view_forward_backward():
    torch.manual_seed(0)
    model = SwinMSJEPA(integrated_cfg()).train()
    x = dual_views(torch.randn(2, 1, 64, 64))
    loss, logs = model.compute_loss(x, step=0, total_steps=10)
    assert torch.isfinite(loss).all()
    assert logs["mae/s1"] >= 0.0
    assert logs["mae/s4"] >= 0.0
    loss.backward()
    assert any(p.grad is not None for p in model.fusion.parameters())


def test_integrated_band_mae_all_stages():
    torch.manual_seed(0)
    cfg = integrated_cfg(stage_base_weights=(1.0, 1.0, 1.0, 1.0))
    model = SwinMSJEPA(cfg).train()
    x = dual_views(torch.randn(2, 1, 64, 64))
    _, logs = model.compute_loss(x, step=0, total_steps=10)
    for s in range(1, 5):
        assert f"mae/s{s}" in logs
        assert logs[f"lambda/s{s}"] > 0.0


def test_integrated_requires_dual_views():
    model = SwinMSJEPA(integrated_cfg())
    x = torch.randn(2, 1, 64, 64)
    try:
        model.compute_loss(x, step=0, total_steps=10)
        assert False, "expected ValueError for single-view input"
    except ValueError:
        pass


def test_integrated_sigreg_backprops_to_student():
    """SIGReg must regularize grad-enabled student bands, not stop-grad teacher."""
    torch.manual_seed(0)
    cfg = integrated_cfg(
        stage_base_weights=(0.0, 0.0, 0.0, 0.0),
        beta_sig=(0.0, 0.0, 0.0, 1.0),
        s4_cosine_level=0.0,
    )
    model = SwinMSJEPA(cfg).train()
    x = dual_views(torch.randn(2, 1, 64, 64))
    loss, logs = model.compute_loss(x, step=0, total_steps=10)
    assert logs["l_sig"] > 0.0
    assert logs["l_pred"] == 0.0
    loss.backward()
    lat_grad = model.lateral["s4"].weight.grad
    assert lat_grad is not None and lat_grad.abs().sum() > 0
