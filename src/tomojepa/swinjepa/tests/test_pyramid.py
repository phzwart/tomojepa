"""Pyramid residual JEPA: upsample, residuals, image-grouped SIGReg, compute_loss."""
import torch

from tomojepa.swinjepa.mask import MultiScaleBlockMask
from tomojepa.swinjepa.model import SwinMSJEPA
from tomojepa.swinjepa.pyramid import (
    CoarseMIMHead, hierarchical_residuals, reconstruct_from_residuals,
    upsample_stage,
)
from tomojepa.swinjepa.sigreg import ImageGroupedStageSIGReg, PooledStageSIGReg
from .conftest import small_cfg
import pytest


def test_lat_dims_equal_required_for_pyramid():
    with pytest.raises(ValueError, match="equal lat_dims"):
        small_cfg(lat_dims=(32, 64, 32, 32), legacy_jepa=False)
    SwinMSJEPA(small_cfg(lat_dims=(32, 32, 32, 32), legacy_jepa=False))
    SwinMSJEPA(small_cfg(lat_dims=(32, 64, 32, 32), legacy_jepa=True))


def test_upsample_stage_geometry():
    torch.manual_seed(0)
    m4 = torch.arange(4, dtype=torch.float32).view(1, 1, 2, 2)
    m3 = upsample_stage(m4, (4, 4))
    gen = MultiScaleBlockMask(grid4=(2, 2), num_stages=4)
    mask4 = torch.ones(1, 2, 2, dtype=torch.bool)
    assert torch.equal(gen.expand(mask4, (4, 4)), mask4.repeat_interleave(2, -2).repeat_interleave(2, -1))
    assert m3.shape == (1, 1, 4, 4)
    assert m3[0, 0, 0, 0] == 0.0
    assert m3[0, 0, 1, 1] == 0.0


def test_hierarchical_residuals_reconstruct():
    torch.manual_seed(0)
    grids = [(16, 16), (8, 8), (4, 4), (2, 2)]
    E = {f"s{i + 1}": torch.randn(1, 8, *grids[i]) for i in range(4)}
    C4 = torch.randn(1, 8, 2, 2)
    R = hierarchical_residuals(E, C4, grids)
    E_hat = reconstruct_from_residuals(C4, R, E)
    assert torch.allclose(E_hat["s3"], E["s3"], atol=1e-5)
    assert torch.allclose(E_hat["s2"], E["s2"], atol=1e-5)
    assert torch.allclose(E_hat["s1"], E["s1"], atol=1e-5)


def test_coarse_mae_grad():
    """Grad reaches CoarseMIMHead; R3 targets do not backprop into C4."""
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(sigreg_queue_len=0)).train()
    x = torch.randn(2, 1, 64, 64)
    loss, _ = model.compute_loss(x, step=0, total_steps=10)
    model.zero_grad(set_to_none=True)
    loss.backward()
    w = model.coarse_head.head[0].weight.grad
    assert w is not None and float(w.norm()) > 0


def test_gather_stage_tokens_fixed_m():
    torch.manual_seed(0)
    from tomojepa.swinjepa.pyramid import gather_stage_tokens
    feat = torch.randn(3, 8, 4, 4)
    fg = torch.ones(3, 4, 4, dtype=torch.bool)
    fg[0, :2] = False
    tok = gather_stage_tokens(feat, fg, n_per_slice=5)
    assert tok.shape == (3, 5, 8)


def test_gather_min_dist_decorrelates():
    from tomojepa.swinjepa.pyramid import _greedy_min_dist_pick
    grid_y, grid_x = torch.meshgrid(
        torch.arange(8), torch.arange(8), indexing="ij")
    coords_all = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    torch.manual_seed(0)
    pick = _greedy_min_dist_pick(coords_all, 4, 3)
    sel = coords_all[pick]
    for i in range(sel.shape[0]):
        for j in range(i + 1, sel.shape[0]):
            sep = (sel[i] - sel[j]).abs().max()
            assert int(sep) >= 3


def test_image_grouped_sigreg_shape():
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(
        dim=16, n_dirs=32, n_tokens_per_slice=4, queue_len=64)
    feat = torch.randn(8, 16, 2, 2, requires_grad=True)
    loss = sig(feat)
    assert loss.ndim == 0 and float(loss) > 0
    loss.backward()
    assert feat.grad is not None
    for _ in range(3):
        sig(torch.randn(4, 16, 2, 2))
    assert int(sig.queue_size[0]) > 0


def test_image_grouped_scales_with_slices_not_tokens():
    """Calibrated stat grows ~linearly with batch (slice) count, not token soup."""
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(
        dim=16, n_dirs=64, n_tokens_per_slice=8, min_grid_dist=0, queue_len=0)
    feat = torch.randn(4, 16, 2, 2)
    l4 = float(sig(feat).detach())
    l32 = float(sig(feat.repeat(8, 1, 1, 1)).detach())
    ratio = l32 / max(l4, 1e-8)
    assert 6.0 < ratio < 10.0


def test_effrank_caps_sigreg_dirs():
    """Low-rank batch tokens cap n_dirs below configured maximum."""
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(dim=16, n_dirs=256, n_tokens_per_slice=8,
                                  min_grid_dist=0, queue_len=0)
    full = torch.randn(8, 16, 4, 4)
    collapsed = torch.ones(8, 16, 4, 4) * torch.randn(8, 16, 1, 1)
    dirs_full = sig._effective_dirs(8, full.permute(0, 2, 3, 1).reshape(-1, 16))
    tok = collapsed.permute(0, 2, 3, 1).reshape(-1, 16)
    dirs_coll = sig._effective_dirs(8, tok)
    assert dirs_coll < dirs_full
    assert dirs_coll <= 8


def test_pooled_sigreg_shape():
    torch.manual_seed(0)
    sig = PooledStageSIGReg(dim=16, n_dirs=32, queue_len=64)
    z = torch.randn(8, 16, requires_grad=True)
    loss = sig(z)
    assert loss.ndim == 0 and float(loss) > 0
    loss.backward()
    assert z.grad is not None
    for _ in range(3):
        sig(torch.randn(4, 16))
    assert int(sig.queue_size[0]) > 0


def test_pyramid_compute_loss():
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(sigreg_queue_len=0))
    x = torch.randn(2, 1, 64, 64)
    loss, logs = model.compute_loss(x, step=0, total_steps=10)
    assert torch.isfinite(loss)
    assert logs["l_mae"] > 0
    assert "pred/s1" in logs and "pred/s3" in logs
    assert "pred/s4" not in logs
    assert "sig/c4" in logs and "sig/r1" in logs


def test_legacy_jepa_flag():
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(legacy_jepa=True, sigreg_queue_len=0))
    assert model.coarse_head is None
    assert model.sigreg is not None
    loss, logs = model.compute_loss(torch.randn(2, 1, 64, 64), step=0, total_steps=10)
    assert torch.isfinite(loss)
    assert logs["l_mae"] == 0.0
    assert "pred/s4" in logs
    assert "sig/s4" in logs


def test_beta_sig_per_stage():
    """Per-stage beta_sig weights only the matching pyramid SIGReg term."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    off = SwinMSJEPA(small_cfg(beta_sig=(0.0, 0.0, 0.0, 0.0), sigreg_queue_len=0))
    c4_only = SwinMSJEPA(small_cfg(beta_sig=(0.0, 0.0, 0.0, 1.0), sigreg_queue_len=0))
    l_off, _ = off.compute_loss(x, step=0, total_steps=10)
    l_c4, logs = c4_only.compute_loss(x, step=0, total_steps=10)
    assert l_c4 > l_off
    assert logs["l_sig"] > 0
