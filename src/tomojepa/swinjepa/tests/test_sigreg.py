"""Acceptance check 6: SIGReg collapse control (sign / gradient direction)."""
import pytest
import torch

from tomojepa.swinjepa.config import SwinMSJEPAConfig
from tomojepa.swinjepa.sigreg import StageSIGReg, ImageGroupedStageSIGReg


def test_sigreg_penalizes_collapse():
    """A near-constant feature tensor yields a large positive SIGReg value, far
    above that of a standard-normal sample."""
    torch.manual_seed(0)
    sig = StageSIGReg(dim=16, n_dirs=64, n_tokens_cap=0)
    collapsed = 1e-3 * torch.randn(256, 16) + 5.0      # near-constant, off-center
    gaussian = torch.randn(256, 16)
    v_collapsed = float(sig(collapsed).detach())
    v_gaussian = float(sig(gaussian).detach())
    assert v_collapsed > 0
    assert v_collapsed > 5 * v_gaussian


def test_sigreg_gradient_increases_variance():
    """A gradient step against SIGReg increases the per-direction variance of a
    collapsed feature toward 1 (correct sign of the regularizer)."""
    torch.manual_seed(0)
    sig = StageSIGReg(dim=8, n_dirs=64, n_tokens_cap=0)
    z = (1e-2 * torch.randn(256, 8)).clone().requires_grad_(True)
    std_before = float(z.detach().std())
    loss = sig(z)
    loss.backward()
    with torch.no_grad():
        z_new = z - 0.5 * z.grad                        # descend SIGReg
    std_after = float(z_new.std())
    assert std_after > std_before


def test_sigreg_queue_runs():
    """The FIFO queue path runs and keeps gradients on the current features."""
    torch.manual_seed(0)
    sig = StageSIGReg(dim=8, n_dirs=32, n_tokens_cap=0, queue_len=128)
    for _ in range(3):
        z = torch.randn(64, 8, requires_grad=True)
        loss = sig(z)
        loss.backward()
        assert z.grad is not None
    assert int(sig.queue_size[0]) > 0


def test_sigreg_n_dirs_clamped_to_lat_dims():
    with pytest.warns(UserWarning, match="clamping"):
        cfg = SwinMSJEPAConfig(
            sigreg_n_dirs=(256, 256, 384, 512), lat_dims=(64, 64, 64, 64))
    assert cfg.sigreg_n_dirs == (64, 64, 64, 64)


def test_sigreg_n_dirs_unchanged_when_within_lat_dims():
    cfg = SwinMSJEPAConfig(
        sigreg_n_dirs=(32, 32, 32, 32), lat_dims=(64, 64, 64, 64))
    assert cfg.sigreg_n_dirs == (32, 32, 32, 32)


def test_image_grouped_keeps_min_dirs_when_collapsed():
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(
        dim=16, n_dirs=64, n_tokens_per_slice=8, min_grid_dist=0,
        queue_len=0, cap_dirs_by_rank=False, min_dirs=16)
    collapsed = torch.ones(8, 16, 4, 4) * torch.randn(8, 16, 1, 1)
    tok = collapsed.permute(0, 2, 3, 1).reshape(-1, 16)
    assert sig._effective_dirs(8, tok) >= 16


def test_image_grouped_penalizes_collapse():
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(
        dim=16, n_dirs=64, n_tokens_per_slice=16, min_grid_dist=0,
        queue_len=0, cap_dirs_by_rank=False, min_dirs=16)
    collapsed = 1e-3 * torch.randn(4, 16, 4, 4) + 5.0
    gaussian = torch.randn(4, 16, 4, 4)
    v_collapsed = float(sig(collapsed).detach())
    v_gaussian = float(sig(gaussian).detach())
    assert v_collapsed > 0
    assert v_collapsed > 5 * v_gaussian


def test_image_grouped_gradient_increases_variance():
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(
        dim=8, n_dirs=64, n_tokens_per_slice=16, min_grid_dist=0,
        queue_len=0, cap_dirs_by_rank=False, min_dirs=16)
    feat = (1e-2 * torch.randn(4, 8, 4, 4)).clone().requires_grad_(True)
    std_before = float(feat.detach().std())
    loss = sig(feat)
    loss.backward()
    with torch.no_grad():
        feat_new = feat - 0.5 * feat.grad
    std_after = float(feat_new.std())
    assert std_after > std_before


def test_image_grouped_rank_cap_when_enabled():
    torch.manual_seed(0)
    sig = ImageGroupedStageSIGReg(
        dim=16, n_dirs=256, n_tokens_per_slice=8, min_grid_dist=0,
        queue_len=0, cap_dirs_by_rank=True, min_dirs=4)
    full = torch.randn(8, 16, 4, 4)
    collapsed = torch.ones(8, 16, 4, 4) * torch.randn(8, 16, 1, 1)
    dirs_full = sig._effective_dirs(8, full.permute(0, 2, 3, 1).reshape(-1, 16))
    tok = collapsed.permute(0, 2, 3, 1).reshape(-1, 16)
    dirs_coll = sig._effective_dirs(8, tok)
    assert dirs_coll < dirs_full
