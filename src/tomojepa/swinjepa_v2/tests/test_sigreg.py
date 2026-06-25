"""Acceptance tests for Epps-Pulley SIGReg (spec §11)."""
import torch

from tomojepa.swinjepa_v2.config import SigRegCfg
from tomojepa.swinjepa_v2.losses.sigreg import SIGReg


def _sigreg(dim=32):
    return SIGReg(SigRegCfg(num_slices=128, n_knots=17, t_max=5.0), feat_dim=dim)


def test_gaussian_near_floor():
    sig = _sigreg()
    z = torch.randn(8000, 32)
    z_col = torch.zeros(8000, 32)
    loss = float(sig(z, step=0).item())
    loss_col = float(sig(z_col, step=0).item())
    assert loss < loss_col * 0.5


def test_collapsed_large():
    sig = _sigreg()
    z = torch.ones(500, 32)
    loss = float(sig(z, step=1).item())
    z_g = torch.randn(500, 32)
    assert loss > float(sig(z_g, step=1).item())


def test_anisotropic_intermediate():
    sig = _sigreg(dim=16)
    z_iso = torch.randn(4000, 16)
    z_aniso = torch.randn(4000, 16)
    z_aniso[:, 0] *= 20.0
    l_iso = float(sig(z_iso, step=2).item())
    l_aniso = float(sig(z_aniso, step=2).item())
    assert l_aniso > l_iso


def test_gradient_finite():
    sig = _sigreg()
    z = torch.randn(128, 32, requires_grad=True)
    loss = sig(z, step=3)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_annealing_decreases():
    sig = _sigreg(dim=8)
    z_bad = torch.randn(2000, 8)
    z_bad[:, 0] *= 10.0
    l0 = float(sig(z_bad, step=4).item())
    z_good = z_bad.clone()
    z_good[:, 0] *= 0.2
    l1 = float(sig(z_good, step=4).item())
    assert l1 < l0
