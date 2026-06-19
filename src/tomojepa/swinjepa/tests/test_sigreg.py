"""Acceptance check 6: SIGReg collapse control (sign / gradient direction)."""
import torch

from tomojepa.swinjepa.sigreg import StageSIGReg


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
