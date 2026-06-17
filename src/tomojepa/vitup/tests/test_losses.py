"""Loss tests (acceptance criterion 5)."""
import torch

from tomojepa.vitup.losses import normalized_l2, cosine_loss, relational_kl
from tomojepa.vitup.config import ViTUpConfig


def test_l2_zero_when_equal():
    f = torch.randn(2, 10, 8)
    assert normalized_l2(f, f).item() < 1e-10


def test_cosine_zero_when_aligned():
    f = torch.randn(2, 10, 8)
    # same direction (positive scaling) -> cosine 1 -> loss 0
    assert cosine_loss(3.0 * f, f).item() < 1e-6


def test_relational_zero_when_equal():
    f = torch.randn(2, 12, 8)
    assert relational_kl(f, f, tau=0.1).item() < 1e-6


def test_relational_ignores_diagonal():
    # scaling features changes self-similarity but the diagonal is masked out,
    # so identical-direction student/teacher still give ~0 relational loss.
    f = torch.randn(1, 6, 8)
    assert relational_kl(2.0 * f, f, tau=0.07).item() < 1e-5


def test_feature_loss_combines():
    from tomojepa.vitup.losses import feature_loss
    cfg = ViTUpConfig()
    f = torch.randn(2, 9, 8)
    d = feature_loss(f, f, cfg)
    assert d["total"].item() < 1e-5
