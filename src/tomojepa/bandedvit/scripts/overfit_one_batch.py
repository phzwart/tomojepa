"""Overfit one synthetic batch — SIGReg loss should decrease."""
from __future__ import annotations

import torch

from tomojepa.bandedvit.config import BandedJEPAConfig
from tomojepa.bandedvit.model import BandedJEPA


def main():
    torch.manual_seed(0)
    cfg = BandedJEPAConfig(
        img_size=64,
        patch_size=16,
        depth=4,
        embed_dim=128,
        num_heads=4,
        beta_sig=(0.0, 0.0, 0.0, 1.0),
        sigreg_blocks=(-1,),
        sigreg_n_dirs=32,
        sigreg_token_cap=256,
        band_K=1,
        pred_enabled=False,
    )
    model = BandedJEPA(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 64, 64)

    losses = []
    for step in range(200):
        opt.zero_grad()
        loss, logs = model.compute_loss(x, step=step)
        loss.backward()
        opt.step()
        losses.append(logs["total"])
        if step % 50 == 0:
            print(f"step {step} loss={logs['total']:.4f}")

    assert all(torch.isfinite(torch.tensor(losses))), "non-finite losses"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"OK  loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
