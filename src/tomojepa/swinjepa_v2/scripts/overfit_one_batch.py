"""Overfit one batch — prediction loss → ~0 (spec §11)."""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from tomojepa.swinjepa_v2.config import default, load_yaml
from tomojepa.swinjepa_v2.data.dataset import SyntheticHLJEPADataset
from tomojepa.swinjepa_v2.losses.prediction import prediction_loss
from tomojepa.swinjepa_v2.models.model import SwinJEPA


def _collate(batch):
    return {
        "view_ctx": torch.stack([b["view_ctx"] for b in batch]),
        "view_tgt": torch.stack([b["view_tgt"] for b in batch]),
        "band_masks": [torch.stack([b["band_masks"][i] for b in batch]) for i in range(4)],
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args(argv)
    cfg = load_yaml(args.config) if args.config else default()
    cfg.data.image_size = (128, 128)
    cfg.loss.lambda_sig = (0.0, 0.0, 0.0, 0.0)
    cfg.loss.stop_grad_target = True
    cfg.predictor.top_down = False
    cfg.predictor.depth_per_band = 1
    cfg.mask.mask_ratio = 0.4

    ds = SyntheticHLJEPADataset(cfg, length=args.batch_size, seed=0)
    batch = _collate([ds[0] for _ in range(args.batch_size)])

    g = torch.Generator().manual_seed(0)
    from tomojepa.swinjepa_v2.data.masking import masks_for_stages
    fixed_masks = masks_for_stages(128, 4, cfg.mask, g)
    batch["band_masks"] = [m.unsqueeze(0).expand(args.batch_size, -1, -1) for m in fixed_masks]

    model = SwinJEPA(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    for step in range(args.steps):
        opt.zero_grad()
        pred, tgt, _ = model.forward_train(batch["view_ctx"], batch["view_tgt"], batch["band_masks"])
        loss = sum(
            prediction_loss(pred[k], tgt[k], batch["band_masks"][k], cfg.loss)
            for k in range(4)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} pred_loss={loss.detach().item():.6f}")

    with torch.no_grad():
        pred, tgt, _ = model.forward_train(batch["view_ctx"], batch["view_tgt"], batch["band_masks"])
        final = sum(
            float(prediction_loss(pred[k], tgt[k], batch["band_masks"][k], cfg.loss))
            for k in range(4)
        )
    print(f"final pred_loss={final:.6f}")
    assert final < 0.15, f"failed to overfit: {final}"


if __name__ == "__main__":
    main()
