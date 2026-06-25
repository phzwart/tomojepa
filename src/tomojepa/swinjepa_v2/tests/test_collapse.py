"""SIGReg collapse sanity (spec §11)."""
import torch

from tomojepa.swinjepa_v2.config import default
from tomojepa.swinjepa_v2.data.dataset import SyntheticHLJEPADataset
from tomojepa.swinjepa_v2.models.model import SwinJEPA
from tomojepa.swinjepa_v2.nd import flatten_tokens
from tomojepa.swinjepa_v2.train.instrument import effective_rank
from torch.utils.data import DataLoader


def _collate(batch):
    return {
        "view_ctx": torch.stack([b["view_ctx"] for b in batch]),
        "view_tgt": torch.stack([b["view_tgt"] for b in batch]),
        "band_masks": [
            torch.stack([b["band_masks"][i] for b in batch])
            for i in range(4)
        ],
    }


def _train_steps(cfg, steps=50):
    cfg.data.image_size = (128, 128)
    cfg.backbone.model_name = "swin_tiny_patch4_window7_224"
    model = SwinJEPA(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ds = SyntheticHLJEPADataset(cfg, length=8, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=_collate)
    batch = next(iter(loader))
    ranks = []
    for step in range(steps):
        opt.zero_grad()
        loss, _ = model.compute_loss(batch["view_ctx"], batch["view_tgt"], batch["band_masks"], step)
        loss.backward()
        opt.step()
        with torch.no_grad():
            _, tgt, _ = model.forward_train(batch["view_ctx"], batch["view_tgt"], batch["band_masks"])
            ranks.append(effective_rank(flatten_tokens(tgt[-1])))
    return ranks


def test_rank_collapses_without_sigreg():
    cfg = default()
    cfg.loss.lambda_sig = (0.0, 0.0, 0.0, 0.0)
    ranks = _train_steps(cfg, steps=80)
    assert ranks[-1] < ranks[0] * 0.85 or ranks[-1] < 3.0


def test_rank_sustained_with_sigreg():
    cfg = default()
    cfg.loss.lambda_sig = (0.2, 0.2, 0.2, 0.2)
    ranks = _train_steps(cfg, steps=80)
    assert ranks[-1] > 1.5
