"""Frozen linear / kNN probe on encode() bands (spec §12)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _pool_band(band: torch.Tensor) -> torch.Tensor:
    return band.mean(dim=tuple(range(2, band.ndim)))


@torch.no_grad()
def knn_accuracy(train_x: torch.Tensor, train_y: torch.Tensor,
                 test_x: torch.Tensor, test_y: torch.Tensor, k: int = 5) -> float:
    dists = torch.cdist(test_x, train_x)
    _, idx = dists.topk(k, largest=False, dim=1)
    votes = train_y[idx]
    pred = torch.mode(votes, dim=1).values
    return float((pred == test_y).float().mean().item())


def run_probe(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 20,
    lr: float = 1e-3,
) -> Dict[str, float]:
    device = images.device
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    bands: List[torch.Tensor] = model.encode(images)
    results: Dict[str, float] = {}
    for i, band in enumerate(bands):
        feats = _pool_band(band).detach()
        n = feats.shape[0]
        split = max(1, int(0.8 * n))
        probe = LinearProbe(feats.shape[1], num_classes).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        for _ in range(epochs):
            logits = probe(feats[:split])
            loss = nn.functional.cross_entropy(logits, labels[:split])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (probe(feats[split:]).argmax(1) == labels[split:]).float().mean()
        results[f"linear/band{i}"] = float(acc.item())
        results[f"knn/band{i}"] = knn_accuracy(
            feats[:split], labels[:split], feats[split:], labels[split:])
    concat = torch.cat([_pool_band(b).detach() for b in bands], dim=1)
    probe = LinearProbe(concat.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    split = max(1, int(0.8 * concat.shape[0]))
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(probe(concat[:split]), labels[:split])
        opt.zero_grad()
        loss.backward()
        opt.step()
    results["linear/pyramid"] = float(
        (probe(concat[split:]).argmax(1) == labels[split:]).float().mean().item())
    return results
