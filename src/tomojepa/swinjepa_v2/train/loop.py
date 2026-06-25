"""Training loop — no EMA / teacher (spec §9)."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, TextIO

import torch
from torch.utils.data import DataLoader

from tomojepa.core import dist as D

from ..config import HLJEPAConfig, dump_yaml, load_yaml
from ..data.dataset import HLJEPADataset, SyntheticHLJEPADataset
from ..models.model import SwinJEPA
from .instrument import collect_metrics
from .pca_viz import build_probe, run_pca_strip


def _collate(batch):
    return {
        "view_ctx": torch.stack([b["view_ctx"] for b in batch]),
        "view_tgt": torch.stack([b["view_tgt"] for b in batch]),
        "band_masks": [torch.stack([b["band_masks"][i] for b in batch]) for i in range(4)],
        "geom_params": batch[0]["geom_params"],
    }


class _TeeLog:
    """Mirror stdout to run_dir/train.log on the main rank."""

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._file: Optional[TextIO] = None
        self._stdout = sys.stdout

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a", buffering=1)
        sys.stdout = self

    def stop(self) -> None:
        if self._file is not None:
            sys.stdout = self._stdout
            self._file.close()
            self._file = None

    def write(self, data: str) -> int:
        self._stdout.write(data)
        if self._file is not None:
            self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        if self._file is not None:
            self._file.flush()


def _lr_at_epoch(cfg: HLJEPAConfig, epoch: int, steps_per_epoch: int) -> float:
    base = cfg.train.base_lr
    warmup = cfg.train.warmup_epochs
    total = cfg.train.epochs
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: HLJEPAConfig, synthetic: bool = False, pca_every: int = 0,
          pca_samples: int = 4) -> None:
    device, _ = D.init_distributed()

    torch.manual_seed(cfg.train.seed)
    if synthetic or not cfg.train.data_dir:
        ds = SyntheticHLJEPADataset(cfg, length=256, seed=cfg.train.seed)
    else:
        ds = HLJEPADataset(cfg, is_train=True)

    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=not D.is_distributed(),
        num_workers=0,
        drop_last=True,
        collate_fn=_collate,
    )

    model = SwinJEPA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.base_lr,
                            weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp_dtype == "fp16")
    amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16

    run_dir = Path(cfg.train.run_dir)
    tee: Optional[_TeeLog] = None
    if D.is_main():
        run_dir.mkdir(parents=True, exist_ok=True)
        pca_dir = run_dir / "out" / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(cfg, run_dir / "config.yaml")
        (run_dir / "config.json").write_text(json.dumps({
            "cfg": "hljepa",
            "pca_dir": str(pca_dir),
            "log_file": str(run_dir / "train.log"),
            "stop_grad_target": cfg.loss.stop_grad_target,
        }))
        tee = _TeeLog(run_dir / "train.log")
        tee.start()
        print(f"__START__ {datetime.now(timezone.utc).isoformat()}", flush=True)
        print(f"run_dir={run_dir} stop_grad_target={cfg.loss.stop_grad_target}", flush=True)

    pca_probe = None
    if D.is_main() and pca_every > 0 and not synthetic and cfg.train.data_dir:
        pca_probe = build_probe(
            cfg.train.data_dir, cfg.img_size, cfg.data.pattern, cfg.data.backend,
            cfg.data.dataset_key, seed=cfg.train.seed, n_samples=pca_samples,
        )
        print(f"PCA strips -> {run_dir / 'out' / 'pca'} every {pca_every} steps", flush=True)

    global_step = 0
    for epoch in range(cfg.train.epochs):
        lr = _lr_at_epoch(cfg, epoch, len(loader))
        for pg in opt.param_groups:
            pg["lr"] = lr
        for batch in loader:
            view_ctx = batch["view_ctx"].to(device)
            view_tgt = batch["view_tgt"].to(device)
            band_masks = [m.to(device) for m in batch["band_masks"]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss, logs = model.compute_loss(view_ctx, view_tgt, band_masks, step=global_step)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()

            if D.is_main() and global_step % cfg.train.log_every == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(logs.items()))
                print(f"step {global_step} {msg}")

            if D.is_main() and pca_probe is not None and pca_every > 0 and global_step % pca_every == 0:
                path = run_pca_strip(model, pca_probe, global_step, run_dir / "out" / "pca", device)
                print(f"pca saved {path}", flush=True)

            if global_step % cfg.train.metric_every == 0:
                with torch.no_grad():
                    pred, tgt, _ = model.forward_train(view_ctx, view_tgt, band_masks)
                    mlogs = collect_metrics(
                        model, pred, tgt, band_masks, model.sigreg_modules,
                        step=global_step,
                        x=view_tgt[0] if view_tgt.shape[0] else None,
                        geom=batch["geom_params"] if isinstance(batch["geom_params"], object) else None,
                    )
                if D.is_main():
                    print(f"metrics step {global_step}: {mlogs}")

            global_step += 1

        if D.is_main():
            ckpt = run_dir / "ckpt" / f"ckpt_epoch{epoch:04d}.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)

    if tee is not None:
        print(f"__END__ {datetime.now(timezone.utc).isoformat()}", flush=True)
        tee.stop()
    D.cleanup()


def main(argv: Optional[list] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="HL-JEPA pretraining")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--pca_every", type=int, default=0)
    p.add_argument("--pca_samples", type=int, default=4)
    p.add_argument("--stop_grad_target", action="store_true", default=None)
    p.add_argument("--no_stop_grad_target", action="store_false", dest="stop_grad_target")
    args = p.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.stop_grad_target is not None:
        cfg.loss.stop_grad_target = args.stop_grad_target
    if args.data_dir:
        cfg.train.data_dir = args.data_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.run_dir:
        cfg.train.run_dir = args.run_dir
    train(cfg, synthetic=args.synthetic or not cfg.train.data_dir,
          pca_every=args.pca_every, pca_samples=args.pca_samples)
