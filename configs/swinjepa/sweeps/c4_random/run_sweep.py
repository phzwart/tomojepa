#!/usr/bin/env python3
"""Run C4 random augmentation sweep sequentially from a manifest.

Usage:
    python configs/swinjepa/sweeps/c4_random/generate.py --n 100
    python configs/swinjepa/sweeps/c4_random/run_sweep.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_DEFAULT_MANIFEST = _HERE / "manifest.yaml"
_VENV_PYTHON = _REPO / ".venv" / "bin" / "python"

# Fixed training flags (shared across all sweep runs).
_TRAIN_FLAGS = [
    "--data_dir", str(_REPO),
    "--pattern", "petiole.zarr",
    "--backend", "zarr",
    "--img_size", "448",
    "--crop_mode", "resize",
    "--batch_size", "16",
    "--foreground_mask",
    "--fg_mode", "circle",
    "--fg_circle_diameter_frac", "1.0",
    "--stage_base_weights", "0", "0", "0", "1",
    "--sigreg_queue_len", "512",
    "--sigreg_token_frac", "0.25",
    "--pca_es_every", "0",
    "--save_every", "25",
    "--pca_every", "25",
    "--log_every", "10",
]


def _log(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


def _is_complete(run_dir: Path) -> bool:
    ckpt = run_dir / "ckpt" / "ckpt_last.pth"
    log = run_dir / "train.log"
    if not ckpt.is_file() or not log.is_file():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    return "epoch 9:" in text or "epoch 10:" in text or "epoch 9 " in text


def run_one(entry: dict, sweep_log: Path, dry_run: bool = False) -> int:
    run_id = entry["id"]
    run_dir = _REPO / entry["run_dir"]
    job_yaml = _REPO / entry["yaml_path"]
    ckpt_dir = run_dir / "ckpt"
    out_dir = run_dir / "out"
    train_log = run_dir / "train.log"

    cmd = [
        str(_VENV_PYTHON), "-m", "tomojepa.swinjepa.train",
        *_TRAIN_FLAGS,
        "--schedule", str(job_yaml),
        "--ckpt_dir", str(ckpt_dir),
        "--out_dir", str(out_dir),
    ]
    _log(sweep_log, f"run_{run_id:03d} START {entry.get('summary', '')}")
    if dry_run:
        print(" ".join(cmd))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    header = (
        f"=== SWEEP run_{run_id:03d} {datetime.now().isoformat()} "
        f"beta_sig={entry.get('beta_sig')} mask={entry.get('mask_ratio')} ===\n"
    )
    with train_log.open("w", encoding="utf-8") as f:
        f.write(header)
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
    if rc == 0:
        _log(sweep_log, f"run_{run_id:03d} OK")
    else:
        _log(sweep_log, f"run_{run_id:03d} FAILED exit_code={rc}")
    return rc


def main():
    p = argparse.ArgumentParser(description="Sequential C4 random aug sweep runner")
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    p.add_argument("--start", type=int, default=1, help="First run id (inclusive)")
    p.add_argument("--end", type=int, default=0, help="Last run id (0 = all in manifest)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose train.log shows epoch 9+ complete")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop sweep on first training failure")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = p.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run generate.py first.", file=sys.stderr)
        return 1
    if not args.dry_run and not _VENV_PYTHON.is_file():
        print(f"Python not found: {_VENV_PYTHON}", file=sys.stderr)
        return 1

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    runs = manifest.get("runs", [])
    runs_root = _REPO / manifest.get("runs_root", "runs/c4_random_sweep")
    sweep_log = runs_root / "sweep.log"

    end_id = args.end if args.end > 0 else max(r["id"] for r in runs)
    selected = [r for r in runs if args.start <= r["id"] <= end_id]
    if not selected:
        print("No runs matched start/end range.", file=sys.stderr)
        return 1

    _log(sweep_log, f"SWEEP START n={len(selected)} manifest={args.manifest.name}")
    t0 = time.time()
    failures = 0
    for i, entry in enumerate(selected):
        run_dir = _REPO / entry["run_dir"]
        if args.skip_existing and _is_complete(run_dir):
            _log(sweep_log, f"run_{entry['id']:03d} SKIP (complete)")
            continue
        rc = run_one(entry, sweep_log, dry_run=args.dry_run)
        if rc != 0:
            failures += 1
            if args.stop_on_error:
                _log(sweep_log, f"SWEEP ABORT after failure on run_{entry['id']:03d}")
                return rc
        if not args.dry_run and i == 0 and len(selected) > 1:
            elapsed = time.time() - t0
            eta_h = elapsed / 3600.0 * (len(selected) - 1)
            _log(sweep_log, f"ETA ~{eta_h:.1f}h for remaining {len(selected) - 1} runs")

    _log(sweep_log, f"SWEEP DONE failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
