#!/usr/bin/env python3
"""Generate random C4 sweep job YAMLs and a manifest.

Usage:
    python configs/swinjepa/sweeps/c4_random/generate.py --n 100 --seed 20260620
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_JOBS_DIR = _HERE / "jobs"
_MANIFEST = _HERE / "manifest.yaml"

_STAGE_OFF = {
    "active": [{"progress": 0.0, "value": 0.0}],
    "beta_sig": [{"progress": 0.0, "value": 0.0}],
    "freeze": [{"progress": 0.0, "value": False}],
}


def _sample_sparse_p(rng: np.random.Generator, force_zero_prob: float = 0.4) -> float:
    if rng.random() < force_zero_prob:
        return 0.0
    return float(rng.uniform(0.0, 0.8))


def sample_run(rng: np.random.Generator, run_id: int, base_seed: int) -> Dict[str, Any]:
    beta_sig = float(10 ** rng.uniform(-1.0, 2.0))
    mask_ratio = float(rng.uniform(0.35, 0.85))
    rotate_on = bool(rng.random() < 0.5)
    hflip_on = bool(rng.random() < 0.5)
    vflip_on = bool(rng.random() < 0.5)
    jitter_on = bool(rng.random() < 0.5)
    intensity_on = bool(rng.random() < 0.7)

    if jitter_on:
        d = float(rng.uniform(0.05, 0.15))
        resize_jitter = [round(1.0 - d, 4), round(1.0 + d, 4)]
    else:
        resize_jitter = None

    aug: Dict[str, Any] = {
        "variant": "tomo2",
        "crop_mode": "resize",
        "global_views": 1,
        "local_views": 0,
        "random_rotate_deg": 180.0 if rotate_on else 0.0,
        "resize_jitter": resize_jitter,
        "hflip_p": 0.5 if hflip_on else 0.0,
        "vflip_p": 0.5 if vflip_on else 0.0,
        "intensity_augment": intensity_on,
        "window": {"p_low": 0.01, "p_high": 0.99, "sample_size": 100000},
        "blur": {"kernel_size": 7, "sigma": [0.1, 1.0]},
        "poisson": {"scale": 10000},
        "pixel_mask": {"ratio": 0.15},
        "normalize": {"mean": 0.5, "std": 0.5},
    }
    if intensity_on:
        aug["equalize_p"] = _sample_sparse_p(rng)
        aug["blur"]["p"] = _sample_sparse_p(rng)
        aug["poisson"]["p"] = _sample_sparse_p(rng)
        aug["pixel_mask"]["p"] = _sample_sparse_p(rng)
    else:
        aug["equalize_p"] = 0.0
        aug["blur"]["p"] = 0.0
        aug["poisson"]["p"] = 0.0
        aug["pixel_mask"]["p"] = 0.0

    seed = int(base_seed + run_id)
    name = f"c4_sweep_{run_id:03d}"
    job = {
        "name": name,
        "run": {
            "epochs": 10,
            "mask_ratio": round(mask_ratio, 4),
            "seed": seed,
        },
        "stages": {
            "s1": dict(_STAGE_OFF),
            "s2": dict(_STAGE_OFF),
            "s3": dict(_STAGE_OFF),
            "s4": {
                "active": [{"progress": 0.0, "value": 1.0}],
                "beta_sig": [{"progress": 0.0, "value": round(beta_sig, 6)}],
                "freeze": [{"progress": 0.0, "value": False}],
            },
        },
        "augmentations": aug,
    }
    summary = (
        f"beta_sig={beta_sig:.4g}; mask={mask_ratio:.3f}; "
        f"rot={'on' if rotate_on else 'off'}; "
        f"hflip={'on' if hflip_on else 'off'}; vflip={'on' if vflip_on else 'off'}; "
        f"jitter={'on' if jitter_on else 'off'}; intensity={'on' if intensity_on else 'off'}"
    )
    return {
        "id": run_id,
        "name": name,
        "seed": seed,
        "beta_sig": beta_sig,
        "mask_ratio": mask_ratio,
        "summary": summary,
        "job": job,
    }


def generate(n: int, seed: int, jobs_dir: Path, manifest_path: Path) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for run_id in range(1, n + 1):
        sampled = sample_run(rng, run_id, base_seed=seed)
        yaml_path = jobs_dir / f"run_{run_id:03d}.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sampled["job"], f, default_flow_style=False, sort_keys=False)
        run_dir = f"runs/c4_random_sweep/run_{run_id:03d}"
        try:
            rel_yaml = yaml_path.relative_to(_REPO)
        except ValueError:
            rel_yaml = yaml_path
        entries.append({
            "id": run_id,
            "name": sampled["name"],
            "yaml_path": str(rel_yaml),
            "run_dir": run_dir,
            "seed": sampled["seed"],
            "beta_sig": round(sampled["beta_sig"], 6),
            "mask_ratio": round(sampled["mask_ratio"], 4),
            "summary": sampled["summary"],
        })

    manifest = {
        "sweep": "c4_random",
        "n_runs": n,
        "generator_seed": seed,
        "runs_root": "runs/c4_random_sweep",
        "runs": entries,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)
    return entries


def main():
    p = argparse.ArgumentParser(description="Generate C4 random augmentation sweep configs")
    p.add_argument("--n", type=int, default=100, help="Number of runs to generate")
    p.add_argument("--seed", type=int, default=20260620, help="RNG seed for sampling")
    p.add_argument("--jobs-dir", type=Path, default=_JOBS_DIR)
    p.add_argument("--manifest", type=Path, default=_MANIFEST)
    args = p.parse_args()
    entries = generate(args.n, args.seed, args.jobs_dir, args.manifest)
    print(f"Wrote {len(entries)} job YAMLs -> {args.jobs_dir}")
    print(f"Manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
