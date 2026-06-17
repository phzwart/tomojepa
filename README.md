# tomojepa

Self-supervised representation learning for microCT tomography, packaged for any
machine (CPU, x86 NVIDIA, or NVIDIA DGX Spark) and any scale (single GPU or
multi-GPU via `torchrun`). Three subsystems share one ViT backbone and data
loader:

- **`tomojepa.ssl`** — LeJEPA / DINOv3 self-supervised pre-training (SIGReg
  collapse prevention, optional masked-latent-prediction + residual
  factorization) and label-free intrinsic validation.
- **`tomojepa.vitup`** — [ViT-Up](https://arxiv.org/abs/2606.14024) faithful
  feature upsampling: multi-scale distillation training + dense-feature inference.
- **`tomojepa.patchdb`** — FAISS + DuckDB cross-image patch-retrieval engine with
  a CLI, a FastAPI service, and an MCP server.

## Install

`torch` is **not** pinned by the package — install the build that matches your
hardware first (using the matching file in [`constraints/`](constraints/)), then
the package:

| Hardware | Step 1: torch | Step 2: package |
|----------|---------------|-----------------|
| CPU (x86 or arm64) | `pip install -r constraints/cpu.txt` | `pip install -e ".[retrieval]"` |
| x86 + NVIDIA GPU | `pip install -r constraints/cuda-x86.txt` | `pip install -e ".[retrieval]"` |
| DGX Spark (aarch64/Blackwell) | `pip install -r constraints/spark-cu128.txt` | `pip install -e ".[retrieval]"` |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r constraints/cpu.txt          # pick your platform file
pip install -e ".[retrieval]"               # add ,analysis ,wandb ,dev as needed
```

Optional extras: `retrieval` (patchdb engine), `wandb` (logging), `analysis`
(shape probes — scikit-image/scikit-learn), `dev` (pytest, ruff), `all`.

Prefer containers? See [`docker/README.md`](docker/README.md) — one Dockerfile,
CPU / x86-CUDA / Spark flavors, runnable under Docker **or** Podman.

## Quickstart

Everything is reachable through the unified `tomojepa` CLI (`tomojepa --help`).

```bash
# 1) Self-supervised pre-training on a directory of .h5/.zarr volumes
tomojepa train-ssl --data_dir /path/to/volumes --pattern 'recon_*.zarr' \
    --epochs 15 --batch_size 8

# 2) Label-free validation of the resulting checkpoints
tomojepa validate --run_dir runs/my_run --data_dir /path/to/volumes \
    --pattern 'recon_*.zarr' --backend zarr

# 3) ViT-Up feature upsampling: distill from a trained encoder, then visualize
tomojepa train-vitup --teacher_ckpt runs/my_run/ckpt/ckpt_last.pth \
    --data_dir /path/to/volumes --pattern '*.zarr' --backend zarr
tomojepa infer-vitup --ckpt checkpoints_vitup/ckpt_last.pth --data_dir /path/to/volumes

# 4) Patch retrieval: build a collection, query it, serve it
tomojepa patchdb build --collection soil --run_dir runs/my_run --pattern '*.zarr'
tomojepa patchdb query --collection soil --image 0 --bbox '8 8 4 4' --out match.png
tomojepa patchdb serve            # FastAPI on :8077; `patchdb mcp` for the MCP server
```

`tomojepa <cmd> --help` forwards to each tool's full option set. Visualization
and analysis helpers live under `tomojepa viz` (cascade PCA maps, A/B curves,
shape probes).

## Multi-GPU (single node)

Multi-GPU is opt-in via `torchrun`; launch the training module directly (the
process group is set up from `torchrun`'s env vars). Without `torchrun`
everything runs single-process exactly as before.

```bash
# SSL: SIGReg is evaluated on the GLOBAL batch (projections are gathered across
# ranks); the exact full-batch gradient is reconstructed by summing partials.
torchrun --nproc_per_node=4 -m tomojepa.ssl.train \
    --data_dir /path/to/volumes --epochs 15 --batch_size 8

# ViT-Up: standard data-parallel (per-sample distillation loss, gradients averaged)
torchrun --nproc_per_node=4 -m tomojepa.vitup.train \
    --teacher_ckpt runs/my_run/ckpt/ckpt_last.pth --data_dir /path/to/volumes
```

Notes: `--amp_dtype fp16` and `--probe` are not supported with multi-GPU SSL
(use `bf16`/`fp32`); rank 0 owns logging, visualization, and checkpointing.
See [the DDP design notes](src/tomojepa/core/dist.py) for why we synchronize
gradients manually instead of wrapping in `DistributedDataParallel`.

## Repository layout

```
src/tomojepa/
  core/      model, dataset, augmentations, dist (shared by all subsystems)
  ssl/       train.py (LeJEPA/DINOv3) + validate.py
  vitup/     ViT-Up model, train.py, infer.py, tests/
  patchdb/   FAISS+DuckDB retrieval engine (CLI / service / MCP)
  viz/       PCA maps, A/B comparison, shape probes, legacy token-DB tools
  cli.py     unified `tomojepa` entry point
constraints/ per-platform torch wheel sets
docker/      Dockerfile, compose, container docs
docs/        RUN_LOCAL.md (design background), NEXT_MODEL_DESIGN.md
```

More background: [`docs/RUN_LOCAL.md`](docs/RUN_LOCAL.md) (objectives, residual
factorization, DGX Spark notes) and the patchdb
[README](src/tomojepa/patchdb/README.md).

## Data format

Volumes are HDF5 (`.h5`) or Zarr (`.zarr`), each holding a `(D, H, W)` array
under `--dataset_key` (default `reconstruction`); the backend is auto-detected
from the extension. A flat slice index maps to `(file, depth-slice)`. Data is
never baked into images — mount it at runtime.

## Development

```bash
pip install -r constraints/cpu.txt && pip install -e ".[dev]"
ruff check src
pytest                       # ViT-Up unit tests (run on CPU with random weights)
```

## License

MIT — see [LICENSE](LICENSE).
