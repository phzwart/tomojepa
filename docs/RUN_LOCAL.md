# LeJEPA — local single-GPU training (DGX Spark)

Refactored from the Perlmutter/Slurm version to run as a single process on one GPU.
No Hydra, no DDP, no Slurm, no file-renaming.

## Setup

Tested on a DGX Spark (GB10 / Blackwell, aarch64, Ubuntu 24.04, Python 3.12). The
`requirements.txt` pins the CUDA 12.8 aarch64 torch build with Blackwell kernels.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The torch/torchvision `+cu128` wheels are pulled from PyTorch's index (declared via
`--extra-index-url` in `requirements.txt`); everything else comes from PyPI. These
wheels are arm64 + CUDA 12.8 specific — on other hardware, adjust the pins/index.

### Docker

A `Dockerfile` is included for packaging. The cu128 wheels bundle the CUDA userspace,
so it builds from a slim Python base and relies on the host driver at runtime via the
NVIDIA Container Toolkit. Build and run **on the Spark (aarch64)**:

```bash
docker build -t lejepa-tomo .
docker run --gpus all --rm -it \
    -v /path/to/data:/data \
    -v "$PWD/checkpoints:/app/checkpoints" \
    -v "$PWD/outputs:/app/outputs" \
    lejepa-tomo --data_dir /data --pattern 'recon_*.zarr' --epochs 15 --batch_size 8
```

## Run

```bash
python main.py --data_dir /path/to/your/h5 --epochs 15 --batch_size 8
```

The loader expects `recon_*.h5` files, each with a `reconstruction` dataset of shape
`(D, H, W)`. Override with `--pattern` / `--dataset_key` if yours differ.

Zarr is supported too: point `--pattern` at your stores (e.g. `recon_*.zarr`) and the
backend is auto-detected from the extension. Force it with `--backend {h5,zarr}` if
your files use a non-standard suffix. A `.zarr` store can be a group holding the array
under `--dataset_key`, or a bare array. Needs the `zarr` package installed.

Useful flags:
- `--augment {tomo,tomo2}`  pick the augmentation pipeline (default `tomo2`; no renaming)
- `--in_chans 1`            grayscale tomography (use `3` for an RGB baseline)
- `--global_views N`        wide-area views per sample (default 2, scale `--global_scale`)
- `--local_views N`         aggressive zoomed-in views per sample (default 2, scale `--local_scale`)
- `--global_scale MIN MAX`  crop area band for global views (default `0.4 1.0`)
- `--local_scale MIN MAX`   crop area band for local views (default `0.1 0.4`)
- `--batch_size`            per-step batch; effective forward batch is `batch_size * (global+local)`
- `--accum_steps N`         accumulate N microbatches per optimizer step via GradCache;
                            SIGReg is computed over the full `batch_size * N` effective
                            batch at single-microbatch memory (bf16/fp32 only)
- `--amp_dtype bf16`        default; Blackwell-friendly, no GradScaler needed
- `--mim_weight W`          enable masked latent prediction (MAE); `0` = off (default)
- `--residual_local`        route LeJEPA invariance+SIGReg through the residual (see below)
- `--mask_ratio`            fraction of patch tokens masked for MIM (default `0.5`)
- `--mask_blocks`           number of rectangular blocks for block masking (default `4`)
- `--no_mim_target_norm`    disable per-token layer-norm of the MAE target (on by default)
- `--indep_weight W`        optional decorrelation penalty (needs `--residual_local`); `0` = off
- `--wandb`                 opt in to W&B (off by default — runs with no account)
- `--probe`                 online linear probe; OFF by default (see note below)

Checkpoints land in `checkpoints/` (`ckpt_last.pth` + per-epoch). Resume is automatic.
PCA token visualizations are written to `outputs/` as PNGs (and to W&B if `--wandb`).

## Masked latent prediction + residual factorization

`--mim_weight > 0` adds a Masked Image Modeling objective: a learnable `[MASK]`
token replaces a block-masked subset of patch embeddings, the (same, no-EMA)
encoder produces a context field, and a small predictor head maps it to a *smooth*
latent field `C` that is matched (smooth-L1, stop-grad normalized target) to the
full-image patch latents at the masked positions. The predictor + mask token live
in a **separate** module, so the encoder `state_dict()` is unchanged and still loads
into the eval notebook.

Two ways to combine it with LeJEPA:

- **Additive** (`--mim_weight W` alone): `total = lejepa(proj(emb)) + W * mae`. The
  invariance/SIGReg objective is unchanged; MAE is just an extra term.
- **Residual** (`--mim_weight W --residual_local`): factorize the representation into
  a smooth, globally-interpolated context `C` (trained by MAE) and an
  augmentation-invariant **residual** `R = T - stopgrad(C)`. LeJEPA's invariance +
  SIGReg are then applied to `z_local = proj(mean_patches(R))`, so the invariant
  features sit *on top of* the MAE predictor and are (structurally) conditionally
  independent of — i.e. complementary to — the predicted context. `--indep_weight`
  optionally adds a cross-covariance penalty between `z_local` and pooled `C` to
  reinforce that independence.

Cost note: MIM adds **2 backbone forwards per view** (full `T` + masked context) on
top of the `V` LeJEPA forwards (residual mode runs them on all `V` views; additive
mode only on the global views). It is fully GradCache-compatible: with
`--accum_steps > 1` the residual feeds the cached SIGReg gradient while the
per-sample MAE term is back-propagated per microbatch (scaled by `1/accum`).

## DGX Spark notes

- **One GPU, ~128 GB unified memory.** All the distributed machinery from the
  original (NCCL / DistributedSampler / `torchrun --nproc_per_node=4`) is gone.
- **PyTorch build matters.** GB10 is Blackwell (sm_121); you need a torch built for
  CUDA 12.8+/aarch64 (recent stable, a nightly, or NVIDIA's NGC/DGX-OS PyTorch
  container). Stock x86 wheels won't have the right kernels.
- **bf16 is the default** and is the right call on Blackwell. `fp16` (+GradScaler)
  and `fp32` are available via `--amp_dtype` if needed.
- Unified memory means you can push `--batch_size` higher than a 40/80 GB discrete
  GPU, but it's compute-bound, so raise it gradually and watch throughput.

## Why the probe is off by default

The online probe trains on the loader's labels, which are all dummy `0`s — so it
just learns to predict class 0 and tells you nothing. It's also `.detach()`ed from
the backbone, so leaving it off changes the representation not at all and only saves
compute. Turn it on with `--probe` only once you wire real labels into the dataset.

## Kept deliberately

`DINOv3ViTEncoder` still builds the backbone with `num_classes=384` (an extra linear
head on the 384-d ViT-S feature). That matches the architecture the released
checkpoints were trained with, so a model you train here loads cleanly into
`evaluate_lejepa_segmentation.ipynb`.
