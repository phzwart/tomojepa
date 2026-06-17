# Container images (Docker / Podman)

One `Dockerfile`, three flavors selected at build time via `TORCH_CONSTRAINTS`.
Everything is OCI, so the same commands work under **Docker** or **Podman**
(swap `docker` for `podman`). Build from the **repo root**.

## Build

| Target | Command |
|--------|---------|
| CPU (x86 or arm64) | `docker build -f docker/Dockerfile --build-arg TORCH_CONSTRAINTS=constraints/cpu.txt -t tomojepa:cpu .` |
| x86 + NVIDIA GPU | `docker build -f docker/Dockerfile --build-arg TORCH_CONSTRAINTS=constraints/cuda-x86.txt -t tomojepa:cuda .` |
| DGX Spark (aarch64, build on the Spark) | `docker build -f docker/Dockerfile --build-arg TORCH_CONSTRAINTS=constraints/spark-cu128.txt -t tomojepa:spark .` |

`--build-arg EXTRAS=retrieval` (default) installs the FAISS/DuckDB/FastAPI/MCP
engine too; use `EXTRAS=""` for a training-only image.

Multi-arch with buildx: `docker buildx build --platform linux/amd64,linux/arm64 ...`
(use a CPU constraints file; CUDA wheels are arch-specific).

## Run training

The entrypoint is the `tomojepa` CLI. Mount data read-only and bind outputs:

```bash
# Docker (single GPU)
docker run --gpus all --rm -it \
    -v /path/to/volumes:/data:ro \
    -v "$PWD/checkpoints:/app/checkpoints" \
    -v "$PWD/outputs:/app/outputs" \
    tomojepa:cuda train-ssl --data_dir /data --pattern 'recon_*.zarr' \
    --epochs 15 --batch_size 8
```

```bash
# Podman (single GPU via CDI; set up with `nvidia-ctk cdi generate`)
podman run --device nvidia.com/gpu=all --rm -it \
    -v /path/to/volumes:/data:ro \
    -v "$PWD/checkpoints:/app/checkpoints" \
    -v "$PWD/outputs:/app/outputs" \
    tomojepa:cuda train-ssl --data_dir /data --epochs 15 --batch_size 8
```

### Multi-GPU (single node)

Launch `torchrun` inside the container with all GPUs visible. Override the
entrypoint so we invoke the module directly:

```bash
docker run --gpus all --rm -it \
    -v /path/to/volumes:/data:ro -v "$PWD/checkpoints:/app/checkpoints" \
    --entrypoint torchrun \
    tomojepa:cuda --nproc_per_node=4 -m tomojepa.ssl.train \
    --data_dir /data --epochs 15 --batch_size 8
```

Use `--ipc=host` (or a larger `--shm-size`) if DataLoader workers hit
shared-memory limits.

## Patch-retrieval service

```bash
docker run --rm -p 8077:8077 \
    -e PATCHDB_DB=/app/runs/patchdb/patchdb.duckdb \
    -v "$PWD/runs:/app/runs" \
    tomojepa:cpu patchdb serve --host 0.0.0.0 --port 8077
```

## Compose

`docker compose -f docker/docker-compose.yml` (or `podman compose`) provides
`train` and `serve` services; see the comments in `docker-compose.yml` for the
`TORCH_CONSTRAINTS` / `DATA_DIR` env vars.
