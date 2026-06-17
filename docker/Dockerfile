# Multi-arch image for tomojepa. ONE Dockerfile, three flavors selected at build
# time by pointing TORCH_CONSTRAINTS at the matching torch wheel set:
#
#   CPU (x86_64 or aarch64):
#     docker build -f docker/Dockerfile \
#       --build-arg TORCH_CONSTRAINTS=constraints/cpu.txt -t tomojepa:cpu .
#
#   x86_64 + NVIDIA GPU (CUDA 12.x):
#     docker build -f docker/Dockerfile \
#       --build-arg TORCH_CONSTRAINTS=constraints/cuda-x86.txt -t tomojepa:cuda .
#
#   DGX Spark (GB10 / Blackwell, aarch64) -- build ON the Spark:
#     docker build -f docker/Dockerfile \
#       --build-arg TORCH_CONSTRAINTS=constraints/spark-cu128.txt -t tomojepa:spark .
#
# The CUDA pip wheels bundle the CUDA userspace, so a slim Python base is enough;
# the host's NVIDIA driver is injected at runtime (Docker: --gpus all; Podman:
# --device nvidia.com/gpu=all via CDI). See docker/README.md.
#
# Build context is the repo ROOT (note the trailing `.`); .dockerignore keeps
# data / checkpoints / venv out of the image.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

# Which torch wheel set to install (see constraints/). Default: CPU.
ARG TORCH_CONSTRAINTS=constraints/cpu.txt
# Which optional dependency group(s) to install with the package.
ARG EXTRAS=retrieval

# libgomp1: OpenMP runtime for torch/numpy/faiss. ca-certificates: pip over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 1) Install the (large, slow, platform-specific) torch build first so it is
#    cached independently of source changes.
COPY constraints/ ./constraints/
RUN pip install --upgrade pip && pip install -r "${TORCH_CONSTRAINTS}"

# 2) Install the package. torch is already present, so its unpinned dependency
#    in pyproject resolves to the build we just installed.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[${EXTRAS}]"

# `tomojepa --help` lists the subcommands; pass real args after the image name,
# e.g. `... tomojepa train-ssl --data_dir /data --epochs 15`.
ENTRYPOINT ["tomojepa"]
CMD ["--help"]
