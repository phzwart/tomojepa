"""Shared, optional multi-GPU primitives for ``torch.distributed`` (torchrun).

Launched with ``torchrun`` (which sets ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``)::

    torchrun --nproc_per_node=4 -m tomojepa.ssl.train   --data_dir /data ...
    torchrun --nproc_per_node=4 -m tomojepa.vitup.train --teacher_ckpt ...

Two sync strategies are provided, neither of which wraps the model in
``DistributedDataParallel`` (LeJEPA's SIGReg is a distribution-level statistic
and the GradCache path uses a hand-written two-pass backward, both of which
fight DDP's autograd-hook sync):

- :func:`all_gather_cat` + :func:`all_reduce_grads_` (SUM) -- used by SSL to
  evaluate the LeJEPA loss on the *global* batch and reconstruct the exact
  full-batch gradient.
- :func:`average_grads_` -- standard data-parallel averaging for a plain
  per-sample mean loss (used by ViT-Up distillation).

Every helper is a safe no-op when not launched under torchrun (``WORLD_SIZE`` 1).
"""
import os

import torch
import torch.distributed as dist


def _initialized():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if _initialized() else 1


def get_rank():
    return dist.get_rank() if _initialized() else 0


def is_distributed():
    return world_size() > 1


def is_main():
    return get_rank() == 0


def loss_scale():
    """Multiplier for per-sample (local-shard) loss terms before SUM all-reduce."""
    return 1.0 / world_size()


def init_distributed():
    """Set up the process group from torchrun env vars (if any).

    Returns ``(device, local_rank)``. When not launched under torchrun
    (``WORLD_SIZE`` <= 1), this is a no-op returning the usual single-process
    device.
    """
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return device, local_rank


def cleanup():
    if _initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def sync_rng(seed: int):
    """Seed CPU+CUDA RNG identically on all ranks.

    Call right before a stochastic loss (e.g. SIGReg's random sketch directions)
    so the gathered loss is one coherent function across ranks.
    """
    if not is_distributed():
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _GatherCat(torch.autograd.Function):
    """All-gather along a dim; backward returns only this rank's own slice.

    Cross-rank gradient summation is intentionally deferred to
    :func:`all_reduce_grads_` on the parameter gradients (see module docstring).
    """

    @staticmethod
    def forward(ctx, dim, x):
        ctx.dim = dim
        ctx.rank = dist.get_rank()
        ws = dist.get_world_size()
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(ws)]
        dist.all_gather(parts, x)
        parts[ctx.rank] = x          # keep this rank's tensor in the autograd graph
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        ws = dist.get_world_size()
        chunks = grad_output.chunk(ws, dim=ctx.dim)
        return None, chunks[ctx.rank]


def all_gather_cat(x, dim=0):
    """Differentiably gather ``x`` from all ranks and concatenate along ``dim``.

    No-op (returns ``x``) when not distributed. Requires equal shapes across
    ranks (guaranteed by ``DistributedSampler(drop_last=True)`` + constant batch).
    """
    if not is_distributed():
        return x
    return _GatherCat.apply(dim, x)


@torch.no_grad()
def all_reduce_grads_(params):
    """In-place SUM all-reduce of parameter ``.grad`` across ranks."""
    if not is_distributed():
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


@torch.no_grad()
def average_grads_(params):
    """In-place mean all-reduce of parameter ``.grad`` across ranks.

    Standard data-parallel gradient averaging for a plain per-sample mean loss
    (SUM then divide by world size; avoids relying on the AVG op, which the gloo
    CPU backend does not implement).
    """
    if not is_distributed():
        return
    ws = world_size()
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= ws
