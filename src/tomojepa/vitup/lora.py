"""Low-rank adaptation (LoRA) for backbone linear and patch-embed conv layers.

For a frozen projection ``W`` the adapted output is::

    W_adapted(x) = W(x) + (alpha / r) * B(A(dropout(x)))

with ``A, B`` low-rank (rank ``r``) and ``A`` zero-init's partner ``B`` so the
adapter starts as a no-op. Implemented as wrapper modules that hold a reference
to the (frozen) base layer so the backbone ``state_dict`` keys are preserved
under their original names; only ``lora_A`` / ``lora_B`` are trainable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with an additive low-rank adapter."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)
        return out + self.scaling * lora


class LoRAConv2d(nn.Module):
    """Wraps a frozen ``nn.Conv2d`` (e.g. the patch-embed conv) with LoRA.

    The down-projection ``A`` reuses the base kernel size/stride/padding so it
    operates on the same receptive field; the up-projection ``B`` is a 1x1 conv.
    """

    def __init__(self, base: nn.Conv2d, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Conv2d(
            base.in_channels, r, kernel_size=base.kernel_size,
            stride=base.stride, padding=base.padding, dilation=base.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.drop(x)))


def _get_submodule(root: nn.Module, dotted: str):
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def replace_module(root: nn.Module, dotted: str, new_module: nn.Module):
    """Swap ``root.<dotted>`` for ``new_module`` in place."""
    parent, leaf = _get_submodule(root, dotted)
    setattr(parent, leaf, new_module)


def wrap_with_lora(root: nn.Module, dotted: str, r: int, alpha: float,
                   dropout: float) -> nn.Module:
    """Replace the layer at ``dotted`` with its LoRA-wrapped version.

    Dispatches on the base layer type (``nn.Linear`` -> :class:`LoRALinear`,
    ``nn.Conv2d`` -> :class:`LoRAConv2d`). Returns the new wrapper.
    """
    parent, leaf = _get_submodule(root, dotted)
    base = getattr(parent, leaf)
    if isinstance(base, nn.Linear):
        wrapped = LoRALinear(base, r, alpha, dropout)
    elif isinstance(base, nn.Conv2d):
        wrapped = LoRAConv2d(base, r, alpha, dropout)
    else:
        raise TypeError(f"cannot LoRA-wrap {dotted!r} of type {type(base)}")
    setattr(parent, leaf, wrapped)
    return wrapped


def lora_parameters(root: nn.Module):
    """Yield only the LoRA adapter parameters under ``root``."""
    for name, p in root.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield p


def freeze_non_lora(root: nn.Module):
    """Set ``requires_grad`` True only for LoRA params, False for everything else."""
    for name, p in root.named_parameters():
        p.requires_grad_("lora_A" in name or "lora_B" in name)
