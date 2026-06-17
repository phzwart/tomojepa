"""ViT-Up configuration (paper defaults, fully overridable).

A single dataclass holds every documented hyperparameter from the paper. The
defaults track this repo's grayscale microCT DINOv3 backbone (``p=16``,
``C=384``, ``L=12``, single-channel, trained at ``img_size=512``); where the
paper's ImageNet defaults differ they are noted inline as the alternative.

``add_argparse_args`` / ``from_args`` wire the dataclass into the repo's
``argparse`` convention without introducing a parallel config system.
"""
from dataclasses import dataclass, fields, MISSING
from typing import List, Tuple


@dataclass
class ViTUpConfig:
    # ---- backbone (discovered in Phase 0; see BackboneAdapter) -------------
    backbone_name: str = "vit_small_patch16_dinov3"
    input_channels: int = 1                 # grayscale microCT (paper: 3)
    backbone_img_size: int = 512            # native train resolution of teacher

    # ---- vit_up architecture ----------------------------------------------
    num_blocks: int = 6                      # T
    # l[t] = 2t -> backbone layers {2,4,6,8,10,12} (every second layer skipped)
    layer_indices: Tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    internal_dim: int = 0                    # 0 -> use backbone C; double for larger backbones
    query_embed_grid: int = 224              # high-res patch-token grid side (tokens)
    attention_window: int = 7                # cross-window neighborhood (token units)
    num_heads: int = 6                       # internal_dim / head_dim (384/64 = 6)
    featx_posenc_dim: int = 64               # sinusoidal encoding dim for Delta x
    mlp_ratio: float = 4.0
    rope_theta: float = 100.0                # base for ViT-Up's continuous 2D RoPE
    query_chunk_size: int = 4096             # inference/training memory control

    # ---- LoRA backbone adaptation -----------------------------------------
    lora_rank: int = 16                      # r
    lora_alpha: float = 32.0                 # alpha
    lora_dropout: float = 0.05
    # logical targets; BackboneAdapter maps these onto the timm module names
    # (fused attn.qkv covers Q/K/V; attn.proj is O; patch_embed is the conv).
    lora_targets: Tuple[str, ...] = ("patch_embed", "attn.qkv", "attn.proj")

    # ---- multi-scale distillation training --------------------------------
    # paper: [224, 448, 896] -> token grids [14, 28, 56] at p=16; here defaults
    # match the teacher's 512 training resolution -> grids [8, 16, 32].
    teacher_resolutions: Tuple[int, ...] = (128, 256, 512)
    student_canvas: int = 512                # paper: 448
    student_scale: Tuple[float, float] = (0.1, 1.0)
    query_grid: int = 32                     # finest query grid (= finest teacher grid); paper: 56

    lambda_l2: float = 1.0
    lambda_cos: float = 1.0
    lambda_rel: float = 1.0
    rel_temperature: float = 0.1             # tau (paper leaves unpinned)
    eps: float = 1e-6

    # ---- optimization ------------------------------------------------------
    epochs: int = 1
    batch_size: int = 24
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.05                # short warmup before cosine
    max_iters: int = 0                       # 0 -> run full epochs; >0 caps iters (ablation)

    def resolved_internal_dim(self, backbone_dim: int) -> int:
        """Internal dim, defaulting to the backbone width when unset (0)."""
        return self.internal_dim if self.internal_dim > 0 else backbone_dim

    def teacher_token_grids(self, patch_size: int) -> List[int]:
        """Teacher token-grid sizes ``N`` derived from ``S`` and patch size."""
        return [r // patch_size for r in self.teacher_resolutions]


# Tuple-valued fields take ``nargs`` from argparse; map element types here.
_TUPLE_ELEM_TYPE = {
    "layer_indices": int,
    "lora_targets": str,
    "teacher_resolutions": int,
    "student_scale": float,
}


def add_argparse_args(parser):
    """Register every :class:`ViTUpConfig` field as an ``argparse`` option."""
    for f in fields(ViTUpConfig):
        default = f.default if f.default is not MISSING else None
        flag = "--" + f.name
        if f.name in _TUPLE_ELEM_TYPE:
            parser.add_argument(flag, type=_TUPLE_ELEM_TYPE[f.name], nargs="+",
                                default=list(default) if default is not None else None)
        elif f.type is bool or isinstance(default, bool):
            parser.add_argument(flag, action="store_true", default=default)
        elif f.type is int:
            parser.add_argument(flag, type=int, default=default)
        elif f.type is float:
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, type=str, default=default)
    return parser


def from_args(args) -> ViTUpConfig:
    """Build a :class:`ViTUpConfig` from a parsed ``argparse`` namespace."""
    kwargs = {}
    for f in fields(ViTUpConfig):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if f.name in _TUPLE_ELEM_TYPE and val is not None:
            val = tuple(val)
        kwargs[f.name] = val
    return ViTUpConfig(**kwargs)
