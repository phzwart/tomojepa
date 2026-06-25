"""BandedViT JEPA training configuration."""
from __future__ import annotations

from dataclasses import dataclass, fields, MISSING
from typing import Tuple


@dataclass
class BandedJEPAConfig:
    # ---- encoder (ViT-S defaults) -----------------------------------------
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_register_tokens: int = 0
    use_cls_token: bool = True
    rope_theta: float = 100.0

    # ---- distance bands ---------------------------------------------------
    band_K: int = 10
    band_keep_self: bool = True
    # independent | balanced | cyclic | balanced_no_adjacent
    band_sample_mode: str = "balanced"
    band_weights: Tuple[float, ...] = ()  # near/mid/far fractions; empty = equal split
    # Cyclic band schedule: M0 steps full attention, then M1 steps banded (repeat).
    # Both must be > 0 to enable; if either is 0, bands are always on.
    band_m0: int = 0
    band_m1: int = 0

    # ---- foreground FOV mask (training-only) ------------------------------
    foreground_mask: bool = False
    fg_mode: str = "std"                        # std | circle (geometric FOV disk)
    fg_std_thresh: float = 0.05
    fg_circle_diameter_frac: float = 1.0        # circle: diameter = frac * image width
    fg_coverage: float = 0.01                   # min FG fraction per patch footprint
    fg_key: str = ""                            # optional precomputed mask array key

    # ---- SIGReg (per transformer block tap) -------------------------------
    beta_sig: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    sigreg_blocks: Tuple[int, ...] = (-1,)
    sigreg_n_dirs: int = 256
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_w_mean: float = 0.1
    sigreg_token_cap: int = 4096
    sigreg_queue_len: int = 0
    sigreg_min_token_dist: int = 0   # Chebyshev min separation on patch grid (0 = off)
    sigreg_n_tokens_per_slice: int = 32

    # ---- masked latent prediction (Phase 2) --------------------------------
    pred_enabled: bool = False
    mask_ratio: float = 0.45
    mask_mode: str = "random_cell"
    mask_num_blocks: int = 4
    block_scale_range: Tuple[float, float] = (0.1, 0.4)
    pred_loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    pred_block: int = -1

    # ---- optimization -----------------------------------------------------
    epochs: int = 100
    batch_size: int = 16
    lr: float = 2.0e-4
    weight_decay: float = 0.05
    warmup_pct: float = 0.05
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    max_iters: int = 0

    def resolved_sigreg_blocks(self) -> Tuple[int, ...]:
        depth = self.depth
        out = []
        for b in self.sigreg_blocks:
            out.append((b if b >= 0 else depth + b) % depth)
        return tuple(sorted(set(out)))

    def beta_for_block(self, block_idx: int) -> float:
        betas = self.beta_sig
        if len(betas) == 1:
            return float(betas[0])
        if block_idx < len(betas):
            return float(betas[block_idx])
        return 0.0


_TUPLE_ELEM_TYPE = {
    "beta_sig": float,
    "sigreg_blocks": int,
    "block_scale_range": float,
    "band_weights": float,
}


def add_argparse_args(parser):
    for f in fields(BandedJEPAConfig):
        default = f.default if f.default is not MISSING else None
        flag = "--" + f.name
        if f.name in _TUPLE_ELEM_TYPE:
            parser.add_argument(
                flag, type=_TUPLE_ELEM_TYPE[f.name], nargs="+",
                default=list(default) if default is not None else None,
            )
        elif f.type is bool or isinstance(default, bool):
            if default:
                parser.add_argument(flag, action="store_true", default=True)
                parser.add_argument(
                    f"--no_{f.name}", dest=f.name, action="store_false",
                )
            else:
                parser.add_argument(flag, action="store_true", default=False)
        elif f.type is int:
            parser.add_argument(flag, type=int, default=default)
        elif f.type is float:
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, type=str, default=default)
    return parser


def from_args(args) -> BandedJEPAConfig:
    kwargs = {}
    for f in fields(BandedJEPAConfig):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if f.name in _TUPLE_ELEM_TYPE and val is not None:
            val = tuple(val)
        kwargs[f.name] = val
    return BandedJEPAConfig(**kwargs)
