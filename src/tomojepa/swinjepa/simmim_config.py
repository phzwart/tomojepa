"""Configuration for SwinSimMIM — dual-aug masked latent SimMIM (no pyramid residuals)."""
from dataclasses import dataclass, fields, MISSING
from typing import List, Optional, Tuple
import warnings


@dataclass
class SwinSimMIMConfig:
    # ---- input -------------------------------------------------------------
    backbone_name: str = "swin_tiny_patch4_window7_224"
    backbone_embed_dim: Optional[int] = None
    in_chans: int = 1
    img_size: int = 224
    drop_path_rate: float = 0.1
    use_rope: bool = True
    rope_theta: float = 100.0

    # ---- mask (stage-4 grid) ----------------------------------------------
    mask_ratio: float = 0.45
    mask_mode: str = "random_cell"
    mask_num_blocks: int = 4
    block_scale_range: Tuple[float, float] = (0.1, 0.4)
    use_mask_token: bool = True

    # ---- targets / loss ----------------------------------------------------
    target_norm: str = "ln"
    pred_loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    stage_base_weights: Tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

    # ---- per-stage latent (1x1 lateral) ------------------------------------
    lat_dims: Tuple[int, ...] = (64, 64, 64, 64)

    # ---- SIGReg ------------------------------------------------------------
    beta_sig: Tuple[float, ...] = (0.05, 0.05, 0.05, 0.01)
    sigreg_n_dirs: Tuple[int, ...] = (64, 64, 64, 64)
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_w_mean: float = 0.1
    sigreg_tokens_per_slice: int = 0
    sigreg_token_frac: float = 0.0
    sigreg_min_token_dist: int = 2
    sigreg_queue_len: int = 512
    sigreg_cap_dirs_by_rank: bool = False
    sigreg_min_dirs: int = 16
    sigreg_scale: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    sigreg_rebalance_by_dim: bool = False
    sigreg_pooled: bool = True

    # ---- cos-gated SIGReg on s4 -------------------------------------------
    s4_cosine_level: float = 0.0
    s4_cosine_ema_decay: float = 0.9
    s4_sigreg_fallback_progress: float = 0.30
    s4_sigreg_ramp_progress: float = 0.10
    sigreg_cos_gate_all_stages: bool = False

    # ---- foreground mask ---------------------------------------------------
    foreground_mask: bool = False
    fg_mode: str = "std"
    fg_std_thresh: float = 0.05
    fg_circle_diameter_frac: float = 1.0
    fg_coverage: float = 0.01
    fg_key: str = ""

    # ---- optimization ------------------------------------------------------
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_pct: float = 0.10
    grad_clip: float = 3.0
    max_iters: int = 0
    freeze_after_epoch: Tuple[int, ...] = (0, 0, 0, 0)

    @property
    def num_stages(self) -> int:
        return 4

    def stage_scale(self, out_chans: List[int]) -> List[float]:
        sc = list(self.sigreg_scale)
        if self.sigreg_rebalance_by_dim:
            sc = [s / c for s, c in zip(sc, out_chans)]
        return sc

    def __post_init__(self):
        if len(self.lat_dims) == 1:
            self.lat_dims = self.lat_dims * self.num_stages
        if len(self.lat_dims) != self.num_stages:
            raise ValueError(f"lat_dims must have {self.num_stages} entries, got {self.lat_dims}")
        if any(d <= 0 for d in self.lat_dims):
            raise ValueError(f"lat_dims entries must be positive, got {self.lat_dims}")
        if self.backbone_embed_dim is not None and self.backbone_embed_dim <= 0:
            raise ValueError(f"backbone_embed_dim must be positive, got {self.backbone_embed_dim}")
        if len(self.beta_sig) == 1:
            self.beta_sig = self.beta_sig * self.num_stages
        if len(self.beta_sig) != self.num_stages:
            raise ValueError(f"beta_sig must have {self.num_stages} entries, got {self.beta_sig}")
        if len(self.stage_base_weights) == 1:
            self.stage_base_weights = self.stage_base_weights * self.num_stages
        if len(self.stage_base_weights) != self.num_stages:
            raise ValueError(
                f"stage_base_weights must have {self.num_stages} entries, "
                f"got {self.stage_base_weights}")
        if len(self.sigreg_n_dirs) == 1:
            self.sigreg_n_dirs = self.sigreg_n_dirs * self.num_stages
        if len(self.sigreg_n_dirs) != self.num_stages:
            raise ValueError(
                f"sigreg_n_dirs must have {self.num_stages} entries, got {self.sigreg_n_dirs}")
        clamped = tuple(min(d, c) for d, c in zip(self.sigreg_n_dirs, self.lat_dims))
        if clamped != self.sigreg_n_dirs:
            warnings.warn(
                f"sigreg_n_dirs {self.sigreg_n_dirs} exceeds lat_dims {self.lat_dims}; "
                f"clamping to {clamped}", stacklevel=2)
            self.sigreg_n_dirs = clamped
        if len(self.sigreg_scale) == 1:
            self.sigreg_scale = self.sigreg_scale * self.num_stages
        if len(self.freeze_after_epoch) == 1:
            self.freeze_after_epoch = self.freeze_after_epoch * self.num_stages
        if not (0.0 <= self.s4_cosine_level <= 1.0):
            raise ValueError(f"s4_cosine_level must be in [0, 1], got {self.s4_cosine_level}")


_TUPLE_ELEM_TYPE = {
    "block_scale_range": float,
    "stage_base_weights": float,
    "freeze_after_epoch": int,
    "sigreg_n_dirs": int,
    "sigreg_scale": float,
    "beta_sig": float,
    "lat_dims": int,
}


def add_argparse_args(parser):
    for f in fields(SwinSimMIMConfig):
        default = f.default if f.default is not MISSING else None
        flag = "--" + f.name
        if f.name == "backbone_embed_dim":
            parser.add_argument(flag, type=int, default=default,
                                help="timm embed_dim override; stages=[e,2e,4e,8e]")
            continue
        if f.name in _TUPLE_ELEM_TYPE:
            parser.add_argument(flag, type=_TUPLE_ELEM_TYPE[f.name], nargs="+",
                                default=list(default) if default is not None else None)
        elif f.type is bool or isinstance(default, bool):
            parser.add_argument(flag, dest=f.name, action="store_true", default=default)
            parser.add_argument("--no_" + f.name, dest=f.name, action="store_false")
        elif f.type is int:
            parser.add_argument(flag, type=int, default=default)
        elif f.type is float:
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, type=str, default=default)
    return parser


def from_args(args) -> SwinSimMIMConfig:
    kwargs = {}
    for f in fields(SwinSimMIMConfig):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if f.name in _TUPLE_ELEM_TYPE and val is not None:
            val = tuple(val)
        kwargs[f.name] = val
    return SwinSimMIMConfig(**kwargs)
