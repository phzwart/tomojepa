"""Swin multi-scale latent-JEPA configuration (design defaults, fully overridable).

A single dataclass holds every documented hyperparameter. Defaults track the
design's Swin-Tiny geometry and the repo's tile-based, single-channel microCT
data. ``add_argparse_args`` / ``from_args`` wire the dataclass into the repo's
``argparse`` convention (the same mechanism :mod:`tomojepa.vitup.config` uses),
so no parallel config system is introduced.

Per-stage knobs that the design specifies as dicts (``n_dirs``, ``scale_s``,
base loss weights) are carried as length-``S`` tuples ordered ``s1..s4`` so they
map cleanly onto ``argparse`` ``nargs``.
"""
from dataclasses import dataclass, fields, MISSING
from typing import List, Optional, Tuple
import warnings


@dataclass
class SwinMSJEPAConfig:
    # ---- input -------------------------------------------------------------
    backbone_name: str = "swin_tiny_patch4_window7_224"
    backbone_embed_dim: Optional[int] = None    # timm embed_dim override; stages=[e,2e,4e,8e]
    in_chans: int = 1                         # grayscale microCT
    img_size: int = 224                       # set from repo tile size (fallback 224)
    drop_path_rate: float = 0.1
    use_rope: bool = True                       # 2D RoPE on backbone window attention
    rope_theta: float = 100.0                   # RoPE frequency base

    # ---- mask (defined on the stage-4 grid) -------------------------------
    mask_ratio: float = 0.75
    mask_mode: str = "random_cell"            # random_cell | block
    mask_num_blocks: int = 4                  # block mode
    block_scale_range: Tuple[float, float] = (0.1, 0.4)

    # ---- cross-scale predictor --------------------------------------------
    pred_dim: int = 384                       # D_pred
    pred_depth: int = 4                       # N_pred decoder blocks
    pred_heads: int = 6
    pred_mlp_ratio: float = 4.0
    predictor_cross_scale: bool = True        # attend across all stages
    predictor_enabled: bool = True            # False -> pure data2vec path (cross_attn only)
    coarse_mim_mode: str = "integrated"       # integrated | cross_attn | conv
    fusion_depth: int = 1                     # integrated: cross-scale fusion blocks
    fusion_heads: int = 4
    fusion_mlp_ratio: float = 4.0
    fusion_cross_scale: bool = True           # attend to same + coarser stages
    dual_view: bool = True                    # student view 0 / teacher view 1

    # ---- target construction ----------------------------------------------
    target_norm: str = "ln"                   # ln | whiten | none

    # ---- prediction loss + curriculum -------------------------------------
    pred_loss: str = "smooth_l1"              # smooth_l1 | mse
    smooth_l1_beta: float = 1.0
    warmup_frac: float = 0.25                  # ramp length (fraction of total steps)
    fine_min_w: float = 0.1                    # min weight for not-yet-active stages
    # per-stage base loss weights (s1..s4); coarse/fine split editable here.
    stage_base_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # Curriculum: fine_in = legacy (s3,s4 first, s1,s2 ramp in);
    # coarse_in = s4 (lowest res) first, then coarse_ramp_stages stir in.
    stage_curriculum: str = "coarse_in"        # fine_in | coarse_in
    coarse_ramp_stages: Tuple[int, ...] = (3, 2, 1)   # 1-based, after s4
    # Per-stage freeze epoch (s1..s4, 0=never). When ``epoch >= N``, that stage
    # stops receiving loss (pred / SIGReg / coarse MAE), its lateral conv is
    # frozen, and student/target latents are detached on that stage. Mirror
    # coarse_in: e.g. ``(0, 0, 0, 5)`` locks coarse after epoch 5.
    freeze_after_epoch: Tuple[int, ...] = (0, 0, 0, 0)
    # 1-based stage ids that are *fine* (fine_in mode only).
    fine_stages: Tuple[int, ...] = (1, 2)

    # ---- pyramid residual (default) vs legacy full-latent JEPA ------------
    legacy_jepa: bool = False                 # True -> pre-pyramid token SIGReg + E targets
    sigreg_pooled: bool = True                 # pyramid: enable cross-batch SIGReg FIFO queue

    # ---- SIGReg (per stage) -----------------------------------------------
    # Per-stage SIGReg weight (s1..s4). Legacy: scales token SIGReg at each E_s.
    # Pyramid: s4 regularizes the coarse base (``sigreg_s4_on``); s3..s1 weight
    # hierarchical residuals R3..R1. NOTE: beta_sig scale differs between legacy
    # (per-token) and pyramid (per-slice) SIGReg paths -- not comparable across modes.
    sigreg_s4_on: str = "e4"                  # e4 | c4 -- pyramid coarse SIGReg target
    beta_sig: Tuple[float, ...] = (0.05, 0.05, 0.05, 0.01)
    sigreg_n_dirs: Tuple[int, ...] = (64, 64, 64, 64)  # bounded by lat_dims[s]
    sigreg_knots: int = 17                      # CF quadrature knots (repo SIGReg)
    sigreg_t_max: float = 3.0                   # CF quadrature range (repo SIGReg)
    sigreg_w_mean: float = 0.1                  # light explicit mean penalty
    sigreg_n_tokens_cap: int = 4096             # legacy token cap (0 = use all)
    sigreg_tokens_per_slice: int = 32           # pyramid: FG tokens per image on s4 (0 = all s4 FG)
    sigreg_token_frac: float = 0.0              # >0: subsample this frac of s4 grid per slice (overrides tok=0)
    sigreg_min_token_dist: int = 2              # min Chebyshev grid dist (0 = random)
    sigreg_queue_len: int = 512                 # FIFO queue length (slices, not pooled rows)
    sigreg_cap_dirs_by_rank: bool = False        # cap n_dirs by batch effective rank
    sigreg_min_dirs: int = 16                   # floor on projection directions
    sigreg_scale: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)   # scale_s
    sigreg_rebalance_by_dim: bool = False       # multiply scale_s by 1/C_s

    # ---- cos-gated SIGReg on s4 (pyramid cross-attn MAE) ------------------
    # When s4_cosine_level > 0, s4 beta_sig is scaled by a one-way gate: MAE-only
    # until mae/cos EMA exceeds the level (or fallback progress), then ramped in.
    s4_cosine_level: float = 0.0                # 0 = disabled; e.g. 0.60
    s4_cosine_ema_decay: float = 0.9
    s4_sigreg_fallback_progress: float = 0.30   # latch SIGReg if cos never hits level
    s4_sigreg_ramp_progress: float = 0.10     # beta scale 0->1 over this progress span

    # ---- pyramid residual parent definition --------------------------------
    strict_laplacian: bool = False               # pool(child) parents when True

    # ---- foreground FOV mask (training-only) ------------------------------
    foreground_mask: bool = False
    fg_mode: str = "std"                        # std | circle (geometric FOV disk)
    fg_std_thresh: float = 0.05                 # std mode: intensity-variation threshold
    fg_circle_diameter_frac: float = 1.0        # circle: diameter = frac * image width
    fg_coverage: float = 0.01                   # min FG fraction per token footprint
    fg_key: str = ""                            # optional precomputed mask array key

    # ---- per-stage JEPA latent (1x1 lateral projections) ------------------
    # Length S tuple (s1..s4): backbone C_s -> lat_dims[s]. Per-stage widths
    # are allowed in pyramid mode; cross-scale residual parents are aligned via
    # 1x1 convs when adjacent stage widths differ.
    lat_dims: Tuple[int, ...] = (64, 64, 64, 64)

    # ---- optimization ------------------------------------------------------
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1.5e-4
    weight_decay: float = 0.05                 # AdamW decoupled weight decay
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_pct: float = 0.10
    grad_clip: float = 3.0
    max_iters: int = 0                          # 0 -> full epochs; >0 caps iters

    # -- derived helpers -----------------------------------------------------
    @property
    def num_stages(self) -> int:
        return 4

    def stage_scale(self, out_chans: List[int]) -> List[float]:
        """Resolved per-stage SIGReg scale (optionally 1/C_s rebalanced)."""
        sc = list(self.sigreg_scale)
        if self.sigreg_rebalance_by_dim:
            sc = [s / c for s, c in zip(sc, out_chans)]
        return sc

    def __post_init__(self):
        if len(self.lat_dims) == 1:
            self.lat_dims = self.lat_dims * self.num_stages
        if len(self.lat_dims) != self.num_stages:
            raise ValueError(
                f"lat_dims must have {self.num_stages} entries (s1..s4), "
                f"got {len(self.lat_dims)}: {self.lat_dims}")
        if any(d <= 0 for d in self.lat_dims):
            raise ValueError(f"lat_dims entries must be positive, got {self.lat_dims}")
        if self.backbone_embed_dim is not None and self.backbone_embed_dim <= 0:
            raise ValueError(
                f"backbone_embed_dim must be positive when set, got {self.backbone_embed_dim}")
        if self.coarse_mim_mode == "cross_attn":
            if self.pred_dim > max(self.lat_dims) * 4:
                warnings.warn(
                    f"pred_dim={self.pred_dim} is much larger than max(lat_dims)="
                    f"{max(self.lat_dims)}; cross-attn predictor memory scales with pred_dim",
                    stacklevel=2)
            if self.pred_dim % self.pred_heads != 0:
                raise ValueError(
                    f"pred_dim={self.pred_dim} must be divisible by "
                    f"pred_heads={self.pred_heads}")
            if (self.pred_dim // self.pred_heads) % 4 != 0:
                raise ValueError(
                    f"pred_dim / pred_heads = {self.pred_dim // self.pred_heads} must be "
                    f"divisible by 4 for RoPE (pred_dim={self.pred_dim}, "
                    f"pred_heads={self.pred_heads})")
        if self.coarse_mim_mode == "integrated":
            if not self.dual_view:
                raise ValueError("coarse_mim_mode='integrated' requires dual_view=True")
            for c in self.lat_dims:
                if c % self.fusion_heads != 0:
                    raise ValueError(
                        f"each lat_dims entry must be divisible by fusion_heads="
                        f"{self.fusion_heads}, got lat_dims={self.lat_dims}")
                if self.use_rope and (c // self.fusion_heads) % 4 != 0:
                    raise ValueError(
                        f"lat_dim {c} / fusion_heads {self.fusion_heads} must yield "
                        f"head_dim divisible by 4 for RoPE")
            if self.fusion_depth < 1:
                raise ValueError(f"fusion_depth must be >= 1, got {self.fusion_depth}")
        if len(self.beta_sig) == 1:
            self.beta_sig = self.beta_sig * self.num_stages
        if len(self.beta_sig) != self.num_stages:
            raise ValueError(
                f"beta_sig must have {self.num_stages} entries (s1..s4), "
                f"got {len(self.beta_sig)}: {self.beta_sig}")
        if len(self.sigreg_n_dirs) == 1:
            self.sigreg_n_dirs = self.sigreg_n_dirs * self.num_stages
        if len(self.sigreg_n_dirs) != self.num_stages:
            raise ValueError(
                f"sigreg_n_dirs must have {self.num_stages} entries (s1..s4), "
                f"got {len(self.sigreg_n_dirs)}: {self.sigreg_n_dirs}")
        clamped = tuple(min(d, c) for d, c in zip(self.sigreg_n_dirs, self.lat_dims))
        if clamped != self.sigreg_n_dirs:
            warnings.warn(
                f"sigreg_n_dirs {self.sigreg_n_dirs} exceeds lat_dims {self.lat_dims}; "
                f"clamping to {clamped}",
                stacklevel=2)
            self.sigreg_n_dirs = clamped
        if self.fg_mode not in ("std", "circle"):
            raise ValueError(f"fg_mode must be 'std' or 'circle', got {self.fg_mode!r}")
        if self.fg_circle_diameter_frac <= 0:
            raise ValueError(
                f"fg_circle_diameter_frac must be positive, got {self.fg_circle_diameter_frac}")
        if len(self.freeze_after_epoch) == 1:
            self.freeze_after_epoch = self.freeze_after_epoch * self.num_stages
        if len(self.freeze_after_epoch) != self.num_stages:
            raise ValueError(
                f"freeze_after_epoch must have {self.num_stages} entries (s1..s4), "
                f"got {len(self.freeze_after_epoch)}: {self.freeze_after_epoch}")
        if any(e < 0 for e in self.freeze_after_epoch):
            raise ValueError(
                f"freeze_after_epoch entries must be non-negative, got {self.freeze_after_epoch}")
        if self.sigreg_s4_on not in ("e4", "c4"):
            raise ValueError(
                f"sigreg_s4_on must be 'e4' or 'c4', got {self.sigreg_s4_on!r}")
        if self.coarse_mim_mode not in ("integrated", "cross_attn", "conv"):
            raise ValueError(
                f"coarse_mim_mode must be 'integrated', 'cross_attn', or 'conv', "
                f"got {self.coarse_mim_mode!r}")
        if (not self.legacy_jepa and self.coarse_mim_mode == "cross_attn"
                and not self.predictor_enabled):
            raise ValueError(
                "coarse_mim_mode='cross_attn' requires predictor_enabled=True")
        if self.coarse_mim_mode == "integrated" and self.legacy_jepa:
            raise ValueError(
                "coarse_mim_mode='integrated' is incompatible with legacy_jepa=True")
        if not (0.0 <= self.s4_cosine_level <= 1.0):
            raise ValueError(
                f"s4_cosine_level must be in [0, 1], got {self.s4_cosine_level}")
        if not (0.0 < self.s4_cosine_ema_decay < 1.0):
            raise ValueError(
                f"s4_cosine_ema_decay must be in (0, 1), got {self.s4_cosine_ema_decay}")
        if not (0.0 < self.s4_sigreg_fallback_progress <= 1.0):
            raise ValueError(
                f"s4_sigreg_fallback_progress must be in (0, 1], "
                f"got {self.s4_sigreg_fallback_progress}")
        if not (0.0 <= self.s4_sigreg_ramp_progress <= 1.0):
            raise ValueError(
                f"s4_sigreg_ramp_progress must be in [0, 1], "
                f"got {self.s4_sigreg_ramp_progress}")


# Tuple-valued fields take ``nargs`` from argparse; map element types here.
_TUPLE_ELEM_TYPE = {
    "block_scale_range": float,
    "stage_base_weights": float,
    "fine_stages": int,
    "coarse_ramp_stages": int,
    "freeze_after_epoch": int,
    "sigreg_n_dirs": int,
    "sigreg_scale": float,
    "beta_sig": float,
    "lat_dims": int,
}


def add_argparse_args(parser):
    """Register every :class:`SwinMSJEPAConfig` field as an ``argparse`` option.

    Bool fields get a ``--flag`` / ``--no_flag`` pair so a ``True`` default is
    still overridable from the command line (mirrors ``ssl/train.py``).
    """
    for f in fields(SwinMSJEPAConfig):
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


def from_args(args) -> SwinMSJEPAConfig:
    """Build a :class:`SwinMSJEPAConfig` from a parsed ``argparse`` namespace."""
    kwargs = {}
    for f in fields(SwinMSJEPAConfig):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if f.name in _TUPLE_ELEM_TYPE and val is not None:
            val = tuple(val)
        kwargs[f.name] = val
    return SwinMSJEPAConfig(**kwargs)
