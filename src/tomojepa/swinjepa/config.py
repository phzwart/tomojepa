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
from typing import List, Tuple


@dataclass
class SwinMSJEPAConfig:
    # ---- input -------------------------------------------------------------
    backbone_name: str = "swin_tiny_patch4_window7_224"
    in_chans: int = 1                         # grayscale microCT
    img_size: int = 224                       # set from repo tile size (fallback 224)
    drop_path_rate: float = 0.1
    use_rope: bool = True                       # 2D RoPE on backbone window attention
    rope_theta: float = 100.0                   # RoPE frequency base

    # ---- mask (defined on the stage-4 grid) -------------------------------
    mask_ratio: float = 0.55
    mask_mode: str = "random_cell"            # random_cell | block
    mask_num_blocks: int = 4                  # block mode
    block_scale_range: Tuple[float, float] = (0.1, 0.4)

    # ---- cross-scale predictor --------------------------------------------
    pred_dim: int = 384                       # D_pred
    pred_depth: int = 4                       # N_pred decoder blocks
    pred_heads: int = 6
    pred_mlp_ratio: float = 4.0
    predictor_cross_scale: bool = True        # attend across all stages
    predictor_enabled: bool = True            # False -> pure data2vec path

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
    # 1-based stage ids that are *fine* (fine_in mode only).
    fine_stages: Tuple[int, ...] = (1, 2)

    # ---- pyramid residual (default) vs legacy full-latent JEPA ------------
    legacy_jepa: bool = False                 # True -> pre-pyramid token SIGReg + E targets
    sigreg_pooled: bool = True                 # pyramid: enable cross-batch SIGReg FIFO queue

    # ---- SIGReg (per stage) -----------------------------------------------
    # Per-stage SIGReg weight (s1..s4). Legacy: scales token SIGReg at each E_s.
    # Pyramid: s4 weights C4; s1..s3 weight R1..R3. Lower s4 lets MAE shape C4.
    beta_sig: Tuple[float, ...] = (0.05, 0.05, 0.05, 0.01)
    sigreg_n_dirs: Tuple[int, ...] = (256, 256, 384, 512)
    sigreg_knots: int = 17                      # CF quadrature knots (repo SIGReg)
    sigreg_t_max: float = 3.0                   # CF quadrature range (repo SIGReg)
    sigreg_w_mean: float = 0.1                  # light explicit mean penalty
    sigreg_n_tokens_cap: int = 4096             # legacy token cap (0 = use all)
    sigreg_tokens_per_slice: int = 32           # pyramid: FG tokens subsampled per image
    sigreg_min_token_dist: int = 2              # min Chebyshev grid dist (0 = random)
    sigreg_queue_len: int = 512                 # FIFO queue length (slices, not pooled rows)
    sigreg_scale: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)   # scale_s
    sigreg_rebalance_by_dim: bool = False       # multiply scale_s by 1/C_s

    # ---- foreground FOV mask (training-only) ------------------------------
    foreground_mask: bool = False
    fg_std_thresh: float = 0.05                 # intensity-variation threshold
    fg_coverage: float = 0.01                   # min FG fraction per token footprint
    fg_key: str = ""                            # optional precomputed mask array key

    # ---- per-stage JEPA latent (1x1 lateral projections) ------------------
    # Length S tuple (s1..s4): each stage maps backbone C_s -> lat_dims[s].
    lat_dims: Tuple[int, ...] = (64, 64, 64, 64)

    # ---- optimization ------------------------------------------------------
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1.5e-4
    weight_decay: float = 0.05
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
        if len(self.lat_dims) != self.num_stages:
            raise ValueError(
                f"lat_dims must have {self.num_stages} entries (s1..s4), "
                f"got {len(self.lat_dims)}: {self.lat_dims}")
        if any(d <= 0 for d in self.lat_dims):
            raise ValueError(f"lat_dims entries must be positive, got {self.lat_dims}")
        if not self.legacy_jepa and len(set(self.lat_dims)) != 1:
            raise ValueError(
                "pyramid residual algebra requires equal lat_dims across s1..s4 "
                "(R_s = E_s - up(parent) subtracts tensors of the same width); "
                "use legacy_jepa=True for per-stage widths or set lat_dims to one shared value")
        if len(self.beta_sig) == 1:
            self.beta_sig = self.beta_sig * self.num_stages
        if len(self.beta_sig) != self.num_stages:
            raise ValueError(
                f"beta_sig must have {self.num_stages} entries (s1..s4), "
                f"got {len(self.beta_sig)}: {self.beta_sig}")


# Tuple-valued fields take ``nargs`` from argparse; map element types here.
_TUPLE_ELEM_TYPE = {
    "block_scale_range": float,
    "stage_base_weights": float,
    "fine_stages": int,
    "coarse_ramp_stages": int,
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
