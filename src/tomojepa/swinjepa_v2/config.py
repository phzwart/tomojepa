"""HL-JEPA configuration dataclasses (spec §2) with YAML load/dump."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


def _default_photometric() -> Dict[str, Any]:
    return {
        "noise_std": (0.0, 0.05),
        "intensity_scale": (0.85, 1.15),
        "contrast_scale": (0.85, 1.15),
        "gamma": (0.85, 1.15),
        "blur_sigma": (0.0, 1.0),
        "blur_p": 0.3,
    }


def _default_geometric() -> Dict[str, Any]:
    return {
        "crop_scale": (0.4, 1.0),
        "hflip_p": 0.5,
        "vflip_p": 0.5,
        "rot90_p": 0.5,
    }


@dataclass
class DataCfg:
    ndim: int = 2
    image_size: Tuple[int, ...] = (256, 256)
    patch_size: int = 4
    photometric: Dict[str, Any] = field(default_factory=_default_photometric)
    geometric: Dict[str, Any] = field(default_factory=_default_geometric)
    in_chans: int = 1
    dataset_key: str = "reconstruction"
    pattern: str = "recon_*.h5"
    backend: str = "auto"
    crop_mode: str = "resize"


@dataclass
class MaskCfg:
    strategy: str = "multiblock"
    mask_ratio: float = 0.6
    num_blocks: int = 4
    block_scale: Tuple[float, float] = (0.15, 0.25)
    block_aspect: Tuple[float, float] = (0.5, 2.0)
    pool_rule: str = "fraction"
    pool_thresh: float = 0.5


@dataclass
class BackboneCfg:
    model_name: str = "swin_tiny_patch4_window7_224"
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 8
    in_chans: int = 1
    drop_path_rate: float = 0.1
    use_rope: bool = True
    rope_theta: float = 100.0
    pretrained: bool = False


@dataclass
class BandCfg:
    mode: str = "laplacian"
    band_dim: int = 256
    use_proj_head: bool = False


@dataclass
class PredictorCfg:
    depth_per_band: int = 3
    embed_dim: int = 256
    num_heads: int = 8
    top_down: bool = True
    mlp_ratio: float = 4.0


@dataclass
class LossCfg:
    w_pred: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    lambda_sig: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1)
    pred_loss: str = "smooth_l1"
    pred_l2norm: bool = False
    stop_grad_target: bool = False


@dataclass
class SigRegCfg:
    num_slices: int = 256
    t_max: float = 5.0
    n_knots: int = 17
    sigma: float = 1.0
    fold_N_into_lambda: bool = True
    per_token: bool = True


@dataclass
class TrainCfg:
    epochs: int = 300
    batch_size: int = 64
    base_lr: float = 1.5e-4
    warmup_epochs: int = 20
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    log_every: int = 50
    metric_every: int = 500
    seed: int = 0
    run_dir: str = "runs/hljepa"
    data_dir: str = ""


@dataclass
class HLJEPAConfig:
    data: DataCfg = field(default_factory=DataCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    band: BandCfg = field(default_factory=BandCfg)
    predictor: PredictorCfg = field(default_factory=PredictorCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    sigreg: SigRegCfg = field(default_factory=SigRegCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    @property
    def num_bands(self) -> int:
        return len(self.backbone.depths)

    @property
    def img_size(self) -> int:
        if self.data.ndim == 2:
            return int(self.data.image_size[0])
        return int(self.data.image_size[1])


def default() -> HLJEPAConfig:
    return HLJEPAConfig()


def _tupleize(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_tupleize(x) for x in v)
    return v


def _from_dict(cls: type, data: Dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        if is_dataclass(f.type) or (hasattr(f.type, "__origin__") is False and is_dataclass(getattr(f.type, "__args__", [None])[0] if hasattr(f.type, "__args__") else None)):
            inner = f.type
            if hasattr(f.type, "__origin__"):
                pass
            elif is_dataclass(f.type):
                kwargs[f.name] = _from_dict(f.type, val) if isinstance(val, dict) else val
                continue
        ft = f.type
        if isinstance(val, dict) and is_dataclass(ft):
            kwargs[f.name] = _from_dict(ft, val)
        elif isinstance(val, list) and f.name in (
            "image_size", "depths", "num_heads", "w_pred", "lambda_sig",
            "block_scale", "block_aspect",
        ):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_yaml(path: Union[str, Path]) -> HLJEPAConfig:
    if yaml is None:
        raise ImportError("PyYAML required for load_yaml") from _YAML_IMPORT_ERROR
    from dataclasses import replace
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = default()
    for section in ("data", "mask", "backbone", "band", "predictor", "loss", "sigreg", "train"):
        if section not in raw:
            continue
        cur = asdict(getattr(cfg, section))
        cur.update(raw[section])
        for key in ("image_size", "depths", "num_heads", "w_pred", "lambda_sig",
                    "block_scale", "block_aspect"):
            if key in cur and isinstance(cur[key], list):
                cur[key] = tuple(cur[key])
        setattr(cfg, section, type(getattr(cfg, section))(**cur))
    return cfg


def dump_yaml(cfg: HLJEPAConfig, path: Union[str, Path]) -> None:
    if yaml is None:
        raise ImportError("PyYAML required for dump_yaml") from _YAML_IMPORT_ERROR
    data = asdict(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
