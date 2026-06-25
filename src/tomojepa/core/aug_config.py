"""YAML-driven augmentation config and optional progress schedules.

Augmentations can be declared in the training schedule YAML under
``augmentations:`` (static defaults + optional ``schedule:`` knots), or in a
standalone file via ``--aug_config``. CLI flags override YAML when explicitly
passed on the command line.

Schedulable channels (piecewise linear in progress, same knot format as
:class:`tomojepa.swinjepa.schedule.TrainingSchedule`):

- ``random_rotate_deg``, ``rotate_p``, ``equalize_p``, ``blur_p``, ``poisson_p``,
  ``pixel_mask_p``
- ``resize_jitter`` — ``[lo, hi]`` pair; ``[0, 0]`` or ``null`` disables
- ``global_scale``, ``local_scale`` — ``[min, max]`` RandomResizedCrop bands
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None

from tomojepa.core.schedule_interp import interp_bool_sticky, interp_numeric, parse_knot

# Backward-compatible aliases for tests / internal callers
_parse_knot = parse_knot
_interp_numeric = interp_numeric
_interp_bool_sticky = interp_bool_sticky

_SCHEDULABLE_NUMERIC = (
    "random_rotate_deg",
    "rotate_p",
    "equalize_p",
    "blur_p",
    "poisson_p",
    "pixel_mask_p",
)
_SCHEDULABLE_PAIR = (
    "resize_jitter",
    "global_scale",
    "local_scale",
)
_SCHEDULABLE_BOOL = ("intensity_augment",)
_ALL_SCHEDULE_CHANNELS = _SCHEDULABLE_NUMERIC + _SCHEDULABLE_PAIR + _SCHEDULABLE_BOOL


@dataclass(frozen=True)
class AugmentationConfig:
    """Static augmentation defaults (train + probe geometry selection)."""

    variant: str = "tomo2"
    crop_mode: str = "resized"
    global_views: int = 1
    local_views: int = 0
    # Apply global_tf once and replicate for every global view (SimMIM: mask-only diff).
    shared_global_aug: bool = False
    global_scale: Tuple[float, float] = (0.4, 1.0)
    local_scale: Tuple[float, float] = (0.1, 0.4)
    # None -> legacy tomo2 half-turn; 0 -> off; >0 -> symmetric +/- deg
    random_rotate_deg: Optional[float] = 180.0
    rotate_deg_range: Optional[Tuple[float, float]] = None  # e.g. (5, 175) overrides +/- deg
    rotate_p: float = 1.0
    # None or (0,0) -> disabled
    resize_jitter: Optional[Tuple[float, float]] = (0.9, 1.1)
    hflip_p: float = 0.0
    vflip_p: float = 0.0
    window_p_low: float = 0.01
    window_p_high: float = 0.99
    window_sample_size: int = 100_000
    equalize_p: float = 0.5
    blur_p: float = 0.5
    blur_kernel_size: int = 7
    blur_sigma: Tuple[float, float] = (0.1, 1.0)
    poisson_p: float = 0.5
    poisson_scale: float = 10000.0
    pixel_mask_p: float = 0.5
    pixel_mask_ratio: float = 0.15
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    intensity_augment: bool = True

    def __post_init__(self):
        if self.variant not in ("tomo", "tomo2"):
            raise ValueError(f"variant must be 'tomo' or 'tomo2', got {self.variant!r}")
        if self.crop_mode not in ("resized", "native", "resize", "crop_down"):
            raise ValueError(f"unknown crop_mode {self.crop_mode!r}")
        if self.global_views < 0 or self.local_views < 0:
            raise ValueError("global_views and local_views must be non-negative")

    def resize_jitter_scale(self) -> Optional[Tuple[float, float]]:
        if self.resize_jitter is None:
            return None
        lo, hi = float(self.resize_jitter[0]), float(self.resize_jitter[1])
        if lo <= 0 or hi <= 0:
            return None
        return (lo, hi)

    def summary_line(self) -> str:
        rj = self.resize_jitter_scale()
        rot = (
            f"rotate=[{self.rotate_deg_range[0]:g},{self.rotate_deg_range[1]:g}]deg"
            if self.rotate_deg_range
            else (f"rotate=+/-{self.random_rotate_deg}deg" if self.random_rotate_deg
                  else ("rotate=legacy" if self.random_rotate_deg is None else "rotate=off"))
        )
        parts = [
            f"variant={self.variant}",
            f"crop={self.crop_mode}",
            f"views=g{self.global_views}/l{self.local_views}",
            rot,
        ]
        if rj:
            parts.append(f"resize_jitter=[{rj[0]:g},{rj[1]:g}]")
        if self.hflip_p > 0:
            parts.append(f"hflip_p={self.hflip_p:g}")
        if self.vflip_p > 0:
            parts.append(f"vflip_p={self.vflip_p:g}")
        if self.rotate_p < 1.0 and (
            self.rotate_deg_range or (self.random_rotate_deg or 0) != 0
        ):
            parts.append(f"rotate_p={self.rotate_p:g}")
        if not self.intensity_augment:
            parts.append("intensity=off")
        if self.shared_global_aug:
            parts.append("shared_global_aug")
        return "; ".join(parts)


@dataclass(frozen=True)
class AugmentationState:
    """Resolved augmentation scalars at a training progress point."""

    random_rotate_deg: Optional[float]
    resize_jitter: Optional[Tuple[float, float]]
    global_scale: Tuple[float, float]
    local_scale: Tuple[float, float]
    rotate_p: float
    equalize_p: float
    blur_p: float
    poisson_p: float
    pixel_mask_p: float
    intensity_augment: bool


def _parse_knot_pair(raw: Any, channel: str) -> Tuple[float, Optional[Tuple[float, float]]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{channel} knot must be a mapping, got {type(raw).__name__}")
    prog = raw.get("progress", raw.get("p"))
    val = raw.get("value", raw.get("v"))
    if prog is None:
        raise ValueError(f"{channel} knot needs progress: {raw!r}")
    p = float(prog)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{channel} progress must be in [0, 1], got {p}")
    if val is None:
        return p, None
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return p, (float(val[0]), float(val[1]))
    raise ValueError(f"{channel} value must be null or [lo, hi], got {val!r}")


class AugmentationSchedule:
    """Piecewise augmentation overrides over training progress."""

    def __init__(self, knots: Dict[str, list], progress_scope: str = "run"):
        self._knots = knots
        if progress_scope not in ("run", "epoch"):
            raise ValueError(f"progress_scope must be 'run' or 'epoch', got {progress_scope!r}")
        self.progress_scope = progress_scope

    @classmethod
    def from_dict(cls, data: Mapping[str, Any],
                    progress_scope: str = "run") -> "AugmentationSchedule":
        raw = data.get("schedule")
        if raw is None:
            return cls({}, progress_scope=progress_scope)
        if not isinstance(raw, Mapping):
            raise ValueError("augmentations.schedule must be a mapping")
        knots: Dict[str, list] = {}
        for channel, knots_raw in raw.items():
            if channel not in _ALL_SCHEDULE_CHANNELS:
                raise ValueError(
                    f"unknown augmentation schedule channel {channel!r}; "
                    f"expected one of {_ALL_SCHEDULE_CHANNELS}")
            if not isinstance(knots_raw, list) or not knots_raw:
                raise ValueError(f"augmentations.schedule.{channel} must be a non-empty list")
            if channel in _SCHEDULABLE_PAIR:
                knots[channel] = [_parse_knot_pair(k, channel) for k in knots_raw]
            elif channel in _SCHEDULABLE_BOOL:
                knots[channel] = [(p, bool(v)) for p, v in
                                  [parse_knot(k, channel) for k in knots_raw]]
            else:
                knots[channel] = [parse_knot(k, channel) for k in knots_raw]
        scope = str(data.get("progress_scope", progress_scope))
        return cls(knots, progress_scope=scope)

    def _progress(self, step: int, total_steps: int,
                  steps_per_epoch: Optional[int] = None) -> float:
        if self.progress_scope == "epoch":
            spe = max(1, int(steps_per_epoch or 1))
            progress = float(step % spe) / spe
        else:
            progress = float(step) / max(1, total_steps)
        return min(1.0, max(0.0, progress))

    @staticmethod
    def _interp_pair(knots: Sequence[Tuple[float, Any]], progress: float,
                     default: Tuple[float, float]) -> Tuple[float, float]:
        if not knots:
            return default
        xs = [_interp_numeric([(p, float(v[0])) for p, v in knots], progress),
              _interp_numeric([(p, float(v[1])) for p, v in knots], progress)]
        return (xs[0], xs[1])

    @staticmethod
    def _interp_optional_pair(knots: Sequence[Tuple[float, Any]], progress: float,
                              default: Optional[Tuple[float, float]]
                              ) -> Optional[Tuple[float, float]]:
        if not knots:
            return default
        val = _interp_numeric([(p, 1.0 if v is not None else 0.0)
                               for p, v in knots], progress)
        if val <= 0:
            return None
        pair_knots = [(p, v) for p, v in knots if v is not None]
        if not pair_knots:
            return None
        lo = _interp_numeric([(p, float(v[0])) for p, v in pair_knots], progress)
        hi = _interp_numeric([(p, float(v[1])) for p, v in pair_knots], progress)
        if lo <= 0 or hi <= 0:
            return None
        return (lo, hi)

    def at(self, base: AugmentationConfig, step: int, total_steps: int,
           steps_per_epoch: Optional[int] = None) -> AugmentationState:
        progress = self._progress(step, total_steps, steps_per_epoch)
        k = self._knots

        def num(channel: str, default: float) -> float:
            if channel not in k:
                return default
            return _interp_numeric(k[channel], progress)

        def pair(channel: str, default: Tuple[float, float]) -> Tuple[float, float]:
            if channel not in k:
                return default
            return self._interp_pair(k[channel], progress, default)

        def opt_pair(channel: str,
                     default: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
            if channel not in k:
                return default
            return self._interp_optional_pair(k[channel], progress, default)

        intensity = base.intensity_augment
        if "intensity_augment" in k:
            intensity = _interp_bool_sticky(k["intensity_augment"], progress)

        rot = base.random_rotate_deg
        if "random_rotate_deg" in k:
            rot = num("random_rotate_deg", float(base.random_rotate_deg or 0))

        return AugmentationState(
            random_rotate_deg=rot,
            resize_jitter=opt_pair("resize_jitter", base.resize_jitter_scale()),
            global_scale=pair("global_scale", base.global_scale),
            local_scale=pair("local_scale", base.local_scale),
            rotate_p=num("rotate_p", base.rotate_p),
            equalize_p=num("equalize_p", base.equalize_p),
            blur_p=num("blur_p", base.blur_p),
            poisson_p=num("poisson_p", base.poisson_p),
            pixel_mask_p=num("pixel_mask_p", base.pixel_mask_p),
            intensity_augment=intensity,
        )

    def summary_lines(self) -> List[str]:
        if not self._knots:
            return []
        lines = [f"  aug schedule (progress_scope={self.progress_scope}):"]
        for ch, pts in sorted(self._knots.items()):
            if ch in _SCHEDULABLE_BOOL:
                seg = ", ".join(f"{p:.2g}->{int(v)}" for p, v in pts)
            else:
                seg = ", ".join(f"{p:.2g}->{v!r}" for p, v in pts)
            lines.append(f"    {ch}: [{seg}]")
        return lines


def _pair_from_yaml(raw: Any, name: str) -> Tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{name} must be a length-2 list, got {raw!r}")
    return (float(raw[0]), float(raw[1]))


def _optional_pair_from_yaml(raw: Any, name: str) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = float(raw[0]), float(raw[1])
        if lo <= 0 or hi <= 0:
            return None
        return (lo, hi)
    raise ValueError(f"{name} must be null or a length-2 list, got {raw!r}")


def augmentation_config_from_dict(data: Optional[Mapping[str, Any]]) -> AugmentationConfig:
    if not data:
        return AugmentationConfig()
    if not isinstance(data, Mapping):
        raise ValueError(f"augmentations must be a mapping, got {type(data).__name__}")

    kwargs: Dict[str, Any] = {}
    if "variant" in data:
        kwargs["variant"] = str(data["variant"])
    if "crop_mode" in data:
        kwargs["crop_mode"] = str(data["crop_mode"])
    if "global_views" in data:
        kwargs["global_views"] = int(data["global_views"])
    if "local_views" in data:
        kwargs["local_views"] = int(data["local_views"])
    if "shared_global_aug" in data:
        kwargs["shared_global_aug"] = bool(data["shared_global_aug"])
    if "global_scale" in data:
        kwargs["global_scale"] = _pair_from_yaml(data["global_scale"], "global_scale")
    if "local_scale" in data:
        kwargs["local_scale"] = _pair_from_yaml(data["local_scale"], "local_scale")
    if "random_rotate_deg" in data:
        v = data["random_rotate_deg"]
        kwargs["random_rotate_deg"] = None if v is None else float(v)
    if "rotate_p" in data:
        kwargs["rotate_p"] = float(data["rotate_p"])
    if "rotate" in data:
        r = data["rotate"]
        if not isinstance(r, Mapping):
            raise ValueError("augmentations.rotate must be a mapping")
        if "p" in r:
            kwargs["rotate_p"] = float(r["p"])
        if "deg" in r:
            v = r["deg"]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                kwargs["rotate_deg_range"] = (float(v[0]), float(v[1]))
            else:
                kwargs["random_rotate_deg"] = None if v is None else float(v)
    if "resize_jitter" in data:
        kwargs["resize_jitter"] = _optional_pair_from_yaml(
            data["resize_jitter"], "resize_jitter")
    if "hflip_p" in data:
        kwargs["hflip_p"] = float(data["hflip_p"])
    if "vflip_p" in data:
        kwargs["vflip_p"] = float(data["vflip_p"])
    if "window" in data:
        w = data["window"]
        if not isinstance(w, Mapping):
            raise ValueError("augmentations.window must be a mapping")
        if "p_low" in w:
            kwargs["window_p_low"] = float(w["p_low"])
        if "p_high" in w:
            kwargs["window_p_high"] = float(w["p_high"])
        if "sample_size" in w:
            kwargs["window_sample_size"] = int(w["sample_size"])
    if "equalize_p" in data:
        kwargs["equalize_p"] = float(data["equalize_p"])
    if "blur" in data:
        b = data["blur"]
        if not isinstance(b, Mapping):
            raise ValueError("augmentations.blur must be a mapping")
        if "p" in b:
            kwargs["blur_p"] = float(b["p"])
        if "kernel_size" in b:
            kwargs["blur_kernel_size"] = int(b["kernel_size"])
        if "sigma" in b:
            kwargs["blur_sigma"] = _pair_from_yaml(b["sigma"], "blur.sigma")
    if "poisson" in data:
        p = data["poisson"]
        if not isinstance(p, Mapping):
            raise ValueError("augmentations.poisson must be a mapping")
        if "p" in p:
            kwargs["poisson_p"] = float(p["p"])
        if "scale" in p:
            kwargs["poisson_scale"] = float(p["scale"])
    if "pixel_mask" in data:
        pm = data["pixel_mask"]
        if not isinstance(pm, Mapping):
            raise ValueError("augmentations.pixel_mask must be a mapping")
        if "p" in pm:
            kwargs["pixel_mask_p"] = float(pm["p"])
        if "ratio" in pm:
            kwargs["pixel_mask_ratio"] = float(pm["ratio"])
    if "normalize" in data:
        n = data["normalize"]
        if not isinstance(n, Mapping):
            raise ValueError("augmentations.normalize must be a mapping")
        if "mean" in n:
            kwargs["normalize_mean"] = float(n["mean"])
        if "std" in n:
            kwargs["normalize_std"] = float(n["std"])
    if "intensity_augment" in data:
        kwargs["intensity_augment"] = bool(data["intensity_augment"])
    return AugmentationConfig(**kwargs)


def parse_augmentations_block(data: Optional[Mapping[str, Any]]
                              ) -> Tuple[AugmentationConfig, Optional[AugmentationSchedule]]:
    cfg = augmentation_config_from_dict(data)
    if not data or "schedule" not in data:
        return cfg, None
    scope = str(data.get("progress_scope", "run"))
    return cfg, AugmentationSchedule.from_dict(data, progress_scope=scope)


def load_augmentation_yaml(path: Union[str, Path]
                           ) -> Tuple[AugmentationConfig, Optional[AugmentationSchedule]]:
    if yaml is None:
        raise ImportError(
            "PyYAML is required for augmentation configs; pip install pyyaml"
        ) from _YAML_IMPORT_ERROR
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"augmentation config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"augmentation config root must be a mapping, got {type(data).__name__}")
    block = data.get("augmentations", data)
    return parse_augmentations_block(block if isinstance(block, Mapping) else None)


def merge_augmentation_cli(base: AugmentationConfig, args, argv: Optional[Sequence[str]] = None
                           ) -> AugmentationConfig:
    """Apply CLI overrides when the corresponding flag appears in ``argv``."""
    import sys
    argv = argv if argv is not None else sys.argv
    argv_set = set(argv)
    kwargs: Dict[str, Any] = {}

    def seen(*flags: str) -> bool:
        return any(f in argv_set for f in flags)

    if seen("--augment"):
        kwargs["variant"] = args.augment
    if seen("--crop_mode"):
        kwargs["crop_mode"] = args.crop_mode
    if seen("--global_views"):
        kwargs["global_views"] = int(args.global_views)
    if seen("--local_views"):
        kwargs["local_views"] = int(args.local_views)
    if seen("--global_scale"):
        kwargs["global_scale"] = tuple(args.global_scale)
    if seen("--local_scale"):
        kwargs["local_scale"] = tuple(args.local_scale)
    if seen("--random_rotate_deg"):
        kwargs["random_rotate_deg"] = float(args.random_rotate_deg)
    if seen("--resize_jitter"):
        lo, hi = args.resize_jitter
        kwargs["resize_jitter"] = None if lo <= 0 or hi <= 0 else (float(lo), float(hi))
    return replace(base, **kwargs) if kwargs else base


def state_to_dynamic_dict(state: AugmentationState) -> Dict[str, Any]:
    rj = state.resize_jitter
    return {
        "random_rotate_deg": state.random_rotate_deg,
        "resize_jitter_lo": rj[0] if rj else 0.0,
        "resize_jitter_hi": rj[1] if rj else 0.0,
        "global_scale_lo": state.global_scale[0],
        "global_scale_hi": state.global_scale[1],
        "local_scale_lo": state.local_scale[0],
        "local_scale_hi": state.local_scale[1],
        "rotate_p": state.rotate_p,
        "equalize_p": state.equalize_p,
        "blur_p": state.blur_p,
        "poisson_p": state.poisson_p,
        "pixel_mask_p": state.pixel_mask_p,
        "intensity_augment": state.intensity_augment,
    }
