"""Declarative training schedules as YAML (progress in [0, 1] over total steps).

Each stage (``s1``..``s4``) can define piecewise schedules for:

- ``active`` — SIGReg loss multiplier (and prediction when ``pred_active`` omitted)
- ``pred_active`` — optional prediction / s4 MAE multiplier (defaults to ``active``)
- ``beta_sig`` — per-step SIGReg weight (replaces static ``beta_sig``)
- ``freeze`` — sticky bool; once true at progress ``p``, stage stays frozen

Knot format::

    - {progress: 0.0, value: 0.1}
    - {progress: 0.25, value: 1.0}

Numeric channels use linear interpolation between knots. Boolean ``freeze``
channels flip at each knot when ``progress >= knot.progress`` (last matching
knot wins).

These files are plain YAML (no Hydra required). They can later be composed
with Hydra config groups, e.g. ``python -m tomojepa.swinjepa.train schedule=petiole_448``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised when PyYAML missing
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None

_STAGE_KEYS = ("s1", "s2", "s3", "s4")
_NUMERIC_CHANNELS = ("active", "beta_sig")
_OPTIONAL_NUMERIC_CHANNELS = ("pred_active",)
_BOOL_CHANNELS = ("freeze",)


@dataclass(frozen=True)
class StageScheduleState:
    active: float
    pred_active: float
    beta_sig: float
    frozen: bool


@dataclass(frozen=True)
class ScheduleState:
    progress: float
    stages: Dict[str, StageScheduleState]


from tomojepa.core.schedule_interp import interp_bool_sticky, interp_numeric, parse_knot

# Backward-compatible aliases for tests / internal callers
_parse_knot = parse_knot
_interp_numeric = interp_numeric
_interp_bool_sticky = interp_bool_sticky


class TrainingSchedule:
    """Piecewise schedule over training progress.

    ``progress_scope``:

    - ``run`` (default): ``progress = step / total_steps`` over the full job
    - ``epoch``: ``progress = (step % steps_per_epoch) / steps_per_epoch`` so
      each epoch repeats the coarse→fine block cycle
    """

    def __init__(self, name: str, stage_knots: Dict[str, Dict[str, list]],
                 progress_scope: str = "run"):
        self.name = name
        self._stage_knots = stage_knots
        if progress_scope not in ("run", "epoch"):
            raise ValueError(f"progress_scope must be 'run' or 'epoch', got {progress_scope!r}")
        self.progress_scope = progress_scope

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingSchedule":
        if yaml is None:
            raise ImportError(
                "PyYAML is required for training schedules; pip install pyyaml"
            ) from _YAML_IMPORT_ERROR
        name = str(data.get("name", "unnamed"))
        progress_scope = str(data.get("progress_scope", "run"))
        raw_stages = data.get("stages")
        if not isinstance(raw_stages, Mapping):
            raise ValueError("schedule YAML must contain a 'stages' mapping")
        stage_knots: Dict[str, Dict[str, list]] = {}
        for key in _STAGE_KEYS:
            if key not in raw_stages:
                raise ValueError(f"schedule missing stage {key!r}")
            spec = raw_stages[key]
            if not isinstance(spec, Mapping):
                raise ValueError(f"stages.{key} must be a mapping")
            parsed: Dict[str, list] = {}
            for channel in _NUMERIC_CHANNELS + _BOOL_CHANNELS:
                knots_raw = spec.get(channel)
                if knots_raw is None:
                    raise ValueError(f"stages.{key}.{channel} is required")
                if not isinstance(knots_raw, list) or not knots_raw:
                    raise ValueError(f"stages.{key}.{channel} must be a non-empty list")
                parsed[channel] = [_parse_knot(k, channel) for k in knots_raw]
                if channel in _BOOL_CHANNELS:
                    parsed[channel] = [(p, bool(v)) for p, v in parsed[channel]]
            for channel in _OPTIONAL_NUMERIC_CHANNELS:
                knots_raw = spec.get(channel)
                if knots_raw is None:
                    continue
                if not isinstance(knots_raw, list) or not knots_raw:
                    raise ValueError(f"stages.{key}.{channel} must be a non-empty list")
                parsed[channel] = [_parse_knot(k, channel) for k in knots_raw]
            if "pred_active" not in parsed:
                parsed["pred_active"] = list(parsed["active"])
            stage_knots[key] = parsed
        return cls(name, stage_knots, progress_scope=progress_scope)

    def _progress(self, step: int, total_steps: int,
                  steps_per_epoch: Optional[int] = None) -> float:
        if self.progress_scope == "epoch":
            spe = max(1, int(steps_per_epoch or 1))
            progress = float(step % spe) / spe
        else:
            progress = float(step) / max(1, total_steps)
        return min(1.0, max(0.0, progress))

    def at(self, step: int, total_steps: int,
           steps_per_epoch: Optional[int] = None) -> ScheduleState:
        progress = self._progress(step, total_steps, steps_per_epoch)
        stages: Dict[str, StageScheduleState] = {}
        for key in _STAGE_KEYS:
            spec = self._stage_knots[key]
            stages[key] = StageScheduleState(
                active=_interp_numeric(spec["active"], progress),
                pred_active=_interp_numeric(spec["pred_active"], progress),
                beta_sig=_interp_numeric(spec["beta_sig"], progress),
                frozen=_interp_bool_sticky(spec["freeze"], progress),
            )
        return ScheduleState(progress=progress, stages=stages)

    def summary_lines(self) -> List[str]:
        lines = [f"schedule: {self.name} (progress_scope={self.progress_scope})"]
        for key in _STAGE_KEYS:
            spec = self._stage_knots[key]
            act = ", ".join(f"{p:.2g}->{v:g}" for p, v in spec["active"])
            frz = ", ".join(f"{p:.2g}->{int(v)}" for p, v in spec["freeze"])
            lines.append(f"  {key}: active [{act}]; freeze [{frz}]")
        return lines


_RUN_OVERRIDE_KEYS = ("epochs", "img_size", "mask_ratio", "seed", "batch_size")


def parse_run_block(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract optional ``run:`` training overrides from a job YAML root."""
    run = data.get("run")
    if run is None:
        return {}
    if not isinstance(run, Mapping):
        raise ValueError("job YAML 'run' must be a mapping")
    out: Dict[str, Any] = {}
    for key in _RUN_OVERRIDE_KEYS:
        if key not in run:
            continue
        val = run[key]
        if key in ("epochs", "img_size", "seed", "batch_size"):
            out[key] = int(val)
        else:
            out[key] = float(val)
    return out


def apply_job_run_overrides(args, run_overrides: Mapping[str, Any]):
    """Apply ``run:`` block fields onto argparse ``args``."""
    for key, val in run_overrides.items():
        if key not in _RUN_OVERRIDE_KEYS:
            raise ValueError(f"unsupported run override {key!r}")
        setattr(args, key, val)
    return args


def job_meta_from_dict(data: Mapping[str, Any], path: Optional[Union[str, Path]] = None
                       ) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "name": str(data.get("name", "unnamed")),
        "run": parse_run_block(data),
    }
    if path is not None:
        meta["path"] = str(path)
    return meta


def load_training_schedule(path: Union[str, Path]) -> TrainingSchedule:
    if yaml is None:
        raise ImportError(
            "PyYAML is required for training schedules; pip install pyyaml"
        ) from _YAML_IMPORT_ERROR
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"schedule file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"schedule root must be a mapping, got {type(data).__name__}")
    return TrainingSchedule.from_dict(data)


def load_job_yaml(path: Union[str, Path]):
    """Load training schedule, augmentation config, and job metadata from one YAML file.

    Returns:
        ``(schedule, aug_cfg, aug_sched, job_meta)`` where ``job_meta`` includes
        ``name``, ``run`` overrides, and ``path``.
    """
    from tomojepa.core.aug_config import parse_augmentations_block

    if yaml is None:
        raise ImportError(
            "PyYAML is required for job configs; pip install pyyaml"
        ) from _YAML_IMPORT_ERROR
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"job config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"job config root must be a mapping, got {type(data).__name__}")
    sched = TrainingSchedule.from_dict(data) if "stages" in data else None
    aug_block = data.get("augmentations")
    aug_cfg, aug_sched = parse_augmentations_block(
        aug_block if isinstance(aug_block, Mapping) else None)
    meta = job_meta_from_dict(data, path=p)
    return sched, aug_cfg, aug_sched, meta
