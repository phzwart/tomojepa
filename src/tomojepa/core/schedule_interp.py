"""Shared piecewise schedule knot parsing and interpolation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple, Union


def parse_knot(raw: Any, channel: str) -> Tuple[float, Union[float, bool]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{channel} knot must be a mapping, got {type(raw).__name__}")
    prog = raw.get("progress", raw.get("p"))
    val = raw.get("value", raw.get("v"))
    if prog is None or val is None:
        raise ValueError(f"{channel} knot needs progress/value keys: {raw!r}")
    p = float(prog)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{channel} progress must be in [0, 1], got {p}")
    if isinstance(val, bool):
        return p, val
    return p, float(val)


def interp_numeric(knots: Sequence[Tuple[float, float]], progress: float) -> float:
    if not knots:
        raise ValueError("numeric schedule requires at least one knot")
    pts = sorted(knots, key=lambda kv: kv[0])
    if progress <= pts[0][0]:
        return pts[0][1]
    if progress >= pts[-1][0]:
        return pts[-1][1]
    for (p0, v0), (p1, v1) in zip(pts[:-1], pts[1:]):
        if p0 <= progress <= p1:
            if p1 == p0:
                return v1
            t = (progress - p0) / (p1 - p0)
            return v0 + t * (v1 - v0)
    return pts[-1][1]


def interp_bool_sticky(knots: Sequence[Tuple[float, bool]], progress: float) -> bool:
    """Latch ``True`` once any knot with ``value: true`` is crossed."""
    if not knots:
        return False
    frozen = False
    for p, v in sorted(knots, key=lambda kv: kv[0]):
        if progress >= p and v:
            frozen = True
    return frozen
