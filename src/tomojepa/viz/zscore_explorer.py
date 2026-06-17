"""Interactive Plotly Dash explorer for ViT-Up feature similarity maps.

Workflow
--------
1. Pick a reference region on the left image (radius click / flood fill / lasso).
2. Save it as a named **probe** (with an optional free-text descriptor).
3. Combine probes into a **score field**:
   - ``aggregate``: pool all probe pixels, recompute one mean/sigma, score once.
   - ``max``: score each probe separately and take the per-pixel maximum.
   Each score field has its own threshold -> a binary mask.
4. Combine score-field masks logically (AND / OR / XOR) into a **final mask**.

Display controls (colorscale, gamma, clipping) only affect rendering -- the
similarity metric is never modified. Zoom/pan is coupled between panels.

Example:
    tomojepa viz zscore --ckpt runs/vitup_soil_1024_5ep/ckpt/ckpt_last.pth \\
        --slice 871 --data_dir . --pattern soild_stack.zarr
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path as FsPath

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from matplotlib.path import Path
from plotly.colors import sample_colorscale
from scipy.stats import norm
from torch.amp import autocast

from ..core.dataset import TomographyDataset
from ..vitup.infer import load_vitup

COLORSCALES = [
    "gray", "Viridis", "Inferno", "Magma", "Plasma", "Cividis",
    "Hot", "Jet", "Turbo", "Greys",
]
LOGICAL_OPS = ["OR", "AND", "XOR"]


def parse_args():
    p = argparse.ArgumentParser(description="ViT-Up similarity Dash explorer")
    p.add_argument("--ckpt", default="runs/vitup_soil_1024_5ep/ckpt/ckpt_last.pth")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="soild_stack.zarr")
    p.add_argument("--backend", choices=["auto", "h5", "zarr"], default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--slice", type=int, default=871, dest="slice_idx")
    p.add_argument("--backbone_res", type=int, default=512)
    p.add_argument("--upsample", type=int, default=512)
    p.add_argument("--query_chunk_size", type=int, default=32768)
    p.add_argument("--probes_path", default=None,
                   help="JSON file to load/save probes + score fields")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


@torch.no_grad()
def extract_slice_features(args, device):
    """Return grayscale image ``[H,W]`` and dense features ``[H,W,C]``."""
    vitup, _cfg = load_vitup(args.ckpt, device)
    ds = TomographyDataset(
        data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
        global_views=1, local_views=0, variant="tomo2",
        img_size=max(args.backbone_res, args.upsample), is_train=False,
        backend=args.backend,
    )
    img = ds[int(args.slice_idx)][0].unsqueeze(0).to(device)
    img_in = F.interpolate(img, size=(args.backbone_res, args.backbone_res),
                           mode="bilinear", align_corners=False)
    use_amp = device.type == "cuda"
    with autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        dense = vitup.upsample(img_in, args.upsample, args.upsample,
                               chunk_size=args.query_chunk_size)[0]
    image = img_in[0, 0].float().cpu().numpy()
    features = dense.float().cpu().numpy()
    return image, features


# --------------------------------------------------------------------------- #
# Geometry / masks
# --------------------------------------------------------------------------- #
def disk_mask(h: int, w: int, cy: int, cx: int, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def flood_mask(image: np.ndarray, cy: int, cx: int, tolerance: float) -> np.ndarray:
    """Connected component containing ``(cy,cx)`` with |I - I_seed| <= tolerance."""
    h, w = image.shape
    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))
    seed_val = float(image[cy, cx])
    candidate = np.abs(image - seed_val) <= tolerance
    out = np.zeros((h, w), dtype=bool)
    if not candidate[cy, cx]:
        out[cy, cx] = True
        return out
    q = deque([(cy, cx)])
    out[cy, cx] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not out[ny, nx] and candidate[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def mask_from_pixels(pixels: list, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for item in pixels:
        x, y = int(round(item[0])), int(round(item[1]))
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = True
    return mask


def mask_from_lasso(points: list, h: int, w: int) -> np.ndarray:
    path = Path([(float(p[0]), float(p[1])) for p in points])
    yy, xx = np.mgrid[:h, :w]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return path.contains_points(coords).reshape(h, w)


def probe_to_mask(probe: dict, image: np.ndarray) -> np.ndarray:
    """Recompute a boolean mask for a stored probe definition."""
    h, w = image.shape
    t = probe.get("type")
    if t == "radius":
        return disk_mask(h, w, int(probe["cy"]), int(probe["cx"]),
                         float(probe["radius"]))
    if t == "mask":
        return flood_mask(image, int(probe["cy"]), int(probe["cx"]),
                          float(probe["tolerance"]))
    if t == "lasso":
        poly = probe.get("polygon") or []
        if len(poly) >= 3:
            return mask_from_lasso(poly, h, w)
    return np.zeros((h, w), dtype=bool)


# --------------------------------------------------------------------------- #
# Similarity metric
# --------------------------------------------------------------------------- #
def reference_stats(features: np.ndarray, mask: np.ndarray, eps: float = 1e-6):
    sel = features[mask]
    if sel.size == 0:
        c = features.shape[-1]
        return np.zeros(c, dtype=np.float32), np.ones(c, dtype=np.float32)
    mu = sel.mean(axis=0)
    sigma = np.maximum(sel.std(axis=0), eps)
    return mu, sigma


def rms_zscore_map(features: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (features - mu) / sigma
    return np.sqrt(np.mean(z ** 2, axis=-1))


def similarity_map(rms: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sel = rms[mask]
    if sel.size == 0:
        return np.zeros_like(rms, dtype=np.float32)
    mu_r = float(sel.mean())
    sig_r = float(sel.std())
    if sig_r < 1e-5:
        rmin = float(sel.min())
        span = max(float(rms.max()) - rmin, eps)
        return np.clip(1.0 - (rms - rmin) / span, 0.0, 1.0).astype(np.float32)
    t = (rms - mu_r) / sig_r
    return (1.0 - norm.cdf(t)).astype(np.float32)


def similarity_for_mask(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Full similarity map for a single reference mask."""
    mu, sigma = reference_stats(features, mask)
    rms = rms_zscore_map(features, mu, sigma)
    return similarity_map(rms, mask)


def score_field_map(features: np.ndarray, masks: list[np.ndarray],
                    mode: str) -> np.ndarray:
    """Combine probe masks into one score field in [0, 1].

    ``aggregate``: union all pixels, recompute a single mean/sigma, score once.
    ``max``: score each probe independently, take per-pixel maximum.
    """
    valid = [m for m in masks if m is not None and m.any()]
    if not valid:
        h, w = features.shape[:2]
        return np.zeros((h, w), dtype=np.float32)
    if mode == "aggregate":
        union = np.zeros_like(valid[0], dtype=bool)
        for m in valid:
            union |= m
        return similarity_for_mask(features, union)
    acc = None
    for m in valid:
        s = similarity_for_mask(features, m)
        acc = s if acc is None else np.maximum(acc, s)
    return acc.astype(np.float32)


def threshold_mask(score: np.ndarray, lo: float, hi: float = 1.0) -> np.ndarray:
    """Binary mask for score values within ``[lo, hi]``."""
    return (score >= lo) & (score <= hi)


def combine_masks(masks: list[np.ndarray], op: str) -> np.ndarray | None:
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    res = valid[0].copy()
    for m in valid[1:]:
        if op == "AND":
            res &= m
        elif op == "XOR":
            res ^= m
        else:  # OR
            res |= m
    return res


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def apply_display_gamma(sim: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law mapping of values in [0,1]; gamma < 1 boosts low values.

    Retained for tests/reference. Display warps the colorscale instead.
    """
    g = max(float(gamma), 0.05)
    out = np.array(sim, dtype=np.float32, copy=True)
    valid = np.isfinite(out)
    out[valid] = np.power(np.clip(out[valid], 0.0, 1.0), g)
    return out


def gamma_colorscale(base: str, gamma: float, n: int = 64) -> list[list]:
    """Plotly colorscale whose color *positions* are gamma-warped.

    Data values are unchanged; only the value->color mapping is distorted.
    ``gamma < 1`` devotes more of the colormap to low values.
    """
    g = max(float(gamma), 0.05)
    ts = np.linspace(0.0, 1.0, n)
    us = np.clip(np.power(ts, g), 0.0, 1.0)
    colors = sample_colorscale(base, [float(u) for u in us])
    return [[float(t), c] for t, c in zip(ts, colors)]


def _point_on_input_panel(point: dict) -> bool:
    ax = str(point.get("xaxis", point.get("xref", "x")))
    return "x2" not in ax and "x3" not in ax


def lasso_selection(selected_data, h: int, w: int) -> list[list[float]] | None:
    """Polygon vertices from lasso (``lassoPoints``) or box (``range``)."""
    if not selected_data:
        return None
    lp = selected_data.get("lassoPoints")
    if lp and lp.get("x") and lp.get("y"):
        pts = [[float(x), float(y)] for x, y in zip(lp["x"], lp["y"])]
        return pts if len(pts) >= 3 else None
    rng = selected_data.get("range")
    if rng and rng.get("x") and rng.get("y"):
        (x0, x1), (y0, y1) = rng["x"], rng["y"]
        return [[float(x0), float(y0)], [float(x1), float(y0)],
                [float(x1), float(y1)], [float(x0), float(y1)]]
    pts = []
    for p in selected_data.get("points", []):
        if not _point_on_input_panel(p):
            continue
        pts.append([float(p["x"]), float(p["y"])])
    return pts if len(pts) >= 3 else None


def _parse_click(click_data) -> tuple[int, int] | None:
    if not click_data or not click_data.get("points"):
        return None
    for pt in click_data["points"]:
        if not _point_on_input_panel(pt):
            continue
        return int(round(float(pt["y"]))), int(round(float(pt["x"])))
    return None


def _empty_similarity(h: int, w: int) -> np.ndarray:
    return np.full((h, w), np.nan, dtype=np.float32)


def _axis_layout(h: int, w: int) -> dict:
    return dict(
        xaxis=dict(constrain="domain", range=[-0.5, w - 0.5], fixedrange=False),
        yaxis=dict(autorange="reversed", range=[h - 0.5, -0.5],
                   scaleanchor="x", scaleratio=1, fixedrange=False),
    )


def extract_view(relayout: dict | None):
    """Return ``"auto"``, ``([x0,x1],[y0,y1])``, or ``None`` from relayoutData."""
    if not relayout:
        return None
    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        return "auto"
    keys = ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]
    if all(k in relayout for k in keys):
        return ([relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]],
                [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]])
    if "xaxis.range" in relayout and "yaxis.range" in relayout:
        return (list(relayout["xaxis.range"]), list(relayout["yaxis.range"]))
    return None


def _add_overlay(fig, mask, color, xs, ys):
    if mask is None or not mask.any():
        return
    overlay = np.where(mask, 1.0, np.nan)
    fig.add_trace(go.Heatmap(
        z=overlay, x=xs, y=ys,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, color]],
        showscale=False, hoverinfo="skip",
    ))


def build_input_figure(image, mask, mode, title, colorscale="gray", gamma=1.0,
                       zmin=None, zmax=None, extra_mask=None,
                       extra_color="rgba(80,160,255,0.45)") -> go.Figure:
    h, w = image.shape
    xs, ys = list(range(w)), list(range(h))
    if zmin is None:
        zmin = float(np.nanmin(image))
    if zmax is None:
        zmax = float(np.nanmax(image))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=image, x=xs, y=ys, colorscale=gamma_colorscale(colorscale, gamma),
        zmin=zmin, zmax=zmax, xgap=0, ygap=0,
        colorbar=dict(title="I", len=0.85, x=1.02),
        hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.4f}<extra></extra>",
    ))
    _add_overlay(fig, extra_mask, extra_color, xs, ys)
    _add_overlay(fig, mask, "rgba(255,80,80,0.55)", xs, ys)
    panel_w = 520
    fig.update_layout(
        title=title, width=panel_w + 90, height=int(panel_w * h / w) + 80,
        margin=dict(l=10, r=70, t=40, b=10),
        dragmode="lasso" if mode == "lasso" else "zoom",
        uirevision=f"input-{mode}", **_axis_layout(h, w),
    )
    return fig


def build_score_figure(field, colorscale="Viridis", gamma=1.0, zmin=0.0,
                       zmax=1.0, title="Score field", binary=False) -> go.Figure:
    h, w = field.shape
    xs, ys = list(range(w)), list(range(h))
    fig = go.Figure()
    if binary:
        fig.add_trace(go.Heatmap(
            z=field.astype(np.float32), x=xs, y=ys,
            colorscale=[[0, "rgb(20,20,30)"], [1, "rgb(250,220,60)"]],
            zmin=0, zmax=1, xgap=0, ygap=0,
            colorbar=dict(title="mask", len=0.85, x=1.02),
            hovertemplate="x=%{x}<br>y=%{y}<br>m=%{z:.0f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Heatmap(
            z=field, x=xs, y=ys, colorscale=gamma_colorscale(colorscale, gamma),
            zmin=zmin, zmax=zmax, xgap=0, ygap=0,
            colorbar=dict(title="score", len=0.85, x=1.02),
            hovertemplate="x=%{x}<br>y=%{y}<br>s=%{z:.3f}<extra></extra>",
        ))
    panel_w = 520
    fig.update_layout(
        title=title, width=panel_w + 90, height=int(panel_w * h / w) + 80,
        margin=dict(l=10, r=70, t=40, b=10), uirevision="sim",
        **_axis_layout(h, w),
    )
    return fig


_INPUT_GRAPH_CONFIG = {
    "displayModeBar": True, "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToAdd": ["lasso2d", "select2d"],
    "modeBarButtonsToRemove": ["autoScale2d"],
}
_SIM_GRAPH_CONFIG = {
    "displayModeBar": True, "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _cscale_options():
    return [{"label": c, "value": c} for c in COLORSCALES]


def _probe_options(probes):
    out = []
    for p in probes:
        desc = f" — {p['descriptor']}" if p.get("descriptor") else ""
        out.append({"label": f"{p['name']} [{p['type']}]{desc} (n={p.get('n', 0)})",
                    "value": p["id"]})
    return out


def _field_options(fields):
    return [{"label": f"{f['name']} [{f['mode']}] thr={f['threshold']:.2f}",
             "value": f["id"]} for f in fields]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def build_app(image, features, slice_idx, probes_path=None) -> Dash:
    h, w = image.shape
    rmax = min(h, w) // 2
    img_lo, img_hi = float(np.nanmin(image)), float(np.nanmax(image))
    img_step = max((img_hi - img_lo) / 200.0, 1e-6)
    app = Dash(__name__)
    app.title = f"ViT-Up similarity explorer (slice {slice_idx})"
    _features = features
    empty_sim = _empty_similarity(h, w)

    init_probes, init_fields = [], []
    if probes_path and FsPath(probes_path).exists():
        try:
            blob = json.loads(FsPath(probes_path).read_text())
            init_probes = blob.get("probes", [])
            init_fields = blob.get("fields", [])
        except Exception:
            pass

    init_input = build_input_figure(image, None, "radius",
                                    "Input — click or lasso to select reference",
                                    "gray", 1.0, img_lo, img_hi)
    init_sim = build_score_figure(empty_sim, "Viridis", 1.0, 0.0, 1.0,
                                  "Score field (none yet)")

    ctrl = {"marginTop": "8px"}
    section = {"borderTop": "1px solid #ddd", "marginTop": "12px", "paddingTop": "8px"}

    app.layout = html.Div([
        html.H3(f"ViT-Up similarity explorer — slice {slice_idx}"),
        html.Div([
            html.Div([
                # ---- selection ----
                html.B("Selection"),
                dcc.Dropdown(id="mode", clearable=False, value="radius", options=[
                    {"label": "Radius (disk, click)", "value": "radius"},
                    {"label": "Mask (flood-fill, click)", "value": "mask"},
                    {"label": "Lasso (draw region)", "value": "lasso"},
                ]),
                html.Label("Radius (px)", style=ctrl),
                dcc.Input(id="radius-input", type="number", value=20, min=1,
                          max=rmax, step=1, debounce=True, style={"width": "80px"}),
                dcc.Slider(id="radius", min=1, max=rmax, step=1, value=20,
                           marks={5: "5", 50: "50", 100: "100"}),
                html.Label("Mask tolerance", style=ctrl),
                dcc.Slider(id="tolerance", min=0.0001, max=0.05, step=0.0001,
                           value=0.002, marks={0.005: "0.005", 0.02: "0.02"}),

                # ---- probes ----
                html.Div([
                    html.B("Probes"),
                    dcc.Input(id="descriptor", type="text", placeholder="descriptor (optional)",
                              debounce=True, style={"width": "100%", "marginTop": "6px"}),
                    html.Button("Add current selection", id="add-probe",
                                n_clicks=0, style={"marginTop": "6px", "width": "100%"}),
                    dcc.Checklist(id="probe-list", options=_probe_options(init_probes),
                                  value=[], style={"marginTop": "6px", "maxHeight": "140px",
                                                   "overflowY": "auto", "fontSize": "12px"}),
                    html.Button("Remove checked probes", id="remove-probe",
                                n_clicks=0, style={"marginTop": "4px", "width": "100%"}),
                ], style=section),

                # ---- score fields ----
                html.Div([
                    html.B("Score field (from checked probes)"),
                    dcc.Input(id="field-name", type="text", placeholder="field name",
                              debounce=True, style={"width": "100%", "marginTop": "6px"}),
                    dcc.RadioItems(id="field-mode", value="aggregate", options=[
                        {"label": "aggregate (pool mu/sigma)", "value": "aggregate"},
                        {"label": "max (per-pixel max score)", "value": "max"},
                    ], style={"fontSize": "12px", "marginTop": "4px"}),
                    html.Button("Create score field", id="create-field",
                                n_clicks=0, style={"marginTop": "4px", "width": "100%"}),
                    html.Label("Edit score field", style=ctrl),
                    dcc.Dropdown(id="field-select", options=_field_options(init_fields),
                                 value=None, placeholder="select a field"),
                    html.Label("Threshold (score in [lo, 1])", style=ctrl),
                    dcc.Slider(id="threshold", min=0.0, max=1.0, step=0.01, value=0.5,
                               marks={0: "0", 0.5: "0.5", 1: "1"}),
                    html.Button("Remove selected field", id="remove-field",
                                n_clicks=0, style={"marginTop": "4px", "width": "100%"}),
                ], style=section),

                # ---- final mask ----
                html.Div([
                    html.B("Final mask (logical combine)"),
                    dcc.Checklist(id="final-fields", options=_field_options(init_fields),
                                  value=[], style={"marginTop": "6px", "fontSize": "12px"}),
                    dcc.RadioItems(id="final-op", value="OR",
                                   options=[{"label": o, "value": o} for o in LOGICAL_OPS],
                                   inline=True, style={"fontSize": "12px"}),
                ], style=section),

                # ---- right view + display ----
                html.Div([
                    html.B("Right panel shows"),
                    dcc.RadioItems(id="right-view", value="live", options=[
                        {"label": "Live similarity", "value": "live"},
                        {"label": "Score field", "value": "field"},
                        {"label": "Field mask", "value": "fieldmask"},
                        {"label": "Final mask", "value": "final"},
                    ], style={"fontSize": "12px"}),
                ], style=section),

                html.Div([
                    html.B("Tomogram display"),
                    dcc.Dropdown(id="img-cscale", options=_cscale_options(),
                                 value="gray", clearable=False, style={"marginTop": "4px"}),
                    html.Label("Gamma (color only)", style=ctrl),
                    dcc.Slider(id="img-gamma", min=0.15, max=3.0, step=0.05, value=1.0,
                               marks={0.3: "0.3", 1.0: "1", 2.0: "2"}),
                    html.Label("Clip (intensity)", style=ctrl),
                    dcc.RangeSlider(id="img-clip", min=img_lo, max=img_hi, step=img_step,
                                    value=[img_lo, img_hi], marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True}),
                ], style=section),
                html.Div([
                    html.B("Score display"),
                    dcc.Dropdown(id="sim-cscale", options=_cscale_options(),
                                 value="Viridis", clearable=False, style={"marginTop": "4px"}),
                    html.Label("Gamma (color only)", style=ctrl),
                    dcc.Slider(id="sim-gamma", min=0.15, max=3.0, step=0.05, value=1.0,
                               marks={0.3: "0.3", 1.0: "1", 2.0: "2"}),
                    html.Label("Clip (score)", style=ctrl),
                    dcc.RangeSlider(id="sim-clip", min=0.0, max=1.0, step=0.01,
                                    value=[0.0, 1.0], marks={0: "0", 1: "1"},
                                    tooltip={"placement": "bottom", "always_visible": True}),
                ], style=section),

                # ---- save / load ----
                html.Div([
                    html.B("Save / load"),
                    dcc.Input(id="save-path", type="text",
                              value=probes_path or "probes.json",
                              style={"width": "100%", "marginTop": "6px"}),
                    html.Div([
                        html.Button("Save", id="save-btn", n_clicks=0,
                                    style={"width": "49%"}),
                        html.Button("Load", id="load-btn", n_clicks=0,
                                    style={"width": "49%", "marginLeft": "2%"}),
                    ], style={"marginTop": "4px"}),
                    html.Div(id="io-status", style={"fontSize": "11px", "color": "#666",
                                                    "marginTop": "4px"}),
                ], style=section),

                html.Div(id="stats", style={"marginTop": "12px", "fontFamily": "monospace",
                                            "whiteSpace": "pre-wrap", "fontSize": "12px"}),
            ], style={"width": "320px", "paddingRight": "16px", "flexShrink": 0,
                      "maxHeight": "95vh", "overflowY": "auto"}),
            html.Div([
                html.Div([dcc.Graph(id="input-graph", figure=init_input,
                                    config=_INPUT_GRAPH_CONFIG)],
                         style={"display": "inline-block", "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="sim-graph", figure=init_sim,
                                    config=_SIM_GRAPH_CONFIG)],
                         style={"display": "inline-block", "verticalAlign": "top"}),
            ]),
        ], style={"display": "flex"}),
        dcc.Store(id="click-store", data=None),
        dcc.Store(id="lasso-store", data=None),
        dcc.Store(id="probes-store", data=init_probes),
        dcc.Store(id="fields-store", data=init_fields),
    ])

    # ---- radius sync ----
    @app.callback(
        Output("radius", "value"), Output("radius-input", "value"),
        Input("radius", "value"), Input("radius-input", "value"),
        prevent_initial_call=True,
    )
    def sync_radius(slider_val, input_val):
        tid = ctx.triggered_id
        if tid == "radius-input" and input_val is not None:
            v = int(np.clip(int(input_val), 1, rmax))
            return v, v
        if slider_val is not None:
            return int(slider_val), int(slider_val)
        return no_update, no_update

    # ---- capture live selection ----
    @app.callback(
        Output("click-store", "data"), Output("lasso-store", "data"),
        Input("input-graph", "clickData"), Input("input-graph", "selectedData"),
        Input("mode", "value"),
        State("click-store", "data"), State("lasso-store", "data"),
        prevent_initial_call=True,
    )
    def capture_selection(click_data, selected_data, mode, click, lasso):
        trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if trig == "mode.value":
            return None, None
        if trig == "input-graph.clickData" and mode != "lasso":
            parsed = _parse_click(click_data)
            if parsed is not None:
                return {"x": parsed[1], "y": parsed[0]}, None
        if trig == "input-graph.selectedData" and mode == "lasso":
            pts = lasso_selection(selected_data, h, w)
            if pts:
                return None, pts
        return no_update, no_update

    def _live_mask(mode, click, lasso, radius, tolerance):
        if mode == "lasso":
            if not lasso or len(lasso) < 3:
                return None, "lasso (draw a region)"
            m = mask_from_lasso(lasso, h, w)
            return m, f"lasso n={int(m.sum())}"
        if not click:
            return None, "no selection"
        cy = int(np.clip(int(click["y"]), 0, h - 1))
        cx = int(np.clip(int(click["x"]), 0, w - 1))
        if mode == "mask":
            m = flood_mask(image, cy, cx, tolerance)
            return m, f"flood tol={tolerance:.4f} @({cx},{cy})"
        m = disk_mask(h, w, cy, cx, radius)
        return m, f"radius={radius} @({cx},{cy})"

    # ---- probe management ----
    @app.callback(
        Output("probes-store", "data"),
        Input("add-probe", "n_clicks"), Input("remove-probe", "n_clicks"),
        State("probes-store", "data"), State("probe-list", "value"),
        State("mode", "value"), State("click-store", "data"),
        State("lasso-store", "data"), State("radius", "value"),
        State("tolerance", "value"), State("descriptor", "value"),
        prevent_initial_call=True,
    )
    def manage_probes(add_n, rm_n, probes, checked, mode, click, lasso,
                      radius, tolerance, descriptor):
        probes = list(probes or [])
        tid = ctx.triggered_id
        if tid == "remove-probe":
            checked = set(checked or [])
            return [p for p in probes if p["id"] not in checked]
        # add
        mask, _ = _live_mask(mode, click, lasso, radius, tolerance)
        if mask is None or not mask.any():
            return no_update
        nid = (max([p["id"] for p in probes]) + 1) if probes else 1
        probe = {"id": nid, "name": f"probe {nid}", "type": mode,
                 "descriptor": (descriptor or "").strip(), "n": int(mask.sum())}
        if mode == "lasso":
            probe["polygon"] = lasso
        else:
            probe["cx"] = int(click["x"])
            probe["cy"] = int(click["y"])
            probe["radius"] = int(radius)
            probe["tolerance"] = float(tolerance)
        probes.append(probe)
        return probes

    # ---- field management ----
    @app.callback(
        Output("fields-store", "data"),
        Input("create-field", "n_clicks"), Input("remove-field", "n_clicks"),
        Input("threshold", "value"),
        State("fields-store", "data"), State("probe-list", "value"),
        State("field-name", "value"), State("field-mode", "value"),
        State("field-select", "value"),
        prevent_initial_call=True,
    )
    def manage_fields(create_n, rm_n, threshold, fields, checked_probes,
                      field_name, field_mode, field_sel):
        fields = list(fields or [])
        tid = ctx.triggered_id
        if tid == "threshold":
            if field_sel is None:
                return no_update
            for f in fields:
                if f["id"] == field_sel:
                    f["threshold"] = float(threshold)
            return fields
        if tid == "remove-field":
            if field_sel is None:
                return no_update
            return [f for f in fields if f["id"] != field_sel]
        # create
        if not checked_probes:
            return no_update
        nid = (max([f["id"] for f in fields]) + 1) if fields else 1
        name = (field_name or "").strip() or f"field {nid}"
        fields.append({"id": nid, "name": name, "mode": field_mode or "aggregate",
                       "probe_ids": list(checked_probes),
                       "threshold": float(threshold) if threshold is not None else 0.5})
        return fields

    # ---- option syncing ----
    @app.callback(Output("probe-list", "options"), Input("probes-store", "data"))
    def _probe_opts(probes):
        return _probe_options(probes or [])

    @app.callback(
        Output("field-select", "options"), Output("final-fields", "options"),
        Input("fields-store", "data"),
    )
    def _field_opts(fields):
        opts = _field_options(fields or [])
        return opts, opts

    @app.callback(
        Output("threshold", "value"),
        Input("field-select", "value"), State("fields-store", "data"),
        prevent_initial_call=True,
    )
    def _load_threshold(field_sel, fields):
        for f in (fields or []):
            if f["id"] == field_sel:
                return f["threshold"]
        return no_update

    # ---- save / load ----
    @app.callback(
        Output("io-status", "children"),
        Output("probes-store", "data", allow_duplicate=True),
        Output("fields-store", "data", allow_duplicate=True),
        Input("save-btn", "n_clicks"), Input("load-btn", "n_clicks"),
        State("save-path", "value"), State("probes-store", "data"),
        State("fields-store", "data"),
        prevent_initial_call=True,
    )
    def save_load(save_n, load_n, path, probes, fields):
        tid = ctx.triggered_id
        if not path:
            return "no path", no_update, no_update
        fp = FsPath(path)
        if tid == "save-btn":
            fp.write_text(json.dumps({"probes": probes or [], "fields": fields or []},
                                     indent=2))
            return f"saved {len(probes or [])} probes, {len(fields or [])} fields", \
                no_update, no_update
        if tid == "load-btn":
            if not fp.exists():
                return f"not found: {path}", no_update, no_update
            blob = json.loads(fp.read_text())
            return (f"loaded {len(blob.get('probes', []))} probes, "
                    f"{len(blob.get('fields', []))} fields"), \
                blob.get("probes", []), blob.get("fields", [])
        return no_update, no_update, no_update

    # ---- render ----
    @app.callback(
        Output("input-graph", "figure"), Output("sim-graph", "figure"),
        Output("stats", "children"),
        Input("click-store", "data"), Input("lasso-store", "data"),
        Input("mode", "value"), Input("radius", "value"), Input("tolerance", "value"),
        Input("img-cscale", "value"), Input("img-gamma", "value"),
        Input("img-clip", "value"), Input("sim-cscale", "value"),
        Input("sim-gamma", "value"), Input("sim-clip", "value"),
        Input("right-view", "value"), Input("field-select", "value"),
        Input("threshold", "value"), Input("final-fields", "value"),
        Input("final-op", "value"), Input("probes-store", "data"),
        Input("fields-store", "data"),
        prevent_initial_call=True,
    )
    def render(click, lasso, mode, radius, tolerance, img_cscale, img_gamma,
               img_clip, sim_cscale, sim_gamma, sim_clip, right_view, field_sel,
               threshold, final_fields, final_op, probes, fields):
        img_gamma = float(img_gamma or 1.0)
        sim_gamma = float(sim_gamma or 1.0)
        img_zmin, img_zmax = (img_clip or [img_lo, img_hi])
        sim_zmin, sim_zmax = (sim_clip or [0.0, 1.0])
        probes = probes or []
        fields = fields or []
        probe_by_id = {p["id"]: p for p in probes}
        field_by_id = {f["id"]: f for f in fields}

        live_mask, live_label = _live_mask(mode, click, lasso, radius, tolerance)

        def field_masks(field):
            return [probe_to_mask(probe_by_id[pid], image)
                    for pid in field.get("probe_ids", []) if pid in probe_by_id]

        def field_score(field):
            return score_field_map(_features, field_masks(field), field["mode"])

        # ---- right panel content ----
        stats_lines = [f"live: {live_label}", f"probes: {len(probes)}  fields: {len(fields)}"]
        extra_mask = None

        if right_view == "live":
            if live_mask is not None and live_mask.any():
                score = similarity_for_mask(_features, live_mask)
                sim_fig = build_score_figure(score, sim_cscale, sim_gamma,
                                             float(sim_zmin), float(sim_zmax),
                                             "Live similarity")
                sel = score[live_mask]
                stats_lines.append(f"sim in ref: mean={sel.mean():.3f} min={sel.min():.3f}")
            else:
                sim_fig = build_score_figure(empty_sim, sim_cscale, sim_gamma,
                                             float(sim_zmin), float(sim_zmax),
                                             "Live similarity (no selection)")
        elif right_view in ("field", "fieldmask"):
            field = field_by_id.get(field_sel)
            if field is None:
                sim_fig = build_score_figure(empty_sim, sim_cscale, sim_gamma,
                                             float(sim_zmin), float(sim_zmax),
                                             "Score field (select one)")
            else:
                score = field_score(field)
                if right_view == "field":
                    sim_fig = build_score_figure(score, sim_cscale, sim_gamma,
                                                 float(sim_zmin), float(sim_zmax),
                                                 f"Score: {field['name']} [{field['mode']}]")
                else:
                    thr = float(threshold if threshold is not None else field["threshold"])
                    bm = threshold_mask(score, thr)
                    extra_mask = bm
                    sim_fig = build_score_figure(bm, title=f"Mask: {field['name']} thr={thr:.2f}",
                                                 binary=True)
                    stats_lines.append(f"field mask px: {int(bm.sum())} @thr={thr:.2f}")
        else:  # final
            masks = []
            for fid in (final_fields or []):
                f = field_by_id.get(fid)
                if f is None:
                    continue
                masks.append(threshold_mask(field_score(f), float(f["threshold"])))
            final = combine_masks(masks, final_op)
            if final is None:
                sim_fig = build_score_figure(np.zeros((h, w)), title="Final mask (none)",
                                             binary=True)
            else:
                extra_mask = final
                sim_fig = build_score_figure(final, title=f"Final mask ({final_op}, "
                                             f"{len(masks)} fields)", binary=True)
                stats_lines.append(f"final mask px: {int(final.sum())}")

        in_fig = build_input_figure(image, live_mask, mode, f"Input — {live_label}",
                                    img_cscale, img_gamma, float(img_zmin),
                                    float(img_zmax), extra_mask=extra_mask)
        return in_fig, sim_fig, "\n".join(stats_lines)

    # ---- coupled zoom/pan ----
    @app.callback(
        Output("input-graph", "figure", allow_duplicate=True),
        Output("sim-graph", "figure", allow_duplicate=True),
        Input("input-graph", "relayoutData"), Input("sim-graph", "relayoutData"),
        prevent_initial_call=True,
    )
    def sync_axes(in_relayout, sim_relayout):
        trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if trig.startswith("input-graph"):
            view, target = extract_view(in_relayout), "sim"
        elif trig.startswith("sim-graph"):
            view, target = extract_view(sim_relayout), "input"
        else:
            raise PreventUpdate
        if view is None:
            raise PreventUpdate
        patch = Patch()
        if view == "auto":
            patch["layout"]["xaxis"]["autorange"] = True
            patch["layout"]["yaxis"]["autorange"] = True
        else:
            xr, yr = view
            patch["layout"]["xaxis"]["autorange"] = False
            patch["layout"]["xaxis"]["range"] = xr
            patch["layout"]["yaxis"]["autorange"] = False
            patch["layout"]["yaxis"]["range"] = yr
        if target == "sim":
            return no_update, patch
        return patch, no_update

    return app


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading slice {args.slice_idx} on {device}...", flush=True)
    image, features = extract_slice_features(args, device)
    print(f"  image {image.shape}, features {features.shape}", flush=True)
    app = build_app(image, features, args.slice_idx, probes_path=args.probes_path)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
