"""Interactive Plotly Dash explorer for ViT-Up feature similarity maps.

Load one slice + upsampled dense features. Pick a reference region by click
(radius / flood-fill) or lasso, then color every pixel by similarity to that
reference (normal-CDF score in [0, 1]; 1 = very similar).

Display controls (colorscale, gamma, clipping) only affect how values are
rendered -- the underlying similarity metric is never modified. Zoom and pan
are coupled between the two panels.

Example:
    tomojepa viz zscore --ckpt runs/vitup_soil_1024_5ep/ckpt/ckpt_last.pth \\
        --slice 871 --data_dir . --pattern soild_stack.zarr
"""
from __future__ import annotations

import argparse
from collections import deque

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


def apply_display_gamma(sim: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law mapping of values in [0,1]; gamma < 1 boosts low values.

    Retained for reference/tests. Display now warps the *colorscale* instead,
    leaving stored values untouched (see :func:`gamma_colorscale`).
    """
    g = max(float(gamma), 0.05)
    out = np.array(sim, dtype=np.float32, copy=True)
    valid = np.isfinite(out)
    out[valid] = np.power(np.clip(out[valid], 0.0, 1.0), g)
    return out


def gamma_colorscale(base: str, gamma: float, n: int = 64) -> list[list]:
    """Build a Plotly colorscale whose color *positions* are gamma-warped.

    The data values are unchanged; only the mapping value->color is distorted.
    ``gamma < 1`` devotes more of the colormap to low values (more contrast in
    low-similarity / low-intensity regions); ``gamma > 1`` does the opposite.
    """
    g = max(float(gamma), 0.05)
    ts = np.linspace(0.0, 1.0, n)
    us = np.clip(np.power(ts, g), 0.0, 1.0)
    colors = sample_colorscale(base, [float(u) for u in us])
    return [[float(t), c] for t, c in zip(ts, colors)]


def _point_on_input_panel(point: dict) -> bool:
    """True when a Plotly event point belongs to the left (input) panel."""
    ax = str(point.get("xaxis", point.get("xref", "x")))
    return "x2" not in ax and "x3" not in ax


def lasso_selection(selected_data, h: int, w: int) -> list[list[float]] | None:
    """Return polygon vertices ``[[x, y], ...]`` from a lasso/box selection.

    Heatmap traces do not emit per-cell entries in ``selectedData["points"]``;
    Plotly instead reports the drawn shape under ``lassoPoints`` (free-form
    lasso) or ``range`` (box select). We return those polygon vertices so the
    interior can be filled with :func:`mask_from_lasso`.
    """
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


def _parse_click(click_data) -> tuple[int, int] | None:
    if not click_data or not click_data.get("points"):
        return None
    for pt in click_data["points"]:
        if not _point_on_input_panel(pt):
            continue
        cx = int(round(float(pt["x"])))
        cy = int(round(float(pt["y"])))
        return cy, cx
    return None


def _empty_similarity(h: int, w: int) -> np.ndarray:
    return np.full((h, w), np.nan, dtype=np.float32)


def _axis_layout(h: int, w: int) -> dict:
    return dict(
        xaxis=dict(constrain="domain", range=[-0.5, w - 0.5], fixedrange=False),
        yaxis=dict(
            autorange="reversed",
            range=[h - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
            fixedrange=False,
        ),
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


def build_input_figure(
    image: np.ndarray,
    mask: np.ndarray | None,
    mode: str,
    title: str,
    colorscale: str = "gray",
    gamma: float = 1.0,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    h, w = image.shape
    xs = list(range(w))
    ys = list(range(h))
    if zmin is None:
        zmin = float(np.nanmin(image))
    if zmax is None:
        zmax = float(np.nanmax(image))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=image, x=xs, y=ys,
        colorscale=gamma_colorscale(colorscale, gamma),
        zmin=zmin, zmax=zmax,
        xgap=0, ygap=0,
        colorbar=dict(title="I", len=0.85, x=1.02),
        hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.4f}<extra></extra>",
    ))
    if mask is not None and mask.any():
        overlay = np.where(mask, 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            z=overlay, x=xs, y=ys,
            colorscale=[[0, "rgba(255,80,80,0)"], [1, "rgba(255,80,80,0.55)"]],
            showscale=False, hoverinfo="skip",
        ))
    panel_w = 520
    fig.update_layout(
        title=title,
        width=panel_w + 90,
        height=int(panel_w * h / w) + 80,
        margin=dict(l=10, r=70, t=40, b=10),
        dragmode="lasso" if mode == "lasso" else "zoom",
        uirevision=f"input-{mode}",
        **_axis_layout(h, w),
    )
    return fig


def build_sim_figure(
    similarity: np.ndarray,
    colorscale: str = "Viridis",
    gamma: float = 1.0,
    zmin: float = 0.0,
    zmax: float = 1.0,
    title: str = "Similarity (1 = like reference)",
) -> go.Figure:
    h, w = similarity.shape
    xs = list(range(w))
    ys = list(range(h))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=similarity, x=xs, y=ys,
        colorscale=gamma_colorscale(colorscale, gamma),
        zmin=zmin, zmax=zmax,
        xgap=0, ygap=0,
        colorbar=dict(title="sim", len=0.85, x=1.02),
        hovertemplate="x=%{x}<br>y=%{y}<br>sim=%{z:.3f}<extra></extra>",
    ))
    panel_w = 520
    fig.update_layout(
        title=title,
        width=panel_w + 90,
        height=int(panel_w * h / w) + 80,
        margin=dict(l=10, r=70, t=40, b=10),
        uirevision="sim",
        **_axis_layout(h, w),
    )
    return fig


_INPUT_GRAPH_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["lasso2d", "select2d"],
    "modeBarButtonsToRemove": ["autoScale2d"],
}

_SIM_GRAPH_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _cscale_options():
    return [{"label": c, "value": c} for c in COLORSCALES]


def build_app(image: np.ndarray, features: np.ndarray, slice_idx: int) -> Dash:
    h, w = image.shape
    rmax = min(h, w) // 2
    img_lo = float(np.nanmin(image))
    img_hi = float(np.nanmax(image))
    img_step = max((img_hi - img_lo) / 200.0, 1e-6)
    app = Dash(__name__)
    app.title = f"ViT-Up similarity explorer (slice {slice_idx})"

    empty_sim = _empty_similarity(h, w)
    init_input = build_input_figure(
        image, None, "radius", "Input — click or lasso to select reference",
        colorscale="gray", gamma=1.0, zmin=img_lo, zmax=img_hi,
    )
    init_sim = build_sim_figure(empty_sim, "Viridis", 1.0, 0.0, 1.0)

    ctrl_style = {"marginTop": "10px"}
    section_style = {"borderTop": "1px solid #ddd", "marginTop": "14px",
                     "paddingTop": "8px"}

    app.layout = html.Div([
        html.H3(f"ViT-Up similarity explorer — slice {slice_idx}"),
        html.P(
            "Select a reference region on the left image (click + radius, "
            "intensity flood-fill, or lasso). The right panel shows similarity "
            "(1 = very similar, 0 = dissimilar). Display controls only affect "
            "rendering; zoom/pan is coupled between panels."
        ),
        html.Div([
            html.Div([
                html.Label("Selection mode"),
                dcc.Dropdown(
                    id="mode",
                    options=[
                        {"label": "Radius (disk, click)", "value": "radius"},
                        {"label": "Mask (intensity flood-fill, click)", "value": "mask"},
                        {"label": "Lasso (draw region)", "value": "lasso"},
                    ],
                    value="radius", clearable=False,
                ),
                html.Label("Radius (px)", style=ctrl_style),
                dcc.Input(id="radius-input", type="number", value=20, min=1, max=rmax,
                          step=1, debounce=True, style={"width": "80px"}),
                dcc.Slider(id="radius", min=1, max=rmax, step=1, value=20,
                           marks={5: "5", 20: "20", 50: "50", 100: "100"}),
                html.Label("Mask tolerance (intensity)", style=ctrl_style),
                dcc.Slider(id="tolerance", min=0.0001, max=0.05, step=0.0001, value=0.002,
                           marks={0.001: "0.001", 0.005: "0.005", 0.02: "0.02"}),

                html.Div([
                    html.B("Tomogram display"),
                    html.Label("Colorscale", style=ctrl_style),
                    dcc.Dropdown(id="img-cscale", options=_cscale_options(),
                                 value="gray", clearable=False),
                    html.Label("Gamma (color only)", style=ctrl_style),
                    dcc.Slider(id="img-gamma", min=0.15, max=3.0, step=0.05, value=1.0,
                               marks={0.3: "0.3", 1.0: "1", 2.0: "2"}),
                    html.Label("Clip (intensity)", style=ctrl_style),
                    dcc.RangeSlider(id="img-clip", min=img_lo, max=img_hi,
                                    step=img_step, value=[img_lo, img_hi],
                                    marks=None,
                                    tooltip={"placement": "bottom",
                                             "always_visible": True}),
                ], style=section_style),

                html.Div([
                    html.B("Similarity display"),
                    html.Label("Colorscale", style=ctrl_style),
                    dcc.Dropdown(id="sim-cscale", options=_cscale_options(),
                                 value="Viridis", clearable=False),
                    html.Label("Gamma (color only)", style=ctrl_style),
                    dcc.Slider(id="sim-gamma", min=0.15, max=3.0, step=0.05, value=1.0,
                               marks={0.3: "0.3", 1.0: "1", 2.0: "2"}),
                    html.Label("Clip (similarity)", style=ctrl_style),
                    dcc.RangeSlider(id="sim-clip", min=0.0, max=1.0, step=0.01,
                                    value=[0.0, 1.0], marks={0: "0", 0.5: "0.5", 1: "1"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": True}),
                ], style=section_style),

                html.Div(
                    id="stats",
                    children="Click the left image to select a reference region.",
                    style={"marginTop": "16px", "fontFamily": "monospace",
                             "whiteSpace": "pre-wrap", "fontSize": "12px"},
                ),
            ], style={"width": "300px", "paddingRight": "16px", "flexShrink": 0}),
            html.Div([
                html.Div([
                    dcc.Graph(id="input-graph", figure=init_input,
                              config=_INPUT_GRAPH_CONFIG),
                ], style={"display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    dcc.Graph(id="sim-graph", figure=init_sim,
                              config=_SIM_GRAPH_CONFIG),
                ], style={"display": "inline-block", "verticalAlign": "top"}),
            ]),
        ], style={"display": "flex"}),
        dcc.Store(id="click-store", data=None),
        dcc.Store(id="lasso-store", data=None),
        dcc.Store(id="sim-store", data=None),
    ])

    _features = features
    _sel_triggers = {
        "input-graph.clickData", "input-graph.selectedData",
        "mode.value", "radius.value", "tolerance.value",
    }

    @app.callback(
        Output("radius", "value"),
        Output("radius-input", "value"),
        Input("radius", "value"),
        Input("radius-input", "value"),
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

    def _resolve_mask(mode, click, lasso_pts, radius, tolerance):
        if mode == "lasso":
            if not lasso_pts or len(lasso_pts) < 3:
                return None, "lasso (draw a region)"
            mask = mask_from_lasso(lasso_pts, h, w)
            return mask, f"lasso, n={int(mask.sum())}"
        if not click:
            return None, "no selection"
        cy, cx = int(click["y"]), int(click["x"])
        cy = int(np.clip(cy, 0, h - 1))
        cx = int(np.clip(cx, 0, w - 1))
        if mode == "mask":
            mask = flood_mask(image, cy, cx, tolerance)
            return mask, f"mask tol={tolerance:.4f} @ ({cx},{cy})"
        mask = disk_mask(h, w, cy, cx, radius)
        return mask, f"radius={radius}px @ ({cx},{cy})"

    @app.callback(
        Output("input-graph", "figure"),
        Output("sim-graph", "figure"),
        Output("stats", "children"),
        Output("click-store", "data"),
        Output("lasso-store", "data"),
        Output("sim-store", "data"),
        Input("input-graph", "clickData"),
        Input("input-graph", "selectedData"),
        Input("mode", "value"),
        Input("radius", "value"),
        Input("tolerance", "value"),
        Input("img-cscale", "value"),
        Input("img-gamma", "value"),
        Input("img-clip", "value"),
        Input("sim-cscale", "value"),
        Input("sim-gamma", "value"),
        Input("sim-clip", "value"),
        State("click-store", "data"),
        State("lasso-store", "data"),
        State("sim-store", "data"),
        prevent_initial_call=True,
    )
    def update_map(click_data, selected_data, mode, radius, tolerance,
                   img_cscale, img_gamma, img_clip,
                   sim_cscale, sim_gamma, sim_clip,
                   stored_click, stored_lasso, stored_sim):
        trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        img_gamma = float(img_gamma if img_gamma is not None else 1.0)
        sim_gamma = float(sim_gamma if sim_gamma is not None else 1.0)
        img_zmin, img_zmax = (img_clip if img_clip else [img_lo, img_hi])
        sim_zmin, sim_zmax = (sim_clip if sim_clip else [0.0, 1.0])
        click = dict(stored_click) if stored_click else None
        lasso_data = list(stored_lasso) if stored_lasso else None

        if trig == "input-graph.clickData" and mode != "lasso":
            parsed = _parse_click(click_data)
            if parsed is not None:
                click = {"x": parsed[1], "y": parsed[0]}
                lasso_data = None
        elif trig == "input-graph.selectedData" and mode == "lasso":
            pts = lasso_selection(selected_data, h, w)
            if pts:
                lasso_data = pts
                click = None
        elif trig == "mode.value":
            click = None
            lasso_data = None

        mask, label = _resolve_mask(mode, click, lasso_data, radius, tolerance)

        in_fig = build_input_figure(
            image, mask, mode, f"Input — {label}",
            colorscale=img_cscale, gamma=img_gamma,
            zmin=float(img_zmin), zmax=float(img_zmax),
        )

        if mask is None or not mask.any():
            sim_fig = build_sim_figure(
                empty_sim, sim_cscale, sim_gamma,
                float(sim_zmin), float(sim_zmax),
            )
            hint = (
                "Lasso mode: draw a closed region on the left image."
                if mode == "lasso" else "Click the left image."
            )
            return in_fig, sim_fig, hint, click, lasso_data, None

        recompute = trig in _sel_triggers or stored_sim is None
        if recompute:
            mu, sigma = reference_stats(_features, mask)
            rms = rms_zscore_map(_features, mu, sigma)
            sim_raw = similarity_map(rms, mask)
        else:
            sim_raw = np.array(stored_sim, dtype=np.float32)

        mu, sigma = reference_stats(_features, mask)
        rms = rms_zscore_map(_features, mu, sigma)
        rms_sel = rms[mask]
        sim_sel = sim_raw[mask]
        stats = (
            f"mode: {label}\n"
            f"selected pixels: {int(mask.sum())}\n"
            f"RMS z in ref: mean={rms_sel.mean():.3f}  std={rms_sel.std():.3f}\n"
            f"similarity in ref: mean={sim_sel.mean():.3f}  min={sim_sel.min():.3f}\n"
            f"similarity global: min={np.nanmin(sim_raw):.3f}  max={np.nanmax(sim_raw):.3f}"
        )
        sim_fig = build_sim_figure(
            sim_raw, sim_cscale, sim_gamma, float(sim_zmin), float(sim_zmax),
        )
        return in_fig, sim_fig, stats, click, lasso_data, sim_raw.tolist()

    @app.callback(
        Output("input-graph", "figure", allow_duplicate=True),
        Output("sim-graph", "figure", allow_duplicate=True),
        Input("input-graph", "relayoutData"),
        Input("sim-graph", "relayoutData"),
        prevent_initial_call=True,
    )
    def sync_axes(in_relayout, sim_relayout):
        trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if trig.startswith("input-graph"):
            view = extract_view(in_relayout)
            target = "sim"
        elif trig.startswith("sim-graph"):
            view = extract_view(sim_relayout)
            target = "input"
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
    app = build_app(image, features, args.slice_idx)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
