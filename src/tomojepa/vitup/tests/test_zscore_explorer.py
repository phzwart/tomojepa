"""Sanity checks for ViT-Up similarity explorer metrics."""
import numpy as np

from tomojepa.viz.zscore_explorer import (
    apply_display_gamma,
    extract_view,
    gamma_colorscale,
    lasso_selection,
    mask_from_lasso,
    mask_from_pixels,
    reference_stats,
    rms_zscore_map,
    similarity_map,
)


def test_rms_zscore_reference_near_one():
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((10, 10, 16)).astype(np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    mu, sigma = reference_stats(feat, mask)
    rms = rms_zscore_map(feat, mu, sigma)
    z_sel = rms[mask]
    assert z_sel.mean() < 2.0
    assert z_sel.mean() > 0.3


def test_similarity_high_at_reference_mean():
    feat = np.zeros((5, 5, 8), dtype=np.float32)
    feat[1:4, 1:4] = 2.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    mu, sigma = reference_stats(feat, mask)
    rms = rms_zscore_map(feat, mu, sigma)
    sim = similarity_map(rms, mask)
    # Pixel at reference mean → rms=0 → similarity should be high (near 1).
    assert sim[2, 2] > 0.9
    assert sim[mask].mean() > 0.4


def test_lasso_selection_from_lasso_points():
    sel = {"points": [], "lassoPoints": {
        "x": [10.0, 30.0, 30.0, 10.0],
        "y": [10.0, 10.0, 30.0, 30.0],
    }}
    pts = lasso_selection(sel, 64, 64)
    assert pts is not None
    assert len(pts) == 4
    mask = mask_from_lasso(pts, 64, 64)
    assert mask[20, 20]
    assert not mask[0, 0]


def test_lasso_selection_from_box_range():
    sel = {"points": [], "range": {"x": [10.0, 30.0], "y": [10.0, 30.0]}}
    pts = lasso_selection(sel, 64, 64)
    assert pts is not None
    assert len(pts) == 4
    mask = mask_from_lasso(pts, 64, 64)
    assert mask[20, 20]


def test_lasso_selection_empty():
    assert lasso_selection({"points": []}, 64, 64) is None
    assert lasso_selection(None, 64, 64) is None


def test_lasso_mask_square():
    pts = [[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]
    mask = mask_from_lasso(pts, 6, 6)
    assert mask[2, 2]
    assert not mask[0, 0]


def test_pixel_mask():
    pixels = [[2, 2], [2, 3], [3, 2]]
    mask = mask_from_pixels(pixels, 5, 5)
    assert mask[2, 2] and mask[2, 3] and mask[3, 2]
    assert not mask[0, 0]


def test_display_gamma_boosts_low_values():
    sim = np.array([0.01, 0.1, 0.5, 1.0], dtype=np.float32)
    boosted = apply_display_gamma(sim, 0.3)
    assert boosted[0] > sim[0]
    assert boosted[-1] == 1.0


def test_gamma_colorscale_endpoints_and_monotonic():
    cs = gamma_colorscale("Viridis", 0.3, n=16)
    assert cs[0][0] == 0.0
    assert cs[-1][0] == 1.0
    positions = [p for p, _ in cs]
    assert positions == sorted(positions)
    # gamma=1 is identity sampling; first/last colors match the base ends
    base = gamma_colorscale("Viridis", 1.0, n=16)
    assert base[0][1] == cs[0][1]
    assert base[-1][1] == cs[-1][1]


def test_extract_view_variants():
    assert extract_view(None) is None
    assert extract_view({"xaxis.autorange": True}) == "auto"
    rl = {"xaxis.range[0]": 1, "xaxis.range[1]": 5,
          "yaxis.range[0]": 9, "yaxis.range[1]": 2}
    view = extract_view(rl)
    assert view == ([1, 5], [9, 2])
    assert extract_view({"hovermode": "closest"}) is None
