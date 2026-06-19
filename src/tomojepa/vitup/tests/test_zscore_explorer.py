"""Sanity checks for ViT-Up similarity explorer metrics."""
import numpy as np

from tomojepa.viz.zscore_explorer import (
    apply_display_gamma,
    apply_morphology,
    combine_masks,
    disk_struct,
    extract_view,
    gamma_colorscale,
    lasso_selection,
    mask_from_lasso,
    mask_from_pixels,
    probe_to_mask,
    reference_stats,
    rms_zscore_map,
    score_field_map,
    similarity_map,
    threshold_mask,
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


def test_probe_to_mask_types():
    image = np.zeros((10, 10), dtype=np.float32)
    image[5:8, 5:8] = 1.0
    radius = probe_to_mask({"type": "radius", "cx": 2, "cy": 2, "radius": 1}, image)
    assert radius[2, 2]
    flood = probe_to_mask({"type": "mask", "cx": 6, "cy": 6, "tolerance": 0.1}, image)
    assert flood[6, 6] and not flood[0, 0]
    lasso = probe_to_mask(
        {"type": "lasso", "polygon": [[1, 1], [4, 1], [4, 4], [1, 4]]}, image)
    assert lasso[2, 2]


def test_score_field_aggregate_vs_max():
    rng = np.random.default_rng(1)
    feat = rng.standard_normal((12, 12, 8)).astype(np.float32)
    m1 = np.zeros((12, 12), dtype=bool); m1[1:4, 1:4] = True
    m2 = np.zeros((12, 12), dtype=bool); m2[8:11, 8:11] = True
    agg = score_field_map(feat, [m1, m2], "aggregate")
    mx = score_field_map(feat, [m1, m2], "max")
    assert agg.shape == (12, 12)
    assert mx.shape == (12, 12)
    # max combines two per-probe maps -> high near both references
    assert mx[m1].mean() > 0.3 and mx[m2].mean() > 0.3
    # empty input is all-zero
    assert score_field_map(feat, [], "aggregate").sum() == 0.0


def test_threshold_and_combine_masks():
    score = np.array([[0.1, 0.6], [0.9, 0.3]], dtype=np.float32)
    a = threshold_mask(score, 0.5)
    assert a[0, 1] and a[1, 0] and not a[0, 0]
    b = np.array([[True, False], [True, True]])
    assert combine_masks([a, b], "AND").tolist() == [[False, False], [True, False]]
    assert combine_masks([a, b], "OR").tolist() == [[True, True], [True, True]]
    assert combine_masks([a, b], "XOR").tolist() == [[True, True], [False, True]]
    assert combine_masks([], "OR") is None


def test_morphology_removes_speck_and_keeps_blob():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True   # large blob
    mask[2, 2] = True           # isolated speck (dust)
    cleaned = apply_morphology(mask, [{"op": "open", "size": 1, "iterations": 1}])
    assert not cleaned[2, 2]            # dust removed
    assert cleaned[20, 20]             # blob preserved
    # no-op sequence returns mask unchanged
    same = apply_morphology(mask, [])
    assert same[2, 2] and same[20, 20]


def test_morphology_close_fills_hole():
    mask = np.ones((20, 20), dtype=bool)
    mask[10, 10] = False  # single-pixel hole
    closed = apply_morphology(mask, [{"op": "close", "size": 1, "iterations": 1}])
    assert closed[10, 10]


def test_disk_struct_shape():
    st = disk_struct(2)
    assert st.shape == (5, 5)
    assert st[2, 2]
    assert not st[0, 0]
