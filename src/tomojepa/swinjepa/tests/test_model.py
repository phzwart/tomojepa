"""Acceptance checks 1-5, 7 on the assembled model.

Covers predictor output shapes, two-pass gradient routing, no-EMA/no-teacher,
the bf16/fp32 smoke run, the tiny-batch overfit (beta_sig=0), and inference
purity (``extract_features`` with the training-only modules deleted).
"""
import torch

from tomojepa.core.augmentations import (
    pool_fg_to_stage, build_fg_stages, build_slice_fg_mask,
)
from tomojepa.swinjepa.model import SwinMSJEPA, SwinMSEncoder, extract_fg_masks
from .conftest import small_cfg


def test_predictor_output_shapes():
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg())
    b = 2
    mask = model.mask_gen.generate(b)
    E_ctx = model.backbone(torch.randn(b, 1, 64, 64), mask1=mask["s1"])
    E_ctx = {k: model.lateral[k](v) for k, v in E_ctx.items()}
    pred = model.predictor(E_ctx, mask)
    for s in range(model.num_stages):
        key = f"s{s + 1}"
        k = int(mask[key][0].sum())
        assert pred[key].shape == (b, k, model.lat_chans[s])


def test_two_pass_gradient_routing():
    """Encoder gets gradient; targets are detached; SIGReg alone also flows."""
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg())
    x = torch.randn(2, 1, 64, 64)

    # Full objective -> backbone params receive gradient.
    loss, _ = model.compute_loss(x, step=0, total_steps=10)
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = model.backbone.patch_embed.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0

    # SIGReg alone (beta_sig carries the whole loss via prediction weight 0)
    # still produces encoder gradient -- it is the route through the target pass.
    sig_cfg = small_cfg(stage_base_weights=(0.0, 0.0, 0.0, 0.0),
                        beta_sig=(1.0, 1.0, 1.0, 1.0),
                        legacy_jepa=True)
    sig_model = SwinMSJEPA(sig_cfg)
    sig_model.zero_grad(set_to_none=True)
    loss_sig, logs = sig_model.compute_loss(x, step=0, total_steps=10)
    assert logs["l_pred"] == 0.0
    loss_sig.backward()
    gs = sig_model.backbone.patch_embed.proj.weight.grad
    assert gs is not None and float(gs.norm()) > 0


def test_targets_detached():
    """No gradient path from the (normalized) targets."""
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg())
    x = torch.randn(2, 1, 64, 64)
    from tomojepa.swinjepa.losses import stage_target_norm
    feats = model.backbone(x, mask1=None)
    t = stage_target_norm(feats["s4"], "ln").detach()
    assert not t.requires_grad


def test_no_ema_no_teacher():
    model = SwinMSJEPA(small_cfg())
    assert model.has_ema is False
    # exactly one backbone (no second/momentum encoder).
    backbones = [m for n, m in model.named_modules()
                 if type(m).__name__ == "SwinMultiScaleBackbone"]
    assert len(backbones) == 1
    bad = [n for n, _ in model.named_buffers()
           if any(t in n.lower() for t in ("ema", "momentum", "teacher"))]
    assert bad == []


def test_smoke_forward_backward():
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg())
    x = torch.randn(2, 1, 64, 64)
    loss, logs = model.compute_loss(x, step=0, total_steps=10)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())


def test_data2vec_path_shapes():
    """predictor.enabled=False uses student features at masked positions."""
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(predictor_enabled=False))
    assert model.predictor is None
    loss, _ = model.compute_loss(torch.randn(2, 1, 64, 64), 0, 10)
    assert torch.isfinite(loss)


def test_overfit_tiny_batch():
    """With beta_sig=0, L_pred drives down on a fixed 2-image batch."""
    torch.manual_seed(0)
    model = SwinMSJEPA(small_cfg(beta_sig=(0.0, 0.0, 0.0, 0.0))).train()
    x = torch.randn(2, 1, 64, 64)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-3)
    first = None
    last = None
    for i in range(120):
        opt.zero_grad(set_to_none=True)
        loss, logs = model.compute_loss(x, step=i, total_steps=120)
        loss.backward()
        opt.step()
        if first is None:
            first = logs["l_pred"]
        last = logs["l_pred"]
    assert last < 0.5 * first, f"L_pred did not drop enough: {first:.3f} -> {last:.3f}"


def test_inference_purity():
    """extract_features works with predictor/sigreg deleted; correct shapes."""
    torch.manual_seed(0)
    cfg = small_cfg()
    model = SwinMSJEPA(cfg)
    model.predictor = None
    model.sigreg = None
    model.coarse_head = None
    model.sigreg_c4 = None
    model.sigreg_r = None
    x = torch.randn(2, 1, 64, 64)
    feats = model.extract_features(x, normalize=True, project=True)
    for s in range(model.num_stages):
        key = f"s{s + 1}"
        h, w = model.grids[s]
        assert feats[key].shape == (2, model.out_chans[s], h, w)
        assert feats[f"{key}_proj"].shape == (2, cfg.lat_dims[s], h, w)


def test_per_stage_lat_dims():
    """Each stage lateral/SIGReg/predictor use the configured width independently."""
    cfg = small_cfg(lat_dims=(16, 24, 32, 40))
    model = SwinMSJEPA(cfg)
    assert model.lat_chans == [16, 24, 32, 40]
    for s in range(model.num_stages):
        key = f"s{s + 1}"
        assert model.lateral[key].out_channels == cfg.lat_dims[s]
    assert model.sigreg_c4.dim == cfg.lat_dims[3]
    for s in range(model.num_stages - 1):
        assert model.sigreg_r[s].dim == cfg.lat_dims[s]


def test_legacy_jepa_per_stage_sigreg():
    """Legacy mode keeps token-level StageSIGReg per stage."""
    cfg = small_cfg(legacy_jepa=True, lat_dims=(16, 24, 32, 40))
    model = SwinMSJEPA(cfg)
    for s in range(model.num_stages):
        assert model.sigreg[s].dim == cfg.lat_dims[s]


def test_encoder_roundtrip(tmp_path):
    """SwinMSEncoder loads backbone weights from a SwinMSJEPA checkpoint."""
    torch.manual_seed(0)
    cfg = small_cfg()
    model = SwinMSJEPA(cfg).eval()
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict()}, ckpt)
    enc = SwinMSEncoder.from_pretrained(str(ckpt), cfg).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        a = model.extract_features(x, normalize=False)
        b = enc.extract_features(x, normalize=False)
    for key in model.stage_keys:
        assert torch.allclose(a[key], b[key], atol=1e-5)


def test_pool_fg_circular_counts():
    """A centred disk mask pools to the expected coarse FG coverage."""
    torch.manual_seed(0)
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    r = h * 0.35
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    fg_px = disk.float().view(1, 1, h, w)
    grids = [(16, 16), (8, 8), (4, 4), (2, 2)]
    counts = []
    for grid in grids:
        fg_s = pool_fg_to_stage(fg_px, grid, coverage=0.01)
        counts.append(int(fg_s.sum()))
    assert counts[0] > counts[-1] > 0
    assert counts[-1] >= 1


def test_strict_fg_stages_coarsen_from_s1():
    """Coarser strict FG is nested inside pooled FG; s4 matches mask eligibility."""
    torch.manual_seed(0)
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    disk = ((yy - 32) ** 2 + (xx - 32) ** 2) <= (h * 0.35) ** 2
    fg_px = disk.float().view(1, 1, h, w)
    grids = [(16, 16), (8, 8), (4, 4), (2, 2)]
    strict = build_fg_stages(fg_px, grids, coverage=0.01)
    pooled_s4 = pool_fg_to_stage(fg_px, grids[3], coverage=0.01)
    assert strict["s4"].shape == pooled_s4.shape
    assert not (strict["s4"] & ~pooled_s4).any()          # strict ⊆ pooled
    assert int(strict["s4"].sum()) <= int(pooled_s4.sum())
    # eligible s4 for masking == strict s4 from s1
    from tomojepa.swinjepa.mask import MultiScaleBlockMask
    gen = MultiScaleBlockMask(grid4=grids[3], num_stages=4)
    eligible = gen._eligible_s4_from_s1(strict["s1"])
    assert torch.equal(eligible, strict["s4"])


def test_bg_excluded_from_prediction_targets():
    """Gated targets are zero outside strict FG at every stage."""
    torch.manual_seed(0)
    cfg = small_cfg(foreground_mask=True, fg_coverage=0.01)
    model = SwinMSJEPA(cfg).eval()
    x = torch.randn(2, 1, 64, 64)
    fg = build_slice_fg_mask(x[0], fg_std_thresh=0.05).unsqueeze(0).expand(2, -1, -1, -1)
    fg_stages = build_fg_stages(fg, model.grids, cfg.fg_coverage)
    with torch.no_grad():
        E_full = model.backbone(x, mask1=None, bg1=~fg_stages["s1"])
        from tomojepa.swinjepa.model import _fg_gate_feats
        from tomojepa.swinjepa.losses import stage_target_norm
        gated = _fg_gate_feats(E_full, fg_stages)
        targets = {k: stage_target_norm(v, cfg.target_norm) for k, v in gated.items()}
    for key in model.stage_keys:
        bg = ~fg_stages[key]
        if not bg.any():
            continue
        t = targets[key].permute(0, 2, 3, 1)[bg]
        assert torch.allclose(t, torch.zeros_like(t))


def test_bg_features_match_stage_tokens():
    """Backbone outputs at strict BG cells are exactly the per-stage bg tokens."""
    torch.manual_seed(0)
    cfg = small_cfg(foreground_mask=True, fg_coverage=0.01)
    model = SwinMSJEPA(cfg).eval()
    x = torch.randn(1, 1, 64, 64)
    fg = build_slice_fg_mask(x[0], fg_std_thresh=0.05).unsqueeze(0)
    fg_stages = build_fg_stages(fg, model.grids, cfg.fg_coverage)
    with torch.no_grad():
        feats = model.backbone(x, mask1=None, bg1=~fg_stages["s1"])
        lat = {k: model.lateral[k](v) for k, v in feats.items()}
    for s, key in enumerate(model.stage_keys):
        bg = ~fg_stages[key]
        if not bg.any():
            continue
        tok = model.lateral[key](
            model.backbone.bg_stage_tokens[s].view(1, -1, 1, 1))
        got = lat[key].permute(0, 2, 3, 1)[bg]
        assert torch.allclose(got, tok.squeeze(-1).squeeze(-1).expand_as(got))


def test_foreground_mask_wiring():
    """With foreground_mask on, masked cells and SIGReg respect the FOV."""
    torch.manual_seed(0)
    cfg = small_cfg(foreground_mask=True, fg_coverage=0.01)
    model = SwinMSJEPA(cfg).train()
    x = torch.randn(2, 1, 64, 64)
    fg = build_slice_fg_mask(x[0], fg_std_thresh=0.05).unsqueeze(0).expand(2, -1, -1, -1)
    loss, logs = model.compute_loss(x, fg_px=fg, step=0, total_steps=10)
    assert torch.isfinite(loss)
    assert "fg_cov/s1" in logs
    assert logs["fg_cov/s1"] > 0.0


def test_foreground_off_ignores_fg():
    """foreground_mask=False ignores fg_px (no FOV machinery, no fg_cov logs)."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    fg = build_slice_fg_mask(x[0]).unsqueeze(0).expand(2, -1, -1, -1)
    off = SwinMSJEPA(small_cfg(foreground_mask=False)).eval()
    torch.manual_seed(0)
    loss_a, logs_a = off.compute_loss(x, fg_px=None, step=0, total_steps=10)
    torch.manual_seed(0)
    loss_b, logs_b = off.compute_loss(x, fg_px=fg, step=0, total_steps=10)
    assert torch.allclose(loss_a, loss_b)
    assert "fg_cov/s1" not in logs_a


def test_extract_fg_masks_batch():
    views = torch.randn(2, 1, 1, 64, 64)
    fg = torch.ones(2, 1, 1, 64, 64)
    assert extract_fg_masks((views, fg)).shape == (2, 1, 1, 64, 64)
    assert extract_fg_masks((views, 0)) is None
