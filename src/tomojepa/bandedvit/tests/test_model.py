"""Tests for BandedViT training model."""
import torch

from tomojepa.bandedvit.bvit import build_bg_attn_bias
from tomojepa.bandedvit.config import BandedJEPAConfig
from tomojepa.bandedvit.mask import PatchBlockMask
from tomojepa.bandedvit.model import BandedJEPA
from tomojepa.core.augmentations import build_circle_fg_mask, pool_fg_to_stage


def _small_cfg(**kwargs) -> BandedJEPAConfig:
    base = dict(
        img_size=64,
        patch_size=16,
        depth=4,
        embed_dim=128,
        num_heads=4,
        beta_sig=(0.0, 0.0, 0.0, 1.0),
        sigreg_blocks=(-1,),
        sigreg_n_dirs=16,
        sigreg_token_cap=128,
        pred_enabled=False,
    )
    base.update(kwargs)
    return BandedJEPAConfig(**base)


def test_sigreg_only_forward():
    cfg = _small_cfg()
    model = BandedJEPA(cfg).eval()
    x = torch.randn(2, 1, 64, 64)
    loss, logs = model.compute_loss(x, step=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logs["l_sig"] > 0
    assert logs["l_pred"] == 0.0


def test_simmim_forward():
    cfg = _small_cfg(pred_enabled=True, mask_ratio=0.4)
    model = BandedJEPA(cfg).eval()
    x = torch.randn(2, 2, 1, 64, 64)
    loss, logs = model.compute_loss(x, step=0)
    assert torch.isfinite(loss)
    assert logs["l_pred"] > 0
    assert logs["l_sig"] > 0


def test_extract_patch_tokens_shape():
    cfg = _small_cfg()
    model = BandedJEPA(cfg).eval()
    x = torch.randn(1, 1, 64, 64)
    tok = model.extract_patch_tokens(x)
    grid = cfg.img_size // cfg.patch_size
    assert tok.shape == (1, grid * grid, cfg.embed_dim)


def test_foreground_mask_sigreg():
    cfg = _small_cfg(foreground_mask=True, fg_mode="circle")
    model = BandedJEPA(cfg).eval()
    x = torch.randn(2, 1, 64, 64)
    fg = build_circle_fg_mask(torch.zeros(1, 64, 64), diameter_frac=1.0)
    fg = fg.unsqueeze(0).expand(2, -1, -1, -1)
    loss_on, logs_on = model.compute_loss(x, fg_px=fg, step=0)
    loss_off, logs_off = model.compute_loss(x, fg_px=None, step=0)
    assert torch.isfinite(loss_on) and torch.isfinite(loss_off)
    assert "fg_cov" in logs_on and logs_on["fg_cov"] > 0
    assert "fg_cov" not in logs_off
    assert loss_on != loss_off


def test_simmim_never_masks_background():
    cfg = _small_cfg(pred_enabled=True, foreground_mask=True, fg_mode="circle", mask_ratio=0.4)
    model = BandedJEPA(cfg).eval()
    x = torch.randn(2, 2, 1, 64, 64)
    fg = build_circle_fg_mask(torch.zeros(1, 64, 64), diameter_frac=1.0)
    fg = fg.unsqueeze(0).unsqueeze(0).expand(2, 2, -1, -1, -1)
    loss, logs = model.compute_loss(x, fg_px=fg, step=0)
    assert torch.isfinite(loss)
    assert logs["l_pred"] > 0
    grid = cfg.img_size // cfg.patch_size
    fg_patch = pool_fg_to_stage(fg[:, 0].float(), (grid, grid), cfg.fg_coverage)
    mask = PatchBlockMask(grid=grid, mask_ratio=0.4).sample(2, torch.device("cpu"), fg=fg_patch)
    assert not (mask & ~fg_patch).any()


def test_bg_attention_isolated_from_foreground():
    cfg = _small_cfg(foreground_mask=True, fg_mode="circle")
    model = BandedJEPA(cfg).eval()
    x = torch.randn(1, 1, 64, 64)
    fg = build_circle_fg_mask(torch.zeros(1, 64, 64), diameter_frac=1.0).unsqueeze(0)
    _, bg = model._fg_from_px(fg)
    bg_flat = bg.reshape(1, -1)
    attn = model.encoder.blocks[0].attn
    t = model.encoder.num_prefix + model.encoder.num_patches
    xin = torch.randn(1, t, cfg.embed_dim)
    bg_bias = build_bg_attn_bias(bg_flat, model.encoder.num_prefix, torch.float32)
    with torch.no_grad():
        qkv = attn.qkv(xin).reshape(1, t, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        qq, kk, _ = qkv[0], qkv[1], qkv[2]
        aw = ((qq @ kk.transpose(-2, -1)) * attn.scale + bg_bias).softmax(-1).mean(1)[0]
    p = model.encoder.num_prefix
    fg_rows = torch.arange(p, t)[~bg_flat[0]]
    bg_cols = torch.arange(p, t)[bg_flat[0]]
    if fg_rows.numel() > 0 and bg_cols.numel() > 0:
        assert aw[fg_rows[:, None], bg_cols[None, :]].max() < 1e-6
