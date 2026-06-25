"""Self-supervised augmentations for microCT tomography slices.

Pipeline layout and defaults are driven by :class:`tomojepa.core.aug_config.AugmentationConfig`
(YAML / CLI). Optional :class:`tomojepa.core.aug_config.AugmentationSchedule` knots
override scalars during training via a shared dynamic dict on
:class:`tomojepa.core.dataset.TomographyDataset`.

When ``carry_mask=True``, each transform accepts ``(image, fg_mask)`` and
returns the pair. Geometric ops run jointly on ``tv_tensors.Image`` and
``tv_tensors.Mask`` so zoom-out / rotation / crop extend the background as 0
("infinite extension"); intensity ops touch the image only.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms import v2

from .aug_config import AugmentationConfig, AugmentationState, state_to_dynamic_dict


class DynamicAugStore(Mapping[str, Any]):
    """Mutable augmentation scalars (optionally multiprocessing-shared)."""

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = dict(initial or {})

    def update_from_state(self, state: AugmentationState) -> None:
        self._data.update(state_to_dynamic_dict(state))

    def update_from_config(self, cfg: AugmentationConfig) -> None:
        rj = cfg.resize_jitter_scale()
        self._data.update({
            "random_rotate_deg": cfg.random_rotate_deg,
            "rotate_deg_range": cfg.rotate_deg_range,
            "rotate_p": cfg.rotate_p,
            "resize_jitter_lo": rj[0] if rj else 0.0,
            "resize_jitter_hi": rj[1] if rj else 0.0,
            "global_scale_lo": cfg.global_scale[0],
            "global_scale_hi": cfg.global_scale[1],
            "local_scale_lo": cfg.local_scale[0],
            "local_scale_hi": cfg.local_scale[1],
            "equalize_p": cfg.equalize_p,
            "blur_p": cfg.blur_p,
            "poisson_p": cfg.poisson_p,
            "pixel_mask_p": cfg.pixel_mask_p,
            "intensity_augment": cfg.intensity_augment,
        })

    def enable_shared(self) -> None:
        from multiprocessing import Manager
        mgr = Manager()
        shared = mgr.dict()
        shared.update(self._data)
        self._data = shared

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class DynamicRandomResizedCrop(nn.Module):
    """RandomResizedCrop with scale band read from ``DynamicAugStore`` each call."""

    def __init__(self, size: int, dynamic: DynamicAugStore,
                 scale_lo_key: str, scale_hi_key: str):
        super().__init__()
        self.size = int(size)
        self.dynamic = dynamic
        self.scale_lo_key = scale_lo_key
        self.scale_hi_key = scale_hi_key

    def forward(self, sample):
        lo = float(self.dynamic[self.scale_lo_key])
        hi = float(self.dynamic[self.scale_hi_key])
        return v2.RandomResizedCrop(
            self.size, scale=(lo, hi), antialias=True)(sample)


class DynamicRandomRotation(nn.Module):
    def __init__(self, dynamic: DynamicAugStore):
        super().__init__()
        self.dynamic = dynamic

    def forward(self, sample):
        if float(self.dynamic.get("rotate_p", 1.0)) <= 0:
            return sample
        if torch.rand(()) > float(self.dynamic["rotate_p"]):
            return sample
        rdr = self.dynamic.get("rotate_deg_range")
        if rdr is not None:
            lo, hi = float(rdr[0]), float(rdr[1])
            return v2.RandomRotation(degrees=(lo, hi))(sample)
        deg = self.dynamic["random_rotate_deg"]
        if deg is None:
            return v2.RandomRotation(degrees=(0, 180))(sample)
        if float(deg) <= 0:
            return sample
        d = float(deg)
        return v2.RandomRotation(degrees=(-d, d))(sample)


class DynamicRandomResizeJitter(nn.Module):
    def __init__(self, size: int, dynamic: DynamicAugStore):
        super().__init__()
        self.size = int(size)
        self.dynamic = dynamic

    def forward(self, sample):
        lo = float(self.dynamic["resize_jitter_lo"])
        hi = float(self.dynamic["resize_jitter_hi"])
        if lo <= 0 or hi <= 0:
            return sample
        return RandomResizeJitter(self.size, scale=(lo, hi))(sample)


class CustomIntensityWindowing(nn.Module):
    """Clip to [p_low, p_high] quantiles, then normalize to [0, 1].

    Quantiles are estimated on a random subsample for speed on large slices.
    """

    def __init__(self, p_low=0.01, p_high=0.99, sample_size=100_000):
        super().__init__()
        self.p_low = p_low
        self.p_high = p_high
        self.sample_size = sample_size

    def forward(self, img):
        img = torch.nan_to_num(img, nan=0.0)
        flat = img.reshape(-1)
        if flat.numel() > self.sample_size:
            idx = torch.randint(0, flat.numel(), (self.sample_size,))
            flat = flat[idx]
        q = torch.quantile(flat, torch.tensor([self.p_low, self.p_high], dtype=flat.dtype))
        q_low, q_high = q[0], q[1]
        if q_high <= q_low:
            return torch.zeros_like(img)
        return (img.clamp(q_low, q_high) - q_low) / (q_high - q_low)


class RandomPixelMask(nn.Module):
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        mask = torch.bernoulli(torch.full_like(x, self.mask_ratio)).bool()
        x = x.clone()
        x[mask] = 0.0
        return x


class PoissonNoise(nn.Module):
    """Simulate photon shot noise by scaling to counts, sampling, scaling back."""

    def __init__(self, scale=10000.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        x_scaled = x.clamp(0.0, 1.0) * self.scale
        return (torch.poisson(x_scaled) / self.scale).clamp(0.0, 1.0)


class DownsampleHalf(nn.Module):
    """Halve H and W (×0.5 zoom-out) on image and mask before rotate/crop."""

    def forward(self, sample):
        if isinstance(sample, (tuple, list)):
            img, mask = sample
            return self._half(img, is_mask=False), self._half(mask, is_mask=True)
        return self._half(sample, is_mask=False)

    def _half(self, t, is_mask: bool):
        if isinstance(t, tv_tensors.Image):
            data = t.as_subclass(torch.Tensor)
            wrap = lambda x: tv_tensors.Image(x, dtype=t.dtype, device=t.device)
        elif isinstance(t, tv_tensors.Mask):
            data = t.as_subclass(torch.Tensor)
            wrap = lambda x: tv_tensors.Mask(x, dtype=t.dtype, device=t.device)
        else:
            data = t
            wrap = lambda x: x
        if data.dim() == 2:
            data = data.unsqueeze(0)
        _, h, w = data.shape
        nh, nw = max(1, h // 2), max(1, w // 2)
        if is_mask:
            out = F.interpolate(
                data.unsqueeze(0).float(), size=(nh, nw), mode="nearest",
            ).squeeze(0)
            if data.dtype == torch.bool:
                out = out.bool()
            else:
                out = out.to(data.dtype)
        else:
            out = F.interpolate(
                data.unsqueeze(0), size=(nh, nw), mode="bilinear",
                align_corners=False, antialias=True,
            ).squeeze(0)
        return wrap(out)


class RandomResizeJitter(nn.Module):
    """Uniform scale jitter around ``size``, then center-crop or zero-pad back."""

    def __init__(self, size, scale=(0.9, 1.1)):
        super().__init__()
        self.size = int(size)
        lo, hi = float(scale[0]), float(scale[1])
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(f"invalid resize jitter scale ({lo}, {hi})")
        self.scale = (lo, hi)

    def forward(self, sample):
        lo, hi = self.scale
        s = lo + (hi - lo) * torch.rand(()).item()
        out = max(1, round(self.size * s))
        if isinstance(sample, (tuple, list)):
            img, mask = sample
            return (self._resize(img, out, is_mask=False),
                    self._resize(mask, out, is_mask=True))
        return self._resize(sample, out, is_mask=False)

    def _resize(self, t, out, is_mask: bool):
        t = self._interpolate(t, out, is_mask)
        return self._fit(t, is_mask)

    def _interpolate(self, t, out, is_mask: bool):
        data, wrap = self._unwrap(t)
        if data.dim() == 2:
            data = data.unsqueeze(0)
        _, h, w = data.shape
        if h == out and w == out:
            return wrap(data)
        if is_mask:
            x = F.interpolate(
                data.unsqueeze(0).float(), size=(out, out), mode="nearest",
            ).squeeze(0)
            if data.dtype == torch.bool:
                x = x.bool()
            else:
                x = x.to(data.dtype)
        else:
            x = F.interpolate(
                data.unsqueeze(0), size=(out, out), mode="bilinear",
                align_corners=False, antialias=True,
            ).squeeze(0)
        return wrap(x)

    def _fit(self, t, is_mask: bool):
        data, wrap = self._unwrap(t)
        if data.dim() == 2:
            data = data.unsqueeze(0)
        _, h, w = data.shape
        if h == self.size and w == self.size:
            return wrap(data)
        if h >= self.size and w >= self.size:
            y0 = (h - self.size) // 2
            x0 = (w - self.size) // 2
            data = data[:, y0:y0 + self.size, x0:x0 + self.size]
            return wrap(data)
        pad_y = self.size - h
        pad_x = self.size - w
        py0, py1 = pad_y // 2, pad_y - pad_y // 2
        px0, px1 = pad_x // 2, pad_x - pad_x // 2
        if is_mask:
            data = F.pad(data.unsqueeze(0), (px0, px1, py0, py1), value=0).squeeze(0)
            if data.dtype == torch.bool:
                data = data.bool()
        else:
            data = F.pad(data.unsqueeze(0), (px0, px1, py0, py1), value=0.0).squeeze(0)
        return wrap(data)

    @staticmethod
    def _unwrap(t):
        if isinstance(t, tv_tensors.Image):
            return t.as_subclass(torch.Tensor), lambda x: tv_tensors.Image(
                x, dtype=t.dtype, device=t.device)
        if isinstance(t, tv_tensors.Mask):
            return t.as_subclass(torch.Tensor), lambda x: tv_tensors.Mask(
                x, dtype=t.dtype, device=t.device)
        return t, lambda x: x


class RandomFloatEqualize(nn.Module):
    """Histogram-equalize a float image in [0, 1] with probability ``p``."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.eq = v2.RandomEqualize(p=1.0)

    def forward(self, x):
        if torch.rand(()) > self.p:
            return x
        x_uint8 = (x.clamp(0.0, 1.0) * 255).to(torch.uint8)
        return self.eq(x_uint8).to(x.dtype) / 255.0


def build_circle_fg_mask(img: torch.Tensor,
                         diameter_frac: float = 1.0) -> torch.Tensor:
    """Per-slice foreground mask ``[1, H, W]`` from a centered circular FOV.

    Foreground = inside a disk centered on the slice. The disk diameter is
    ``diameter_frac * W`` where ``W`` is the image width (last spatial axis).
    ``diameter_frac=1.0`` gives the standard inscribed tomography FOV (diameter
    equals the frame width; square corners are background). ``diameter_frac=2.0``
    uses diameter ``2*W`` (the disk fully contains a square ``W×W`` slice).
    """
    if img.dim() == 3:
        _, h, w = img.shape
    elif img.dim() == 2:
        h, w = img.shape
    else:
        raise ValueError(f"expected [H,W] or [1,H,W], got {tuple(img.shape)}")
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid slice shape {(h, w)}")
    radius = 0.5 * float(diameter_frac) * w
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device),
        torch.arange(w, device=img.device),
        indexing="ij",
    )
    dist_sq = (yy.float() - cy) ** 2 + (xx.float() - cx) ** 2
    fg = (dist_sq <= radius * radius).to(torch.float32)
    return fg.unsqueeze(0)


def build_slice_fg_mask(img: torch.Tensor, fg_std_thresh: float = 0.05,
                        patch_size: int = 16) -> torch.Tensor:
    """Per-slice foreground mask ``[1, H, W]`` from local intensity variation.

    Constant background (holder / out-of-FOV) has near-zero block std; textured
    sample interior is foreground. The mask is built at ``patch_size`` blocks and
    nearest-upsampled to pixels so it is cheap on large slices and stable under
    later rescaling (the geometric transform then carries it exactly).
    """
    if img.dim() == 3:
        x = img.mean(0)
    else:
        x = img
    h, w = x.shape
    ph = pw = max(1, int(patch_size))
    h_trim, w_trim = (h // ph) * ph, (w // pw) * pw
    if h_trim == 0 or w_trim == 0:
        return torch.ones((1, h, w), dtype=torch.float32, device=x.device)
    blocks = (x[:h_trim, :w_trim]
              .reshape(h_trim // ph, ph, w_trim // pw, pw)
              .float())
    std = blocks.std(dim=(1, 3))
    fg = (std > fg_std_thresh).float()
    fg = fg.repeat_interleave(ph, dim=0).repeat_interleave(pw, dim=1)
    out = torch.zeros(h, w, dtype=torch.float32, device=x.device)
    out[:h_trim, :w_trim] = fg
    if not bool(out.any()):
        out.fill_(1.0)
    return out.unsqueeze(0)


def build_fg_mask(img: torch.Tensor, fg_mode: str = "std",
                  fg_std_thresh: float = 0.05,
                  fg_circle_diameter_frac: float = 1.0,
                  patch_size: int = 16) -> torch.Tensor:
    """Build a pixel foreground mask ``[1, H, W]`` at the slice's native resolution.

    Call once on the decoded slice, then carry the mask through geometric
    transforms (``carry_mask=True``) so BG tokens track zoom/crop/rotate.
    Do not re-call on downsampled or cropped tiles.
    """
    if fg_mode == "circle":
        return build_circle_fg_mask(img, fg_circle_diameter_frac)
    if fg_mode == "std":
        return build_slice_fg_mask(img, fg_std_thresh, patch_size)
    raise ValueError(f"unknown fg_mode {fg_mode!r}; expected 'std' or 'circle'")


class _ImageOnlyTransform(nn.Module):
    """Apply a transform to the image channel of an ``(image, mask)`` pair."""

    def __init__(self, tf):
        super().__init__()
        self.tf = tf

    def forward(self, sample):
        img, mask = sample
        return self.tf(img), mask


class _ImageMaskTransform(nn.Module):
    """Three-phase train transform: pre-intensity, joint geometry, intensity tail."""

    def __init__(self, pre_intensity, geom, intensity):
        super().__init__()
        self.pre_intensity = pre_intensity
        self.geom = geom
        self.intensity = intensity

    def forward(self, sample):
        img, mask = sample
        img = self.pre_intensity(img)
        img, mask = self.geom((img, mask))
        img = self.intensity(img)
        return img, mask


class _ImageTransform(nn.Module):
    """Image-only train path (no foreground mask carried)."""

    def __init__(self, pre_intensity, geom, intensity):
        super().__init__()
        self.pre_intensity = pre_intensity
        self.geom = geom
        self.intensity = intensity

    def forward(self, img):
        img = self.pre_intensity(img)
        img = self.geom(img)
        img = self.intensity(img)
        return img


def _rotation_op(variant, random_rotate_deg=None, rotate_deg_range=None):
    """In-plane rotation before the final crop (after any downsample in crop_down)."""
    if rotate_deg_range is not None:
        return v2.RandomRotation(degrees=(rotate_deg_range[0], rotate_deg_range[1]))
    if random_rotate_deg is None:
        if variant == "tomo2":
            return v2.RandomRotation(degrees=(0, 180))
        return None
    if random_rotate_deg > 0:
        return v2.RandomRotation(degrees=(-random_rotate_deg, random_rotate_deg))
    return None


def _crop_op(img_size, scale, crop_mode, crop_size=None,
             dynamic: Optional[DynamicAugStore] = None,
             scale_lo_key: str = "global_scale_lo",
             scale_hi_key: str = "global_scale_hi"):
    if crop_mode == "native":
        return v2.RandomCrop(img_size, pad_if_needed=True)
    if crop_mode == "resized":
        if dynamic is not None:
            return DynamicRandomResizedCrop(
                img_size, dynamic, scale_lo_key, scale_hi_key)
        return v2.RandomResizedCrop(img_size, scale=tuple(scale), antialias=True)
    if crop_mode == "resize":
        return v2.Resize((img_size, img_size), antialias=True)
    if crop_mode == "crop_down":
        return v2.RandomCrop(img_size, pad_if_needed=True)
    raise ValueError(f"unknown crop_mode: {crop_mode!r}")


def _append_rotation(ops, cfg: AugmentationConfig,
                     dynamic: Optional[DynamicAugStore] = None):
    if dynamic is not None:
        ops.append(DynamicRandomRotation(dynamic))
        return
    rot = _rotation_op(cfg.variant, cfg.random_rotate_deg, cfg.rotate_deg_range)
    if rot is None:
        return
    if cfg.rotate_p >= 1.0:
        ops.append(rot)
    elif cfg.rotate_p > 0:
        ops.append(v2.RandomApply([rot], p=cfg.rotate_p))


def _geom_ops(cfg: AugmentationConfig, img_size: int, scale_band: str,
              dynamic: Optional[DynamicAugStore] = None, crop_size=None):
    """Geometry on the slice before intensity aug."""
    crop_mode = cfg.crop_mode
    lo_key = f"{scale_band}_scale_lo"
    hi_key = f"{scale_band}_scale_hi"
    ops = []
    if crop_mode == "crop_down":
        ops.append(DownsampleHalf())
    _append_rotation(ops, cfg, dynamic=dynamic)
    if cfg.hflip_p > 0:
        ops.append(v2.RandomHorizontalFlip(p=cfg.hflip_p))
    if cfg.vflip_p > 0:
        ops.append(v2.RandomVerticalFlip(p=cfg.vflip_p))
    if dynamic is not None:
        scale = (float(dynamic[lo_key]), float(dynamic[hi_key]))
    else:
        scale = cfg.global_scale if scale_band == "global" else cfg.local_scale
    ops.append(_crop_op(img_size, scale, crop_mode, crop_size=crop_size,
                        dynamic=dynamic, scale_lo_key=lo_key, scale_hi_key=hi_key))
    if dynamic is not None:
        ops.append(DynamicRandomResizeJitter(img_size, dynamic))
    elif cfg.resize_jitter_scale() is not None:
        ops.append(RandomResizeJitter(img_size, scale=cfg.resize_jitter_scale()))
    return ops


def _intensity_ops(cfg: AugmentationConfig, is_train=True,
                   dynamic: Optional[DynamicAugStore] = None):
    norm = v2.Normalize(mean=[cfg.normalize_mean], std=[cfg.normalize_std])
    if not is_train:
        return [norm]
    if dynamic is not None:
        return [_DynamicIntensityTail(cfg, dynamic, norm)]

    if not cfg.intensity_augment:
        return [norm]
    ops = [
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=cfg.blur_kernel_size,
                             sigma=tuple(cfg.blur_sigma))],
            p=cfg.blur_p),
        v2.RandomApply([PoissonNoise(scale=cfg.poisson_scale)], p=cfg.poisson_p),
        v2.RandomApply([RandomPixelMask(mask_ratio=cfg.pixel_mask_ratio)],
                       p=cfg.pixel_mask_p),
        norm,
    ]
    if cfg.variant == "tomo2":
        ops = [RandomFloatEqualize(p=cfg.equalize_p)] + ops
    return ops


class _DynamicIntensityTail(nn.Module):
    """Intensity augmentations with probabilities read from ``DynamicAugStore``."""

    def __init__(self, cfg: AugmentationConfig, dynamic: DynamicAugStore, norm):
        super().__init__()
        self.cfg = cfg
        self.dynamic = dynamic
        self.norm = norm
        self.equalize = RandomFloatEqualize(p=1.0)
        self.blur = v2.GaussianBlur(
            kernel_size=cfg.blur_kernel_size, sigma=tuple(cfg.blur_sigma))
        self.poisson = PoissonNoise(scale=cfg.poisson_scale)
        self.pixel_mask = RandomPixelMask(mask_ratio=cfg.pixel_mask_ratio)

    def forward(self, x):
        if not self.dynamic["intensity_augment"]:
            return self.norm(x)
        if self.cfg.variant == "tomo2" and torch.rand(()) <= float(self.dynamic["equalize_p"]):
            x = self.equalize(x)
        if torch.rand(()) <= float(self.dynamic["blur_p"]):
            x = self.blur(x)
        if torch.rand(()) <= float(self.dynamic["poisson_p"]):
            x = self.poisson(x)
        if torch.rand(()) <= float(self.dynamic["pixel_mask_p"]):
            x = self.pixel_mask(x)
        return self.norm(x)


def _build_train_tf(cfg: AugmentationConfig, img_size: int, scale_band: str = "global",
                    carry_mask=False, crop_size=None,
                    dynamic: Optional[DynamicAugStore] = None):
    pre = CustomIntensityWindowing(
        p_low=cfg.window_p_low, p_high=cfg.window_p_high,
        sample_size=cfg.window_sample_size)
    geom_ops = _geom_ops(cfg, img_size, scale_band, dynamic=dynamic, crop_size=crop_size)
    if carry_mask:
        geom = v2.Compose(geom_ops)
        intensity = v2.Compose(_intensity_ops(cfg, is_train=True, dynamic=dynamic))
        return _ImageMaskTransform(pre, geom, intensity)

    geom = v2.Compose(geom_ops)
    intensity = v2.Compose(_intensity_ops(cfg, is_train=True, dynamic=dynamic))
    return _ImageTransform(pre, geom, intensity)


def _build_test_tf(cfg: AugmentationConfig, img_size: int, carry_mask=False,
                   crop_size=None):
    pre = CustomIntensityWindowing(
        p_low=cfg.window_p_low, p_high=cfg.window_p_high,
        sample_size=cfg.window_sample_size)
    crop_mode = cfg.crop_mode
    if crop_mode == "native":
        geom_ops = [v2.CenterCrop(img_size)]
    elif crop_mode == "resize":
        geom_ops = [v2.Resize((img_size, img_size), antialias=True)]
    elif crop_mode == "crop_down":
        geom_ops = [DownsampleHalf(), v2.CenterCrop(img_size)]
    else:
        geom_ops = [v2.Resize(img_size, antialias=True), v2.CenterCrop(img_size)]
    if carry_mask:
        geom = v2.Compose(geom_ops)
        intensity = v2.Compose(_intensity_ops(cfg, is_train=False))
        return _ImageMaskTransform(pre, geom, intensity)
    return _ImageTransform(pre, v2.Compose(geom_ops),
                           v2.Compose(_intensity_ops(cfg, is_train=False)))


def build_augmentations(cfg: AugmentationConfig, img_size: int, carry_mask=False,
                        crop_size=None, dynamic: Optional[DynamicAugStore] = None):
    """Return ``(global_tf, local_tf, test_tf)`` from a config object."""
    global_tf = _build_train_tf(cfg, img_size, "global", carry_mask, crop_size, dynamic)
    local_tf = _build_train_tf(cfg, img_size, "local", carry_mask, crop_size, dynamic)
    test_tf = _build_test_tf(cfg, img_size, carry_mask, crop_size=crop_size)
    return global_tf, local_tf, test_tf


def build_probe_augmentation(cfg: AugmentationConfig, img_size: int,
                             carry_mask=False, crop_size=None):
    """PCA / eval probe: deterministic resize/downsample only."""
    return _build_test_tf(cfg, img_size, carry_mask, crop_size=crop_size)


def _legacy_config_from_kwargs(variant="tomo2", crop_mode="resized",
                               global_scale=(0.4, 1.0), local_scale=(0.1, 0.4),
                               random_rotate_deg=None, resize_jitter_scale=None,
                               ) -> AugmentationConfig:
    rj = resize_jitter_scale
    if rj is not None:
        rj = (float(rj[0]), float(rj[1]))
    rot = random_rotate_deg
    if rot is not None:
        rot = float(rot)
    return AugmentationConfig(
        variant=variant,
        crop_mode=crop_mode,
        global_scale=tuple(global_scale),
        local_scale=tuple(local_scale),
        random_rotate_deg=rot,
        resize_jitter=rj,
    )


def get_probe_augmentation(variant="tomo2", img_size=512, global_scale=(0.4, 1.0),
                           crop_mode="resized", carry_mask=False,
                           random_rotate_deg=None, crop_size=None,
                           resize_jitter_scale=None, aug_config=None):
    """PCA probe: deterministic eval geometry (legacy kwargs or ``aug_config``)."""
    cfg = aug_config or _legacy_config_from_kwargs(
        variant, crop_mode, global_scale, global_scale,
        random_rotate_deg, resize_jitter_scale)
    return build_probe_augmentation(cfg, img_size, carry_mask, crop_size=crop_size)


def get_augmentations(variant="tomo2", img_size=512,
                      global_scale=(0.4, 1.0), local_scale=(0.1, 0.4),
                      crop_mode="resized", carry_mask=False,
                      random_rotate_deg=None, crop_size=None,
                      resize_jitter_scale=None, aug_config=None,
                      dynamic: Optional[DynamicAugStore] = None):
    """Return ``(global_tf, local_tf, test_tf)`` (legacy kwargs or ``aug_config``)."""
    cfg = aug_config or _legacy_config_from_kwargs(
        variant, crop_mode, global_scale, local_scale,
        random_rotate_deg, resize_jitter_scale)
    return build_augmentations(cfg, img_size, carry_mask, crop_size, dynamic)



def wrap_image_mask(img: torch.Tensor, mask: torch.Tensor):
    """Wrap tensors as v2 ``Image`` / ``Mask`` for joint geometric transforms."""
    return tv_tensors.Image(img), tv_tensors.Mask(mask)


def unwrap_image_mask(sample):
    """Extract plain ``[C,H,W]`` image and ``[1,H,W]`` mask from a v2 sample."""
    img, mask = sample
    if isinstance(img, tv_tensors.Image):
        img = img.as_subclass(torch.Tensor)
    if isinstance(mask, tv_tensors.Mask):
        mask = mask.as_subclass(torch.Tensor)
    if mask.dtype != torch.float32:
        mask = mask.float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return img, mask


def pool_fg_to_stage(fg_px: torch.Tensor, grid_hw: tuple,
                     coverage: float = 0.01) -> torch.Tensor:
    """Average-pool a pixel FG mask ``[B,1,H,W]`` to a stage token grid.

    Returns ``[B, h, w]`` bool where True means at least ``coverage`` fraction
    of the token footprint is foreground (i.e. exclude tokens that are >99%
    background when ``coverage=0.01``).
    """
    b, _, h, w = fg_px.shape
    hs, ws = grid_hw
    if hs <= 0 or ws <= 0:
        raise ValueError(f"invalid stage grid {grid_hw}")
    pooled = F.adaptive_avg_pool2d(fg_px, (hs, ws)).squeeze(1)
    return pooled >= coverage


def strict_fg_stages_from_s1(fg_s1: torch.Tensor,
                             stage_grids: list) -> dict:
    """Coarsen a stage-1 FG grid with an *all-children* rule.

    A token at stage ``s+1`` is foreground only when every finer child token it
    merges from stage ``s`` is foreground. Background (``~fg_s1``) therefore
    never appears in ``fg_stages["s2".."s4"]`` unless the entire subtree is FG.
    """
    if fg_s1.dim() != 3:
        raise ValueError(f"expected fg_s1 [B,h,w], got {tuple(fg_s1.shape)}")
    out = {"s1": fg_s1}
    prev = fg_s1
    for s, grid in enumerate(stage_grids[1:], start=2):
        h_prev, w_prev = prev.shape[-2:]
        h, w = grid
        if h_prev % h or w_prev % w:
            raise ValueError(
                f"stage s{s} grid {(h, w)} not a clean factor of prev {(h_prev, w_prev)}")
        fh, fw = h_prev // h, w_prev // w
        b = prev.shape[0]
        coarsened = prev.view(b, h, fh, w, fw).all(dim=(2, 4))
        out[f"s{s}"] = coarsened
        prev = coarsened
    return out


def strict_bg_stages_from_s1(bg_s1: torch.Tensor,
                             stage_grids: list) -> dict:
    """Coarsen a stage-1 BG grid with an *any-child* rule.

    A token at stage ``s+1`` is background when any finer child token it merges
    from stage ``s`` is background. This is the complement of
    ``strict_fg_stages_from_s1`` when ``bg_s1 == ~fg_s1``.
    """
    if bg_s1.dim() != 3:
        raise ValueError(f"expected bg_s1 [B,h,w], got {tuple(bg_s1.shape)}")
    out = {"s1": bg_s1}
    prev = bg_s1
    for s, grid in enumerate(stage_grids[1:], start=2):
        h_prev, w_prev = prev.shape[-2:]
        h, w = grid
        if h_prev % h or w_prev % w:
            raise ValueError(
                f"stage s{s} grid {(h, w)} not a clean factor of prev {(h_prev, w_prev)}")
        fh, fw = h_prev // h, w_prev // w
        b = prev.shape[0]
        coarsened = prev.view(b, h, fh, w, fw).any(dim=(2, 4))
        out[f"s{s}"] = coarsened
        prev = coarsened
    return out


def build_fg_stages(fg_px: torch.Tensor, stage_grids: list,
                    coverage: float = 0.01) -> dict:
    """Pixel FG mask -> strict per-stage FG grids ``s1..sN``.

    Stage 1 uses ``pool_fg_to_stage`` on pixels; coarser stages require every
    child token in the pyramid to be foreground (see ``strict_fg_stages_from_s1``).
    """
    fg_s1 = pool_fg_to_stage(fg_px, stage_grids[0], coverage)
    return strict_fg_stages_from_s1(fg_s1, stage_grids)
