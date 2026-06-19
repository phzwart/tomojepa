"""Self-supervised augmentations for microCT tomography slices.

Two variants, selected at runtime (no file renaming required):

  "tomo"  : intensity windowing -> resized crop / flips -> Gaussian blur,
            Poisson noise, random pixel masking.
  "tomo2" : everything in "tomo" plus random rotation and random
            histogram (float) equalization.

Pick the variant from main.py via ``--augment {tomo,tomo2}``.

Multi-scale views: the train pipeline is built per *scale band* so the dataset
can emit "global" (wide-area) and aggressive "local" (zoomed-in) crops. Both
tiers render at ``img_size`` (so they stay stackable), differing only in the
``RandomResizedCrop`` area-scale range. Tune via ``--global_scale`` /
``--local_scale`` and the per-tier view counts in main.py.

When ``carry_mask=True``, each transform accepts ``(image, fg_mask)`` and
returns the pair. Geometric ops run jointly on ``tv_tensors.Image`` and
``tv_tensors.Mask`` so zoom-out / rotation / crop extend the background as 0
("infinite extension"); intensity ops touch the image only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms import v2


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


def _geom_ops(variant, img_size, scale, crop_mode):
    if crop_mode == "native":
        crop_op = v2.RandomCrop(img_size, pad_if_needed=True)
    elif crop_mode == "resized":
        crop_op = v2.RandomResizedCrop(img_size, scale=tuple(scale), antialias=True)
    elif crop_mode == "resize":
        crop_op = v2.Resize((img_size, img_size), antialias=True)
    else:
        raise ValueError(f"unknown crop_mode: {crop_mode!r}")
    ops = [
        crop_op,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ]
    if variant == "tomo2":
        ops.append(v2.RandomRotation(degrees=(0, 180)))
    return ops


def _intensity_ops(variant, is_train=True):
    if not is_train:
        return [v2.Normalize(mean=[0.5], std=[0.5])]
    ops = [
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 1.0))], p=0.5),
        v2.RandomApply([PoissonNoise(scale=10000.0)], p=0.5),
        v2.RandomApply([RandomPixelMask(mask_ratio=0.15)], p=0.5),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
    if variant == "tomo2":
        ops = [RandomFloatEqualize(p=0.5)] + ops
    return ops


def _build_train_tf(variant, img_size, scale, crop_mode="resized", carry_mask=False):
    """Train transform for one crop scale band.

    ``carry_mask=True`` returns ``(image, fg_mask)``; otherwise image only.
    """
    pre = CustomIntensityWindowing(p_low=0.01, p_high=0.99)
    if carry_mask:
        geom = v2.Compose(_geom_ops(variant, img_size, scale, crop_mode))
        intensity = v2.Compose(_intensity_ops(variant, is_train=True))
        return _ImageMaskTransform(pre, geom, intensity)

    geom = v2.Compose(_geom_ops(variant, img_size, scale, crop_mode))
    intensity = v2.Compose(_intensity_ops(variant, is_train=True))
    return _ImageTransform(pre, geom, intensity)


def _build_test_tf(variant, img_size, crop_mode, carry_mask=False):
    pre = CustomIntensityWindowing(p_low=0.01, p_high=0.99)
    if crop_mode == "native":
        geom_ops = [v2.CenterCrop(img_size)]
    elif crop_mode == "resize":
        geom_ops = [v2.Resize((img_size, img_size), antialias=True)]
    else:
        geom_ops = [v2.Resize(img_size, antialias=True), v2.CenterCrop(img_size)]
    if carry_mask:
        geom = v2.Compose(geom_ops)
        intensity = v2.Compose(_intensity_ops(variant, is_train=False))
        return _ImageMaskTransform(pre, geom, intensity)
    return _ImageTransform(pre, v2.Compose(geom_ops),
                           v2.Compose(_intensity_ops(variant, is_train=False)))


def get_augmentations(variant="tomo2", img_size=512,
                      global_scale=(0.4, 1.0), local_scale=(0.1, 0.4),
                      crop_mode="resized", carry_mask=False):
    """Return ``(global_tf, local_tf, test_tf)`` for the requested variant."""
    if variant not in ("tomo", "tomo2"):
        raise ValueError(f"unknown augmentation variant: {variant!r}")

    global_tf = _build_train_tf(variant, img_size, global_scale, crop_mode, carry_mask)
    local_tf = _build_train_tf(variant, img_size, local_scale, crop_mode, carry_mask)
    test_tf = _build_test_tf(variant, img_size, crop_mode, carry_mask)
    return global_tf, local_tf, test_tf


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
