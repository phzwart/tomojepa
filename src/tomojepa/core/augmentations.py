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
"""
import torch
import torch.nn as nn
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


def _build_train_tf(variant, img_size, scale):
    """Train transform for one crop scale band (everything but the crop is shared)."""
    pre = [CustomIntensityWindowing(p_low=0.01, p_high=0.99)]

    train_ops = [
        v2.RandomResizedCrop(img_size, scale=tuple(scale), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ]
    if variant == "tomo2":
        train_ops += [
            v2.RandomRotation(degrees=(0, 180)),
            RandomFloatEqualize(p=0.5),
        ]
    train_ops += [
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 1.0))], p=0.5),
        v2.RandomApply([PoissonNoise(scale=10000.0)], p=0.5),
        v2.RandomApply([RandomPixelMask(mask_ratio=0.15)], p=0.5),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
    return v2.Compose(pre + train_ops)


def get_augmentations(variant="tomo2", img_size=512,
                      global_scale=(0.4, 1.0), local_scale=(0.1, 0.4)):
    """Return ``(global_tf, local_tf, test_tf)`` for the requested variant.

    ``global_tf`` and ``local_tf`` are identical except for the crop area-scale
    band, giving wide-context vs. aggressively zoomed-in views (both at
    ``img_size``).
    """
    if variant not in ("tomo", "tomo2"):
        raise ValueError(f"unknown augmentation variant: {variant!r}")

    global_tf = _build_train_tf(variant, img_size, global_scale)
    local_tf = _build_train_tf(variant, img_size, local_scale)
    test = v2.Compose([
        CustomIntensityWindowing(p_low=0.01, p_high=0.99),
        v2.Resize(img_size, antialias=True),
        v2.CenterCrop(img_size),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])
    return global_tf, local_tf, test
