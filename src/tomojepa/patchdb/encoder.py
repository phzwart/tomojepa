"""Encoder: load a checkpoint, fit a shared PCA basis, encode images to codes.

Reuses the residual/LeJEPA backbone from ``model.py``. The token-extraction,
shared-basis fit, and projection mirror the prototype in ``cascade_rgb.py`` /
``build_token_db.py`` so the codes stored here match the visual analyses.
"""
import os
import re

import numpy as np
import torch

from ..core.model import DINOv3ViTEncoder, foreground_tokens
from ..core.dataset import TomographyDataset


def resolve_ckpt(run_dir, ckpt_subdir, spec):
    """Resolve a checkpoint: a path, an int epoch, or 'last'/'best'."""
    import glob
    d = os.path.join(run_dir, ckpt_subdir)
    if spec not in (None, "last") and os.path.exists(str(spec)):
        return str(spec)
    if spec in (None, "last"):
        cand = os.path.join(d, "ckpt_last.pth")
        if os.path.exists(cand):
            return cand
        eps = glob.glob(os.path.join(d, "ckpt_epoch_*.pth"))
        if not eps:
            raise FileNotFoundError(f"no ckpt_epoch_*.pth under {d}")
        return max(eps, key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)))
    return os.path.join(d, f"ckpt_epoch_{int(spec)}.pth")


def ckpt_epoch_tag(ckpt_path):
    m = re.search(r"epoch_(\d+)", os.path.basename(ckpt_path))
    return m.group(1) if m else "last"


def load_net(ckpt_path, img_size=512, in_chans=1, proj_dim=16, device="cuda"):
    net = DINOv3ViTEncoder(proj_dim=proj_dim, img_size=img_size,
                           in_chans=in_chans, pretrained=False).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)["net"])
    net.eval()
    return net


def make_dataset(data_dir, pattern, backend, dataset_key, img_size):
    return TomographyDataset(
        data_dir=data_dir, dataset_key=dataset_key, pattern=pattern,
        global_views=1, local_views=0, variant="tomo2", img_size=img_size,
        is_train=False, backend=backend,
    )


def get_view(ds, i):
    """Return a single image view tensor shaped [1, C, H, W]."""
    item = ds[i]
    view = item[0] if isinstance(item, (list, tuple)) else item
    if isinstance(view, (list, tuple)):
        view = view[0]
    return view.unsqueeze(0)


@torch.no_grad()
def extract_tokens(net, view, device, fg_thresh=None):
    """L2-normalized patch tokens + foreground mask for one image.

    Returns ``(tokens[P, D], grid, fg[P] bool)`` on ``device``.
    """
    feat = net.backbone.forward_features(view.to(device))
    tokens = feat[:, net.backbone.num_prefix_tokens:].squeeze(0).float()
    tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    P = tokens.shape[0]
    grid = int(round(P ** 0.5))
    if fg_thresh is not None:
        fg = foreground_tokens(view.to(device), grid, fg_thresh).squeeze(0)
    else:
        fg = torch.ones(P, dtype=torch.bool, device=tokens.device)
    return tokens, grid, fg


def fit_shared_basis(fg_tokens, k, outlier_pct=2.0):
    """Robust PCA basis on pooled foreground tokens from many images.

    Returns ``(mu[1, D], Vh[k, D], explained_var_ratio[k])`` (all torch on the
    input device). A directional-outlier reject keeps a few artifact patches from
    hijacking the shared basis.
    """
    fit = fg_tokens
    med = fit.median(0, keepdim=True).values
    dist = (fit - med).norm(dim=-1)
    if outlier_pct > 0 and fit.shape[0] > 20:
        keep = dist <= torch.quantile(dist, 1.0 - outlier_pct / 100.0)
        fit = fit[keep]
    mu = fit.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(fit - mu, full_matrices=False)
    k = min(k, S.numel())
    ev = (S.square() / S.square().sum()).cpu().numpy()[:k]
    return mu, Vh[:k], ev


def project_view(net, view, mu, Vh, device, fg_thresh=None):
    """Encode one view onto a shared basis.

    Returns ``(codes[G, G, K] float32, fg[G, G] bool)`` as numpy arrays.
    """
    tokens, grid, fg = extract_tokens(net, view, device, fg_thresh=fg_thresh)
    K = Vh.shape[0]
    proj = ((tokens - mu) @ Vh.T).reshape(grid, grid, K)
    return proj.cpu().numpy().astype(np.float32), fg.reshape(grid, grid).cpu().numpy()


class SharedBasisEncoder:
    """Holds a loaded net + shared basis for on-the-fly encoding (e.g. service).

    ``mu``/``Vh`` are torch tensors on ``device``; ``ev`` is a numpy array.
    """

    def __init__(self, net, mu, Vh, ev, grid, patch_size, img_size,
                 fg_thresh, device):
        self.net = net
        self.mu = mu
        self.Vh = Vh
        self.ev = ev
        self.grid = grid
        self.patch_size = patch_size
        self.img_size = img_size
        self.fg_thresh = fg_thresh
        self.device = device

    @torch.no_grad()
    def encode_view(self, view):
        return project_view(self.net, view, self.mu, self.Vh, self.device,
                            fg_thresh=self.fg_thresh)
