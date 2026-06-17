import torch
import torch.nn as nn
from torchvision.ops import MLP
import timm


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (the LeJEPA collapse-prevention term).

    Matches the empirical characteristic function of the projected embeddings,
    measured along ``n_sketches`` random 1D directions, to that of an isotropic
    standard Gaussian. Replaces the teacher/EMA machinery of classic JEPA.
    """

    def __init__(self, knots=17, t_max=3.0, n_sketches=256):
        super().__init__()
        self.n_sketches = n_sketches
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt                       # trapezoidal quadrature
        phi = torch.exp(-t.square() / 2.0)          # target Gaussian char. fn
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, proj):
        # proj: [V, N, D]
        A = torch.randn(proj.size(-1), self.n_sketches,
                        device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)            # unit-norm directions
        x_t = (proj @ A).unsqueeze(-1) * self.t            # [V, N, S, K]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class DINOv3ViTEncoder(nn.Module):
    """timm DINOv3 ViT backbone + projection head.

    NOTE: the backbone is built with ``num_classes=embed_dim`` (an extra linear
    head onto the 384-d ViT-S feature). This is intentional and matches the
    architecture the released checkpoints were trained with -- keep it so a
    newly trained model loads cleanly into the evaluation notebook.
    """

    def __init__(self, proj_dim=128, img_size=512, in_chans=1, pretrained=False,
                 model_name="vit_small_patch16_dinov3", embed_dim=384,
                 drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=embed_dim,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            in_chans=in_chans,
        )
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        # x: [N, V, C, H, W]
        n, v = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))                       # [N*V, embed_dim]
        proj = self.proj(emb).reshape(n, v, -1).transpose(0, 1)    # [V, N, proj_dim]
        return emb, proj


def lejepa_projections(encoder, views, grid, fg_thresh=None):
    """Projections fed to the LeJEPA invariance + SIGReg terms.

    ``views``: ``[N, V, C, H, W]``. Returns ``(emb, proj)`` where ``proj`` is
    ``[V, N, proj_dim]``. When ``fg_thresh`` is set, each view's patch tokens
    are mean-pooled over foreground positions (``foreground_tokens``) before
    the projection head, so LeJEPA does not spend capacity on flat
    holder/background patches. When ``fg_thresh`` is ``None``, delegates to
    :meth:`DINOv3ViTEncoder.forward` (global backbone embedding).
    """
    if fg_thresh is None:
        return encoder(views)
    n, v = views.shape[:2]
    prefix = encoder.backbone.num_prefix_tokens
    z_list = []
    for vi in range(v):
        view = views[:, vi]
        tokens = encoder.backbone.forward_features(view)[:, prefix:]   # [N, P, D]
        fg = foreground_tokens(view, grid, fg_thresh)
        pooled = masked_mean(tokens, fg)                               # [N, D]
        z_list.append(encoder.proj(pooled))
    return None, torch.stack(z_list, 0)


def encode_masked(backbone, img, mask, mask_token):
    """Encode ``img`` after replacing masked patch embeddings with ``mask_token``.

    Mirrors timm's EVA ``forward_features`` (eva.py:1014) but injects a learnable
    ``[MASK]`` token at the masked patch positions right after ``patch_embed`` and
    before position embedding (iBOT/BEiT-style). Returns the patch tokens (prefix
    tokens dropped) so callers get a ``[B, P, D]`` context field.

    Args:
        backbone: the timm EVA/ViT backbone (``DINOv3ViTEncoder.backbone``).
        img:   ``[B, C, H, W]`` input.
        mask:  ``[B, P]`` bool, True where the patch is masked (P = grid*grid).
        mask_token: ``[1, 1, D]`` learnable parameter.
    """
    x = backbone.patch_embed(img)                  # [B, P, D] or [B, H, W, D]
    mt = mask_token.to(x.dtype).reshape(*([1] * (x.dim() - 1)), -1)
    if x.dim() == 4:                               # NHWC patch grid
        m = mask.view(x.shape[0], x.shape[1], x.shape[2], 1)
    else:
        m = mask.view(x.shape[0], x.shape[1], 1)
    x = torch.where(m, mt, x)
    x, rope = backbone._pos_embed(x)
    x = backbone.norm_pre(x)
    if getattr(backbone, "rope_mixed", False) and rope is not None:
        for i, blk in enumerate(backbone.blocks):
            x = blk(x, rope=rope[i])
    else:
        for blk in backbone.blocks:
            x = blk(x, rope=rope)
    x = backbone.norm(x)
    return x[:, backbone.num_prefix_tokens:]                        # [B, P, D]


class MaskedLatentPredictor(nn.Module):
    """Masked image-modeling head for the residual factorization.

    Holds the learnable ``[MASK]`` token and a small MLP that maps context
    tokens (from :func:`encode_masked`) to the *smooth* latent field ``C`` over
    **all** patch positions. ``C`` serves two roles: (a) the masked-position
    predictions are matched to the stop-grad full-image target (the MAE loss),
    and (b) the full field is subtracted from the unmasked token latents to form
    the augmentation-invariant residual.

    Kept separate from :class:`DINOv3ViTEncoder` so the encoder ``state_dict()``
    stays identical to the released architecture.
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.head = MLP(embed_dim, [embed_dim, embed_dim])

    def forward(self, ctx):
        # ctx: [B, P, D] context tokens -> smooth field C [B, P, D]
        return self.head(ctx)


def foreground_tokens(view, grid, thresh):
    """Per-patch foreground (sample-ROI) mask from a view.

    The sample sits on a flat surround (the imaging frame / holder); background
    patches are near-constant so their per-patch intensity std ~ 0, while the
    textured sample has high std. Returns ``[B, P]`` bool (True = foreground).

    Robust to fully-interior crops: an absolute std threshold keeps every patch
    when there is no background (rather than splitting texture). If a view ends
    up all-background, it falls back to all-foreground to avoid empty pooling.
    """
    B, C, H, W = view.shape
    ph, pw = H // grid, W // grid
    x = view.mean(1)[:, :grid * ph, :grid * pw]                 # [B, grid*ph, grid*pw]
    x = x.reshape(B, grid, ph, grid, pw).permute(0, 1, 3, 2, 4).reshape(B, grid, grid, ph * pw)
    std = x.float().std(dim=-1).reshape(B, grid * grid)         # [B, P]
    fg = std > thresh
    empty = ~fg.any(dim=1)
    if empty.any():
        fg[empty] = True
    return fg


def masked_mean(tokens, fg):
    """Mean-pool ``tokens`` [B, P, D] over foreground positions ``fg`` [B, P]."""
    w = fg.unsqueeze(-1).to(tokens.dtype)
    return (tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
