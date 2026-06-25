"""
bvit.py
=======
Flat, self-contained ViT-S with 2D axial RoPE and an *injectable per-block
attention bias* ("distance band"), plus a `BandManager` that owns the band
geometry / sampling schedule and pushes biases into the blocks on the fly.

Exploratory-phase posture (deliberate): ONE file, fully owned, edit in place.
Duplication over abstraction until the design is locked. The single line that
matters for banding is marked  `# <<< BIAS INJECTION POINT >>>`.

Scope (as agreed across design):
  * This is the ENCODER. It implements DISTANCE-banded attention (the masks)
    and returns per-block hidden-state taps for downstream SIGReg / probes.
  * RESOLUTION-band formation (Laplacian / multi-scale) and the JEPA prediction
    wrap are deliberately NOT here — they sit on top of this.

Three layers, one-directional dependency (manager -> blocks -> scorer-line):
  Attention.bias            : passive slot, consumes whatever bias is set.
  Attention.forward         : rope(q,k) -> q@k^T*scale -> + bias -> softmax -> @v
  BandManager               : owns geometry + schedule; pushes bias every step.

Hard band  = bias in {0, -inf}  (masked_fill, non-learned, no grad through mask).
Soft band  = real-valued bias    (could carry grad; same injection slot).
Both are the same `+bias` interface; the scorer does not care which.

Run `python -m tomojepa.bandedvit.bvit` to execute the sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ============================================================================
# 2D axial RoPE  —  QUARANTINED. Standard construction; treat as a black box.
# head_dim must be divisible by 4 (half the rotation pairs go to each spatial
# axis). NeoX-style: emb = cat([angles, angles]); rotate_half splits in halves.
# ============================================================================
def build_axial_rope_cossin(
    grid_h: int,
    grid_w: int,
    head_dim: int,
    theta: float = 100.0,
    device=None,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) each of shape (grid_h*grid_w, head_dim) for the patch tokens.

    Row-major token order: token index = row * grid_w + col. First head_dim/4
    rotation pairs encode the ROW coordinate, next head_dim/4 encode the COL
    coordinate, then duplicated to head_dim for the rotate_half convention.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D axial RoPE"
    d_quarter = head_dim // 4  # rotation pairs per spatial axis
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, d_quarter, device=device, dtype=dtype) / d_quarter)
    )
    ys = torch.arange(grid_h, device=device, dtype=dtype)
    xs = torch.arange(grid_w, device=device, dtype=dtype)
    ang_y = torch.outer(ys, inv_freq)                       # (H, d/4)
    ang_x = torch.outer(xs, inv_freq)                       # (W, d/4)
    ang_y = ang_y[:, None, :].expand(grid_h, grid_w, d_quarter)
    ang_x = ang_x[None, :, :].expand(grid_h, grid_w, d_quarter)
    ang = torch.cat([ang_y, ang_x], dim=-1)                 # (H, W, d/2)
    ang = ang.reshape(grid_h * grid_w, head_dim // 2)       # (N, d/2)
    emb = torch.cat([ang, ang], dim=-1)                     # (N, d)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, nh, N, hd);  cos/sin: (N, hd)  ->  rotated x (norm-preserving)."""
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


def _masked_attn_fill(dtype: torch.dtype) -> float:
    if dtype in (torch.float32, torch.float64):
        return float("-inf")
    return torch.finfo(dtype).min


def build_bg_attn_bias(
    bg_flat: torch.Tensor,
    num_prefix: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-sample additive attention bias isolating background patch tokens.

    ``bg_flat``: ``[B, N]`` bool, True = background patch.
    Returns ``[B, 1, T, T]`` with masked positions at ``finfo.min`` / ``-inf``.

    Rules (prefix tokens are never background):
      * FG / prefix queries cannot attend to BG keys.
      * BG queries cannot attend to FG keys.
      * BG queries cannot attend to other BG keys (diagonal self kept).
      * BG queries may attend to prefix keys (cls floor).
    """
    if bg_flat.dim() != 2:
        raise ValueError(f"bg_flat must be [B, N], got {tuple(bg_flat.shape)}")
    b, n_patches = bg_flat.shape
    t = num_prefix + n_patches
    device = bg_flat.device
    is_bg = torch.zeros(b, t, dtype=torch.bool, device=device)
    is_bg[:, num_prefix:] = bg_flat
    is_fg = ~is_bg

    # FG/prefix query -> BG key
    col_mask = is_fg.unsqueeze(-1) & is_bg.unsqueeze(-2)
    # BG query -> FG key
    row_mask = is_bg.unsqueeze(-1) & is_fg.unsqueeze(-2)
    # BG query -> other BG keys (keep diagonal for softmax floor)
    bg_bg = is_bg.unsqueeze(-1) & is_bg.unsqueeze(-2)
    eye = torch.eye(t, dtype=torch.bool, device=device)
    bg_bg_offdiag = bg_bg & ~eye.unsqueeze(0)

    blocked = col_mask | row_mask | bg_bg_offdiag
    bias = torch.zeros(b, 1, t, t, device=device, dtype=dtype)
    bias.masked_fill_(blocked.unsqueeze(1), _masked_attn_fill(dtype))
    return bias


# ============================================================================
# Attention  —  the one bespoke piece. RoPE above the product, `+bias` slot
# inside it. Explicit (non-fused) path so the bias is always injectable.
# ============================================================================
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # Pushed by BandManager. None => plain full attention. Shape (T,T) or
        # (num_heads,T,T); broadcasts over batch (and heads if (T,T)).
        self.register_buffer("attn_bias", None, persistent=False)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.attn_bias

    def set_bias(self, bias: Optional[torch.Tensor]) -> None:
        self.attn_bias = bias

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        num_prefix: int = 0,
        extra_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B, nh, T, hd)

        # RoPE on patch tokens only; prefix (cls/register) tokens pass through.
        if rope_cos is not None:
            qp = apply_rope(q[:, :, num_prefix:, :], rope_cos, rope_sin)
            kp = apply_rope(k[:, :, num_prefix:, :], rope_cos, rope_sin)
            q = torch.cat([q[:, :, :num_prefix, :], qp], dim=2)
            k = torch.cat([k[:, :, :num_prefix, :], kp], dim=2)

        s = (q @ k.transpose(-2, -1)) * self.scale          # (B, nh, T, T)
        if self.attn_bias is not None:
            s = s + self.attn_bias.to(dtype=s.dtype)        # <<< BIAS INJECTION POINT >>>
        if extra_bias is not None:
            s = s + extra_bias.to(dtype=s.dtype)
        a = self.attn_drop(s.softmax(dim=-1))
        out = (a @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x, rope_cos=None, rope_sin=None, num_prefix=0, extra_bias=None):
        x = x + self.attn(
            self.norm1(x), rope_cos, rope_sin, num_prefix, extra_bias=extra_bias,
        )
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# The ViT (ViT-S defaults). Single resolution; returns per-block taps.
# ============================================================================
@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 1             # microCT: single channel
    embed_dim: int = 384          # ViT-S
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_register_tokens: int = 0
    use_cls_token: bool = True
    rope_theta: float = 100.0


class BandedViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.img_size % cfg.patch_size == 0
        self.grid = cfg.img_size // cfg.patch_size
        self.num_patches = self.grid * self.grid
        self.num_prefix = (1 if cfg.use_cls_token else 0) + cfg.num_register_tokens

        self.patch_embed = nn.Conv2d(
            cfg.in_chans, cfg.embed_dim, cfg.patch_size, cfg.patch_size
        )
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if cfg.use_cls_token else None
        )
        self.reg_tokens = (
            nn.Parameter(torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim))
            if cfg.num_register_tokens > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        cos, sin = build_axial_rope_cossin(
            self.grid, self.grid, cfg.embed_dim // cfg.num_heads, cfg.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.reg_tokens is not None:
            nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def attentions(self) -> List[Attention]:
        """The injection points the BandManager pushes biases into."""
        return [blk.attn for blk in self.blocks]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_token: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
        bg_token: Optional[torch.Tensor] = None,
        return_taps: bool = True,
    ):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, C)
        if bg is not None:
            if bg_token is None:
                raise ValueError("bg_token required when bg is set")
            if bg.shape != (B, self.num_patches):
                raise ValueError(
                    f"bg must be [B, num_patches]={B, self.num_patches}, got {tuple(bg.shape)}"
                )
            if mask is not None and (mask & bg).any():
                raise ValueError("mask and bg must not overlap")
            bt = bg_token.expand(B, 1, -1)
            x = torch.where(bg.unsqueeze(-1), bt, x)
        if mask is not None:
            if mask_token is None:
                raise ValueError("mask_token required when mask is set")
            if mask.shape != (B, self.num_patches):
                raise ValueError(
                    f"mask must be [B, num_patches]={B, self.num_patches}, got {tuple(mask.shape)}"
                )
            mt = mask_token.expand(B, 1, -1)
            x = torch.where(mask.unsqueeze(-1), mt, x)
        prefix = []
        if self.cls_token is not None:
            prefix.append(self.cls_token.expand(B, -1, -1))
        if self.reg_tokens is not None:
            prefix.append(self.reg_tokens.expand(B, -1, -1))
        if prefix:
            x = torch.cat(prefix + [x], dim=1)              # (B, P+N, C)

        bg_attn_bias = None
        if bg is not None:
            bg_attn_bias = build_bg_attn_bias(bg, self.num_prefix, x.dtype)

        taps: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(
                x, self.rope_cos, self.rope_sin, self.num_prefix,
                extra_bias=bg_attn_bias,
            )
            if return_taps:
                taps.append(x)
        x = self.norm(x)
        return (x, taps) if return_taps else x


# ============================================================================
# BandManager  —  the management/injection layer. Owns geometry + schedule.
# Pushes additive biases into the block attentions every step; resamples every
# K steps. Frozen-vs-sampled is decided HERE, not in the blocks.
# ============================================================================
BAND_SAMPLE_MODES = ("independent", "balanced", "cyclic", "balanced_no_adjacent")


def weighted_band_counts(n_blocks: int, weights: Sequence[float]) -> List[int]:
    """Largest-remainder allocation of ``n_blocks`` across ``weights``."""
    n_bands = len(weights)
    if n_blocks <= 0 or n_bands <= 0:
        return []
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"band weights must sum to > 0, got {weights}")
    norm = [w / total for w in weights]
    raw = [n_blocks * w for w in norm]
    floors = [int(x) for x in raw]
    rem = [r - f for r, f in zip(raw, floors)]
    leftover = n_blocks - sum(floors)
    for j in sorted(range(n_bands), key=lambda i: rem[i], reverse=True)[:leftover]:
        floors[j] += 1
    return floors


@dataclass
class BandConfig:
    # Distance bands as (r_lo, r_hi) in grid-cell units; one is sampled per
    # block per resample. A "far" band (large r_lo) is the long-range enforcer.
    bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
    )
    keep_self: bool = True          # always allow the diagonal (query sees itself)
    sample_mode: str = "balanced"   # see BAND_SAMPLE_MODES
    weights: Tuple[float, ...] = () # per-band mix; empty => equal counts
    seed: int = 0


class BandManager(nn.Module):
    def __init__(
        self,
        model: BandedViT,
        K: "int | Callable[[int], int]" = 1,
        band_cfg: Optional[BandConfig] = None,
        off_steps: int = 0,
        on_steps: int = 0,
    ):
        super().__init__()
        self.attns = model.attentions()
        self.num_prefix = model.num_prefix
        self.grid = model.grid
        self.T = self.num_prefix + model.num_patches
        self.K: Callable[[int], int] = (lambda step: K) if isinstance(K, int) else K
        self.off_steps = off_steps
        self.on_steps = on_steps
        self._bands_active = False
        self.cfg = band_cfg or BandConfig()
        if self.cfg.sample_mode not in BAND_SAMPLE_MODES:
            raise ValueError(
                f"band sample_mode must be one of {BAND_SAMPLE_MODES}, "
                f"got {self.cfg.sample_mode!r}"
            )

        # Floor guarantee: every query must keep >=1 allowed key (else softmax
        # over an all -inf row -> NaN). Prefix tokens are always-attendable; if
        # there are none, keep_self must hold so the diagonal is the floor.
        assert self.num_prefix > 0 or self.cfg.keep_self, (
            "no prefix tokens and keep_self=False -> a far band can fully mask a "
            "row and NaN the softmax. Add a register/cls token or set keep_self."
        )

        self.register_buffer("dist", self._build_distance(), persistent=False)
        self.register_buffer("is_prefix_pair", self._build_prefix_pair(), persistent=False)
        self._biases: List[Optional[torch.Tensor]] = [None] * len(self.attns)
        self._gen = torch.Generator().manual_seed(self.cfg.seed)

    # ---- geometry (computed once) -----------------------------------------
    def _patch_coords(self) -> torch.Tensor:
        idx = torch.arange(self.grid * self.grid)
        return torch.stack([idx // self.grid, idx % self.grid], dim=1).float()  # (N,2)

    def _build_distance(self) -> torch.Tensor:
        T, P = self.T, self.num_prefix
        D = torch.zeros(T, T)
        coords = self._patch_coords()
        D[P:, P:] = torch.cdist(coords, coords)             # Euclidean grid distance
        return D

    def _build_prefix_pair(self) -> torch.Tensor:
        T, P = self.T, self.num_prefix
        m = torch.zeros(T, T, dtype=torch.bool)
        if P > 0:
            m[:P, :] = True      # prefix tokens (as queries) attend everything
            m[:, :P] = True      # every query attends the prefix tokens  <- the floor
        return m

    # ---- mask construction -------------------------------------------------
    def _activation_dtype(self) -> torch.dtype:
        return self.attns[0].qkv.weight.dtype

    def _band_mask(self, r_lo: float, r_hi: float) -> torch.Tensor:
        dev = self.dist.device
        dtype = self._activation_dtype()
        allowed = (self.dist >= r_lo) & (self.dist < r_hi)
        allowed = allowed | self.is_prefix_pair
        if self.cfg.keep_self:
            allowed = allowed | torch.eye(self.T, dtype=torch.bool, device=dev)
        bias = torch.zeros(self.T, self.T, device=dev, dtype=dtype)
        bias.masked_fill_(~allowed, _masked_attn_fill(dtype))
        return bias

    def _band_counts(self, n_blocks: int, n_bands: int) -> List[int]:
        if self.cfg.weights:
            if len(self.cfg.weights) != n_bands:
                raise ValueError(
                    f"band weights length {len(self.cfg.weights)} != num bands {n_bands}"
                )
            return weighted_band_counts(n_blocks, self.cfg.weights)
        base = n_blocks // n_bands
        rem = n_blocks % n_bands
        return [base + (1 if j < rem else 0) for j in range(n_bands)]

    def _shuffle_indices(self, indices: List[int]) -> List[int]:
        perm = torch.randperm(len(indices), generator=self._gen).tolist()
        return [indices[i] for i in perm]

    def _assign_independent(self, n_blocks: int, n_bands: int) -> List[int]:
        if self.cfg.weights:
            w = torch.tensor(self.cfg.weights, dtype=torch.float64)
            w = w / w.sum()
            return torch.multinomial(
                w, n_blocks, replacement=True, generator=self._gen,
            ).tolist()
        return [
            int(torch.randint(n_bands, (1,), generator=self._gen).item())
            for _ in range(n_blocks)
        ]

    def _assign_balanced(self, n_blocks: int, n_bands: int) -> List[int]:
        counts = self._band_counts(n_blocks, n_bands)
        indices: List[int] = []
        for j, c in enumerate(counts):
            indices.extend([j] * c)
        return self._shuffle_indices(indices)

    def _assign_cyclic(self, n_blocks: int, n_bands: int) -> List[int]:
        offset = int(torch.randint(n_bands, (1,), generator=self._gen).item())
        labels = list(range(n_bands))
        perm = torch.randperm(n_bands, generator=self._gen).tolist()
        labels = [labels[i] for i in perm]
        return [labels[(offset + i) % n_bands] for i in range(n_blocks)]

    def _assign_balanced_no_adjacent(self, n_blocks: int, n_bands: int) -> List[int]:
        counts = self._band_counts(n_blocks, n_bands)
        for _ in range(256):
            remaining = counts.copy()
            out: List[int] = []
            prev = -1
            ok = True
            for _ in range(n_blocks):
                choices = [j for j in range(n_bands) if remaining[j] > 0 and j != prev]
                if not choices:
                    ok = False
                    break
                j = int(choices[int(torch.randint(len(choices), (1,), generator=self._gen).item())])
                out.append(j)
                remaining[j] -= 1
                prev = j
            if ok:
                return out
        # Fallback: cyclic always satisfies adjacency when n_bands >= 2.
        return self._assign_cyclic(n_blocks, n_bands)

    def _assign_band_indices(self, n_blocks: int) -> List[int]:
        n_bands = len(self.cfg.bands)
        if n_blocks <= 0 or n_bands <= 0:
            return []
        mode = self.cfg.sample_mode
        if mode == "independent":
            return self._assign_independent(n_blocks, n_bands)
        if mode == "balanced":
            return self._assign_balanced(n_blocks, n_bands)
        if mode == "cyclic":
            return self._assign_cyclic(n_blocks, n_bands)
        if mode == "balanced_no_adjacent":
            return self._assign_balanced_no_adjacent(n_blocks, n_bands)
        raise ValueError(f"unknown band sample_mode {mode!r}")

    def sample(self, step: int) -> List[torch.Tensor]:
        del step
        indices = self._assign_band_indices(len(self.attns))
        return [self._band_mask(*self.cfg.bands[j]) for j in indices]

    # ---- the on-the-fly control surface -----------------------------------
    def use_bands(self, step: int) -> bool:
        """True when distance-band masks should be active at ``step``."""
        period = self.off_steps + self.on_steps
        if period <= 0 or self.off_steps <= 0 or self.on_steps <= 0:
            return True
        return (step % period) >= self.off_steps

    def maybe_resample(self, step: int) -> None:
        """Call once per training step. Regenerates masks every K steps;
        (re)pushes the current masks into every block on every call."""
        if not self.use_bands(step):
            for attn in self.attns:
                attn.set_bias(None)
            self._biases = [None] * len(self.attns)
            self._bands_active = False
            return

        entering = not self._bands_active
        if entering or step % self.K(step) == 0:
            self._biases = self.sample(step)
        self._bands_active = True
        for attn, b in zip(self.attns, self._biases):
            attn.set_bias(b)

    def freeze(self, biases: Optional[List[torch.Tensor]] = None) -> None:
        """Lock a fixed arrangement for deployment (stop calling maybe_resample)."""
        if biases is not None:
            self._biases = biases
        for attn, b in zip(self.attns, self._biases):
            attn.set_bias(b)

    def clear(self) -> None:
        """Remove all biases -> plain full attention."""
        for attn in self.attns:
            attn.set_bias(None)


# ============================================================================
# Sanity checks (the contract). Run: python -m tomojepa.bandedvit.bvit
# ============================================================================
def sanity_check():
    torch.manual_seed(0)
    cfg = ViTConfig(
        img_size=64, patch_size=16, in_chans=1, embed_dim=384, depth=4,
        num_heads=6, use_cls_token=True, num_register_tokens=2,
    )
    model = BandedViT(cfg).eval()
    mgr = BandManager(model, K=1, band_cfg=BandConfig())
    x = torch.randn(2, 1, 64, 64)
    T = model.num_prefix + model.num_patches

    # [1] plain forward, taps, shapes, finiteness
    mgr.clear()
    out, taps = model(x)
    assert out.shape == (2, T, 384), out.shape
    assert len(taps) == cfg.depth
    assert torch.isfinite(out).all()
    print(f"[1] forward OK   tokens={T} (prefix={model.num_prefix}, "
          f"patches={model.num_patches})   taps={len(taps)}   out={tuple(out.shape)}")

    # [2] RoPE is a proper rotation (norm-preserving) and is actually applied
    hd = cfg.embed_dim // cfg.num_heads
    q = torch.randn(1, cfg.num_heads, model.num_patches, hd)
    qr = apply_rope(q, model.rope_cos, model.rope_sin)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4), "RoPE must preserve norm"
    assert not torch.allclose(q, qr), "RoPE must actually change q"
    print(f"[2] RoPE OK      norm-preserving (max norm drift "
          f"{(q.norm(dim=-1) - qr.norm(dim=-1)).abs().max():.2e}), "
          f"max|Δq|={(q - qr).abs().max():.3f}")

    # [3] bias injection: every masked (q,k) pair gets exactly ~0 attention,
    #     rows still sum to 1. Tested on block-0 attention's own math.
    A = model.blocks[0].attn
    allow = torch.zeros(T, T, dtype=torch.bool)
    allow[:, model.num_prefix] = True            # allow exactly one patch key...
    allow |= torch.eye(T, dtype=torch.bool)      # ...plus the diagonal
    bias = torch.zeros(T, T).masked_fill_(~allow, float("-inf"))
    A.set_bias(bias)
    with torch.no_grad():
        xin = torch.randn(1, T, 384)
        qkv = A.qkv(xin).reshape(1, T, 3, A.num_heads, A.head_dim).permute(2, 0, 3, 1, 4)
        qq, kk, _ = qkv[0], qkv[1], qkv[2]
        aw = ((qq @ kk.transpose(-2, -1)) * A.scale + A.bias).softmax(-1)  # (1,nh,T,T)
    aw_mean = aw.mean(1)[0]                       # (T,T) averaged over heads
    inf_mask = torch.isinf(bias)
    max_leak = aw_mean[inf_mask].max().item()
    row_sums = aw_mean.sum(-1)
    assert max_leak < 1e-6, f"masked keys leak attention: {max_leak}"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    print(f"[3] bias OK      max attn on masked keys={max_leak:.2e}   "
          f"rows sum to 1 (min={row_sums.min():.4f}, max={row_sums.max():.4f})")

    # [4] floor: a far-only band (no keep_self) must never fully mask a row,
    #     and the full model must stay finite under it (no NaN).
    mgr.cfg = BandConfig(bands=[(5.0, 1e9)], keep_self=False)  # rely on prefix floor
    mgr.maybe_resample(0)
    b0 = model.blocks[0].attn.bias
    finite_per_row = torch.isfinite(b0).sum(-1)
    assert (finite_per_row >= 1).all(), "a query has zero allowed keys -> NaN risk"
    out2, _ = model(x)
    assert torch.isfinite(out2).all(), "non-finite output under far-only banding"
    print(f"[4] floor OK     far-only band, min allowed keys/row="
          f"{finite_per_row.min().item()} (prefix floor holds)   output finite")

    # [5] banding actually changes the representation vs full attention,
    #     and resampling on the K-schedule changes the masks.
    mgr.cfg = BandConfig()
    mgr.maybe_resample(0)
    rep_banded, _ = model(x)
    mgr.clear()
    rep_full, _ = model(x)
    diff = (rep_banded - rep_full).abs().mean().item()
    assert diff > 1e-4, "banding had no effect on the representation"
    print(f"[5] effect OK    mean|Δrepr| banded-vs-full={diff:.4f}")

    # [6] gradient flows through q/k/v but NOT through a hard (-inf) mask
    model.train()
    mgr.maybe_resample(0)
    out3, _ = model(x)
    out3.float().sum().backward()
    g = model.blocks[0].attn.qkv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert model.blocks[0].attn.bias.grad is None  # hard mask is a constant, no grad
    print(f"[6] grad OK      qkv.grad finite & nonzero (|g|sum={g.abs().sum():.1f}); "
          f"hard mask carries no grad")

    # [7] fp16: RoPE buffers stay fp32 but activations may be half
    model.eval()
    model.half()
    mgr.maybe_resample(0)
    x16 = x.half()
    out_fp16, _ = model(x16)
    assert torch.isfinite(out_fp16).all(), "non-finite output under fp16"
    print(f"[7] fp16 OK      output finite under model.half()")

    # [8] device: bias buffers follow .cuda() before maybe_resample
    if torch.cuda.is_available():
        model_fp32 = BandedViT(cfg).cuda().eval()
        mgr_cuda = BandManager(model_fp32, K=1, band_cfg=BandConfig()).cuda()
        mgr_cuda.maybe_resample(0)
        x_cuda = torch.randn(1, 1, 64, 64, device="cuda")
        b_cuda = model_fp32.blocks[0].attn.bias
        assert b_cuda is not None and b_cuda.device.type == "cuda"
        out_cuda, _ = model_fp32(x_cuda)
        assert torch.isfinite(out_cuda).all(), "non-finite output on CUDA"
        print(f"[8] cuda OK      bias on {b_cuda.device}, output finite")
    else:
        print("[8] cuda SKIP    CUDA not available")

    # [9] background attention isolation: FG queries must not attend to BG keys.
    mgr.clear()
    model.float().eval()
    n = model.num_patches
    bg_flat = torch.zeros(2, n, dtype=torch.bool)
    bg_flat[:, : n // 4] = True
    bg_bias = build_bg_attn_bias(bg_flat, model.num_prefix, torch.float32)
    A = model.blocks[0].attn
    with torch.no_grad():
        xin = torch.randn(2, T, 384)
        qkv = A.qkv(xin).reshape(2, T, 3, A.num_heads, A.head_dim).permute(2, 0, 3, 1, 4)
        qq, kk, _ = qkv[0], qkv[1], qkv[2]
        aw = ((qq @ kk.transpose(-2, -1)) * A.scale + bg_bias).softmax(-1).mean(1)[0]
    p = model.num_prefix
    fg_rows = torch.arange(p, T)[~bg_flat[0]]
    bg_cols = torch.arange(p, T)[bg_flat[0]]
    leak = aw[fg_rows[:, None], bg_cols[None, :]].max().item() if (
        fg_rows.numel() > 0 and bg_cols.numel() > 0
    ) else 0.0
    assert leak < 1e-6, f"FG queries leak attention to BG keys: {leak}"
    bg_tok = torch.zeros(1, 1, 384)
    out_bg, _ = model(x, bg=bg_flat, bg_token=bg_tok)
    assert torch.isfinite(out_bg).all(), "non-finite output under bg attention mask"
    print(f"[9] bg attn OK  max FG->BG attn={leak:.2e}; output finite")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nALL CHECKS PASSED   (ViT-S/{cfg.patch_size} demo, {cfg.depth} blocks, "
          f"{n_params/1e6:.1f}M params)")


if __name__ == "__main__":
    sanity_check()