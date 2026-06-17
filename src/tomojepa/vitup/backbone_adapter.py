"""Backbone adapter -- the only coupling point between ViT-Up and the host ViT.

Exposes the minimal interface ViT-Up depends on (paper section 0 / section 6):
patch size ``p``, model dim ``C``, layer count ``L``, the patch-embedding
module, intermediate hidden states ``H_l`` for a requested subset of layers, and
the low-resolution patch-token center coordinates ``X``. Optionally applies LoRA
to the patch-embed conv and attention Q/K/V/O projections.

Everything downstream (query embedding, blocks, model) talks to this adapter and
nothing else about the backbone, so a different ViT only needs a new adapter.
"""
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from .lora import wrap_with_lora, freeze_non_lora


def build_backbone(name: str, in_chans: int, img_size: int,
                   dynamic_img_size: bool = True) -> nn.Module:
    """Create a headless timm ViT that accepts variable input resolutions."""
    import timm
    return timm.create_model(
        name, pretrained=False, num_classes=0, in_chans=in_chans,
        img_size=img_size, dynamic_img_size=dynamic_img_size,
    )


def load_backbone_state(backbone: nn.Module, ckpt_path: str,
                        device="cpu") -> None:
    """Load backbone weights from a repo checkpoint (``DINOv3ViTEncoder``).

    The checkpoint stores the encoder under ``"net"`` with backbone params
    prefixed ``"backbone."`` and an unused projection head under ``"proj."``;
    we keep only the ``backbone.*`` subset and load non-strictly (the timm head
    was dropped via ``num_classes=0``).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("net", ckpt)
    bb = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            bb[k[len("backbone."):]] = v
    missing, unexpected = backbone.load_state_dict(bb, strict=False)
    # head.* (classifier) is expected to be unexpected; warn on anything else.
    leftover = [u for u in unexpected if not u.startswith("head.")]
    if leftover:
        print(f"[backbone_adapter] unexpected keys: {leftover[:8]}"
              f"{' ...' if len(leftover) > 8 else ''}", flush=True)
    real_missing = [m for m in missing if "lora_" not in m]
    if real_missing:
        print(f"[backbone_adapter] missing keys: {real_missing[:8]}"
              f"{' ...' if len(real_missing) > 8 else ''}", flush=True)


class BackboneAdapter(nn.Module):
    """Uniform access to a timm ViT's structure and intermediate features."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        pe = backbone.patch_embed
        ps = pe.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.embed_dim = int(getattr(backbone, "embed_dim",
                                     getattr(backbone, "num_features")))
        self.num_layers = len(backbone.blocks)
        self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 0))

    # -- properties expected by ViT-Up (p, C, L) ----------------------------
    @property
    def p(self) -> int:
        return self.patch_size

    @property
    def C(self) -> int:
        return self.embed_dim

    @property
    def L(self) -> int:
        return self.num_layers

    @property
    def patch_embed(self) -> nn.Module:
        return self.backbone.patch_embed

    # -- LoRA ----------------------------------------------------------------
    def apply_lora(self, targets: Sequence[str], r: int, alpha: float,
                   dropout: float) -> List[nn.Module]:
        """Wrap the requested projections with LoRA and freeze everything else.

        ``targets`` are logical names: ``patch_embed`` (the conv), ``attn.qkv``
        (the fused Q/K/V projection, or separate ``q_proj/k_proj/v_proj`` when
        the backbone is not fused), and ``attn.proj`` (the output O projection).
        """
        wrapped: List[nn.Module] = []
        for t in targets:
            if t == "patch_embed":
                wrapped.append(wrap_with_lora(self.backbone, "patch_embed.proj",
                                              r, alpha, dropout))
            elif t == "attn.qkv":
                for i in range(self.num_layers):
                    attn = self.backbone.blocks[i].attn
                    if getattr(attn, "qkv", None) is not None:
                        wrapped.append(wrap_with_lora(
                            self.backbone, f"blocks.{i}.attn.qkv", r, alpha, dropout))
                    else:  # non-fused fallback: q_proj / k_proj / v_proj
                        for sub in ("q_proj", "k_proj", "v_proj"):
                            if getattr(attn, sub, None) is not None:
                                wrapped.append(wrap_with_lora(
                                    self.backbone, f"blocks.{i}.attn.{sub}",
                                    r, alpha, dropout))
            elif t == "attn.proj":
                for i in range(self.num_layers):
                    wrapped.append(wrap_with_lora(
                        self.backbone, f"blocks.{i}.attn.proj", r, alpha, dropout))
            else:
                raise ValueError(f"unknown LoRA target: {t!r}")
        freeze_non_lora(self.backbone)
        return wrapped

    # -- feature extraction --------------------------------------------------
    def _grid_size(self, img: torch.Tensor):
        h = img.shape[-2] // self.patch_size
        w = img.shape[-1] // self.patch_size
        return h, w

    def embedding_grid(self, img: torch.Tensor) -> torch.Tensor:
        """The embedding-layer feature ``H_0`` as ``[B, C, h, w]``.

        This is the patch-embed output (the space ``q_0`` lives in), used as the
        ``t = 0`` (``l[0] = 0``) supervision target.
        """
        x = self.backbone.patch_embed(img)
        h, w = self._grid_size(img)
        return self._to_nchw(x, h, w)

    def hidden_states(self, img: torch.Tensor,
                      layer_indices: Sequence[int]) -> Dict[int, torch.Tensor]:
        """Intermediate hidden states ``H_l`` for 1-based backbone layers.

        Returns ``{l: [B, C, h, w]}``. Layer ``0`` (if requested) is the
        embedding-layer grid; layer ``l >= 1`` is the output of block ``l-1``.
        """
        out: Dict[int, torch.Tensor] = {}
        block_layers = [l for l in layer_indices if l >= 1]
        if 0 in layer_indices:
            out[0] = self.embedding_grid(img)
        if block_layers:
            block_idx = [l - 1 for l in block_layers]
            feats = self.backbone.forward_intermediates(
                img, indices=block_idx, norm=False, output_fmt="NCHW",
                intermediates_only=True,
            )
            for l, f in zip(block_layers, feats):
                out[l] = f
        return out

    def token_centers(self, h: int, w: int, device=None,
                      dtype=torch.float32) -> torch.Tensor:
        """Patch-token center coordinates ``X in R^{h x w x 2}`` (token units).

        Coordinates are ``(row + 0.5, col + 0.5)`` in token-grid units, i.e. the
        center of each patch measured in patch widths. Order is ``(y, x)``.
        """
        ys = torch.arange(h, device=device, dtype=dtype) + 0.5
        xs = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx], dim=-1)

    @staticmethod
    def _to_nchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if x.dim() == 4:                       # NHWC or NCHW
            if x.shape[1] == h and x.shape[2] == w:   # NHWC
                return x.permute(0, 3, 1, 2).contiguous()
            return x                                  # already NCHW
        # NLC -> NCHW
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()
