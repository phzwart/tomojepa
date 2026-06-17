"""FeatX -- local sub-token feature extractor (paper section 2.3, Fig. 2D).

Windowed attention blurs high-frequency detail, especially in shallow layers.
FeatX recovers it by conditioning the *nearest* patch-token feature on the
sub-token offset between the query and that token's center:

    k_nn      = argmin_k || x_q - X[k] ||     (nearest patch-token)
    x_nn      = X[k_nn]                         (its center coordinate)
    h_nn      = H[k_nn]                         (its feature)
    Delta x   = (x_q - x_nn) / p                (offset, token-grid units)
    p_dx      = E_pos(Delta x)                  (sinusoidal encoding, dim 64)
    gamma,beta= MLP_FiLM(p_dx)
    h~_nn     = (1 + gamma) * LN(h_nn) + beta   (FiLM modulation)
    x_subtok  = MLP_subtoken(h~_nn)
"""
import torch
import torch.nn as nn


class SinusoidalPosEnc(nn.Module):
    """Coordinate-field sinusoidal encoding ``R^{in_dim} -> R^{dim}``."""

    def __init__(self, dim: int = 64, in_dim: int = 2):
        super().__init__()
        if dim % (2 * in_dim) != 0:
            raise ValueError(f"dim {dim} must be divisible by 2*in_dim {2 * in_dim}")
        self.dim = dim
        self.in_dim = in_dim
        num_freqs = dim // (2 * in_dim)
        bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("bands", bands * torch.pi, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim] -> [..., dim]
        proj = x.unsqueeze(-1) * self.bands.to(x.dtype)         # [..., in_dim, F]
        proj = proj.flatten(-2)                                 # [..., in_dim*F]
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class FeatX(nn.Module):
    def __init__(self, backbone_dim: int, internal_dim: int,
                 posenc_dim: int = 64, hidden_ratio: float = 1.0):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.internal_dim = internal_dim
        self.posenc = SinusoidalPosEnc(posenc_dim, in_dim=2)
        film_hidden = max(backbone_dim, 128)
        self.mlp_film = nn.Sequential(
            nn.Linear(posenc_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * backbone_dim),
        )
        nn.init.zeros_(self.mlp_film[-1].weight)               # start as identity FiLM
        nn.init.zeros_(self.mlp_film[-1].bias)
        self.ln = nn.LayerNorm(backbone_dim)
        sub_hidden = int(max(backbone_dim, internal_dim) * hidden_ratio)
        self.mlp_subtoken = nn.Sequential(
            nn.Linear(backbone_dim, sub_hidden),
            nn.GELU(),
            nn.Linear(sub_hidden, internal_dim),
        )

    def forward(self, h_grid: torch.Tensor, q_coords: torch.Tensor,
                patch_size: int) -> torch.Tensor:
        """Args:
            h_grid: backbone hidden state ``[B, C, h, w]``.
            q_coords: query coordinates ``[B, Q, 2]`` (token-grid units).
            patch_size: ``p`` (kept for the documented ``Delta x / p`` form;
                coordinates are already in token units so this divides 1 token = p
                pixels back out -- here ``q_coords`` are token units so we pass
                the offset directly).
        Returns:
            ``x_subtoken`` of shape ``[B, Q, internal_dim]``.
        """
        B, C, h, w = h_grid.shape
        h_flat = h_grid.permute(0, 2, 3, 1).reshape(B, h * w, C)   # [B,hw,C]

        qi = torch.clamp(torch.floor(q_coords[..., 0]).long(), 0, h - 1)  # [B,Q]
        qj = torch.clamp(torch.floor(q_coords[..., 1]).long(), 0, w - 1)
        flat = (qi * w + qj)                                       # [B,Q]
        h_nn = torch.gather(h_flat, 1, flat.unsqueeze(-1).expand(B, flat.shape[1], C))
        x_nn = torch.stack([qi.to(q_coords.dtype) + 0.5,
                            qj.to(q_coords.dtype) + 0.5], dim=-1)  # [B,Q,2]

        # q_coords already in token units; (x_q - x_nn) is the in-token offset
        # which equals the paper's (x_q - x_nn)/p when x is measured in pixels.
        dx = q_coords - x_nn                                       # [B,Q,2]
        p_dx = self.posenc(dx)                                     # [B,Q,posenc_dim]
        gamma, beta = self.mlp_film(p_dx).chunk(2, dim=-1)         # each [B,Q,C]
        h_tilde = (1.0 + gamma) * self.ln(h_nn) + beta
        return self.mlp_subtoken(h_tilde)                         # [B,Q,internal_dim]
