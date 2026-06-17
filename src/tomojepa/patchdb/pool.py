"""Window pooling + whitening helpers (numpy, no torch)."""
import numpy as np


def whiten_scale(ev, whiten):
    """Per-component scale vector: ``1/sqrt(ev)`` when whitening, else ones.

    Whitening counteracts the dominant leading component (e.g. porosity at ~97%
    variance) so finer structure drives similarity instead of overall density.
    """
    ev = np.asarray(ev, dtype=np.float32)
    if not whiten:
        return np.ones_like(ev)
    return (1.0 / np.sqrt(np.clip(ev, 1e-8, None))).astype(np.float32)


def normalize_codes(x, scale):
    """Apply per-component ``scale`` then L2-normalize along the last axis."""
    x = np.asarray(x, dtype=np.float32) * scale
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def integral_window_mean(codes, fg, h, w):
    """Foreground-weighted mean of ``codes`` over every ``h x w`` window.

    ``codes``: [N, G, G, K], ``fg``: [N, G, G]. Returns ``(desc, frac)`` where
    ``desc[n, y, x]`` is the mean of foreground codes in window
    ``(y..y+h, x..x+w)`` with shape [N, Y, X, K], and ``frac`` is the foreground
    fraction of each window. Computed with integral images in O(N*G*G*K).
    """
    codes = np.asarray(codes, dtype=np.float32)
    fg = np.asarray(fg)
    N, G, _, K = codes.shape
    # float32 integral images: grids are tiny (<=32) and values are O(1), so the
    # cumulative sums stay well within float32 precision while halving cost.
    wsum = codes * fg[..., None]
    ii = np.zeros((N, G + 1, G + 1, K), dtype=np.float32)
    ii[:, 1:, 1:] = wsum.cumsum(1).cumsum(2)
    cw = np.zeros((N, G + 1, G + 1), dtype=np.float32)
    cw[:, 1:, 1:] = fg.astype(np.float32).cumsum(1).cumsum(2)

    def rect(A):
        return A[:, h:, w:] - A[:, :-h, w:] - A[:, h:, :-w] + A[:, :-h, :-w]

    csum = rect(ii)                                  # [N, Y, X, K]
    cnt = rect(cw)                                    # [N, Y, X]
    desc = csum / np.clip(cnt, 1.0, None)[..., None]
    frac = cnt / float(h * w)
    return desc.astype(np.float32), frac.astype(np.float32)


def window_mean_single(codes, fg, r, c, h, w):
    """Foreground-weighted mean of a single ``h x w`` window of one image.

    ``codes``: [G, G, K], ``fg``: [G, G]. Returns a [K] vector.
    """
    codes = np.asarray(codes, dtype=np.float32)
    fg = np.asarray(fg)
    sub = codes[r:r + h, c:c + w]
    m = fg[r:r + h, c:c + w]
    denom = max(float(m.sum()), 1.0)
    return (sub * m[..., None]).reshape(-1, sub.shape[-1]).sum(0) / denom
