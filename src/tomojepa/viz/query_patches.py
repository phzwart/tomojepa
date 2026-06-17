"""Cross-image patch retrieval over a shared-basis token database.

NOTE: superseded by the ``patchdb`` package (DuckDB + FAISS). Prefer
``python -m patchdb.cli query``. Kept as a standalone npz-based prototype.


Given a rectangular region of patches (ANY size) in a query image, find the most
similar regions of the same footprint anywhere in the database. Each region is
summarized by the foreground-weighted mean of its per-token codes (the top-K
shared-basis projections from ``build_token_db.py``); similarity is cosine in
that K-dim space.

Because PC1 typically dominates the spectrum (e.g. porosity at ~97% variance),
``--whiten`` (default on) rescales each component by ``1/sqrt(eigenvalue)`` so
finer structure -- not just overall density -- drives the match. Use
``--no_whiten`` to retrieve by the raw (variance-weighted) coordinates instead.

Sliding-window means over every image are computed in O(G^2 * K) with integral
images, so querying the whole database is fast.

Usage:
    python query_patches.py --db runs/soil_residual_fg/token_db_ep14.npz \
        --query_image 484 --bbox 12 12 5 5 --topk 12
    # bbox = top-left patch row, col, height, width (in PATCH units)
    # add --px to give --bbox in pixel units instead
"""
import os
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..core.dataset import TomographyDataset


def integral_window_mean(codes, fg, h, w):
    """Foreground-weighted mean of ``codes`` over every ``h x w`` window.

    codes: [N, G, G, K], fg: [N, G, G]. Returns ``(desc, frac)`` where
    ``desc[n, y, x] = mean of foreground codes in window (y..y+h, x..x+w)`` with
    shape [N, Y, X, K], and ``frac`` is the foreground fraction of each window.
    """
    N, G, _, K = codes.shape
    wsum = codes * fg[..., None]
    # integral images (pad a leading zero row/col)
    ii = np.zeros((N, G + 1, G + 1, K), dtype=np.float64)
    ii[:, 1:, 1:] = wsum.cumsum(1).cumsum(2)
    cw = np.zeros((N, G + 1, G + 1), dtype=np.float64)
    cw[:, 1:, 1:] = fg.astype(np.float64).cumsum(1).cumsum(2)

    def rect(A):
        return (A[:, h:, w:] - A[:, :-h, w:] - A[:, h:, :-w] + A[:, :-h, :-w])

    csum = rect(ii)                                  # [N, Y, X, K]
    cnt = rect(cw)                                    # [N, Y, X]
    desc = csum / np.clip(cnt, 1.0, None)[..., None]
    frac = cnt / float(h * w)
    return desc.astype(np.float32), frac.astype(np.float32)


def load_image(ds, idx):
    item = ds[idx]
    view = item[0] if isinstance(item, (list, tuple)) else item
    if isinstance(view, (list, tuple)):
        view = view[0]
    return view.mean(0).cpu().numpy()                # [H, W] grayscale


def main():
    p = argparse.ArgumentParser(description="Cross-image patch retrieval")
    p.add_argument("--db", required=True, help="token_db .npz from build_token_db.py")
    p.add_argument("--query_image", type=int, required=True,
                   help="dataset index of the query image")
    p.add_argument("--bbox", type=int, nargs=4, required=True,
                   metavar=("ROW", "COL", "H", "W"),
                   help="query region top-left (row,col) + size (h,w)")
    p.add_argument("--px", action="store_true",
                   help="interpret --bbox in pixel units (default: patch units)")
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--whiten", dest="whiten", action="store_true", default=True)
    p.add_argument("--no_whiten", dest="whiten", action="store_false")
    p.add_argument("--min_fg_frac", type=float, default=0.6,
                   help="skip candidate windows with less foreground than this")
    p.add_argument("--per_image", action="store_true",
                   help="keep only the single best window per image")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    db = np.load(args.db, allow_pickle=True)
    codes = db["codes"].astype(np.float32)          # [N, G, G, K]
    fg = db["fg"]                                     # [N, G, G]
    ev = db["ev"].astype(np.float32)                 # [K]
    image_ids = db["image_ids"]
    G = int(db["grid"]); ps = int(db["patch_size"])
    K = int(db["k"])
    N = codes.shape[0]

    # resolve query bbox into patch units
    r, c, h, w = args.bbox
    if args.px:
        r, c, h, w = r // ps, c // ps, max(1, h // ps), max(1, w // ps)
    h = max(1, min(h, G)); w = max(1, min(w, G))
    r = max(0, min(r, G - h)); c = max(0, min(c, G - w))

    # locate the query image row in the db
    qrows = np.where(image_ids == args.query_image)[0]
    if len(qrows) == 0:
        raise SystemExit(f"query image {args.query_image} not in db "
                         f"(image_ids range {image_ids.min()}..{image_ids.max()})")
    qrow = int(qrows[0])

    # whitening scale (down-weight dominant comps so texture matters)
    scale = (1.0 / np.sqrt(np.clip(ev, 1e-8, None))) if args.whiten else np.ones(K, np.float32)

    def normd(x):                                     # whiten + L2-normalize
        x = x * scale
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)

    # query descriptor: foreground-weighted mean over the query window
    qcodes = codes[qrow, r:r + h, c:c + w]            # [h, w, K]
    qfg = fg[qrow, r:r + h, c:c + w]
    qvec = (qcodes * qfg[..., None]).reshape(-1, K).sum(0) / max(qfg.sum(), 1)
    qn = normd(qvec)

    # sliding-window descriptors for the whole db, same footprint
    desc, frac = integral_window_mean(codes, fg, h, w)   # [N, Y, X, K], [N, Y, X]
    Y, X = desc.shape[1], desc.shape[2]
    dn = normd(desc.reshape(-1, K)).reshape(N, Y, X, K)
    sim = dn @ qn                                      # [N, Y, X] cosine

    valid = frac >= args.min_fg_frac
    sim = np.where(valid, sim, -2.0)

    # suppress the exact query window
    if r < Y and c < X:
        sim[qrow, r, c] = -2.0

    # rank windows; greedy spatial NMS within each image so we don't return a
    # cluster of overlapping near-duplicates around one spot
    flat = sim.reshape(-1)
    order = np.argsort(flat)[::-1]
    picks = []
    taken = {}                                        # n -> list of (y,x)
    for idx in order:
        if flat[idx] <= -1.0:
            break
        n, y, x = np.unravel_index(idx, sim.shape)
        n, y, x = int(n), int(y), int(x)
        if args.per_image and n in taken and len(taken[n]) >= 1:
            continue
        ok = True
        for (yy, xx) in taken.get(n, []):
            if abs(yy - y) < h and abs(xx - x) < w:   # overlapping window
                ok = False
                break
        if not ok:
            continue
        taken.setdefault(n, []).append((y, x))
        picks.append((float(flat[idx]), n, y, x))
        if len(picks) >= args.topk:
            break

    # report
    print(f"query img {args.query_image} patch-bbox (row={r},col={c},h={h},w={w}) "
          f"= px [{r*ps}:{(r+h)*ps}, {c*ps}:{(c+w)*ps}]  whiten={args.whiten}")
    for rank, (s, n, y, x) in enumerate(picks):
        print(f"  #{rank+1:>2} sim={s:.3f}  img={int(image_ids[n])}  "
              f"patch(row={y},col={x})  px[{y*ps}:{(y+h)*ps}, {x*ps}:{(x+w)*ps}]")

    # visualize: query crop (with box on the full image) + top matches
    ds = TomographyDataset(
        data_dir=str(db["data_dir"]), dataset_key=str(db["dataset_key"]),
        pattern=str(db["pattern"]), global_views=1, local_views=0,
        variant="tomo2", img_size=int(db["img_size"]), is_train=False,
        backend=str(db["backend"]),
    )
    ncols = 4
    ntiles = 1 + len(picks)
    nrows = int(np.ceil(ntiles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.array(axes).reshape(-1)

    def draw(ax, img_idx, yy, xx, title, color):
        img = load_image(ds, img_idx)
        ax.imshow(img, cmap="gray")
        rectp = mpatches.Rectangle((xx * ps, yy * ps), w * ps, h * ps,
                                   fill=False, edgecolor=color, linewidth=2.0)
        ax.add_patch(rectp)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    draw(axes[0], args.query_image, r, c, f"QUERY img {args.query_image}", "lime")
    for t, (s, n, y, x) in enumerate(picks):
        draw(axes[t + 1], int(image_ids[n]), y, x,
             f"#{t+1} img{int(image_ids[n])} sim={s:.2f}", "red")
    for j in range(ntiles, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"patch retrieval  ({h}x{w} patches = {h*ps}x{w*ps}px)  "
                 f"whiten={args.whiten}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(os.path.dirname(args.db),
                                   f"query_img{args.query_image}_r{r}c{c}_{h}x{w}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
