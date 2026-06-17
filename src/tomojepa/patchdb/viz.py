"""Render a query window and its retrieved matches as a PNG grid."""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def render_query(engine, results, *, bbox, query_image_id=None,
                 query_img=None, out_path, title=None, ncols=4):
    """``results`` is the list from ``RetrievalEngine.query*``.

    Provide either ``query_image_id`` (in-DB) or ``query_img`` ([H,W] array).
    ``bbox`` is (row, col, h, w) in patch units.
    """
    ps = engine.ps
    r, c, h, w = bbox
    if query_img is None:
        query_img = engine.load_image(query_image_id)

    ntiles = 1 + len(results)
    nrows = int(np.ceil(ntiles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.array(axes).reshape(-1)

    def draw(ax, img, yy, xx, hh, ww, ttl, color):
        ax.imshow(img, cmap="gray")
        ax.add_patch(mpatches.Rectangle((xx * ps, yy * ps), ww * ps, hh * ps,
                                        fill=False, edgecolor=color, linewidth=2.0))
        ax.set_title(ttl, fontsize=8)
        ax.axis("off")

    qlabel = f"QUERY img {query_image_id}" if query_image_id is not None else "QUERY (external)"
    draw(axes[0], query_img, r, c, h, w, qlabel, "lime")
    for t, m in enumerate(results):
        img = engine.load_image(m["image_id"])
        p = m["patch"]
        draw(axes[t + 1], img, p["row"], p["col"], p["h"], p["w"],
             f"#{m['rank']} img{m['image_id']} sim={m['similarity']:.2f}", "red")
    for j in range(ntiles, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title or f"patch retrieval ({h}x{w} patches = {h*ps}x{w*ps}px)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
