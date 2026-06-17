"""Can the encoder representation see object eccentricity (balls vs eggs)?

The test volume is 3D balls and eggs. In a 2D slice a ball always cuts to a
circle; an egg cuts to an ellipse whose eccentricity depends on the cut plane
(and a perpendicular egg cut also looks circular). Density encodes the material
class (ball vs egg). We:

  1. segment objects per slice (reusing analyze_shapes) -> per-object mask +
     measured 2D eccentricity + density,
  2. pool the encoder's patch tokens over each object mask -> object embedding,
  3. linearly decode from the embedding:
       - eccentricity (RidgeCV, cross-validated R^2),
       - ball/egg material (LogisticRegression, CV accuracy/AUC),
       - eccentricity *within eggs only* (controls for the density cue),
  4. compare across runs (e.g. baseline vs residual).

A density-only baseline is reported so we can tell whether the representation
carries shape beyond the trivial material cue.
"""
import os
import glob
import re
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.filters import threshold_otsu
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

from .analyze_shapes import open_raw, window, segment, measure_slice
from ..core.dataset import TomographyDataset
from ..core.model import DINOv3ViTEncoder


def resolve_ckpt(run_dir, ckpt_subdir="ckpt"):
    d = os.path.join(run_dir, ckpt_subdir)
    last = os.path.join(d, "ckpt_last.pth")
    if os.path.exists(last):
        return last
    eps = glob.glob(os.path.join(d, "ckpt_epoch_*.pth"))
    if not eps:
        raise FileNotFoundError(f"no checkpoints under {d}")
    return max(eps, key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)))


@torch.no_grad()
def token_grid(net, view, device):
    feat = net.backbone.forward_features(view.unsqueeze(0).to(device))
    tok = feat[:, net.backbone.num_prefix_tokens:].squeeze(0).float().cpu().numpy()
    g = int(round(tok.shape[0] ** 0.5))
    return tok.reshape(g, g, -1), g


def pool_objects(tok_grid, g, lab, rows):
    """Mean-pool tokens over each object's mask (downsampled to the patch grid)."""
    H, W = lab.shape
    embs = []
    for r in rows:
        m = lab == r["label"]
        mg = resize(m.astype(np.float32), (g, g), order=1) > 0.5
        if not mg.any():                                   # tiny obj -> centroid patch
            gy = min(int(r["cy"] / H * g), g - 1)
            gx = min(int(r["cx"] / W * g), g - 1)
            mg[gy, gx] = True
        embs.append(tok_grid[mg].mean(0))
    return np.stack(embs) if embs else np.zeros((0, tok_grid.shape[-1]))


def cv_regress(X, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 4, 13)))
    pred = cross_val_predict(model, X, y, cv=kf)
    return r2_score(y, pred)


def cv_classify(X, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    pred = cross_val_predict(model, X, y, cv=skf)
    prob = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    return accuracy_score(y, pred), roc_auc_score(y, prob)


def main():
    p = argparse.ArgumentParser(description="Probe encoder for object eccentricity")
    p.add_argument("--runs", nargs="+", required=True, help="run dirs to compare")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="upsampled_1024.zarr")
    p.add_argument("--backend", default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--in_chans", type=int, default=1)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--n_slices", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min_area", type=int, default=40)
    p.add_argument("--max_area", type=int, default=20000)
    p.add_argument("--min_solidity", type=float, default=0.85)
    p.add_argument("--open_radius", type=int, default=1)
    p.add_argument("--denoise_radius", type=int, default=2)
    p.add_argument("--max_fg_frac", type=float, default=0.30)
    p.add_argument("--umap", action="store_true", help="also render a UMAP embedding")
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--no_probe", action="store_true",
                   help="skip the (slow) linear probes; just embed + visualize")
    p.add_argument("--out_dir", default="runs/shape_probe")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. segment objects once (model-independent ground truth) ----
    arr = open_raw(os.path.join(args.data_dir, args.pattern), args.dataset_key)
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(arr.shape[0], size=min(args.n_slices, arr.shape[0]),
                             replace=False).tolist())
    per_slice = {}      # si -> (lab, rows)
    ecc, dens, area = [], [], []
    obj_index = []      # (si, row_pos) in collection order
    for si in idxs:
        raw = np.asarray(arr[si], dtype=np.float32)
        if raw.ndim > 2:
            raw = raw[0]
        win = window(raw)
        lab, _, fg = segment(win, args.min_area, args.open_radius, True, args.denoise_radius)
        if fg > args.max_fg_frac:
            continue
        rows = measure_slice(raw, lab, args.min_solidity, args.max_area)
        if not rows:
            continue
        per_slice[si] = (lab, rows)
        for j, r in enumerate(rows):
            ecc.append(r["eccentricity"]); dens.append(r["mean_intensity"])
            area.append(r["area"]); obj_index.append((si, j))
    ecc = np.array(ecc); dens = np.array(dens); area = np.array(area)
    n = len(ecc)
    print(f"collected {n} objects over {len(per_slice)} grain slices")

    # ball/egg label from the bimodal density (material cue)
    dthr = threshold_otsu(dens)
    egg = (dens > dthr).astype(int)        # 1 = high-density (egg), 0 = ball
    print(f"density split @ {dthr:.3f}: balls={int((egg==0).sum())}  eggs={int(egg.sum())}")

    # density-only controls (does shape need the representation at all?)
    if not args.no_probe:
        r2_ecc_from_dens = cv_regress(dens[:, None], ecc, args.seed)
        print(f"\n[control] eccentricity from density alone:  R2 = {r2_ecc_from_dens:+.3f}")

    # ---- 2-4. per run: embed, probe ----
    summary = []
    embeds = {}
    for run in args.runs:
        ck = resolve_ckpt(run)
        net = DINOv3ViTEncoder(proj_dim=args.proj_dim, img_size=args.img_size,
                               in_chans=args.in_chans, pretrained=False).to(device)
        net.load_state_dict(torch.load(ck, map_location=device)["net"])
        net.eval()
        ds = TomographyDataset(
            data_dir=args.data_dir, dataset_key=args.dataset_key, pattern=args.pattern,
            global_views=1, local_views=0, variant="tomo2", img_size=args.img_size,
            is_train=False, backend=args.backend,
        )
        X = np.zeros((n, 0))
        per_slice_emb = {}
        for si, (lab, rows) in per_slice.items():
            item = ds[si]
            view = item[0] if isinstance(item, (list, tuple)) else item
            tg, g = token_grid(net, view, device)
            per_slice_emb[si] = pool_objects(tg, g, lab, rows)
        D = next(iter(per_slice_emb.values())).shape[1]
        X = np.zeros((n, D), dtype=np.float32)
        for i, (si, j) in enumerate(obj_index):
            X[i] = per_slice_emb[si][j]
        embeds[run] = X
        name = os.path.basename(run.rstrip("/"))

        if not args.no_probe:
            r2_all = cv_regress(X, ecc, args.seed)
            acc, auc = cv_classify(X, egg, args.seed)
            # eccentricity within eggs only (controls for material/density cue)
            mask = egg == 1
            r2_egg = cv_regress(X[mask], ecc[mask], args.seed) if mask.sum() > 15 else float("nan")
            summary.append((name, r2_all, r2_egg, acc, auc))
            print(f"\n[{name}]  ckpt={os.path.basename(ck)}")
            print(f"    eccentricity R2 (all objects):   {r2_all:+.3f}")
            print(f"    eccentricity R2 (eggs only):     {r2_egg:+.3f}")
            print(f"    ball/egg accuracy:               {acc:.3f}")
            print(f"    ball/egg AUC:                    {auc:.3f}")
        else:
            print(f"[{name}]  embedded {X.shape[0]} objects (probe skipped)")

    # ---- comparison table ----
    if not args.no_probe:
        print("\n=== summary ===")
        print(f"{'run':>16} | ecc R2(all) | ecc R2(eggs) | ball/egg acc | AUC")
        print("-" * 70)
        print(f"{'density-only':>16} | {r2_ecc_from_dens:+10.3f} | "
              f"{'':>12} | {'':>12} | {'':>4}")
        for name, r2a, r2e, acc, auc in summary:
            print(f"{name:>16} | {r2a:+10.3f} | {r2e:+12.3f} | {acc:12.3f} | {auc:.3f}")

    # ---- PCA of object embeddings, coloured by eccentricity ----
    nr = len(args.runs)
    fig, axes = plt.subplots(1, nr, figsize=(5.2 * nr, 4.6), squeeze=False)
    for ax, run in zip(axes[0], args.runs):
        X = embeds[run]
        Z = PCA(2).fit_transform(StandardScaler().fit_transform(X))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ecc, cmap="viridis", s=24,
                        alpha=0.8, edgecolors="none")
        ax.set_title(os.path.basename(run.rstrip("/")))
        ax.set_xticks([]); ax.set_yticks([])
        # mark balls (discs) with a faint ring
        m = egg == 0
        ax.scatter(Z[m, 0], Z[m, 1], facecolors="none", edgecolors="r",
                   s=70, linewidths=0.7, alpha=0.25, label="ball")
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, label="eccentricity")
    fig.suptitle("object embeddings (PCA), coloured by eccentricity; red ring = ball")
    fp = os.path.join(args.out_dir, "embed_pca_ecc.png")
    fig.savefig(fp, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fp}")

    # ---- UMAP of object embeddings, coloured by eccentricity ----
    if args.umap:
        import umap
        fig, axes = plt.subplots(1, nr, figsize=(5.2 * nr, 4.6), squeeze=False)
        for ax, run in zip(axes[0], args.runs):
            X = embeds[run]
            reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                                min_dist=args.umap_min_dist,
                                random_state=args.seed)
            Z = reducer.fit_transform(StandardScaler().fit_transform(X))
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=ecc, cmap="viridis", s=24,
                            alpha=0.85, edgecolors="none")
            m = egg == 0
            ax.scatter(Z[m, 0], Z[m, 1], facecolors="none", edgecolors="r",
                       s=70, linewidths=0.7, alpha=0.25)
            ax.set_title(os.path.basename(run.rstrip("/")))
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, label="eccentricity")
        fig.suptitle("object embeddings (UMAP), coloured by eccentricity; red ring = ball")
        fp = os.path.join(args.out_dir, "embed_umap_ecc.png")
        fig.savefig(fp, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
