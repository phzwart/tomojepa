"""Thresholded instance segmentation of input slices.

Pulls clean slices through the same pipeline the model sees, segments objects
(Otsu threshold -> connected components), and characterises each object by
shape (eccentricity, minor/major axis ratio) and density (mean intensity).
The point is to confirm the data actually contains differently shaped objects
(discs vs ellipses) at different densities -- a prerequisite for asking whether
the encoder distinguishes them.

Outputs (under --out_dir):
  - shape_overlays.png : slices with instance masks colour-coded by shape class
  - shape_scatter.png  : eccentricity vs density scatter (+ marginal histograms)
  - shapes.csv         : per-object measurements
"""
import os
import argparse

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu, median
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, opening, disk
from skimage.segmentation import clear_border

# eccentricity bands: 0 == perfect circle, ->1 == elongated line
DISC_MAX_ECC = 0.55
ELLIPSE_MIN_ECC = 0.75


def open_raw(path, key):
    root = zarr.open(path, mode="r")
    if hasattr(root, "shape"):
        return root
    return root[key]


def window(raw):
    """Raw [H,W] -> [0,1] by 1/99 percentiles (for thresholding & display only)."""
    lo, hi = np.percentile(raw, [1, 99])
    if hi <= lo:
        return np.zeros_like(raw, dtype=np.float32)
    return np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)


def segment(win, min_area, open_radius, drop_border, denoise_radius):
    """Return a label image of foreground objects (thresholded on windowed img).

    The volume is a connected 3D structure; the high-frequency "speckle" is
    sensor noise, not objects. A median filter removes it before thresholding so
    noise-only regions yield no spurious instances.
    """
    sm = median(win, disk(denoise_radius)) if denoise_radius > 0 else win
    thr = threshold_otsu(sm)
    fg = sm > thr
    fg_frac = float(fg.mean())          # sparse grains -> small; noise-only -> ~0.5
    if open_radius > 0:
        fg = opening(fg, disk(open_radius))
    fg = remove_small_objects(fg, min_size=min_area)
    if drop_border:
        fg = clear_border(fg)
    return label(fg), sm, fg_frac


def classify(ecc):
    if ecc < DISC_MAX_ECC:
        return "disc"
    if ecc > ELLIPSE_MIN_ECC:
        return "ellipse"
    return "intermediate"


CLASS_COLOR = {"disc": (0.20, 0.70, 1.0),        # blue
               "intermediate": (0.65, 0.65, 0.65),
               "ellipse": (1.0, 0.45, 0.10)}      # orange


def measure_slice(raw, lab, min_solidity, max_area):
    """Density measured on raw intensities; ragged/huge blobs filtered out."""
    rows = []
    for r in regionprops(lab, intensity_image=raw):
        if r.solidity < min_solidity or r.area > max_area:
            continue
        ratio = (r.axis_minor_length / r.axis_major_length
                 if r.axis_major_length > 0 else 0.0)
        rows.append({
            "label": r.label,
            "area": float(r.area),
            "eccentricity": float(r.eccentricity),
            "axis_ratio": float(ratio),
            "mean_intensity": float(r.intensity_mean),
            "solidity": float(r.solidity),
            "equiv_diam": float(r.equivalent_diameter_area),
            "cy": float(r.centroid[0]),
            "cx": float(r.centroid[1]),
            "klass": classify(float(r.eccentricity)),
        })
    return rows


def overlay(img, lab, rows):
    """RGB overlay: grayscale slice + object fills tinted by shape class."""
    base = np.stack([img] * 3, axis=-1)
    klass_by_label = {row["label"]: row["klass"] for row in rows}
    out = base.copy()
    for lb, kl in klass_by_label.items():
        m = lab == lb
        col = np.array(CLASS_COLOR[kl])
        out[m] = 0.45 * base[m] + 0.55 * col
    return np.clip(out, 0, 1)


def main():
    p = argparse.ArgumentParser(description="Thresholded instance segmentation of slices")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--pattern", default="upsampled_1024.zarr")
    p.add_argument("--backend", default="zarr")
    p.add_argument("--dataset_key", default="reconstruction")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--n_slices", type=int, default=9)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min_area", type=int, default=40, help="drop blobs smaller than this (px)")
    p.add_argument("--max_area", type=int, default=20000, help="drop blobs larger than this (px)")
    p.add_argument("--min_solidity", type=float, default=0.85,
                   help="drop ragged (non-convex) blobs, e.g. speckle fragments")
    p.add_argument("--open_radius", type=int, default=1, help="morphological opening radius (0=off)")
    p.add_argument("--denoise_radius", type=int, default=2,
                   help="median-filter radius to kill speckle noise before threshold (0=off)")
    p.add_argument("--max_fg_frac", type=float, default=0.30,
                   help="skip slice as noise-only if thresholded foreground exceeds this")
    p.add_argument("--keep_border", action="store_true", help="keep objects touching the edge")
    p.add_argument("--out_dir", default="runs/shape_analysis")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fpath = os.path.join(args.data_dir, args.pattern)
    arr = open_raw(fpath, args.dataset_key)          # (D, H, W) raw intensities
    depth = arr.shape[0]
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(depth, size=min(args.n_slices, depth),
                             replace=False).tolist())

    all_rows = []
    overlays = []
    for si in idxs:
        raw = np.asarray(arr[si], dtype=np.float32)
        if raw.ndim > 2:                              # (C,H,W) -> first channel
            raw = raw[0]
        win = window(raw)
        lab, sm, fg_frac = segment(win, args.min_area, args.open_radius,
                                   not args.keep_border, args.denoise_radius)
        noise = fg_frac > args.max_fg_frac
        if noise:
            lab = np.zeros_like(lab)
            rows = []
        else:
            rows = measure_slice(raw, lab, args.min_solidity, args.max_area)
        for r in rows:
            r["slice"] = si
        all_rows.extend(rows)
        overlays.append((si, sm, lab, rows, noise))

    if not all_rows:
        print("No objects segmented -- try lowering --min_area or --open_radius.")
        return

    # ---- per-slice overlay montage ----
    n = len(overlays)
    cols = int(np.ceil(np.sqrt(n)))
    rows_g = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_g, cols, figsize=(cols * 3.0, rows_g * 3.0))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (si, img, lab, rows, noise) in zip(axes, overlays):
        ax.imshow(overlay(img, lab, rows))
        if noise:
            ax.set_title(f"slice {si}  NOISE (skipped)", fontsize=8, color="red")
        else:
            nd = sum(r["klass"] == "disc" for r in rows)
            ne = sum(r["klass"] == "ellipse" for r in rows)
            ax.set_title(f"slice {si}  n={len(rows)}  disc={nd} ellipse={ne}", fontsize=8)
    fig.suptitle("Instance segmentation  (blue=disc, gray=intermediate, orange=ellipse)",
                 fontsize=11)
    fig.tight_layout()
    fp = os.path.join(args.out_dir, "shape_overlays.png")
    fig.savefig(fp, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fp}")

    # ---- scatter: eccentricity vs density, sized by area ----
    ecc = np.array([r["eccentricity"] for r in all_rows])
    dens = np.array([r["mean_intensity"] for r in all_rows])
    area = np.array([r["area"] for r in all_rows])
    cols_pt = np.array([CLASS_COLOR[r["klass"]] for r in all_rows])
    sizes = 8 + 120 * (area - area.min()) / (np.ptp(area) + 1e-9)

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                          left=0.1, right=0.97, bottom=0.1, top=0.93,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(ecc, dens, s=sizes, c=cols_pt, alpha=0.6, edgecolors="none")
    ax.axvline(DISC_MAX_ECC, color="0.5", ls="--", lw=0.8)
    ax.axvline(ELLIPSE_MIN_ECC, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("eccentricity  (0=circle -> 1=elongated)")
    ax.set_ylabel("mean intensity  (density proxy)")
    axx.hist(ecc, bins=30, color="0.6")
    axx.axis("off")
    axy.hist(dens, bins=30, orientation="horizontal", color="0.6")
    axy.axis("off")
    fig.suptitle(f"shape vs density  ({len(all_rows)} objects, {len(idxs)} slices)",
                 fontsize=11)
    fp = os.path.join(args.out_dir, "shape_scatter.png")
    fig.savefig(fp, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fp}")

    # ---- CSV + summary ----
    keys = ["slice", "label", "area", "eccentricity", "axis_ratio",
            "mean_intensity", "solidity", "equiv_diam", "cy", "cx", "klass"]
    csv = os.path.join(args.out_dir, "shapes.csv")
    with open(csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"wrote {csv}")

    def stat(rs):
        d = np.array([r["mean_intensity"] for r in rs])
        a = np.array([r["equiv_diam"] for r in rs])
        return len(rs), d.mean(), d.std(), a.mean()

    n_noise = sum(1 for o in overlays if o[4])
    print("\n=== summary ===")
    print(f"total objects: {len(all_rows)}  over {len(idxs) - n_noise} grain slices "
          f"({n_noise} noise-only slices skipped)")
    for kl in ("disc", "intermediate", "ellipse"):
        rs = [r for r in all_rows if r["klass"] == kl]
        if rs:
            n_, dm, dsd, am = stat(rs)
            print(f"  {kl:>12}: n={n_:4d}  density={dm:.3f}±{dsd:.3f}  diam~{am:.1f}px")
    # density contrast between classes
    dd = [r["mean_intensity"] for r in all_rows if r["klass"] == "disc"]
    de = [r["mean_intensity"] for r in all_rows if r["klass"] == "ellipse"]
    if dd and de:
        print(f"\n  disc vs ellipse density gap: "
              f"{np.mean(dd) - np.mean(de):+.3f} "
              f"(disc {np.mean(dd):.3f} vs ellipse {np.mean(de):.3f})")


if __name__ == "__main__":
    main()
