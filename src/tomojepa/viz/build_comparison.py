"""Side-by-side A/B comparison: baseline LeJEPA vs residual-MIM.

Reads the per-epoch metrics.json written by validate.py for both runs and
renders effective-rank and augmentation-consistency curves, plus a summary
panel with the final values and the eccentricity-probe outcome.
"""
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    with open(f"runs/{run}/metrics.json") as f:
        m = json.load(f)
    ep = np.array([d["epoch"] for d in m])
    o = np.argsort(ep)
    g = lambda k: np.array([d[k] for d in m])[o]
    return {"epoch": ep[o], "emb": g("emb_effrank"), "tok": g("token_effrank"),
            "cos": g("aug_cos"), "cos_sd": g("aug_cos_std")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="val_baseline")
    p.add_argument("--residual", default="val_residual")
    p.add_argument("--out", default="runs/ab_comparison.png")
    args = p.parse_args()

    b = load(args.baseline)
    r = load(args.residual)
    CB, CR = "#1f77b4", "#d62728"

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].plot(b["epoch"], b["tok"], "-o", color=CB, label="baseline", ms=4)
    ax[0, 0].plot(r["epoch"], r["tok"], "-o", color=CR, label="residual", ms=4)
    ax[0, 0].set_title("patch-token effective rank")
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("eff. rank"); ax[0, 0].legend()

    ax[0, 1].plot(b["epoch"], b["emb"], "-o", color=CB, label="baseline", ms=4)
    ax[0, 1].plot(r["epoch"], r["emb"], "-o", color=CR, label="residual", ms=4)
    ax[0, 1].set_title("pooled-embedding effective rank")
    ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("eff. rank"); ax[0, 1].legend()

    for d, c, lab in [(b, CB, "baseline"), (r, CR, "residual")]:
        ax[1, 0].plot(d["epoch"], d["cos"], "-o", color=c, label=lab, ms=4)
        ax[1, 0].fill_between(d["epoch"], d["cos"] - d["cos_sd"],
                              np.minimum(d["cos"] + d["cos_sd"], 1.0),
                              color=c, alpha=0.15)
    ax[1, 0].set_title("augmentation consistency (cosine, on backbone tokens T)")
    ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("cos sim"); ax[1, 0].legend()

    ax[1, 1].axis("off")
    txt = (
        "FINAL (epoch 14)            baseline   residual\n"
        "-------------------------------------------------\n"
        f"token eff-rank            {b['tok'][-1]:8.1f}  {r['tok'][-1]:8.1f}\n"
        f"emb   eff-rank            {b['emb'][-1]:8.1f}  {r['emb'][-1]:8.1f}\n"
        f"aug-consistency (T)       {b['cos'][-1]:8.3f}  {r['cos'][-1]:8.3f}\n"
        "\n"
        "ECCENTRICITY PROBE  (5-seed mean +/- sd)\n"
        "-------------------------------------------------\n"
        "ball/egg AUC               ~0.99      ~0.99\n"
        "ecc R2, eggs-only      0.30+/-0.05  0.23+/-0.05\n"
        "   (overlapping -> no robust difference)\n"
        "\n"
        "TAKEAWAYS\n"
        "-------------------------------------------------\n"
        "* residual concentrates variance hard: ~3x lower\n"
        "  effective rank (matches 85/13 PCA spectrum).\n"
        "* aug-consistency on T looks lower/noisier for\n"
        "  residual, but T is NOT the invariance-trained\n"
        "  quantity (R = T - sg(C) is) -> eval caveat.\n"
        "* both decode material near-perfectly and carry\n"
        "  real eccentricity; no reliable shape gap.\n"
    )
    ax[1, 1].text(0.0, 1.0, txt, family="monospace", fontsize=10,
                  va="top", ha="left")

    fig.suptitle("A/B: baseline LeJEPA vs residual-MIM", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
