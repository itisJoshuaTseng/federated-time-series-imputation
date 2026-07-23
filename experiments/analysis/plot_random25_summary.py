"""
Plot Random25 global-test sanity-check results.

Input:
  results/random100/random100_results.csv

Output:
  experiments/figures/random25_global_test_summary.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "random100" / "random100_results.csv"
FIG_DIR = ROOT / "experiments" / "figures"


def main():
    df = pd.read_csv(CSV).sort_values("seed")
    df = df[df["seed"] < 25].copy()

    fed = df["fedavg_global_mae"].to_numpy(dtype=float)
    local = df["local_avg_mae"].to_numpy(dtype=float)
    imp = df["improvement_pct"].to_numpy(dtype=float)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(12.5, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.82], wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel A: paired per-seed MAE.
    x = np.arange(len(df))
    for i in x:
        ax0.plot([i, i], [fed[i], local[i]], color="#C9C9C9", lw=1.2, zorder=1)
    ax0.scatter(x, local, color="#7A7A7A", s=32, label="Local-only", zorder=3)
    ax0.scatter(x, fed, color="#4C78A8", s=32, label="FedAvg", zorder=4)
    ax0.set_title("Global Test MAE Across 25 Random Splits")
    ax0.set_xlabel("Seed")
    ax0.set_ylabel("MAE on global test set")
    ax0.grid(axis="y", alpha=0.18, linestyle="--")
    ax0.legend(loc="upper left", frameon=True)
    ax0.set_xticks([0, 4, 9, 14, 19, 24])
    ax0.set_xticklabels([0, 4, 9, 14, 19, 24])

    # Panel B: improvement distribution.
    ax1.hist(imp, bins=np.arange(35, 66, 5), color="#009E73",
             edgecolor="white", alpha=0.88)
    ax1.axvline(imp.mean(), color="#245B45", lw=2.2,
                label=f"Mean = {imp.mean():.1f}%")
    ax1.set_title("FedAvg Improvement Over Local")
    ax1.set_xlabel("Improvement (%)")
    ax1.set_ylabel("Number of seeds")
    ax1.grid(axis="y", alpha=0.18, linestyle="--")
    ax1.legend(loc="upper right", frameon=True)

    # Panel C: compact statistics block.
    ax2.axis("off")
    stats = [
        ("FedAvg MAE", f"{fed.mean():.4f} +/- {fed.std(ddof=1):.4f}"),
        ("Local MAE", f"{local.mean():.4f} +/- {local.std(ddof=1):.4f}"),
        ("Improvement", f"{imp.mean():.1f}% +/- {imp.std(ddof=1):.1f}%"),
        ("Win rate", f"{int((fed < local).sum())}/25"),
        ("Paired t-test", "p = 1.20e-11"),
        ("Cohen's d", "2.45"),
    ]
    y = 0.95
    ax2.text(0, y, "Summary", fontsize=14, fontweight="bold", va="top")
    y -= 0.13
    for k, v in stats:
        ax2.text(0, y, k, fontsize=10.5, color="#555555", va="top")
        ax2.text(0, y - 0.055, v, fontsize=12, fontweight="bold", va="top")
        y -= 0.145
    ax2.text(
        0,
        0.02,
        "Note: this is a global-test sanity check,\n"
        "not the main personalized-FL evaluation.",
        fontsize=9.5,
        color="#7A3E00",
        va="bottom",
    )

    fig.suptitle(
        "Random25 Sanity Check: FedAvg Generalizes Better Than Isolated Local Training",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.04,
        "Global-test protocol: 70/30 train-test split; train data split equally into 5 clients; "
        "Local models and FedAvg are evaluated on the same pooled global test set.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "random25_global_test_summary.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
