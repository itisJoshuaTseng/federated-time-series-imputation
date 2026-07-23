"""
Plot client-local held-out MNAR validation results.

Input logs:
  logs/saits_mnar/heldout_sf0p5_*_seeds_0-4.json

Output:
  figures/phase1_heldout_validation.png
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT.parent / "logs" / "saits_mnar"
FIG_DIR = ROOT / "figures"

METHODS = [
    ("local", "Local", "#7A7A7A"),
    ("fedavg", "FedAvg", "#4C78A8"),
    ("fed_ca", "Fed-CA beta=0.5", "#009E73"),
]

PANEL_ORDER = [
    ("S1", "quantile", 0.3, "S1 quantile rho=0.3"),
    ("S1", "quantile", 0.7, "S1 quantile rho=0.7"),
    ("S1", "logit", 0.7, "S1 logit rho=0.7"),
    ("S4", "logit", 0.7, "S4 logit rho=0.7"),
]


def load_results():
    data = {}
    for path in sorted(glob.glob(str(LOG_DIR / "heldout_sf0p5_*_seeds_0-4.json"))):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        key = (obj["scenario"], obj["mnar_method"], float(obj["missing_rate"]))
        grouped = {}
        for row in obj["results"]:
            grouped.setdefault(row["method"], []).append(row)
        data[key] = grouped
    return data


def mean_std(rows, field):
    values = np.array([r[field] for r in rows], dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def main():
    data = load_results()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.1), sharey=False)

    for ax, (scenario, mnar, rho, title) in zip(axes, PANEL_ORDER):
        key = (scenario, mnar, rho)
        grouped = data.get(key, {})
        xs = np.arange(len(METHODS))

        means, stds = [], []
        for method, _, _ in METHODS:
            rows = grouped.get(method, [])
            if rows:
                mu, sd = mean_std(rows, "mean_mae")
            else:
                mu, sd = np.nan, np.nan
            means.append(mu)
            stds.append(sd)

        colors = [c for _, _, c in METHODS]
        ax.bar(xs, means, yerr=stds, capsize=4, color=colors,
               edgecolor="white", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label, _ in METHODS], rotation=25,
                           ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, linestyle="--")
        ax.set_ylabel("Held-out MAE" if ax is axes[0] else "")

        if grouped.get("fedavg") and grouped.get("fed_ca"):
            fa = np.array([
                r["mean_mae"]
                for r in sorted(grouped["fedavg"], key=lambda x: x["seed"])
            ])
            ca = np.array([
                r["mean_mae"]
                for r in sorted(grouped["fed_ca"], key=lambda x: x["seed"])
            ])
            improvement = float(np.nanmean((fa - ca) / fa * 100.0))
            sign = "+" if improvement >= 0 else ""
            ax.text(
                0.5, 0.96,
                f"CA vs FedAvg: {sign}{improvement:.1f}%",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                color="#245B45" if improvement >= 0 else "#8A3B12",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85"),
            )

    fig.suptitle(
        "Client-local held-out validation: Fed-CA helps only when complementarity is detectable",
        y=1.04,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5, -0.04,
        "Each client is split into train/test cases; training and CA fingerprints use train cases only; "
        "MAE is computed on held-out test MNAR holes. Bars show mean +/- std over five seeds.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()

    out = FIG_DIR / "phase1_heldout_validation.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
