"""
Phase 1 Step 4 — Fingerprint-level summary.

Bar chart of avg maj→C0 pull, grouped by setting × fingerprint method.
Captures the Step 4 finding in a single glance: quantile MNAR is visible
to all three fingerprint methods; logit MNAR is invisible to all, at any ρ.

Numbers hard-coded from compare_fingerprints.py run (reproducible by
re-running that script). Output:
  figures/phase1_step4_fingerprint_summary.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":      ["Heiti TC", "Hiragino Sans", "Arial Unicode MS"],
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "experiments" / "figures"

# (setting_label, {method: maj_to_c0})
DATA = [
    ("S1 q\nρ=0.3",  {"LR": 0.982, "MI": 0.443, "RF": 0.810}),
    ("S1 l\nρ=0.3",  {"LR": 0.145, "MI": 0.253, "RF": 0.240}),
    ("S1 l\nρ=0.5",  {"LR": 0.127, "MI": 0.259, "RF": 0.246}),
    ("S1 l\nρ=0.7",  {"LR": 0.128, "MI": 0.271, "RF": 0.293}),
]

METHODS = ["LR", "MI", "RF"]
METHOD_COLORS = {"LR": "#4878CF", "MI": "#E07B39", "RF": "#6A994E"}


def main():
    n_settings = len(DATA)
    n_methods = len(METHODS)

    x = np.arange(n_settings)
    bar_w = 0.26

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))

    for i, method in enumerate(METHODS):
        vals = [d[1][method] for d in DATA]
        ax.bar(
            x + (i - 1) * bar_w, vals, bar_w,
            color=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.8,
            label=method, zorder=3,
        )
        for j, v in enumerate(vals):
            ax.text(
                x[j] + (i - 1) * bar_w, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#333",
            )

    # Uniform reference line
    ax.axhline(
        0.25, color="#d62728", linestyle="--", linewidth=1.2,
        alpha=0.7, zorder=2, label="uniform baseline (0.25)",
    )
    ax.text(
        n_settings - 0.5, 0.26, "uniform = 1/4 peers",
        ha="right", va="bottom", fontsize=9, color="#d62728",
    )

    # Shade "quantile" vs "logit" region
    ax.axvspan(-0.5, 0.5, color="#f7f2e6", alpha=0.5, zorder=1)
    ax.axvspan(0.5, n_settings - 0.5, color="#f0f4fa", alpha=0.5,
               zorder=1)
    ax.text(0, 1.06, "hard MNAR\n(quantile)", ha="center", va="bottom",
            fontsize=10, color="#8a6d3b", fontweight="bold")
    ax.text((n_settings + 0) / 2, 1.06,
            "soft MNAR (logit) — flat across ρ",
            ha="center", va="bottom",
            fontsize=10, color="#31587a", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in DATA])
    ax.set_ylabel("avg maj→C0  (CA peer-weight to minority)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Fingerprint 看得見什麼？  三種 fingerprint × 四個設定\n"
        "Logit MNAR 在任何 ρ 下 fingerprint 都近 uniform → CA 無法 sharpen"
    )
    ax.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    out = FIG_DIR / "phase1_step4_fingerprint_summary.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
