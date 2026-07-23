"""
Plot S4 safety check for beta=0.5 against FedAvg and beta=4.

Output:
  figures/phase1_s4_safety_beta0p5.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
})

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "saits_mnar"
FIG = REPO / "experiments" / "figures"

RHOS = [0.3, 0.5, 0.7]
METHODS = [
    ("fedavg", "FedAvg", "#7A8793"),
    ("fed_ca_b4", "CA beta=4", "#D55E00"),
    ("fed_ca_b05", "CA beta=0.5", "#009E73"),
]


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)["results"]


def summarize(rows: list[dict], method: str) -> tuple[float, float]:
    vals = np.array([r["mean_mae"] for r in rows if r["method"] == method], dtype=float)
    if len(vals) != 5:
        raise ValueError(f"Expected 5 rows for {method}, got {len(vals)}")
    return float(vals.mean()), float(vals.std(ddof=1))


def get_values(tag: str, rho: float):
    rho_tag = f"rho0p{int(rho * 10)}"
    base = load_rows(LOG / f"cafe_fix_v2_S4_{tag}_{rho_tag}_seeds_0-4.json")
    b05 = load_rows(LOG / f"ablation_sf0p5_S4_{tag}_{rho_tag}_seeds_0-4.json")
    return {
        "fedavg": summarize(base, "fedavg"),
        "fed_ca_b4": summarize(base, "fed_ca"),
        "fed_ca_b05": summarize(b05, "fed_ca"),
    }


def draw_panel(ax, tag: str, title: str):
    vals = {rho: get_values(tag, rho) for rho in RHOS}
    x = np.arange(len(RHOS))
    width = 0.24
    offsets = [-width, 0, width]

    for offset, (key, label, color) in zip(offsets, METHODS):
        means = [vals[rho][key][0] for rho in RHOS]
        stds = [vals[rho][key][1] for rho in RHOS]
        ax.bar(
            x + offset,
            means,
            yerr=stds,
            width=width,
            label=label,
            color=color,
            alpha=0.90,
            capsize=3,
            edgecolor="white",
            linewidth=0.7,
        )
        for xi, yi in zip(x + offset, means):
            ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"rho={rho}" for rho in RHOS])
    ax.set_ylabel("MAE on MNAR holes")
    ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    draw_panel(axes[0], "q", "S4 quantile: no complementarity")
    draw_panel(axes[1], "l", "S4 logit: weak/implicit diversity")
    axes[0].set_ylim(0.70, 1.32)
    axes[1].set_ylim(0.10, 0.56)
    axes[1].set_ylabel("")
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle("S4 safety check: beta=0.5 remains near FedAvg / beta=4", y=1.02)
    fig.tight_layout()
    out = FIG / "phase1_s4_safety_beta0p5.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
