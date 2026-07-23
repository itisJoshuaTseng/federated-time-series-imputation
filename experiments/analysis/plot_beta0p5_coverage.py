"""
Plot the completed beta=0.5 full-grid coverage.

Input:
  logs/saits_mnar/ablation_sf0p5_{S1,S4}_{q,l}_rho0p{3,5,7}_seeds_0-4.json

Output:
  figures/phase1_beta0p5_full_grid_mae.png
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
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
})

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"

SCENARIOS = ["S1", "S4"]
MNARS = [("q", "quantile"), ("l", "logit")]
RHOS = [0.3, 0.5, 0.7]


def result_path(scenario: str, tag: str, rho: float) -> Path:
    rho_tag = f"rho0p{int(rho * 10)}"
    return LOG_DIR / f"ablation_sf0p5_{scenario}_{tag}_{rho_tag}_seeds_0-4.json"


def summarize(path: Path) -> tuple[float, float, float, float]:
    with path.open() as f:
        data = json.load(f)

    rows = [r for r in data["results"] if r["method"] == "fed_ca"]
    if len(rows) != 5:
        raise ValueError(f"Expected 5 fed_ca rows in {path}, got {len(rows)}")

    maes = np.array([r["mean_mae"] for r in rows], dtype=float)
    rmses = np.array([r["mean_rmse"] for r in rows], dtype=float)
    return (
        float(maes.mean()),
        float(maes.std(ddof=1)),
        float(rmses.mean()),
        float(rmses.std(ddof=1)),
    )


def collect():
    table = {}
    for scenario in SCENARIOS:
        for tag, mnar in MNARS:
            vals = []
            for rho in RHOS:
                path = result_path(scenario, tag, rho)
                if not path.exists():
                    raise FileNotFoundError(path)
                vals.append(summarize(path))
            table[(scenario, mnar)] = vals
    return table


def main():
    table = collect()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True)

    color = {
        "quantile": "#2B6CB0",
        "logit": "#C05621",
    }
    face = {
        "S1": "#F7FBFF",
        "S4": "#FFF8F0",
    }

    for r, scenario in enumerate(SCENARIOS):
        for c, (_, mnar) in enumerate(MNARS):
            ax = axes[r, c]
            vals = table[(scenario, mnar)]
            means = [v[0] for v in vals]
            stds = [v[1] for v in vals]
            ax.set_facecolor(face[scenario])
            ax.errorbar(
                RHOS,
                means,
                yerr=stds,
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=4,
                color=color[mnar],
                ecolor=color[mnar],
            )
            for x, y in zip(RHOS, means):
                ax.annotate(
                    f"{y:.3f}",
                    xy=(x, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color="#333333",
                )
            ax.set_title(f"{scenario} {mnar}")
            ax.set_xticks(RHOS)
            ax.set_xticklabels([str(rho) for rho in RHOS])
            ax.set_xlabel("missing rate rho")
            if c == 0:
                ax.set_ylabel("MAE on MNAR holes")

    axes[0, 0].set_ylim(0.55, 0.78)
    axes[0, 1].set_ylim(0.22, 0.48)
    axes[1, 0].set_ylim(0.72, 1.28)
    axes[1, 1].set_ylim(0.12, 0.55)

    fig.suptitle(
        "CA beta=0.5 full-grid coverage (5 seeds each)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = FIG_DIR / "phase1_beta0p5_full_grid_mae.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    print()
    print(f"{'setting':<18} {'MAE mean±std':<20} {'RMSE mean±std':<20}")
    print("-" * 62)
    for scenario in SCENARIOS:
        for _, mnar in MNARS:
            for rho, vals in zip(RHOS, table[(scenario, mnar)]):
                mae_m, mae_s, rmse_m, rmse_s = vals
                print(
                    f"{scenario} {mnar[0]} rho={rho:<3} "
                    f"{mae_m:.4f}±{mae_s:.4f}      "
                    f"{rmse_m:.4f}±{rmse_s:.4f}"
                )


if __name__ == "__main__":
    main()
