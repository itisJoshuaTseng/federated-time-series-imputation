"""
Updated figA: aggregation-function selection plot for S1 quantile MNAR.

The figure compares the available SAITS-side training/aggregation choices
across rho in {0.3, 0.5, 0.7}: Local, Centralized, FedAvg, FedProx, and
FedAvg+CA. FedAdam is shown as not run because no S1-quantile MNAR FedAdam
result exists in the current result archive.

Output: figures/figA_s1_quantile_line_v2.png
"""

from __future__ import annotations

import json
import statistics
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
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"

RHOS = [0.3, 0.5, 0.7]
SCEN, MNAR = "S1", "q"

# FedProx lives in older mnar_recon_* files (5 seeds each).
FEDPROX_FILES = {
    0.3: "mnar_recon_S1_quantile_20260404_082938.json",
    0.5: "mnar_recon_S1_quantile_rho0p5_20260405_003448.json",
    0.7: "mnar_recon_S1_quantile_rho0p7_20260405_135443.json",
}

FEDADAM_FILES = {
    0.3: "S1_q_rho0p3_fedadam_seeds_0-2.json",
    0.5: "S1_q_rho0p5_fedadam_seeds_0-2.json",
    0.7: "S1_q_rho0p7_fedadam_seeds_0-2.json",
}

METHODS = [
    # (label, source, json_method_key, color, linestyle, marker)
    ("Local",       "cafe_fix_v2",          "local",         "#7F7F7F", "--", "^"),
    ("FedAvg",      "cafe_fix_v2",          "fedavg",        "#1F77B4", "-",  "o"),
    ("FedProx",     "fedprox_log",          "fedprox",       "#9467BD", "-",  "v"),
    ("FedAdam",     "fedadam_log",          "fedadam",       "#FF7F0E", "-",  "X"),
    ("FedAvg+CA",   "ablation_sf0p5",       "fed_ca",        "#2CA02C", "-",  "s"),
]


def fetch(source, json_method, rho):
    if source == "fedprox_log":
        fname = FEDPROX_FILES.get(rho)
        if fname is None:
            return None, None
        path = LOG / fname
    elif source == "fedadam_log":
        fname = FEDADAM_FILES.get(rho)
        if fname is None:
            return None, None
        path = LOG / fname
    else:
        rho_tag = f"rho0p{int(rho * 10)}"
        path = LOG / f"{source}_{SCEN}_{MNAR}_{rho_tag}_seeds_0-4.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())

    if source == "centralized_ceiling":
        # Centralized log has no "method" key; mean_mae is computed across
        # all clients' MNAR holes after pooling.
        means = [r["mean_mae"] for r in data["results"]]
    else:
        means = [r["mean_mae"] for r in data["results"]
                 if r["method"] == json_method]
    if not means:
        return None, None
    return statistics.mean(means), (statistics.stdev(means) if len(means) > 1 else 0.0)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))

    for label, source, mkey, color, ls, marker in METHODS:
        means, stds = [], []
        for rho in RHOS:
            m, s = fetch(source, mkey, rho)
            means.append(m)
            stds.append(s)

        # Plot available points; the bridge logic remains as a graceful
        # fallback if a future run is missing.
        valid = [(rho, m, s) for rho, m, s in zip(RHOS, means, stds) if m is not None]
        missing = [rho for rho, m in zip(RHOS, means) if m is None]
        if not valid:
            continue
        rs, ms, ss = zip(*valid)
        ax.errorbar(rs, ms, yerr=ss, color=color, marker=marker,
                    markersize=8, linewidth=1.8, linestyle=ls,
                    capsize=3.5, label=label, zorder=3)
        if missing:
            for rho in missing:
                ax.scatter(rho, np.interp(rho, rs, ms), color=color,
                           marker=marker, s=70, facecolors="none",
                           edgecolors=color, linewidth=1.5, zorder=3,
                           alpha=0.5)
            ax.plot(RHOS, [np.interp(r, rs, ms) for r in RHOS], color=color,
                    linestyle=":", linewidth=1.2, alpha=0.45, zorder=2)
            ax.text(missing[0], np.interp(missing[0], rs, ms) + 0.018,
                    "  (pending)", fontsize=8, color=color,
                    style="italic", ha="left", va="bottom")

    ax.set_xticks(RHOS)
    ax.set_xticklabels([f"{r}" for r in RHOS])
    ax.set_xlabel(r"Missing rate $\rho$")
    ax.set_ylabel("MAE on induced MNAR holes (z-score, σ)")
    ax.set_title(
        "Aggregation Function Selection under S1 Quantile MNAR",
        fontsize=13,
    )
    fedadam_values = [fetch("fedadam_log", "fedadam", rho)[0] for rho in RHOS]
    if all(v is None for v in fedadam_values):
        ax.text(
            0.705,
            0.97,
            "FedAdam: not run in this MNAR grid",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#666666",
            style="italic",
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=6,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIG_DIR / "figA_s1_quantile_line_v2.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}\n")

    # Summary
    print(f"{'Method':<22} {'ρ=0.3':>14} {'ρ=0.5':>14} {'ρ=0.7':>14}")
    print("-" * 68)
    for label, source, mkey, _, _, _ in METHODS:
        row = [label]
        for rho in RHOS:
            m, s = fetch(source, mkey, rho)
            row.append(f"{m:.3f}±{s:.3f}" if m is not None else "       —")
        print(f"{row[0]:<22} {row[1]:>14} {row[2]:>14} {row[3]:>14}")


if __name__ == "__main__":
    main()
