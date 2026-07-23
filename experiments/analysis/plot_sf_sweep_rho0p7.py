"""
Phase 1 Step 4 — scale_factor sweep at S1 q ρ=0.7.

Tests the "saturation" hypothesis from Step 3: does the majority benefit
require strong sharpening (large sf), or does fingerprint ranking alone
(small sf) already capture it?

Data:
  - Δ C0, Δ Maj from ablation_sf*_S1_q_rho0p7_seeds_0-4.json (5 seeds each)
    plus cafe_fix_v2_S1_q_rho0p7_seeds_0-4.json (sf=4, 5 seeds)
  - maj→C0 pull computed from fingerprint (one-shot, deterministic given seed)

Output: figures/phase1_step4_sf_sweep_rho0p7.png
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
    "legend.fontsize":  9.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"

# (sf, filename)
POINTS = [
    (0.5, "ablation_sf0p5_S1_q_rho0p7_seeds_0-4.json"),
    (1.0, "ablation_sf1_S1_q_rho0p7_seeds_0-4.json"),
    (2.0, "ablation_sf2_S1_q_rho0p7_seeds_0-4.json"),
    (3.0, "ablation_sf3_S1_q_rho0p7_seeds_0-4.json"),
    (4.0, "cafe_fix_v2_S1_q_rho0p7_seeds_0-4.json"),
    (8.0, "ablation_sf8_S1_q_rho0p7_seeds_0-4.json"),
]

# From compute-once dump in the console run (reproducible from fingerprints)
MAJ_TO_C0 = {
    0.5: 0.4929,
    1.0: 0.7391,
    2.0: 0.9600,
    3.0: 0.9951,
    4.0: 0.9994,
    8.0: 1.0000,
}


def summarize(path, method="fed_ca"):
    data = json.loads(path.read_text())
    c0s, majs = [], []
    for r in data["results"]:
        if r["method"] != method:
            continue
        ms = r["client_metrics"]
        c0s.append(ms[0]["mae"])
        majs.append(sum(m["mae"] for m in ms[1:]) / 4)
    return c0s, majs


def main():
    # FedAvg baseline
    fa_c0, fa_maj = summarize(
        LOG_DIR / "cafe_fix_v2_S1_q_rho0p7_seeds_0-4.json", "fedavg"
    )
    fa_c0_m, fa_maj_m = statistics.mean(fa_c0), statistics.mean(fa_maj)

    sfs, d_c0, d_maj, err_c0, err_maj, pulls = [], [], [], [], [], []
    for sf, fname in POINTS:
        c0s, majs = summarize(LOG_DIR / fname, "fed_ca")
        c0_m, maj_m = statistics.mean(c0s), statistics.mean(majs)
        c0_s = statistics.stdev(c0s) if len(c0s) > 1 else 0.0
        maj_s = statistics.stdev(majs) if len(majs) > 1 else 0.0
        sfs.append(sf)
        d_c0.append(100 * (fa_c0_m - c0_m) / fa_c0_m)
        d_maj.append(100 * (fa_maj_m - maj_m) / fa_maj_m)
        # Relative std in %
        err_c0.append(100 * c0_s / fa_c0_m)
        err_maj.append(100 * maj_s / fa_maj_m)
        pulls.append(MAJ_TO_C0[sf])

    sfs = np.array(sfs)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.8))

    # --- Left axis: Δ vs FedAvg (two lines with error bars) ---
    ax1.errorbar(
        sfs, d_maj, yerr=err_maj,
        marker="o", markersize=8, linewidth=2.0,
        color="#4878CF", label="Δ Majority vs FedAvg (%)",
        capsize=4, zorder=4,
    )
    ax1.errorbar(
        sfs, d_c0, yerr=err_c0,
        marker="s", markersize=8, linewidth=2.0,
        color="#E07B39", label="Δ C0 vs FedAvg (%)",
        capsize=4, zorder=4,
    )
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks(sfs)
    ax1.set_xticklabels([str(s) for s in sfs])
    ax1.set_xlabel("scale_factor  (power-law exponent)")
    ax1.set_ylabel("Δ vs FedAvg  (%, + = CA better)")
    ax1.set_ylim(-6, 20)

    # Annotate CAFE default
    ax1.axvspan(3.5, 4.5, color="#d62728", alpha=0.08, zorder=1)
    ax1.text(4, 18, "CAFE default",
             ha="center", fontsize=9, color="#d62728")
    # Annotate "best C0" point
    best_c0_idx = int(np.argmax(d_c0))
    ax1.annotate(
        f"best for C0\n({sfs[best_c0_idx]}, +{d_c0[best_c0_idx]:.1f}%)",
        xy=(sfs[best_c0_idx], d_c0[best_c0_idx]),
        xytext=(0.7, 9),
        arrowprops=dict(arrowstyle="->", color="#E07B39",
                        alpha=0.7, lw=1.2),
        fontsize=9, color="#a35429",
        ha="center",
    )

    ax1.legend(loc="center right", framealpha=0.95)

    # --- Right axis: maj→C0 pull ---
    ax2 = ax1.twinx()
    ax2.plot(sfs, pulls, marker="D", markersize=6, linewidth=1.3,
             color="#6A994E", alpha=0.75, label="avg maj→C0 pull",
             linestyle=":", zorder=3)
    ax2.set_ylabel("avg maj→C0 pull", color="#4a7237")
    ax2.tick_params(axis="y", labelcolor="#4a7237")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.25, color="#6A994E", linestyle="--",
                linewidth=0.8, alpha=0.4)
    ax2.text(0.45, 0.27, "uniform (0.25)", fontsize=8, color="#6A994E")
    ax2.grid(False)
    ax2.legend(loc="lower right", framealpha=0.95)

    ax1.set_title(
        "scale_factor sweep @ S1 quantile ρ=0.7\n"
        "Majority benefit saturates at sf ≤ 1; "
        "sharpening further only costs C0"
    )

    fig.tight_layout()
    out = FIG_DIR / "phase1_step4_sf_sweep_rho0p7.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}\n")

    # Summary table
    print(f"{'sf':<5} {'maj→C0':<9} {'Δ C0 %':<10} {'Δ Maj %':<10}")
    print("-" * 40)
    for i, sf in enumerate(sfs):
        print(f"{sf:<5} {pulls[i]:<9.3f} {d_c0[i]:+7.2f}    {d_maj[i]:+7.2f}")


if __name__ == "__main__":
    main()
