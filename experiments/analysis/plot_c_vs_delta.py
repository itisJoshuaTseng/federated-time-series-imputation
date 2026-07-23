"""
Phase 1 Step 1 — Phase A plot: C_score vs CA improvement.

Uses existing S1/S4 × (quant/logit) × rho logs to plot Δ_CA against
the CAFE-style fingerprint complementarity score.

Two curves per scatter:
  - Δ_CA on C0 (minority client in S1)
  - Δ_CA on mean(C1..C4) (majority clients in S1)

Separation between the two curves visualises CA's "spreading" effect.

Prereq: run `experiments/compute_c_score.py` to obtain C_score values
(currently hard-coded into this script from that output).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":      ["Heiti TC", "PingFang TC", "Hiragino Sans",
                         "Arial Unicode MS"],
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
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"

# From compute_c_score.py output
C_SCORE = {
    ("S1", "quantile", 0.3): 0.084,
    ("S1", "quantile", 0.5): 0.144,
    ("S1", "quantile", 0.7): 0.185,
    ("S1", "logit",    0.3): 0.053,
    ("S4", "quantile", 0.3): 0.020,
    ("S4", "quantile", 0.5): 0.020,
    ("S4", "quantile", 0.7): 0.019,
    ("S4", "logit",    0.3): 0.025,
}


def load_per_client(scen, mnar, rho):
    tag = {"quantile": "q", "logit": "l"}[mnar]
    rho_tag = f"rho0p{int(rho * 10)}"
    path = LOG_DIR / f"cafe_fix_v2_{scen}_{tag}_{rho_tag}_seeds_0-4.json"
    with path.open() as f:
        data = json.load(f)
    by_method = {}
    for r in data["results"]:
        by_method.setdefault(r["method"], []).append(
            [cm["mae"] for cm in r["client_metrics"]]
        )
    return {m: np.mean(v, axis=0) for m, v in by_method.items()}


def compute_delta(scen, mnar, rho):
    """Return (delta_minority_pct, delta_majority_pct) for CA vs FedAvg."""
    pc = load_per_client(scen, mnar, rho)
    fa, ca = pc["fedavg"], pc["fed_ca"]
    delta = (fa - ca) / fa * 100.0      # + means CA better
    if scen == "S1":
        minority = delta[0]
        majority = np.mean(delta[1:])
    else:
        minority = delta[0]           # just pick C0 as reference
        majority = np.mean(delta[1:])
    return float(minority), float(majority), float(np.mean(delta))


def main():
    rows = []
    for (scen, mnar, rho), c in C_SCORE.items():
        minority, majority, overall = compute_delta(scen, mnar, rho)
        rows.append({
            "scen": scen, "mnar": mnar, "rho": rho,
            "c_score": c,
            "minority": minority,
            "majority": majority,
            "overall": overall,
        })

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    markers = {"quantile": "o", "logit": "s"}
    for r in rows:
        m = markers[r["mnar"]]
        # Minority (C0) point
        ax.scatter(
            r["c_score"], r["minority"],
            s=120, marker=m, color="#E07B39",
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        # Majority (mean C1-C4) point
        ax.scatter(
            r["c_score"], r["majority"],
            s=120, marker=m, color="#4878CF",
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        # Connect minority ↔ majority for same setting
        ax.plot(
            [r["c_score"], r["c_score"]],
            [r["minority"], r["majority"]],
            color="#aaa", linewidth=1.0, alpha=0.6, zorder=1,
        )
        # Annotate setting
        y_label = max(r["minority"], r["majority"]) + 1.2
        label = f"{r['scen']} {r['mnar'][:1]} ρ={r['rho']}"
        ax.text(
            r["c_score"], y_label, label,
            ha="center", fontsize=8, color="#444",
        )

    # Fit a trend line through majority points (the main story)
    cs = np.array([r["c_score"] for r in rows])
    maj = np.array([r["majority"] for r in rows])
    order = np.argsort(cs)
    ax.plot(
        cs[order], maj[order],
        color="#4878CF", linewidth=1.2, alpha=0.45,
        linestyle=":", zorder=1,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("C_score  (CAFE-style fingerprint complementarity)")
    ax.set_ylabel("Δ_CA  (%, + = CA better than FedAvg)")
    ax.set_title(
        "C_score vs CA 改善量 (per-client)\n"
        "橙色 = minority (C0)，藍色 = majority mean (C1–C4)"
    )
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E07B39",
               markersize=10, label="Minority (C0) — quantile"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#E07B39",
               markersize=10, label="Minority (C0) — logit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4878CF",
               markersize=10, label="Majority mean — quantile"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4878CF",
               markersize=10, label="Majority mean — logit"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "phase1_step1A_cscore_vs_delta.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # Also print a summary table
    print()
    print(f"{'setting':<22} {'C_score':<10} {'Δ minor':<10} {'Δ major':<10}")
    print("-" * 54)
    for r in sorted(rows, key=lambda x: x["c_score"]):
        tag = f"{r['scen']} {r['mnar'][:1]} ρ={r['rho']}"
        print(f"{tag:<22} {r['c_score']:<10.3f} "
              f"{r['minority']:+.2f}%    {r['majority']:+.2f}%")


if __name__ == "__main__":
    main()
