"""
Method comparison on the (Δ C0, Δ Maj) Pareto plane.

Each panel = one (scenario, MNAR, ρ) setting. Each point = one method
(Local, CA-β=4, CA-β=1, CA-β=0.5) positioned by its improvement over
FedAvg on the two client groups. FedAvg sits at the origin by
construction. A dashed trajectory connects the three β values to show
how β tuning moves within the Pareto space.

Four quadrants tell the story at a glance:
  (+, +)  top-right  = Pareto-wins: help both groups         ← goal
  (+, −)  bottom-right = help C0 but damage majority         ← β=4 ρ=0.3
  (−, +)  top-left   = help majority at C0's cost            ← β=4 ρ=0.7
  (−, −)  bottom-left  = strictly worse than FedAvg           ← Local typically

Output: figures/phase1_step5_method_pareto.png
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams.update({
    "font.family":      ["Heiti TC", "Hiragino Sans", "Arial Unicode MS"],
    "font.size":        10.5,
    "axes.titlesize":   11,
    "axes.labelsize":   10.5,
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"

# Panels: (title, scenario_tag, mnar_tag, rho_tag, display_title)
SETTINGS = [
    ("S1 quantile ρ=0.3",  "S1", "q", "rho0p3", "low-signal regime"),
    ("S1 quantile ρ=0.5",  "S1", "q", "rho0p5", "mid-signal regime"),
    ("S1 quantile ρ=0.7",  "S1", "q", "rho0p7", "high-signal regime"),
    ("S1 logit ρ=0.3",     "S1", "l", "rho0p3", "soft-MNAR, weak"),
    ("S1 logit ρ=0.5",     "S1", "l", "rho0p5", "soft-MNAR, mid"),
    ("S1 logit ρ=0.7",     "S1", "l", "rho0p7", "soft-MNAR, high"),
]

METHOD_COLORS = {
    "Local":     "#8C8C8C",
    "CA β=4":    "#B32424",
    "CA β=1":    "#D68A3B",
    "CA β=0.5":  "#2D8F47",
}
METHOD_ORDER = ["Local", "CA β=4", "CA β=1", "CA β=0.5"]


def load_per_client(path):
    data = json.loads(path.read_text())
    by_method = {}
    for r in data["results"]:
        by_method.setdefault(r["method"], []).append(
            [cm["mae"] for cm in r["client_metrics"]]
        )
    return {m: np.mean(v, axis=0) for m, v in by_method.items()}


def delta_vs_fedavg(per_client_mae, fedavg_mae):
    """Return (Δ C0 %, Δ Maj %) where Δ = (FedAvg - method) / FedAvg × 100."""
    d = (fedavg_mae - per_client_mae) / fedavg_mae * 100
    return float(d[0]), float(d[1:].mean())


def collect_points(scen, mnar, rho_tag):
    """Return dict {method_name: (Δ C0 %, Δ Maj %)} for all available methods."""
    base = LOG / f"cafe_fix_v2_{scen}_{mnar}_{rho_tag}_seeds_0-4.json"
    ab1  = LOG / f"ablation_sf1_{scen}_{mnar}_{rho_tag}_seeds_0-4.json"
    abp5 = LOG / f"ablation_sf0p5_{scen}_{mnar}_{rho_tag}_seeds_0-4.json"

    pc = load_per_client(base)
    fa = pc["fedavg"]
    pts = {}
    if "local" in pc:
        pts["Local"] = delta_vs_fedavg(pc["local"], fa)
    if "fed_ca" in pc:
        pts["CA β=4"] = delta_vs_fedavg(pc["fed_ca"], fa)
    if ab1.exists():
        pc1 = load_per_client(ab1)
        if "fed_ca" in pc1:
            pts["CA β=1"] = delta_vs_fedavg(pc1["fed_ca"], fa)
    if abp5.exists():
        pcp = load_per_client(abp5)
        if "fed_ca" in pcp:
            pts["CA β=0.5"] = delta_vs_fedavg(pcp["fed_ca"], fa)
    return pts


def shade_quadrants(ax, xmin, xmax, ymin, ymax):
    """Light background shading to highlight the four quadrants."""
    # Top-right: Pareto win — very light green
    ax.axhspan(0, ymax, xmin=0.5, xmax=1,
               facecolor="#2D8F47", alpha=0.06, zorder=0)
    # Top-left: majority-only win
    ax.axhspan(0, ymax, xmin=0, xmax=0.5,
               facecolor="#4878CF", alpha=0.04, zorder=0)
    # Bottom-right: C0-only (at majority's expense) — faint red warning
    ax.axhspan(ymin, 0, xmin=0.5, xmax=1,
               facecolor="#B32424", alpha=0.05, zorder=0)
    # Bottom-left: strictly worse — faint gray
    ax.axhspan(ymin, 0, xmin=0, xmax=0.5,
               facecolor="#555", alpha=0.04, zorder=0)


def main():
    all_pts = {}
    for label, scen, mnar, rho_tag, note in SETTINGS:
        all_pts[label] = collect_points(scen, mnar, rho_tag)

    # Focus axis on the CA-vs-FedAvg comparison region. Local typically lies
    # far outside (e.g., −74% for ρ=0.7); we clip and annotate instead of
    # letting Local stretch the axes and squash the interesting comparison.
    xlim = 24
    ylim = 24

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), sharex=True,
                             sharey=True)
    axes = axes.flatten()

    for ax_idx, ((label, scen, mnar, rho_tag, note), ax) in enumerate(
            zip(SETTINGS, axes)):
        pts = all_pts[label]

        shade_quadrants(ax, -xlim, xlim, -ylim, ylim)

        # Axes of the Pareto space
        ax.axhline(0, color="black", linewidth=0.9, zorder=2)
        ax.axvline(0, color="black", linewidth=0.9, zorder=2)

        # Diagonal: "equal improvement for both groups"
        ax.plot([-xlim, xlim], [-xlim, xlim],
                color="#bbb", linestyle=":", linewidth=0.9,
                zorder=1)

        # Trajectory β=4 → β=1 → β=0.5 (if available, dashed line)
        beta_path = [m for m in ["CA β=4", "CA β=1", "CA β=0.5"] if m in pts]
        if len(beta_path) >= 2:
            xs_p = [pts[m][0] for m in beta_path]
            ys_p = [pts[m][1] for m in beta_path]
            ax.plot(xs_p, ys_p, color="#444", linestyle="--",
                    linewidth=1.3, alpha=0.55, zorder=2,
                    label="β-sweep trajectory (β: 4 → 1 → 0.5)")

        # Draw points; clip Local to edge if it's off-chart
        for method in METHOD_ORDER:
            if method not in pts:
                continue
            x, y = pts[method]
            color = METHOD_COLORS[method]

            in_range = (-xlim < x < xlim) and (-ylim < y < ylim)

            if not in_range and method == "Local":
                # Clip Local point to nearest axis edge and mark with arrow
                cx = max(-xlim + 0.5, min(xlim - 0.5, x))
                cy = max(-ylim + 0.5, min(ylim - 0.5, y))
                ax.scatter(cx, cy, s=120, marker=">" if x > xlim else
                           ("<" if x < -xlim else ("^" if y > ylim else "v")),
                           color=color, edgecolor="white", linewidth=1.0,
                           zorder=4)
                ax.annotate(
                    f"Local\n({x:+.0f}, {y:+.0f})\n← off-chart",
                    xy=(cx, cy), xytext=(cx - 0.8, cy + 1.2),
                    ha="right" if x < 0 else "left",
                    va="bottom", fontsize=7.5,
                    color=color, zorder=5,
                )
                continue

            marker_size = 210 if method == "CA β=0.5" else 150
            edge = "black" if method == "CA β=0.5" else "white"
            edge_w = 1.6 if method == "CA β=0.5" else 1.0
            ax.scatter(x, y, s=marker_size, color=color,
                       edgecolor=edge, linewidth=edge_w, zorder=4)

            dx = 0.7 if x >= 0 else -0.7
            dy = 1.1 if y >= 0 else -1.1
            ha = "left" if x >= 0 else "right"
            va = "bottom" if y >= 0 else "top"
            ax.annotate(f"{method}\n({x:+.1f}, {y:+.1f})",
                        xy=(x, y), xytext=(x + dx, y + dy),
                        ha=ha, va=va, fontsize=8, color="#222",
                        zorder=5)

        # FedAvg reference label at origin
        ax.plot(0, 0, marker="+", color="black", markersize=14,
                markeredgewidth=2, zorder=3)
        ax.annotate("FedAvg\n(reference)",
                    xy=(0, 0), xytext=(0.6, -0.9),
                    fontsize=7.5, color="#222", zorder=5)

        # Quadrant labels in corners (only once on leftmost panel)
        if ax_idx == 0:
            corner_fs = 8.2
            corner_alpha = 0.55
            ax.text(xlim * 0.95, ylim * 0.95,
                    "雙贏\n(Pareto-win)", ha="right", va="top",
                    fontsize=corner_fs, color="#2D8F47",
                    alpha=corner_alpha, fontweight="bold")
            ax.text(-xlim * 0.95, ylim * 0.95,
                    "Maj 贏\nC0 輸", ha="left", va="top",
                    fontsize=corner_fs, color="#4878CF",
                    alpha=corner_alpha, fontweight="bold")
            ax.text(xlim * 0.95, -ylim * 0.95,
                    "C0 贏\nMaj 輸", ha="right", va="bottom",
                    fontsize=corner_fs, color="#B32424",
                    alpha=corner_alpha, fontweight="bold")
            ax.text(-xlim * 0.95, -ylim * 0.95,
                    "雙輸", ha="left", va="bottom",
                    fontsize=corner_fs, color="#444",
                    alpha=corner_alpha, fontweight="bold")

        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("Δ C0 (minority) vs FedAvg  (%)")
        if ax_idx % 3 == 0:
            ax.set_ylabel("Δ Majority mean vs FedAvg  (%)")
        ax.set_title(f"{label}\n({note})")
        ax.grid(alpha=0.15, linestyle="--", zorder=0)

    # Shared legend at the bottom
    legend_elems = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=METHOD_COLORS["Local"],
               markersize=10, label="Local (each client trains alone)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=METHOD_COLORS["CA β=4"],
               markersize=10, label="CA β=4 (Cafe default)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=METHOD_COLORS["CA β=1"],
               markersize=10, label="CA β=1 (sharpening off)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=METHOD_COLORS["CA β=0.5"],
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=12, label="CA β=0.5 (ours; Pareto-optimal)"),
        Line2D([0], [0], marker="+", color="black",
               markersize=12, markeredgewidth=2,
               linestyle="", label="FedAvg (origin / reference)"),
        Line2D([0], [0], color="#444", linestyle="--",
               linewidth=1.3, label="β-sweep trajectory"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9)

    fig.suptitle(
        "Method comparison on the (Δ C0, Δ Majority) Pareto plane — "
        "full S1 sweep across MNAR mechanism and missing rate.\n"
        "Quantile panels show the strong beta effect; logit panels cluster "
        "near the origin, indicating a near-FedAvg regime.",
        fontsize=11, y=0.99,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = FIG_DIR / "phase1_step5_method_pareto.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}\n")

    # Summary table
    print(f"{'Setting':<22} {'Method':<12} {'Δ C0 %':>8} {'Δ Maj %':>9}")
    print("-" * 54)
    for label, _, _, _, _ in SETTINGS:
        for method in METHOD_ORDER:
            if method in all_pts[label]:
                x, y = all_pts[label][method]
                print(f"{label:<22} {method:<12} {x:>+7.2f}  {y:>+7.2f}")
        print()


if __name__ == "__main__":
    main()
