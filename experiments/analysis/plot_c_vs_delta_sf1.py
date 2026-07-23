"""
Phase 1 Step 3/4 — C_score vs Δ_CA, redesigned for readability.

The previous single-panel overlay was hard to parse because it encoded:
  - client group (minority / majority),
  - MNAR mechanism (quantile / logit),
  - scenario / rho label,
  - and beta (= scale_factor)
all in one view.

This version splits the figure into 2x2 small multiples:
  rows    = client group (majority / minority)
  columns = MNAR mechanism (quantile / logit)

Within each panel:
  - x = C_score
  - y = Δ_CA vs FedAvg
  - color = beta
  - marker = scenario (S1 vs S4)
  - a short gray trajectory connects the same setting across betas

Only settings with >=2 beta values are shown, so the figure focuses on
the beta-ablation story rather than mixing in one-off reference points.

Output: figures/phase1_step3_cscore_vs_delta_sf1.png
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
    "font.size":        10.5,
    "axes.titlesize":   12,
    "axes.labelsize":   10.5,
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"

C_SCORE = {
    ("S1", "quantile", 0.3): 0.084,
    ("S1", "quantile", 0.5): 0.144,
    ("S1", "quantile", 0.7): 0.185,
    ("S1", "logit",    0.3): 0.053,
    ("S1", "logit",    0.5): 0.067,
    ("S1", "logit",    0.7): 0.085,
    ("S4", "quantile", 0.3): 0.020,
    ("S4", "quantile", 0.5): 0.020,
    ("S4", "quantile", 0.7): 0.019,
    ("S4", "logit",    0.3): 0.025,
}

AVAILABLE_BETAS = {
    1.0: {
        ("S1", "quantile", 0.3),
        ("S1", "quantile", 0.5),
        ("S1", "quantile", 0.7),
    },
    0.5: {
        ("S1", "quantile", 0.3),
        ("S1", "quantile", 0.5),
        ("S1", "quantile", 0.7),
        ("S1", "logit",    0.3),
        ("S1", "logit",    0.5),
        ("S1", "logit",    0.7),
        ("S4", "quantile", 0.3),
        ("S4", "quantile", 0.5),
        ("S4", "quantile", 0.7),
        ("S4", "logit",    0.3),
    },
}

BETA_ORDER = [4.0, 1.0, 0.5]

BETA_STYLE = {
    4.0: {
        "label": "beta=4 (CAFE default)",
        "offset": -0.0030,
        "facecolor": "none",
        "edgecolor": "#9AA0A6",
        "size": 140,
        "linewidth": 1.7,
    },
    1.0: {
        "label": "beta=1",
        "offset": 0.0,
        "facecolor": "#E69F00",
        "edgecolor": "white",
        "size": 130,
        "linewidth": 1.2,
    },
    0.5: {
        "label": "beta=0.5 (ours)",
        "offset": 0.0030,
        "facecolor": "#009E73",
        "edgecolor": "#111111",
        "size": 155,
        "linewidth": 1.5,
    },
}

SCENARIO_STYLE = {
    "S1": {"marker": "o", "linestyle": "-",  "label": "S1"},
    "S4": {"marker": "D", "linestyle": "--", "label": "S4"},
}

PANEL_ORDER = [
    ("majority", "quantile", "Quantile MNAR"),
    ("majority", "logit",    "Logit MNAR"),
    ("minority", "quantile", "Quantile MNAR"),
    ("minority", "logit",    "Logit MNAR"),
]

LABEL_OFFSETS = {
    ("majority", "quantile", "S4", 0.3): (-18, 8),
    ("majority", "quantile", "S4", 0.5): (-18, -14),
    ("majority", "quantile", "S4", 0.7): (-18, 8),
    ("majority", "quantile", "S1", 0.3): (-14, -14),
    ("majority", "quantile", "S1", 0.5): (-6, 10),
    ("majority", "quantile", "S1", 0.7): (-4, 10),
    ("majority", "logit",    "S4", 0.3): (-18, 8),
    ("majority", "logit",    "S1", 0.3): (-18, -14),
    ("majority", "logit",    "S1", 0.5): (-10, 10),
    ("majority", "logit",    "S1", 0.7): (8, -12),
    ("minority", "quantile", "S4", 0.3): (-18, 8),
    ("minority", "quantile", "S4", 0.5): (-18, -14),
    ("minority", "quantile", "S4", 0.7): (-18, 8),
    ("minority", "quantile", "S1", 0.3): (-14, 10),
    ("minority", "quantile", "S1", 0.5): (-8, 10),
    ("minority", "quantile", "S1", 0.7): (-6, 10),
    ("minority", "logit",    "S4", 0.3): (-16, 8),
    ("minority", "logit",    "S1", 0.3): (-14, 8),
    ("minority", "logit",    "S1", 0.5): (-10, -14),
    ("minority", "logit",    "S1", 0.7): (8, 8),
}


def load_per_client(path: Path):
    with path.open() as f:
        data = json.load(f)
    by_method = {}
    for r in data["results"]:
        by_method.setdefault(r["method"], []).append(
            [cm["mae"] for cm in r["client_metrics"]]
        )
    return {m: np.mean(v, axis=0) for m, v in by_method.items()}


def baseline_path(scen: str, mnar: str, rho: float) -> Path:
    tag = {"quantile": "q", "logit": "l"}[mnar]
    rho_tag = f"rho0p{int(rho * 10)}"
    return LOG_DIR / f"cafe_fix_v2_{scen}_{tag}_{rho_tag}_seeds_0-4.json"


def ablation_path(scen: str, mnar: str, rho: float, beta: float) -> Path:
    tag = {"quantile": "q", "logit": "l"}[mnar]
    rho_tag = f"rho0p{int(rho * 10)}"
    prefix = {1.0: "ablation_sf1", 0.5: "ablation_sf0p5"}[beta]
    return LOG_DIR / f"{prefix}_{scen}_{tag}_{rho_tag}_seeds_0-4.json"


def compute_delta(scen: str, mnar: str, rho: float, beta: float):
    """Return (Δ minority %, Δ majority %) for CA vs FedAvg."""
    base = baseline_path(scen, mnar, rho)
    if not base.exists():
        return None

    pc_base = load_per_client(base)
    if "fedavg" not in pc_base or "fed_ca" not in pc_base:
        return None

    fa = pc_base["fedavg"]
    if beta == 4.0:
        ca = pc_base["fed_ca"]
    else:
        ab = ablation_path(scen, mnar, rho, beta)
        if not ab.exists():
            return None
        pc_ab = load_per_client(ab)
        if "fed_ca" not in pc_ab:
            return None
        ca = pc_ab["fed_ca"]

    delta = (fa - ca) / fa * 100.0
    return {
        "minority": float(delta[0]),
        "majority": float(np.mean(delta[1:])),
    }


def collect_rows():
    rows = []
    for (scen, mnar, rho), c_score in C_SCORE.items():
        setting = (scen, mnar, rho)
        variants = {}
        for beta in BETA_ORDER:
            if beta != 4.0 and setting not in AVAILABLE_BETAS.get(beta, set()):
                continue
            delta = compute_delta(scen, mnar, rho, beta)
            if delta is None:
                continue
            variants[beta] = delta

        if len(variants) < 2:
            continue

        rows.append({
            "scen": scen,
            "mnar": mnar,
            "rho": rho,
            "c_score": c_score,
            "variants": variants,
        })
    return rows


def draw_panel(ax, rows, group: str, mnar: str):
    panel_rows = [r for r in rows if r["mnar"] == mnar]
    panel_rows.sort(key=lambda r: (r["scen"], r["rho"]))

    ax.axhline(0, color="#333333", linewidth=0.9, zorder=0)
    if mnar == "quantile":
        ax.set_facecolor("#FCFAF6")
        ax.set_xlim(0.012, 0.192)
    else:
        ax.set_facecolor("#F7FAFC")
        ax.set_xlim(0.018, 0.092)

    for row in panel_rows:
        ordered_betas = [b for b in BETA_ORDER if b in row["variants"]]
        xs = [row["c_score"] + BETA_STYLE[b]["offset"] for b in ordered_betas]
        ys = [row["variants"][b][group] for b in ordered_betas]

        scen_style = SCENARIO_STYLE[row["scen"]]
        ax.plot(
            xs, ys,
            color="#B9BDC5",
            linewidth=1.25,
            linestyle=scen_style["linestyle"],
            zorder=1,
        )

        for beta in ordered_betas:
            bstyle = BETA_STYLE[beta]
            x = row["c_score"] + bstyle["offset"]
            y = row["variants"][beta][group]
            ax.scatter(
                x, y,
                s=bstyle["size"],
                marker=scen_style["marker"],
                facecolor=bstyle["facecolor"],
                edgecolor=bstyle["edgecolor"],
                linewidth=bstyle["linewidth"],
                zorder=3,
            )

        anchor_beta = 0.5 if 0.5 in row["variants"] else ordered_betas[-1]
        x_anchor = row["c_score"] + BETA_STYLE[anchor_beta]["offset"]
        y_anchor = row["variants"][anchor_beta][group]
        dx, dy = LABEL_OFFSETS[(group, mnar, row["scen"], row["rho"])]
        ax.annotate(
            f"{row['scen']} rho={row['rho']}",
            xy=(x_anchor, y_anchor),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="#444444",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                boxstyle="round,pad=0.15",
            ),
            zorder=4,
        )


def main():
    rows = collect_rows()

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharey="row")
    axes = np.asarray(axes)

    draw_panel(axes[0, 0], rows, "majority", "quantile")
    draw_panel(axes[0, 1], rows, "majority", "logit")
    draw_panel(axes[1, 0], rows, "minority", "quantile")
    draw_panel(axes[1, 1], rows, "minority", "logit")

    axes[0, 0].set_title("Quantile MNAR")
    axes[0, 1].set_title("Logit MNAR")

    axes[0, 0].set_ylabel("Majority mean Δ_CA (%)")
    axes[1, 0].set_ylabel("Minority (C0) Δ_CA (%)")
    axes[1, 0].set_xlabel("C_score")
    axes[1, 1].set_xlabel("C_score")

    axes[0, 0].set_ylim(-22.5, 16.5)
    axes[1, 0].set_ylim(-6.5, 6.8)

    for ax in axes.flat:
        ax.grid(alpha=0.22, linestyle="--")

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=BETA_STYLE[4.0]["edgecolor"],
               markeredgewidth=BETA_STYLE[4.0]["linewidth"],
               linewidth=0, label="beta=4"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BETA_STYLE[1.0]["facecolor"],
               markeredgecolor=BETA_STYLE[1.0]["edgecolor"],
               markeredgewidth=BETA_STYLE[1.0]["linewidth"],
               linewidth=0, label="beta=1"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BETA_STYLE[0.5]["facecolor"],
               markeredgecolor=BETA_STYLE[0.5]["edgecolor"],
               markeredgewidth=BETA_STYLE[0.5]["linewidth"],
               linewidth=0, label="beta=0.5 (ours)"),
        Line2D([0], [0], color="#B9BDC5", linewidth=1.25,
               label="same setting across betas"),
        Line2D([0], [0], marker=SCENARIO_STYLE["S1"]["marker"], color="none",
               markerfacecolor="white", markeredgecolor="#444444",
               markeredgewidth=1.2, linewidth=0, label="S1"),
        Line2D([0], [0], marker=SCENARIO_STYLE["S4"]["marker"], color="none",
               markerfacecolor="white", markeredgecolor="#444444",
               markeredgewidth=1.2, linewidth=0, label="S4"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    fig.suptitle(
        "C_score vs CA improvement, split for readability\n"
        "Each short trajectory shows the same setting as beta moves 4 → 1 → 0.5",
        fontsize=13,
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = FIG_DIR / "phase1_step3_cscore_vs_delta_sf1.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    print()
    print(f"{'setting':<22} {'beta':<9} {'Delta minor':<12} {'Delta major':<12}")
    print("-" * 64)
    for row in sorted(rows, key=lambda x: (x["mnar"], x["c_score"])):
        tag = f"{row['scen']} {row['mnar'][:1]} rho={row['rho']}"
        for beta in BETA_ORDER:
            if beta not in row["variants"]:
                continue
            vals = row["variants"][beta]
            print(
                f"{tag:<22} {beta:<9.1f} "
                f"{vals['minority']:+7.2f}%     {vals['majority']:+7.2f}%"
            )


if __name__ == "__main__":
    main()
