"""
Phase 1 Step 0: Per-client visualisation of CA-fix experiments.

Generates three figures into experiments/figures/:
  phase1_step0_s1_per_client.png  — S1 settings, per-client MAE bars
  phase1_step0_s4_per_client.png  — S4 settings, per-client MAE bars
  phase1_step0_ca_delta.png       — CA vs FedAvg delta per client

Run from federated learning/:
  python experiments/plot_phase1_step0.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":      ["Heiti TC", "PingFang TC", "Hiragino Sans", "Arial Unicode MS"],
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

C = {
    "local":  "#888888",
    "fedavg": "#4878CF",
    "fed_ca": "#E07B39",
}
LABEL = {
    "local":  "Local",
    "fedavg": "FedAvg",
    "fed_ca": "Fed-CA",
}

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "saits_mnar"
FIG_DIR = REPO / "experiments" / "figures"
FIG_DIR.mkdir(exist_ok=True)

SETTINGS = [
    ("S1", "q", 0.3), ("S1", "q", 0.5), ("S1", "q", 0.7), ("S1", "l", 0.3),
    ("S4", "q", 0.3), ("S4", "q", 0.5), ("S4", "q", 0.7), ("S4", "l", 0.3),
]


def load_setting(scen, mnar, rho):
    rho_tag = f"rho0p{int(rho * 10)}"
    path = LOG_DIR / f"cafe_fix_v2_{scen}_{mnar}_{rho_tag}_seeds_0-4.json"
    with path.open() as f:
        return json.load(f)


def per_client_mean(data):
    """Return {method: [mae_c0, ..., mae_c4]} averaged over seeds."""
    bucket = {}
    for r in data["results"]:
        m = r["method"]
        bucket.setdefault(m, []).append([cm["mae"] for cm in r["client_metrics"]])
    return {m: np.mean(v, axis=0) for m, v in bucket.items()}


def title_for(scen, mnar, rho):
    mnar_full = "quantile" if mnar == "q" else "logit"
    extra = ""
    if scen == "S1":
        extra = "\n(C0=MNAR-Left, C1–4=MNAR-Right)"
    else:
        extra = "\n(all clients MNAR-Left)"
    return f"{scen}  {mnar_full}  ρ={rho}{extra}"


def plot_per_client_group(settings, out_name, suptitle):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=False)
    methods = ["local", "fedavg", "fed_ca"]
    width = 0.25
    x = np.arange(5)

    for ax, (scen, mnar, rho) in zip(axes, settings):
        data = load_setting(scen, mnar, rho)
        per_client = per_client_mean(data)
        for i, m in enumerate(methods):
            if m not in per_client:
                continue
            ax.bar(
                x + (i - 1) * width,
                per_client[m],
                width,
                color=C[m],
                label=LABEL[m],
                edgecolor="white",
                linewidth=0.6,
            )

        if scen == "S1":
            for pos in [-0.5]:
                ax.axvspan(pos, pos + 1, alpha=0.08, color="red", zorder=0)
            ax.text(
                0, ax.get_ylim()[1] * 0.02, "minority",
                ha="center", fontsize=8, color="#b00",
                style="italic",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in range(5)])
        ax.set_ylabel("MAE")
        ax.set_title(title_for(scen, mnar, rho), fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_ca_delta_all():
    """One horizontal bar chart per setting: (FedAvg − Fed-CA) per client,
    positive = CA better, negative = CA worse."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)

    for ax, (scen, mnar, rho) in zip(axes.flat, SETTINGS):
        data = load_setting(scen, mnar, rho)
        per_client = per_client_mean(data)
        if "fedavg" not in per_client or "fed_ca" not in per_client:
            continue
        fa = per_client["fedavg"]
        ca = per_client["fed_ca"]
        delta = (fa - ca) / fa * 100.0

        colors = ["#2ca02c" if d > 0 else "#d62728" for d in delta]
        y = np.arange(5)
        ax.barh(y, delta, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"C{c}" for c in range(5)])
        ax.set_title(title_for(scen, mnar, rho), fontsize=10)
        ax.set_xlabel("CA vs FedAvg  (%, + = CA better)")
        for i, d in enumerate(delta):
            ax.text(
                d + (0.3 if d >= 0 else -0.3),
                i,
                f"{d:+.1f}%",
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=8,
            )
        if scen == "S1":
            ax.axhspan(-0.5, 0.5, alpha=0.08, color="red", zorder=0)

    fig.suptitle(
        "Fed-CA vs FedAvg 的改善分佈（per client）\n"
        "+ 表示 CA 在該 client 上改善；− 表示 CA 退化",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "phase1_step0_ca_delta.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_gap_summary():
    """Compare mean/worst/gap across all settings as grouped bars."""
    rows = []
    for scen, mnar, rho in SETTINGS:
        data = load_setting(scen, mnar, rho)
        per_client = per_client_mean(data)
        for m, arr in per_client.items():
            rows.append({
                "setting": f"{scen} {mnar}\nρ={rho}",
                "method": m,
                "mean": float(np.mean(arr)),
                "worst": float(np.max(arr)),
                "gap": float(np.max(arr) - np.min(arr)),
            })

    methods = ["local", "fedavg", "fed_ca"]
    settings_labels = [f"{s} {m}\nρ={r}" for s, m, r in SETTINGS]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    metrics = [("mean", "Mean MAE (5 clients)"),
               ("worst", "Worst-client MAE"),
               ("gap", "MAE gap (worst − best)")]

    x = np.arange(len(settings_labels))
    width = 0.26

    for ax, (key, title) in zip(axes, metrics):
        for i, m in enumerate(methods):
            vals = [
                next((r[key] for r in rows
                      if r["setting"] == lbl and r["method"] == m), np.nan)
                for lbl in settings_labels
            ]
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                color=C[m],
                label=LABEL[m],
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(settings_labels, rotation=0, fontsize=8)
        ax.set_ylabel("MAE")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "三種 metric 對比：mean 看不出來的，worst 和 gap 會看出來",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "phase1_step0_metric_summary.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    s1 = [s for s in SETTINGS if s[0] == "S1"]
    s4 = [s for s in SETTINGS if s[0] == "S4"]
    plot_per_client_group(
        s1,
        "phase1_step0_s1_per_client.png",
        "S1（互補性場景）— per-client MAE",
    )
    plot_per_client_group(
        s4,
        "phase1_step0_s4_per_client.png",
        "S4（無互補場景）— per-client MAE",
    )
    plot_ca_delta_all()
    plot_gap_summary()


if __name__ == "__main__":
    main()
