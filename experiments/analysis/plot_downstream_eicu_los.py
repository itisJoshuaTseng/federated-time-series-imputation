#!/usr/bin/env python3
"""Plot eICU downstream LOS utility as grouped bar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SERVER_RESULTS = ROOT.parent / "server_results"
RESULT_DIR = SERVER_RESULTS / "downstream_eicu_los_flatten"
SUMMARY_CSV = RESULT_DIR / "eicu_downstream_los_summary.csv"
OUT_DIR = ROOT / "experiments" / "figures" / "downstream_eicu_los"

SCENARIOS = ["S1", "S4"]
METHODS = ["FedSAITS+CA", "FedICE+CA"]
METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("f1", "F1"),
    ("balanced_accuracy", "Balanced Acc."),
]

COLORS = {
    "FedSAITS+CA": "#2ca02c",   # green — matches combined_2x3 figure
    "FedICE+CA":   "#d62728",   # red   — matches combined_2x3 figure
}


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing downstream summary: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    x = np.arange(len(METRICS))
    width = 0.34

    for ax, scenario in zip(axes, SCENARIOS):
        sub = df[df["scenario"] == scenario].copy()
        for idx, method in enumerate(METHODS):
            row = sub[sub["method_label"] == method]
            if row.empty:
                raise ValueError(f"Missing row for {scenario}/{method}")
            row = row.iloc[0]
            means = [row[f"{metric}_mean"] for metric, _ in METRICS]
            stds = [row[f"{metric}_std"] for metric, _ in METRICS]
            offset = (idx - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                label=method,
            )

        ax.set_title(scenario, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in METRICS], rotation=15, ha="right")
        ax.set_ylim(0.45, 0.75)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel("Downstream classification score", fontsize=11)
    fig.suptitle(
        "eICU ICU LOS >= 48h Downstream Utility",
        fontsize=14,
        y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    out = OUT_DIR / "figure4_eicu_downstream_los_grouped_bar"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario]
        for method in METHODS:
            row = sub[sub["method_label"] == method].iloc[0]
            item = {"scenario": scenario, "method": method}
            for metric, label in METRICS:
                item[label] = row[f"{metric}_mean"]
                item[f"{label}_std"] = row[f"{metric}_std"]
            table_rows.append(item)
    pd.DataFrame(table_rows).to_csv(
        OUT_DIR / "figure4_eicu_downstream_los_grouped_bar_table.csv",
        index=False,
    )
    print(f"wrote {out.with_suffix('.png')}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
