#!/usr/bin/env python3
"""
Plot the three core figures for the diagnostic benchmark story.

Inputs:
  experiments/figures/all_mnar_results/all_mnar_results_summary.csv

Outputs:
  experiments/figures/thesis_core/figure1_quantile_core_methods.png
  experiments/figures/thesis_core/figure2_local_ablation_rho0p5.png
  experiments/figures/thesis_core/figure3_logit_robustness.png
  experiments/figures/thesis_core/figure3_logit_robustness_table.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "experiments" / "figures" / "all_mnar_results" / "all_mnar_results_summary.csv"
OUT_DIR = ROOT / "experiments" / "figures" / "thesis_core"

SCENARIOS = ["S1", "S2", "S3", "S4"]
RHOS = [0.3, 0.5, 0.7]

CORE_METHODS = [
    ("FedAvg-SAITS", "FedSAITS"),
    ("Fed-SAITS-CA b=0.5", "FedSAITS+CA"),
    ("FedICE", "FedICE"),
    ("FedICE-CA b=4", "FedICE+CA"),
]

SAITS_ABLATION = [
    ("Local-SAITS", "Local-SAITS"),
    ("FedAvg-SAITS", "FedSAITS"),
    ("Fed-SAITS-CA b=0.5", "FedSAITS+CA"),
]

ICE_ABLATION = [
    ("Local-ICE", "Local-ICE"),
    ("FedICE", "FedICE"),
    ("FedICE-CA b=4", "FedICE+CA"),
]

LOGIT_METHODS = [
    ("Local-SAITS", "Local-SAITS"),
    ("FedAvg-SAITS", "FedSAITS"),
    ("Fed-SAITS-CA b=0.5", "FedSAITS+CA"),
    ("Local-ICE", "Local-ICE"),
    ("FedICE", "FedICE"),
    ("FedICE-CA b=4", "FedICE+CA"),
]

COLORS = {
    "Local-SAITS": "#7f7f7f",
    "FedSAITS": "#1f77b4",
    "FedSAITS+CA": "#2ca02c",
    "Local-ICE": "#8c564b",
    "FedICE": "#9467bd",
    "FedICE+CA": "#d62728",
}

MARKERS = {
    "Local-SAITS": "o",
    "FedSAITS": "o",
    "FedSAITS+CA": "s",
    "Local-ICE": "D",
    "FedICE": "D",
    "FedICE+CA": "^",
}


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df["missing_rate"] = df["missing_rate"].astype(float)
    return df


def row_for(
    df: pd.DataFrame,
    *,
    scenario: str,
    rho: float,
    method: str,
    mnar_method: str,
) -> pd.Series | None:
    rows = df[
        (df["scenario"] == scenario)
        & (df["mnar_method"] == mnar_method)
        & np.isclose(df["missing_rate"], rho)
        & (df["method_label"] == method)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_figure1(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    x = np.arange(len(SCENARIOS))

    for ax, rho in zip(axes, RHOS):
        for method_key, label in CORE_METHODS:
            y, err = [], []
            for scenario in SCENARIOS:
                row = row_for(
                    df,
                    scenario=scenario,
                    rho=rho,
                    method=method_key,
                    mnar_method="quantile",
                )
                y.append(np.nan if row is None else row["mean_mae"])
                err.append(0.0 if row is None else row["std_mae"])

            ax.errorbar(
                x,
                y,
                yerr=err,
                marker=MARKERS[label],
                linewidth=2.2,
                markersize=6,
                capsize=3,
                color=COLORS[label],
                label=label,
            )

        ax.set_title(rf"$\rho={rho:.1f}$", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("MNAR scenario")

    axes[0].set_ylabel("MAE on induced MNAR holes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Figure 1. Quantile MNAR: Backbone Crossover", y=0.98, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.83])
    save_current_figure(OUT_DIR / "figure1_quantile_core_methods.png")


def plot_family_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    methods: list[tuple[str, str]],
    title: str,
) -> None:
    x = np.arange(len(SCENARIOS))
    for method_key, label in methods:
        y, err = [], []
        for scenario in SCENARIOS:
            row = row_for(
                df,
                scenario=scenario,
                rho=0.5,
                method=method_key,
                mnar_method="quantile",
            )
            y.append(np.nan if row is None else row["mean_mae"])
            err.append(0.0 if row is None else row["std_mae"])

        ax.errorbar(
            x,
            y,
            yerr=err,
            marker=MARKERS[label],
            linewidth=2.2,
            markersize=6,
            capsize=3,
            color=COLORS[label],
            label=label,
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("MNAR scenario")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)


def plot_figure2(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    plot_family_panel(axes[0], df, SAITS_ABLATION, "SAITS family")
    plot_family_panel(axes[1], df, ICE_ABLATION, "ICE family")
    axes[0].set_ylabel("MAE on induced MNAR holes")
    fig.suptitle("Figure 2. Local Ablation under Quantile MNAR ($\\rho=0.5$)", y=1.03)
    fig.tight_layout()
    save_current_figure(OUT_DIR / "figure2_local_ablation_rho0p5.png")


def plot_figure3(df: pd.DataFrame) -> None:
    data = df[
        (df["mnar_method"] == "logit")
        & (df["scenario"].isin(["S1", "S4"]))
        & (df["method_label"].isin([m[0] for m in LOGIT_METHODS]))
    ].copy()
    display_map = dict(LOGIT_METHODS)
    data["display_method"] = data["method_label"].map(display_map)
    data["logit_setting"] = data["scenario"].map({"S1": "Logit setting A", "S4": "Logit setting B"})

    table = data[
        [
            "logit_setting",
            "scenario",
            "missing_rate",
            "display_method",
            "mean_mae",
            "std_mae",
            "n_seeds",
            "source_files",
        ]
    ].sort_values(["logit_setting", "missing_rate", "display_method"])
    table.to_csv(OUT_DIR / "figure3_logit_robustness_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    methods = [label for _, label in LOGIT_METHODS]
    width = 0.12
    x = np.arange(len(RHOS))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

    for ax, setting in zip(axes, ["Logit setting A", "Logit setting B"]):
        subset = data[data["logit_setting"] == setting]
        for idx, method in enumerate(methods):
            vals, errs = [], []
            for rho in RHOS:
                rows = subset[
                    np.isclose(subset["missing_rate"], rho)
                    & (subset["display_method"] == method)
                ]
                vals.append(np.nan if rows.empty else rows.iloc[0]["mean_mae"])
                errs.append(0.0 if rows.empty else rows.iloc[0]["std_mae"])
            ax.bar(
                x + offsets[idx],
                vals,
                width=width,
                yerr=errs,
                capsize=2,
                color=COLORS[method],
                label=method,
                alpha=0.92,
            )

        ax.set_title(setting)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{rho:.1f}" for rho in RHOS])
        ax.set_xlabel(r"Missing rate $\rho$")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("MAE on induced MNAR holes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Figure 3. Logit-based MNAR Robustness", y=0.98, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.83])
    save_current_figure(OUT_DIR / "figure3_logit_robustness.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    plot_figure1(df)
    plot_figure2(df)
    plot_figure3(df)
    print(f"Saved thesis core figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
