#!/usr/bin/env python3
"""Plot full rho=0.5 external/local-family ablation figures.

This script combines:
  1. VitalDB rho=0.5 quantile summary from all_mnar_results_summary.csv
  2. open-data/eICU FedSAITS/FedICE results from previous external validation
  3. open-data/eICU Local-SAITS, Local-ICE, FedSAITS+CA, FedICE+CA results
     from server_results/external_full_ablation_rho0p5

Outputs eight figures:
  - open-data SAITS family
  - open-data ICE family
  - eICU-demo SAITS family
  - eICU-demo ICE family
  - VitalDB all families
  - open-data all families
  - eICU-demo all families
  - combined 3x1 CA-method-by-dataset overview
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SERVER_RESULTS = ROOT.parent / "server_results"
FULL_EXTERNAL_DIR = SERVER_RESULTS / "external_full_ablation_rho0p5"
VITALDB_SUMMARY_CSV = (
    ROOT
    / "experiments"
    / "figures"
    / "all_mnar_results"
    / "all_mnar_results_summary.csv"
)
OUT_DIR = ROOT / "experiments" / "figures" / "external_full_ablation_rho0p5"

SCENARIOS = ["S1", "S2", "S3", "S4"]

DATASET_LABELS = {
    "vitaldb": "vitalDB",
    "open_data": "MIMIC-III",
    "eicu_demo": "eICU",
}

METHOD_LABELS = {
    "local": "Local-SAITS",
    "fedavg": "FedSAITS",
    "fed_ca": "FedSAITS+CA",
    "local_ice": "Local-ICE",
    "fedice": "FedICE",
    "fedice_ca": "FedICE+CA",
}

VITALDB_METHOD_LABELS = {
    "Local-SAITS": "Local-SAITS",
    "FedAvg-SAITS": "FedSAITS",
    "Fed-SAITS-CA b=0.5": "FedSAITS+CA",
    "Local-ICE": "Local-ICE",
    "FedICE": "FedICE",
    "FedICE-CA b=4": "FedICE+CA",
}

SAITS_FAMILY = ["Local-SAITS", "FedSAITS", "FedSAITS+CA"]
ICE_FAMILY = ["Local-ICE", "FedICE", "FedICE+CA"]
ALL_METHODS = SAITS_FAMILY + ICE_FAMILY

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

DATASET_ORDER = ["open_data", "vitaldb", "eicu_demo"]


EXTERNAL_FEDAVG_FILES = {
    "open_data": {
        "S1": SERVER_RESULTS
        / "open_data_3clients_allfeat"
        / "open_data_S1_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
        "S2": SERVER_RESULTS
        / "external_s2_s3_rho0p5"
        / "open_data_S2_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
        "S3": SERVER_RESULTS
        / "external_s2_s3_rho0p5"
        / "open_data_S3_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
        "S4": SERVER_RESULTS
        / "open_data_3clients_allfeat"
        / "open_data_S4_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
    },
    "eicu_demo": {
        "S1": SERVER_RESULTS
        / "eicu_demo"
        / "eicu_demo_S1_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
        "S2": SERVER_RESULTS
        / "external_s2_s3_rho0p5"
        / "eicu_demo_S2_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
        "S3": SERVER_RESULTS
        / "external_s2_s3_rho0p5"
        / "eicu_demo_S3_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
        "S4": SERVER_RESULTS
        / "eicu_demo"
        / "eicu_demo_S4_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
    },
}


def _dataset_from_full_filename(path: Path) -> str:
    name = path.name
    if name.startswith("open_data_3clients_allfeat_"):
        return "open_data"
    if name.startswith("eicu_demo_5clients_hr_rr_spo2_"):
        return "eicu_demo"
    raise ValueError(f"Unknown external full-ablation file name: {path.name}")


def _load_json_rows(path: Path, dataset: str, allowed_methods: set[str] | None = None) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = []
    for result in payload["results"]:
        method = result["method"]
        if allowed_methods is not None and method not in allowed_methods:
            continue
        if method not in METHOD_LABELS:
            continue
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "scenario": result.get("scenario", payload["scenario"]),
                "mnar_method": result.get("mnar_method", payload["mnar_method"]),
                "missing_rate": float(result.get("missing_rate", payload["missing_rate"])),
                "seed": int(result["seed"]),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "mean_mae": float(result["mean_mae"]),
                "mean_rmse": float(result["mean_rmse"]),
                "num_clients": int(result.get("num_clients", payload["num_clients"])),
                "source_file": str(path.relative_to(ROOT.parent)),
            }
        )
    return rows


def load_external_rows() -> pd.DataFrame:
    rows: list[dict] = []
    missing: list[str] = []

    for path in sorted(FULL_EXTERNAL_DIR.glob("*.json")):
        rows.extend(_load_json_rows(path, _dataset_from_full_filename(path)))

    for dataset, scenario_paths in EXTERNAL_FEDAVG_FILES.items():
        for scenario, path in scenario_paths.items():
            if not path.exists():
                missing.append(str(path))
                continue
            rows.extend(_load_json_rows(path, dataset, {"fedavg", "fedice"}))

    if missing:
        raise FileNotFoundError("Missing external result files:\n" + "\n".join(missing))
    return pd.DataFrame(rows)


def load_vitaldb_summary_rows() -> pd.DataFrame:
    df = pd.read_csv(VITALDB_SUMMARY_CSV)
    df = df[
        (df["mnar_method"] == "quantile")
        & np.isclose(df["missing_rate"].astype(float), 0.5)
        & df["scenario"].isin(SCENARIOS)
        & df["method_label"].isin(VITALDB_METHOD_LABELS)
    ].copy()
    df["dataset"] = "vitaldb"
    df["dataset_label"] = DATASET_LABELS["vitaldb"]
    df["method_label"] = df["method_label"].map(VITALDB_METHOD_LABELS)
    df["method"] = df["method_label"]
    df["n_seeds"] = df["n_seeds"].astype(int)
    df["source_file"] = df["source_files"]
    return df[
        [
            "dataset",
            "dataset_label",
            "scenario",
            "mnar_method",
            "missing_rate",
            "method",
            "method_label",
            "mean_mae",
            "std_mae",
            "mean_rmse",
            "std_rmse",
            "n_seeds",
            "source_file",
        ]
    ]


def summarize_external(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "dataset",
                "dataset_label",
                "scenario",
                "mnar_method",
                "missing_rate",
                "method",
                "method_label",
            ],
            as_index=False,
        )
        .agg(
            mean_mae=("mean_mae", "mean"),
            std_mae=("mean_mae", "std"),
            mean_rmse=("mean_rmse", "mean"),
            std_rmse=("mean_rmse", "std"),
            n_seeds=("seed", "nunique"),
            source_file=("source_file", lambda x: ";".join(sorted(set(x)))),
        )
    )


def load_all_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    external_per_seed = load_external_rows()
    external_summary = summarize_external(external_per_seed)
    vitaldb_summary = load_vitaldb_summary_rows()
    summary = pd.concat([vitaldb_summary, external_summary], ignore_index=True)
    summary["method_label"] = pd.Categorical(
        summary["method_label"],
        categories=ALL_METHODS,
        ordered=True,
    )
    summary["scenario"] = pd.Categorical(summary["scenario"], categories=SCENARIOS, ordered=True)
    summary = summary.sort_values(["dataset", "scenario", "method_label"]).reset_index(drop=True)
    return external_per_seed, summary


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_methods(
    summary: pd.DataFrame,
    *,
    dataset: str,
    methods: list[str],
    title: str,
    name: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    sub = summary[
        (summary["dataset"] == dataset)
        & (summary["method_label"].isin(methods))
        & (summary["scenario"].isin(SCENARIOS))
    ].copy()

    missing = []
    for scenario in SCENARIOS:
        for method in methods:
            if sub[(sub["scenario"] == scenario) & (sub["method_label"] == method)].empty:
                missing.append(f"{scenario}/{method}")
    if missing:
        print(f"[warn] {DATASET_LABELS[dataset]} missing entries for {name}: {missing}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(SCENARIOS))
    for method in methods:
        rows = sub[sub["method_label"] == method].set_index("scenario")
        y = [rows.loc[scenario, "mean_mae"] if scenario in rows.index else np.nan for scenario in SCENARIOS]
        err = [
            rows.loc[scenario, "std_mae"] if scenario in rows.index and pd.notna(rows.loc[scenario, "std_mae"]) else 0.0
            for scenario in SCENARIOS
        ]
        ax.errorbar(
            x,
            y,
            yerr=err,
            marker=MARKERS[method],
            linewidth=2.25,
            markersize=6.5,
            capsize=3.5,
            color=COLORS[method],
            label=method,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("Quantile MNAR scenario")
    ax.set_ylabel("MAE on induced MNAR holes")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2 if len(methods) > 3 else 1)
    fig.tight_layout()
    save_figure(fig, name)


def plot_combined_family_overview(summary: pd.DataFrame) -> None:
    """Plot a compact 3x1 overview: rows=datasets, CA methods only."""

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 6.2),
        sharex=True,
        sharey=False,
    )
    x = np.arange(len(SCENARIOS))

    methods = ["FedSAITS+CA", "FedICE+CA"]
    legend_handles = {}

    for row, dataset in enumerate(DATASET_ORDER):
        ax = axes[row]
        ax.set_title(DATASET_LABELS[dataset], fontsize=14, pad=7)
        sub = summary[
            (summary["dataset"] == dataset)
            & (summary["method_label"].isin(methods))
        ].copy()

        for method in methods:
            rows = sub[sub["method_label"] == method].set_index("scenario")
            y = [
                rows.loc[scenario, "mean_mae"]
                if scenario in rows.index else np.nan
                for scenario in SCENARIOS
            ]
            err = [
                rows.loc[scenario, "std_mae"]
                if scenario in rows.index and pd.notna(rows.loc[scenario, "std_mae"])
                else 0.0
                for scenario in SCENARIOS
            ]
            handle = ax.errorbar(
                x,
                y,
                yerr=err,
                marker=MARKERS[method],
                linewidth=2.35,
                markersize=6.5,
                capsize=3.2,
                color=COLORS[method],
                label=method,
            )
            legend_handles[method] = handle

        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.grid(axis="y", alpha=0.24)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("")

        values = sub["mean_mae"].dropna().to_numpy(dtype=float)
        errors = sub["std_mae"].fillna(0.0).to_numpy(dtype=float)
        if values.size:
            lower = float(np.nanmin(values - errors))
            upper = float(np.nanmax(values + errors))
            span = max(upper - lower, 0.2)
            ax.set_ylim(max(0, lower - 0.18 * span), upper + 0.22 * span)

    axes[-1].set_xlabel("Quantile MNAR scenario", fontsize=11)
    fig.supylabel("MAE on induced MNAR holes", fontsize=12, x=0.025)

    fig.suptitle(
        "FedSAITS+CA vs FedICE+CA under Quantile MNAR (rho=0.5)",
        fontsize=15,
        y=0.985,
    )
    fig.legend(
        [legend_handles[m] for m in methods],
        methods,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=[0.055, 0.08, 1.0, 0.94], h_pad=1.25)
    save_figure(fig, "combined_2x3_family_dataset_rho0p5")


def plot_original_2x3_family_overview(summary: pd.DataFrame) -> None:
    """Recreate the original 2x3 overview: rows=families, columns=datasets."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.6),
        sharex=True,
        sharey="row",
    )
    x = np.arange(len(SCENARIOS))

    row_specs = [
        ("SAITS family", SAITS_FAMILY, ["Local", "FedSAITS", "FedSAITS+CA"]),
        ("ICE family", ICE_FAMILY, ["Local", "FedICE", "FedICE+CA"]),
    ]

    legend_handles = {}
    for col, dataset in enumerate(DATASET_ORDER):
        axes[0, col].set_title(DATASET_LABELS[dataset], fontsize=14, pad=10)

    for row, (family_name, methods, display_labels) in enumerate(row_specs):
        for col, dataset in enumerate(DATASET_ORDER):
            ax = axes[row, col]
            sub = summary[
                (summary["dataset"] == dataset)
                & (summary["method_label"].isin(methods))
            ].copy()

            for method, display_label in zip(methods, display_labels):
                rows = sub[sub["method_label"] == method].set_index("scenario")
                y = [
                    rows.loc[scenario, "mean_mae"]
                    if scenario in rows.index else np.nan
                    for scenario in SCENARIOS
                ]
                err = [
                    rows.loc[scenario, "std_mae"]
                    if scenario in rows.index and pd.notna(rows.loc[scenario, "std_mae"])
                    else 0.0
                    for scenario in SCENARIOS
                ]
                handle = ax.errorbar(
                    x,
                    y,
                    yerr=err,
                    marker=MARKERS[method],
                    linewidth=2.1,
                    markersize=5.5,
                    capsize=2.8,
                    color=COLORS[method],
                    label=display_label,
                )
                legend_handles[(family_name, display_label)] = handle

            ax.set_xticks(x)
            ax.set_xticklabels(SCENARIOS)
            ax.grid(axis="y", alpha=0.22)
            ax.tick_params(axis="both", labelsize=10)

            if col == 0:
                ax.set_ylabel("MAE", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    top_handles = [
        legend_handles[("SAITS family", label)]
        for label in ["Local", "FedSAITS", "FedSAITS+CA"]
    ]
    ice_handles = [
        legend_handles[("ICE family", label)]
        for label in ["Local", "FedICE", "FedICE+CA"]
    ]

    fig.suptitle(
        "Quantile MNAR External Comparison at rho=0.5",
        fontsize=15,
        y=0.985,
    )
    fig.legend(
        top_handles + ice_handles,
        ["Local", "FedSAITS", "FedSAITS+CA", "Local", "FedICE", "FedICE+CA"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0.02, 0.08, 1.0, 0.94])
    save_figure(fig, "combined_original_2x3_family_dataset_rho0p5")


def plot_combined_ca_overview_horizontal(summary: pd.DataFrame) -> None:
    """Optional 1x3 overview kept for quick side-by-side inspection."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 3.65),
        sharex=True,
        sharey=True,
    )
    x = np.arange(len(SCENARIOS))

    methods = ["FedSAITS+CA", "FedICE+CA"]

    legend_handles = {}
    for col, dataset in enumerate(DATASET_ORDER):
        ax = axes[col]
        ax.set_title(DATASET_LABELS[dataset], fontsize=14, pad=9)
        sub = summary[
            (summary["dataset"] == dataset)
            & (summary["method_label"].isin(methods))
        ].copy()

        for method in methods:
            rows = sub[sub["method_label"] == method].set_index("scenario")
            y = [
                rows.loc[scenario, "mean_mae"]
                if scenario in rows.index else np.nan
                for scenario in SCENARIOS
            ]
            err = [
                rows.loc[scenario, "std_mae"]
                if scenario in rows.index and pd.notna(rows.loc[scenario, "std_mae"])
                else 0.0
                for scenario in SCENARIOS
            ]
            handle = ax.errorbar(
                x,
                y,
                yerr=err,
                marker=MARKERS[method],
                linewidth=2.25,
                markersize=6.0,
                capsize=3.2,
                color=COLORS[method],
                label=method,
            )
            legend_handles[method] = handle

        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("MAE on induced MNAR holes", fontsize=11)
        else:
            ax.set_ylabel("")

    fig.suptitle(
        "FedSAITS+CA vs FedICE+CA under Quantile MNAR (rho=0.5)",
        fontsize=15,
        y=0.985,
    )
    fig.legend(
        [legend_handles[m] for m in methods],
        methods,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=[0.02, 0.13, 1.0, 0.88])
    save_figure(fig, "combined_2x3_family_dataset_rho0p5")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    external_per_seed, summary = load_all_summary()

    external_per_seed.to_csv(OUT_DIR / "external_full_ablation_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "external_full_ablation_summary.csv", index=False)

    # Four family-specific external figures.
    plot_methods(
        summary,
        dataset="open_data",
        methods=SAITS_FAMILY,
        title="open-data: SAITS Family (rho=0.5)",
        name="open_data_saits_family_rho0p5",
    )
    plot_methods(
        summary,
        dataset="open_data",
        methods=ICE_FAMILY,
        title="open-data: ICE Family (rho=0.5)",
        name="open_data_ice_family_rho0p5",
    )
    plot_methods(
        summary,
        dataset="eicu_demo",
        methods=SAITS_FAMILY,
        title="eICU-demo: SAITS Family (rho=0.5)",
        name="eicu_demo_saits_family_rho0p5",
    )
    plot_methods(
        summary,
        dataset="eicu_demo",
        methods=ICE_FAMILY,
        title="eICU-demo: ICE Family (rho=0.5)",
        name="eicu_demo_ice_family_rho0p5",
    )

    # Three all-family figures, one per dataset.
    plot_methods(
        summary,
        dataset="vitaldb",
        methods=ALL_METHODS,
        title="VitalDB: SAITS and ICE Families (rho=0.5)",
        name="vitaldb_all_families_rho0p5",
    )
    plot_methods(
        summary,
        dataset="open_data",
        methods=ALL_METHODS,
        title="open-data: SAITS and ICE Families (rho=0.5)",
        name="open_data_all_families_rho0p5",
    )
    plot_methods(
        summary,
        dataset="eicu_demo",
        methods=ALL_METHODS,
        title="eICU-demo: SAITS and ICE Families (rho=0.5)",
        name="eicu_demo_all_families_rho0p5",
    )
    plot_original_2x3_family_overview(summary)
    plot_combined_ca_overview_horizontal(summary)

    print(f"Saved figures and CSV summaries to: {OUT_DIR}")
    key = summary.pivot_table(
        index=["dataset_label", "scenario"],
        columns="method_label",
        values="mean_mae",
        observed=False,
    ).reset_index()
    print(key.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
