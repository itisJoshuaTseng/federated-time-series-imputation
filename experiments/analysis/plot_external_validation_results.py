#!/usr/bin/env python3
"""Summarize and plot external open-data/eICU validation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SERVER_RESULTS = ROOT.parent / "server_results"
OUT_DIR = ROOT / "experiments" / "figures" / "external_validation"
SCENARIOS = ["S1", "S2", "S3", "S4"]
METHOD_LABELS = {"fedavg": "FedSAITS", "fedice": "FedICE"}
COLORS = {"FedSAITS": "#1f77b4", "FedICE": "#9467bd"}
MARKERS = {"FedSAITS": "o", "FedICE": "D"}


FILES = {
    "open-data": [
        SERVER_RESULTS / "open_data_3clients_allfeat"
        / "open_data_S1_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
        SERVER_RESULTS / "external_s2_s3_rho0p5"
        / "open_data_S2_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
        SERVER_RESULTS / "external_s2_s3_rho0p5"
        / "open_data_S3_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
        SERVER_RESULTS / "open_data_3clients_allfeat"
        / "open_data_S4_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json",
    ],
    "eICU-demo": [
        SERVER_RESULTS / "eicu_demo"
        / "eicu_demo_S1_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
        SERVER_RESULTS / "external_s2_s3_rho0p5"
        / "eicu_demo_S2_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
        SERVER_RESULTS / "external_s2_s3_rho0p5"
        / "eicu_demo_S3_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
        SERVER_RESULTS / "eicu_demo"
        / "eicu_demo_S4_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json",
    ],
}


def load_rows() -> pd.DataFrame:
    rows = []
    missing = []
    for dataset, paths in FILES.items():
        for path in paths:
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            for result in payload["results"]:
                if result["method"] not in METHOD_LABELS:
                    continue
                rows.append({
                    "dataset": dataset,
                    "scenario": payload["scenario"],
                    "missing_rate": payload["missing_rate"],
                    "num_clients": payload["num_clients"],
                    "target_features": ",".join(map(str, payload["target_features"])),
                    "seed": result["seed"],
                    "method": result["method"],
                    "method_label": METHOD_LABELS[result["method"]],
                    "mean_mae": result["mean_mae"],
                    "mean_rmse": result["mean_rmse"],
                    "source_file": path.name,
                })
    if missing:
        raise FileNotFoundError("Missing result files:\n" + "\n".join(missing))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "scenario", "method", "method_label"], as_index=False)
        .agg(
            mean_mae=("mean_mae", "mean"),
            std_mae=("mean_mae", "std"),
            mean_rmse=("mean_rmse", "mean"),
            std_rmse=("mean_rmse", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    return summary


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    x = np.arange(len(SCENARIOS))
    for ax, dataset in zip(axes, ["open-data", "eICU-demo"]):
        sub = summary[summary["dataset"] == dataset]
        for label in ["FedSAITS", "FedICE"]:
            rows = sub[sub["method_label"] == label].set_index("scenario")
            y = [rows.loc[sc, "mean_mae"] for sc in SCENARIOS]
            err = [rows.loc[sc, "std_mae"] for sc in SCENARIOS]
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker=MARKERS[label],
                linewidth=2.2,
                markersize=6.5,
                capsize=3.5,
                color=COLORS[label],
                label=label,
            )
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_xlabel("Quantile MNAR scenario")
        ax.set_ylabel("MAE on induced MNAR holes")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("External Validation at rho=0.5", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_validation_s1_s4_s2_s3_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "external_validation_s1_s4_s2_s3_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    summary = summarize(df)
    df.to_csv(OUT_DIR / "external_validation_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "external_validation_summary.csv", index=False)

    pivot = summary.pivot_table(
        index=["dataset", "scenario"],
        columns="method_label",
        values="mean_mae",
    ).reset_index()
    pivot["winner"] = np.where(
        pivot["FedSAITS"] <= pivot["FedICE"],
        "FedSAITS",
        "FedICE",
    )
    pivot.to_csv(OUT_DIR / "external_validation_wide_summary.csv", index=False)
    plot(summary)
    print(f"Saved external validation summary and figure to: {OUT_DIR}")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
