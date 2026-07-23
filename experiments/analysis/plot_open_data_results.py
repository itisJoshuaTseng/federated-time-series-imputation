#!/usr/bin/env python3
"""Plot open-data external validation results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT.parent / "server_results" / "open_data_3clients_allfeat"
OUT_DIR = ROOT / "experiments" / "figures" / "open_data"
SCENARIOS = ["S1", "S4"]
METHODS = [
    ("fedavg", "FedSAITS", "#1f77b4", "o"),
    ("fedice", "FedICE", "#9467bd", "D"),
]


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULT_DIR.glob("open_data_*.json")):
        payload = json.loads(path.read_text())
        for result in payload["results"]:
            rows.append({
                "scenario": payload["scenario"],
                "missing_rate": payload["missing_rate"],
                "num_clients": payload["num_clients"],
                "target_features": ",".join(map(str, payload["target_features"])),
                "seed": result["seed"],
                "method": result["method"],
                "mean_mae": result["mean_mae"],
                "mean_rmse": result["mean_rmse"],
                "source_file": path.name,
            })
    if not rows:
        raise FileNotFoundError(f"No open-data result JSON files found in {RESULT_DIR}")
    return pd.DataFrame(rows)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["scenario", "method"], as_index=False)
        .agg(
            mean_mae=("mean_mae", "mean"),
            std_mae=("mean_mae", "std"),
            mean_rmse=("mean_rmse", "mean"),
            std_rmse=("mean_rmse", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    labels = {"fedavg": "FedSAITS", "fedice": "FedICE"}
    summary["method_label"] = summary["method"].map(labels)
    return summary


def plot(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SCENARIOS))
    offsets = {"fedavg": -0.035, "fedice": 0.035}

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    for method, label, color, marker in METHODS:
        means = []
        stds = []
        for scenario in SCENARIOS:
            row = summary[
                (summary["scenario"] == scenario)
                & (summary["method"] == method)
            ]
            means.append(float(row["mean_mae"].iloc[0]))
            stds.append(float(row["std_mae"].iloc[0]))

        ax.errorbar(
            x + offsets[method],
            means,
            yerr=stds,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            capsize=4,
            label=label,
        )

        for i, scenario in enumerate(SCENARIOS):
            values = df[
                (df["scenario"] == scenario)
                & (df["method"] == method)
            ].sort_values("seed")
            jitter = np.linspace(-0.018, 0.018, len(values))
            ax.scatter(
                np.full(len(values), x[i] + offsets[method]) + jitter,
                values["mean_mae"],
                s=28,
                color=color,
                alpha=0.45,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("Quantile MNAR scenario")
    ax.set_ylabel("MAE on induced MNAR holes")
    ax.set_title("Open-Data External Check (3 Clients, All 4 Features)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    fig.savefig(OUT_DIR / "open_data_s1_s4_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "open_data_s1_s4_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_rows()
    summary = make_summary(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "open_data_s1_s4_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "open_data_s1_s4_summary.csv", index=False)
    plot(df, summary)
    print(f"Saved open-data figure and tables to: {OUT_DIR}")


if __name__ == "__main__":
    main()
