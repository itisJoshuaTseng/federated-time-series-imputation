#!/usr/bin/env python3
"""
Collect and plot all available MNAR experiment results.

The logs in logs/saits_mnar come from several experiment waves with
slightly different filenames.  This script normalizes them into tidy CSVs
and creates figures for the current diagnostic-benchmark framing.

Outputs:
  experiments/figures/all_mnar_results_raw.csv
  experiments/figures/all_mnar_results_canonical.csv
  experiments/figures/all_mnar_results_summary.csv
  experiments/figures/fig_all_mnar_quantile_methods.png
  experiments/figures/fig_all_mnar_logit_methods.png
  experiments/figures/fig_quantile_family_crossover.png
  experiments/figures/fig_quantile_method_heatmap.png
  experiments/figures/fig_ca_scale_sweep.png
  experiments/figures/ALL_MNAR_RESULTS_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "Local-SAITS",
    "Centralized-SAITS",
    "FedAvg-SAITS",
    "FedProx-SAITS",
    "Fed-SAITS-CA b=0.5",
    "Fed-SAITS-CA b=1",
    "Fed-SAITS-CA b=2",
    "Fed-SAITS-CA b=3",
    "Fed-SAITS-CA b=4",
    "Fed-SAITS-CA b=8",
    "Fed-SAITS-CA old",
    "Local-ICE",
    "FedICE",
    "FedICE-CA b=4",
    "Fed-SAITS-PD",
    "Fed-SAITS-CA+PD",
]

PLOT_METHOD_ORDER = [
    "Local-SAITS",
    "Centralized-SAITS",
    "FedAvg-SAITS",
    "FedProx-SAITS",
    "Fed-SAITS-CA b=0.5",
    "Fed-SAITS-CA b=4",
    "Local-ICE",
    "FedICE",
    "FedICE-CA b=4",
]

COLORS = {
    "Local-SAITS": "#8a8a8a",
    "Centralized-SAITS": "#222222",
    "FedAvg-SAITS": "#1f77b4",
    "FedProx-SAITS": "#17becf",
    "Fed-SAITS-CA b=0.5": "#2ca02c",
    "Fed-SAITS-CA b=1": "#98df8a",
    "Fed-SAITS-CA b=2": "#74c476",
    "Fed-SAITS-CA b=3": "#31a354",
    "Fed-SAITS-CA b=4": "#ff7f0e",
    "Fed-SAITS-CA b=8": "#d95f0e",
    "Fed-SAITS-CA old": "#bcbd22",
    "Local-ICE": "#8c564b",
    "FedICE": "#9467bd",
    "FedICE-CA b=4": "#d62728",
    "Fed-SAITS-PD": "#e377c2",
    "Fed-SAITS-CA+PD": "#7f7f7f",
}

SCENARIOS = ["S1", "S2", "S3", "S4"]
RHOS = [0.3, 0.5, 0.7]


def rho_tag(value: Any) -> str:
    if value is None or pd.isna(value):
        return "rhoNA"
    return f"rho{float(value):.1f}".replace(".", "p")


def normalize_mechanism(value: Any, filename: str = "") -> str | None:
    if value is None:
        if re.search(r"(^|_)q(_|$)", filename) or "quantile" in filename:
            return "quantile"
        if re.search(r"(^|_)l(_|$)", filename) or "logit" in filename:
            return "logit"
        return None
    s = str(value).lower()
    if s in {"q", "quantile"}:
        return "quantile"
    if s in {"l", "logit", "logistic"}:
        return "logit"
    return s


def infer_scale_factor(result: dict[str, Any], filename: str) -> float | None:
    if result.get("ca_scale_factor") is not None:
        return float(result["ca_scale_factor"])

    patterns = [
        (r"sf0p5|b0p5", 0.5),
        (r"sf1(?:_|$)|b1(?:_|$)", 1.0),
        (r"sf2(?:_|$)|b2(?:_|$)", 2.0),
        (r"sf3(?:_|$)|b3(?:_|$)", 3.0),
        (r"sf4(?:_|$)|b4(?:_|$)", 4.0),
        (r"sf8(?:_|$)|b8(?:_|$)", 8.0),
    ]
    for pattern, value in patterns:
        if re.search(pattern, filename):
            return value

    # The CA fix wave used the CAFE power-law beta/scale setting.
    if filename.startswith("cafe_fix_v2_"):
        return 4.0
    return None


def method_label(result: dict[str, Any], filename: str) -> str:
    method = str(result.get("method", "")).lower()

    if method == "local":
        return "Local-SAITS"
    if method == "local_ice":
        return "Local-ICE"
    if method == "centralized":
        return "Centralized-SAITS"
    if method == "fedavg":
        return "FedAvg-SAITS"
    if method == "fedprox":
        return "FedProx-SAITS"
    if method == "fedice":
        return "FedICE"
    if method == "fedice_ca":
        scale = infer_scale_factor(result, filename)
        if scale is not None:
            return f"FedICE-CA b={scale:g}"
        return "FedICE-CA"
    if method == "fed_pd":
        return "Fed-SAITS-PD"
    if method == "fed_ca_pd":
        return "Fed-SAITS-CA+PD"
    if method == "fed_ca":
        if filename.startswith("mnar_fed_ca_"):
            return "Fed-SAITS-CA old"
        scale = infer_scale_factor(result, filename)
        if scale is not None:
            return f"Fed-SAITS-CA b={scale:g}"
        if filename.startswith("cafe_fix_v2_"):
            return "Fed-SAITS-CA b=4"
        return "Fed-SAITS-CA"

    return method or "unknown"


def source_kind(filename: str, experiment: str | None) -> str:
    if filename.startswith(".") or filename.startswith("tmp_"):
        return "tmp"
    if "validation" in filename or filename.startswith("test_"):
        return "validation"
    if filename.startswith("heldout_") or experiment == "mnar_client_local_heldout":
        return "heldout"
    if filename.startswith("centralized_ceiling_"):
        return "centralized"
    if filename.startswith("local_saits_") or filename.startswith("local_ice_"):
        return "local_backbone_gap"
    if filename.startswith("S") and ("_fed" in filename):
        return "current_s23_or_fedice"
    if filename.startswith("ablation_sf"):
        return "scale_ablation"
    if filename.startswith("cafe_fix_v2_"):
        return "cafe_fix_v2"
    if filename.startswith("mnar_recon_"):
        return "mnar_recon"
    if filename.startswith("mnar_fed_ca_"):
        return "old_ca"
    if filename.startswith("tau_sweep_"):
        return "tau_sweep"
    return "other"


def source_priority(filename: str, result: dict[str, Any], kind: str) -> int:
    priority_by_kind = {
        "current_s23_or_fedice": 100,
        "local_backbone_gap": 100,
        "centralized": 96,
        "scale_ablation": 94,
        "cafe_fix_v2": 90,
        "mnar_recon": 70,
        "old_ca": 40,
        "tau_sweep": 30,
        "heldout": 20,
        "validation": 5,
        "tmp": 0,
        "other": 10,
    }
    priority = priority_by_kind.get(kind, 10)
    if "seeds_" in filename:
        priority += 3
    if re.search(r"_seed\d+\.json$", filename):
        priority -= 2
    return priority


def parse_result_file(path: Path) -> list[dict[str, Any]]:
    filename = path.name
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] failed to parse {path}: {exc}")
        return []

    experiment = data.get("experiment")
    kind = source_kind(filename, experiment)
    results = data.get("results")

    if isinstance(results, list):
        entries = results
    elif data.get("mean_mae") is not None:
        entries = [data]
    else:
        return []

    rows: list[dict[str, Any]] = []
    mtime = path.stat().st_mtime
    for item in entries:
        if not isinstance(item, dict):
            continue
        mean_mae = item.get("mean_mae")
        mean_rmse = item.get("mean_rmse")
        if mean_mae is None:
            continue

        scenario = item.get("scenario", data.get("scenario"))
        mechanism = normalize_mechanism(
            item.get("mnar_method", data.get("mnar_method")), filename
        )
        missing_rate = item.get("missing_rate", data.get("missing_rate"))
        seed = item.get("seed")
        label = method_label(item, filename)
        scale = infer_scale_factor(item, filename)

        rows.append({
            "source_file": filename,
            "source_path": str(path),
            "source_kind": kind,
            "source_priority": source_priority(filename, item, kind),
            "source_mtime": mtime,
            "experiment": experiment,
            "eval_protocol": "heldout" if kind == "heldout" else "reconstruction",
            "scenario": scenario,
            "mnar_method": mechanism,
            "missing_rate": float(missing_rate) if missing_rate is not None else np.nan,
            "rho_tag": rho_tag(missing_rate),
            "seed": int(seed) if seed is not None else -1,
            "method_raw": item.get("method"),
            "method_label": label,
            "ca_scale_factor": scale,
            "ca_tau": item.get("ca_tau"),
            "rounds": item.get("rounds", item.get("ice_rounds", item.get("num_rounds"))),
            "mean_mae": float(mean_mae),
            "mean_rmse": float(mean_rmse) if mean_rmse is not None else np.nan,
        })
    return rows


def collect_results(log_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json")):
        rows.extend(parse_result_file(path))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIOS, ordered=True)
    df["method_order"] = df["method_label"].map({
        m: i for i, m in enumerate(METHOD_ORDER)
    }).fillna(999).astype(int)
    return df


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Choose one row per seed/method/config using source priority."""
    keep = df.copy()

    # Raw sweep/validation/tmp/heldout are kept in raw CSV but excluded from
    # the main reconstruction plots. Heldout needs separate figures because its
    # metric is not the same as MNAR-hole reconstruction.
    keep = keep[
        (keep["eval_protocol"] == "reconstruction")
        & (~keep["source_kind"].isin(["tmp", "validation", "tau_sweep"]))
        & (keep["method_label"] != "Fed-SAITS-CA")
    ].copy()

    group_cols = [
        "scenario",
        "mnar_method",
        "missing_rate",
        "method_label",
    ]
    keep["max_group_priority"] = keep.groupby(
        group_cols, observed=True
    )["source_priority"].transform("max")
    keep = keep[keep["source_priority"] == keep["max_group_priority"]].copy()

    keep = keep.sort_values(
        [
            "scenario",
            "mnar_method",
            "missing_rate",
            "method_label",
            "seed",
            "source_priority",
            "source_mtime",
        ],
        ascending=[True, True, True, True, True, False, False],
    )
    dedup_cols = [
        "scenario",
        "mnar_method",
        "missing_rate",
        "method_label",
        "seed",
    ]
    keep = keep.drop_duplicates(dedup_cols, keep="first")
    keep = keep.drop(columns=["max_group_priority"])
    return keep.sort_values([
        "scenario", "mnar_method", "missing_rate", "method_order", "seed"
    ])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["scenario", "mnar_method", "missing_rate", "rho_tag", "method_label"],
            observed=True,
        )
        .agg(
            mean_mae=("mean_mae", "mean"),
            std_mae=("mean_mae", "std"),
            mean_rmse=("mean_rmse", "mean"),
            std_rmse=("mean_rmse", "std"),
            n_seeds=("seed", "nunique"),
            source_files=("source_file", lambda x: ";".join(sorted(set(x)))),
        )
        .reset_index()
    )
    summary["std_mae"] = summary["std_mae"].fillna(0.0)
    summary["std_rmse"] = summary["std_rmse"].fillna(0.0)
    summary["method_order"] = summary["method_label"].map({
        m: i for i, m in enumerate(METHOD_ORDER)
    }).fillna(999).astype(int)
    return summary.sort_values([
        "scenario", "mnar_method", "missing_rate", "method_order"
    ])


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
    })


def plot_method_grid(summary: pd.DataFrame, mechanism: str, out: Path) -> None:
    data = summary[summary["mnar_method"] == mechanism].copy()
    if data.empty:
        return

    methods = [m for m in PLOT_METHOD_ORDER if m in set(data["method_label"])]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    axes = axes.ravel()

    for ax, scenario in zip(axes, SCENARIOS):
        sub = data[data["scenario"].astype(str) == scenario]
        if sub.empty:
            ax.set_title(f"{scenario} (no data)")
            ax.axis("off")
            continue
        for method in methods:
            mdf = sub[sub["method_label"] == method].sort_values("missing_rate")
            if mdf.empty:
                continue
            ax.errorbar(
                mdf["missing_rate"],
                mdf["mean_mae"],
                yerr=mdf["std_mae"],
                marker="o",
                capsize=3,
                label=method,
                color=COLORS.get(method),
            )
        ax.set_title(scenario)
        ax.set_xticks(RHOS)
        ax.set_xlabel("MNAR missing rate (rho)")
        ax.set_ylabel("MAE on MNAR holes")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"All available canonical methods ({mechanism})", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_family_crossover(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["mnar_method"] == "quantile"].copy()
    if data.empty:
        return

    family_rows = []
    for (scenario, rho), g in data.groupby(["scenario", "missing_rate"], observed=True):
        saits = g[g["method_label"].isin([
            "FedAvg-SAITS",
            "Fed-SAITS-CA b=0.5",
            "Fed-SAITS-CA b=4",
        ])]
        ice = g[g["method_label"].isin(["FedICE", "FedICE-CA b=4"])]
        for family, fg in [("SAITS family", saits), ("ICE family", ice)]:
            if fg.empty:
                continue
            best = fg.sort_values("mean_mae").iloc[0]
            family_rows.append({
                "scenario": str(scenario),
                "missing_rate": rho,
                "family": family,
                "mean_mae": best["mean_mae"],
                "std_mae": best["std_mae"],
                "best_method": best["method_label"],
            })
    fam = pd.DataFrame(family_rows)
    if fam.empty:
        return

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    colors = {"SAITS family": "#2ca02c", "ICE family": "#9467bd"}
    for ax, scenario in zip(axes, SCENARIOS):
        sub = fam[fam["scenario"] == scenario]
        for family in ["SAITS family", "ICE family"]:
            fdf = sub[sub["family"] == family].sort_values("missing_rate")
            if fdf.empty:
                continue
            ax.errorbar(
                fdf["missing_rate"],
                fdf["mean_mae"],
                yerr=fdf["std_mae"],
                marker="o",
                capsize=3,
                label=family,
                color=colors[family],
            )
        ax.set_title(scenario)
        ax.set_xticks(RHOS)
        ax.set_xlabel("rho")
    axes[0].set_ylabel("Best family MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Backbone-family crossover (quantile MNAR)", y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    fam.to_csv(out.with_suffix(".csv"), index=False)


def plot_heatmap(summary: pd.DataFrame, out: Path) -> None:
    data = summary[
        (summary["mnar_method"] == "quantile")
        & (summary["method_label"].isin(PLOT_METHOD_ORDER))
    ].copy()
    if data.empty:
        return
    data["row"] = (
        data["scenario"].astype(str) + " " + data["missing_rate"].map(lambda x: f"rho={x:.1f}")
    )
    pivot = data.pivot_table(
        index="row",
        columns="method_label",
        values="mean_mae",
        aggfunc="mean",
        observed=True,
    )
    row_order = [f"{s} rho={r:.1f}" for s in SCENARIOS for r in RHOS]
    col_order = [m for m in PLOT_METHOD_ORDER if m in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(max(8, len(col_order) * 1.25), 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title("Quantile MNAR MAE heatmap (lower is better)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if pd.isna(val):
                ax.text(j, i, "-", ha="center", va="center", color="white", fontsize=8)
            else:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_scale_sweep(summary: pd.DataFrame, out: Path) -> None:
    data = summary[
        summary["method_label"].str.startswith("Fed-SAITS-CA b=", na=False)
    ].copy()
    data = data[data["mnar_method"].isin(["quantile", "logit"])]
    if data.empty:
        return
    data["beta"] = data["method_label"].str.extract(r"b=([0-9.]+)").astype(float)

    # Focus on configurations with at least two beta values.
    counts = (
        data.groupby(["scenario", "mnar_method", "missing_rate"], observed=True)["beta"]
        .nunique()
        .reset_index(name="n_beta")
    )
    multi = counts[counts["n_beta"] >= 2]
    if multi.empty:
        return
    data = data.merge(
        multi[["scenario", "mnar_method", "missing_rate"]],
        on=["scenario", "mnar_method", "missing_rate"],
        how="inner",
    )

    keys = list(data.groupby(["scenario", "mnar_method", "missing_rate"], observed=True).groups)
    n = len(keys)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, key in zip(axes_flat, keys):
        scenario, mech, rho = key
        sub = data[
            (data["scenario"].astype(str) == str(scenario))
            & (data["mnar_method"] == mech)
            & (np.isclose(data["missing_rate"], rho))
        ].sort_values("beta")
        ax.errorbar(
            sub["beta"], sub["mean_mae"], yerr=sub["std_mae"],
            marker="o", capsize=3, color="#2ca02c",
        )
        ax.set_xscale("log")
        ax.set_title(f"{scenario} {mech} rho={rho:.1f}")
        ax.set_xlabel("CA beta / scale factor")
        ax.set_ylabel("MAE")
    for ax in axes_flat[len(keys):]:
        ax.axis("off")
    fig.suptitle("Fed-SAITS-CA scale-factor sweep", y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def format_mean_std(row: pd.Series) -> str:
    return f"{row['mean_mae']:.3f} ± {row['std_mae']:.3f}"


def write_markdown(summary: pd.DataFrame, raw: pd.DataFrame, out: Path) -> None:
    lines = []
    lines.append("# All MNAR Results Summary\n")
    lines.append("Generated by `experiments/plot_all_mnar_results.py`.\n")
    lines.append(f"- Raw parsed rows: {len(raw)}")
    lines.append(f"- Canonical summary rows: {len(summary)}\n")

    for mechanism in ["quantile", "logit"]:
        sub = summary[summary["mnar_method"] == mechanism]
        if sub.empty:
            continue
        lines.append(f"## {mechanism.capitalize()} Results\n")
        for scenario in SCENARIOS:
            ssub = sub[sub["scenario"].astype(str) == scenario]
            if ssub.empty:
                continue
            pivot = ssub.copy()
            pivot["value"] = pivot.apply(format_mean_std, axis=1)
            table = pivot.pivot_table(
                index="missing_rate",
                columns="method_label",
                values="value",
                aggfunc="first",
                observed=True,
            )
            cols = [m for m in PLOT_METHOD_ORDER if m in table.columns]
            extra_cols = [c for c in table.columns if c not in cols]
            table = table[cols + extra_cols]
            lines.append(f"### {scenario}\n")
            lines.append(table.to_markdown())
            lines.append("")

    availability = (
        summary.assign(config=lambda d: d["scenario"].astype(str) + "-" + d["mnar_method"] + "-" + d["rho_tag"])
        .pivot_table(index="config", columns="method_label", values="n_seeds", aggfunc="max", observed=True)
        .fillna("")
    )
    cols = [m for m in METHOD_ORDER if m in availability.columns]
    availability = availability[cols]
    lines.append("## Availability Matrix (number of seeds)\n")
    lines.append(availability.to_markdown())
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default="logs/saits_mnar",
        help="Directory containing MNAR result JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/figures/all_mnar_results",
        help="Output directory for CSVs and figures.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    raw = collect_results(log_dir)
    if raw.empty:
        raise SystemExit(f"No result rows found in {log_dir}")

    canonical = canonicalize(raw)
    summary = summarize(canonical)

    raw.to_csv(out_dir / "all_mnar_results_raw.csv", index=False)
    canonical.to_csv(out_dir / "all_mnar_results_canonical.csv", index=False)
    summary.to_csv(out_dir / "all_mnar_results_summary.csv", index=False)

    plot_method_grid(summary, "quantile", out_dir / "fig_all_mnar_quantile_methods.png")
    plot_method_grid(summary, "logit", out_dir / "fig_all_mnar_logit_methods.png")
    plot_family_crossover(summary, out_dir / "fig_quantile_family_crossover.png")
    plot_heatmap(summary, out_dir / "fig_quantile_method_heatmap.png")
    plot_scale_sweep(summary, out_dir / "fig_ca_scale_sweep.png")
    write_markdown(summary, raw, out_dir / "ALL_MNAR_RESULTS_SUMMARY.md")

    print(f"Parsed raw rows: {len(raw)}")
    print(f"Canonical rows: {len(canonical)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Output directory: {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
