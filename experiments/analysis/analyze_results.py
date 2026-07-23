#!/usr/bin/env python3
"""
Experiment Results Analyzer.

Reads all experiment result JSONs and produces:
  1. Comparison table (LaTeX + terminal)
  2. Per-client fairness analysis
  3. Convergence curves (if matplotlib available)

Usage:
    python experiments/analyze_results.py --log-dir logs/saits
    python experiments/analyze_results.py --log-dir logs/saits --latex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load_all_results(log_dir: str) -> List[dict]:
    """Load all result JSON files from log directory."""
    results = []
    patterns = [
        os.path.join(log_dir, "*.json"),
        os.path.join(log_dir, "**", "*.json"),
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))

    for fp in sorted(files):
        try:
            with open(fp) as f:
                data = json.load(f)
            if "mode" in data or "experiment_name" in data:
                data["_filepath"] = fp
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"Loaded {len(results)} experiment results from {log_dir}")
    return results


def extract_summary(result: dict) -> dict:
    """Extract key metrics from a single experiment result."""
    mode = result.get("mode", "unknown")
    name = result.get("experiment_name", "unnamed")

    summary = {
        "name": name,
        "mode": mode,
    }

    if mode == "centralized":
        tm = result.get("test_metrics", {})
        summary["test_mae"] = tm.get("mae", None)
        summary["test_rmse"] = tm.get("rmse", None)
        summary["test_mre"] = tm.get("mre", None)

    elif mode == "local":
        s = result.get("summary", {})
        summary["test_mae"] = s.get("avg_test_mae", None)
        summary["test_rmse"] = None
        summary["mae_std"] = s.get("std_test_mae", None)
        summary["mae_best"] = s.get("best_test_mae", None)
        summary["mae_worst"] = s.get("worst_test_mae", None)

        # Per-client breakdown
        per_client = result.get("per_client", [])
        if per_client:
            summary["per_client"] = per_client

    elif mode == "federated":
        summary["test_mae"] = result.get("best_mae", None)
        summary["num_rounds"] = result.get("num_rounds", None)

        # Final per-client evaluation
        final = result.get("final_eval", {})
        if final:
            global_m = final.get("global", {})
            summary["test_mae"] = global_m.get("mae", summary["test_mae"])
            summary["test_rmse"] = global_m.get("rmse", None)
            summary["test_mre"] = global_m.get("mre", None)

            fairness = final.get("fairness", {})
            summary["mae_mean_client"] = fairness.get("mae_mean", None)
            summary["mae_std"] = fairness.get("mae_std", None)
            summary["mae_best"] = fairness.get("mae_best", None)
            summary["mae_worst"] = fairness.get("mae_worst", None)
            summary["mae_gap"] = fairness.get("mae_gap", None)
            summary["mae_cv"] = fairness.get("mae_cv", None)

            summary["per_client"] = final.get("per_client", [])

    # Extract split & aggregation from config
    cfg = result.get("config", {})
    summary["split_strategy"] = cfg.get("data", {}).get("split_strategy", "?")
    summary["aggregation"] = cfg.get("federation", {}).get("aggregation", "?")
    summary["num_clients"] = cfg.get("federation", {}).get("num_clients", "?")

    return summary


def print_comparison_table(summaries: List[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT COMPARISON TABLE")
    print("=" * 100)

    header = (f"{'Name':<32s}  {'Mode':<12s}  {'Split':<14s}  {'Agg':<8s}  "
              f"{'Test MAE':>10s}  {'MAE Std':>9s}  {'MAE Gap':>9s}  {'CV':>6s}")
    print(header)
    print("-" * 100)

    for s in summaries:
        mae = f"{s['test_mae']:.6f}" if s.get('test_mae') is not None else "N/A"
        std = f"{s['mae_std']:.6f}" if s.get('mae_std') is not None else "-"
        gap = f"{s['mae_gap']:.6f}" if s.get('mae_gap') is not None else "-"
        cv  = f"{s['mae_cv']:.3f}" if s.get('mae_cv') is not None else "-"

        print(f"{s['name']:<32s}  {s['mode']:<12s}  "
              f"{s.get('split_strategy', '?'):<14s}  "
              f"{s.get('aggregation', '?'):<8s}  "
              f"{mae:>10s}  {std:>9s}  {gap:>9s}  {cv:>6s}")


def print_per_client_detail(summaries: List[dict]):
    """Print per-client metrics for experiments that have them."""
    print("\n" + "=" * 100)
    print("  PER-CLIENT FAIRNESS ANALYSIS")
    print("=" * 100)

    for s in summaries:
        pc = s.get("per_client", [])
        if not pc:
            continue

        print(f"\n--- {s['name']} ({s['mode']}, split={s.get('split_strategy')}) ---")
        for c in pc:
            cid = c.get("client_id", "?")
            n = c.get("num_samples", "?")
            mr = c.get("local_missing_rate", 0)

            # Handle both local and federated formats
            if "test_metrics" in c:
                mae_val = c["test_metrics"].get("mae", 0)
            else:
                mae_val = c.get("mae", 0)

            print(f"  Client {cid:2}: n={str(n):>5s}, "
                  f"miss_rate={mr:.1%}, MAE={mae_val:.6f}")


def print_latex_table(summaries: List[dict]):
    """Generate a LaTeX table for the paper."""
    print("\n% --- LaTeX Table ---")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Imputation performance under different client heterogeneity scenarios}")
    print(r"\label{tab:heterogeneity}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"Method & Split Strategy & Test MAE$\downarrow$ & "
          r"Client MAE Std & MAE Gap & CV \\")
    print(r"\midrule")

    for s in summaries:
        name_clean = s['name'].replace("_", r"\_")
        mae = f"{s['test_mae']:.4f}" if s.get('test_mae') is not None else "—"
        std = f"{s['mae_std']:.4f}" if s.get('mae_std') is not None else "—"
        gap = f"{s['mae_gap']:.4f}" if s.get('mae_gap') is not None else "—"
        cv  = f"{s['mae_cv']:.3f}" if s.get('mae_cv') is not None else "—"

        split = s.get('split_strategy', '—').replace("_", r"\_")
        print(f"{name_clean} & {split} & {mae} & {std} & {gap} & {cv} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--log-dir", type=str, default="logs/saits",
                        help="Directory with result JSONs")
    parser.add_argument("--latex", action="store_true",
                        help="Also print LaTeX table")
    args = parser.parse_args()

    results = load_all_results(args.log_dir)
    if not results:
        print("No results found!")
        return

    summaries = [extract_summary(r) for r in results]

    # Sort: centralized first, then local, then federated
    mode_order = {"centralized": 0, "local": 1, "federated": 2}
    summaries.sort(key=lambda s: (
        mode_order.get(s["mode"], 9),
        s.get("split_strategy", ""),
        s.get("aggregation", ""),
    ))

    print_comparison_table(summaries)
    print_per_client_detail(summaries)

    if args.latex:
        print_latex_table(summaries)


if __name__ == "__main__":
    main()
