"""
Step 0: Per-client re-analysis of existing MNAR CA-fix logs.

Reads `cafe_fix_v2_*_seeds_0-4.json` under `logs/saits_mnar/`,
extracts per-client MAE from each seed, and aggregates three
views that the existing EXPERIMENT_REPORT files do not show:

    mean_mae     : mean across 5 clients, averaged over seeds
                   (this is what the current reports print)
    worst_mae    : max across 5 clients per seed, averaged over seeds
                   (how bad is the weakest client?)
    mae_gap      : max - min across 5 clients per seed, averaged
                   (spread / fairness indicator)

Also prints per-client MAE side-by-side for FedAvg vs fed_ca, so we
can see whether CA's improvement is uniform or concentrated on the
minority client (Client 0 under S1).

Usage:
    python experiments/reanalyze_per_client.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "saits_mnar"


def load_runs(pattern: str = "cafe_fix_v2_*_seeds_0-4.json"):
    """Yield (meta, results) for every consolidated log."""
    for path in sorted(LOG_DIR.glob(pattern)):
        with path.open() as f:
            data = json.load(f)
        yield path.name, data


def group_by_setting(data):
    """Group results by (method) -> {seed: [client_mae,...]}."""
    by_method = {}
    for r in data["results"]:
        method = r["method"]
        seed = r["seed"]
        per_client = [cm["mae"] for cm in r["client_metrics"]]
        by_method.setdefault(method, {})[seed] = per_client
    return by_method


def aggregate_across_seeds(seed_to_client_mae):
    """
    seed_to_client_mae: {seed: [mae_c0, mae_c1, ...]}

    Returns dict with mean_mae / worst_mae / mae_gap / per_client_mean,
    each averaged over seeds.
    """
    seeds = sorted(seed_to_client_mae.keys())
    per_seed_mean = [mean(seed_to_client_mae[s]) for s in seeds]
    per_seed_worst = [max(seed_to_client_mae[s]) for s in seeds]
    per_seed_gap = [
        max(seed_to_client_mae[s]) - min(seed_to_client_mae[s])
        for s in seeds
    ]

    num_clients = len(next(iter(seed_to_client_mae.values())))
    per_client_mean = []
    for c in range(num_clients):
        per_client_mean.append(
            mean(seed_to_client_mae[s][c] for s in seeds)
        )

    def ms(vals):
        return (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)

    return {
        "n_seeds": len(seeds),
        "mean_mae": ms(per_seed_mean),
        "worst_mae": ms(per_seed_worst),
        "mae_gap": ms(per_seed_gap),
        "per_client_mean": per_client_mean,
    }


def fmt(x, y, w=12):
    return f"{x:.4f} ± {y:.4f}".ljust(w + 6)


def report(data, fname):
    scenario = data["scenario"]
    method_mnar = data["mnar_method"]
    rho = data["missing_rate"]
    tag = f"{scenario} / {method_mnar} / rho={rho}"

    grouped = group_by_setting(data)
    agg = {m: aggregate_across_seeds(v) for m, v in grouped.items()}

    print(f"\n=== {tag}   ({fname}) ===")
    print(f"{'method':<12} {'mean_MAE':<22} "
          f"{'worst_MAE':<22} {'gap':<22} n_seeds")
    for m in ("local", "fedavg", "fed_ca"):
        if m not in agg:
            continue
        a = agg[m]
        print(
            f"{m:<12} "
            f"{fmt(*a['mean_mae'], 14):<22} "
            f"{fmt(*a['worst_mae'], 14):<22} "
            f"{fmt(*a['mae_gap'], 14):<22} "
            f"{a['n_seeds']}"
        )

    methods_for_per_client = [m for m in ("local", "fedavg", "fed_ca")
                              if m in agg]
    if len(methods_for_per_client) >= 2:
        n = len(agg[methods_for_per_client[0]]["per_client_mean"])
        header = "per-client MAE  | " + " | ".join(
            f"C{c}" for c in range(n)
        )
        print(header)
        for m in methods_for_per_client:
            row = agg[m]["per_client_mean"]
            cells = " | ".join(f"{v:.4f}" for v in row)
            print(f"  {m:<12} | {cells}")

        if "fedavg" in agg and "fed_ca" in agg:
            delta = [
                (agg["fedavg"]["per_client_mean"][c]
                 - agg["fed_ca"]["per_client_mean"][c])
                for c in range(n)
            ]
            rel = [
                delta[c] / agg["fedavg"]["per_client_mean"][c] * 100.0
                for c in range(n)
            ]
            cells = " | ".join(f"{v:+.2%}" for v in
                               (d / 100 for d in rel))
            print(f"  CA vs FedAvg | {cells}   "
                  f"(+ means CA is better on that client)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="cafe_fix_v2_*_seeds_0-4.json")
    args = p.parse_args()

    count = 0
    for fname, data in load_runs(args.pattern):
        report(data, fname)
        count += 1

    if count == 0:
        print(f"No logs matched under {LOG_DIR} "
              f"with pattern '{args.pattern}'")


if __name__ == "__main__":
    main()
