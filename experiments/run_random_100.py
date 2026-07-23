#!/usr/bin/env python3
"""
100× Random Seed Experiment — Statistical Validation of FL Benefits.

Research Question:
    Does Federated Averaging (FedAvg) consistently outperform Local-only
    training for time-series imputation across different random data splits?

Design:
    - 100 different random seeds → 100 different 5-client IID partitions
    - For each seed: run FedAvg (50 rounds) vs Local-only (same total epochs)
    - Record: global test MAE, per-client MAE, training time
    - Output: CSV for statistical analysis (paired t-test, Wilcoxon, etc.)

Expected runtime (per seed):
    - FedAvg:     ~12 min (50 rounds × 5 clients × 5 local epochs)
    - Local-only: ~12 min (5 clients × 250 total epochs)
    - Total per seed: ~24 min
    - Total 100 seeds: ~40 hours (RTX4060 recommended)

Features:
    - Resume support: skips seeds that already have results
    - CSV output: one row per seed per method, ready for pandas/R
    - JSON backup: full detailed results for each seed
    - Progress reporting: ETA, running stats

Usage:
    # Run all 100 seeds (default)
    python experiments/run_random_100.py

    # Run specific seed range (for parallel execution)
    python experiments/run_random_100.py --start 0 --end 50

    # Use GPU
    python experiments/run_random_100.py --device cuda

    # Quick test (3 seeds, 5 rounds)
    python experiments/run_random_100.py --quick

    # Dry run (print what would be run)
    python experiments/run_random_100.py --dry-run

    # Resume from where we left off
    python experiments/run_random_100.py --resume

    # Analyze completed results
    python experiments/run_random_100.py --analyze-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ================================================================
# Configuration
# ================================================================

DEFAULT_CONFIG = {
    "data": {
        "type": "vitaldb",
        "seq_length": 300,
        "num_features": 14,
        "missing_rate": 0.2,
        "missing_mechanism": "mcar",
        "tensor_dir": "../2026_vitalDB/tensor-file-for-4feature-20260304T112438Z-3-001/tensor-file-for-4feature/vitaldb_14feats_tensor_T300",
        "split_strategy": "random",
        "train_ratio": 0.7,
        "eval_missing_rate": 0.1,
        "normalize": True,
    },
    "vitaldb": {
        "num_cases": 2500,
        "use_extended_tracks": False,
        "normalize": True,
        "split_strategy": "random",
    },
    "federation": {
        "num_clients": 5,
        "rounds": 50,
        "frac_clients": 1.0,
        "aggregation": "fedavg",
    },
    "saits": {
        "n_layers": 2,
        "d_model": 256,
        "n_heads": 4,
        "d_ffn": 128,
        "d_k": 64,
        "d_v": 64,
        "dropout": 0.1,
        "attn_dropout": 0.0,
        "diagonal_attention_mask": True,
        "ORT_weight": 1.0,
        "MIT_weight": 1.0,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience": 3,
    },
    "training": {
        "eval_every": 1,
        "early_stop_patience": 10,
        "checkpoint_dir": "checkpoints/random100",
    },
    "logging": {
        "log_dir": "logs/random100",
    },
}


# ================================================================
# Single Seed Runner
# ================================================================

def run_single_seed(
    seed: int,
    config: dict,
    device: str = "cpu",
    rounds: int = 50,
) -> Dict:
    """
    Run FedAvg + Local-only for a single seed.

    Returns:
        dict with results for this seed
    """
    import random
    import torch
    from src.data.dataset import create_federated_datasets
    from src.federation.saits_client import SAITSClient
    from src.federation.saits_server import SAITSFederatedServer

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = json.loads(json.dumps(config))  # deep copy
    cfg["training"]["seed"] = seed
    cfg["federation"]["rounds"] = rounds
    cfg["data"]["split_strategy"] = "random"
    # Suppress checkpoint saving for batch runs
    cfg["training"]["checkpoint_dir"] = f"checkpoints/random100/seed_{seed}"

    # ---- Load Data ----
    datasets = create_federated_datasets(cfg, seed=seed)
    client_datasets = datasets["client_datasets"]
    test_dataset = datasets.get("test_dataset")
    num_clients = len(client_datasets)

    actual_features = client_datasets[0].data.shape[2]
    actual_seq_len = client_datasets[0].data.shape[1]
    cfg["data"]["num_features"] = actual_features
    cfg["data"]["seq_length"] = actual_seq_len

    client_sizes = [len(ds) for ds in client_datasets]

    result = {
        "seed": seed,
        "num_clients": num_clients,
        "client_sizes": client_sizes,
    }

    # ================================================================
    # 1. FedAvg Training
    # ================================================================
    t0 = time.time()
    print(f"\n  [Seed {seed}] FedAvg training ({rounds} rounds)...")

    cfg_fedavg = json.loads(json.dumps(cfg))
    cfg_fedavg["federation"]["aggregation"] = "fedavg"

    clients = []
    for i, ds in enumerate(client_datasets):
        client = SAITSClient(
            client_id=i, train_data=ds, val_data=None,
            config=cfg_fedavg, device=device,
        )
        clients.append(client)

    server = SAITSFederatedServer(
        clients=clients, test_data=test_dataset,
        config=cfg_fedavg, device=device,
    )
    train_results = server.train()

    fedavg_time = time.time() - t0
    fedavg_eval = train_results.get("final_eval", {})
    fedavg_global_mae = fedavg_eval.get("global", {}).get("mae", float("nan"))
    fedavg_global_rmse = fedavg_eval.get("global", {}).get("rmse", float("nan"))
    fedavg_per_client = fedavg_eval.get("per_client", [])
    fedavg_fairness = fedavg_eval.get("fairness", {})

    result["fedavg"] = {
        "global_mae": fedavg_global_mae,
        "global_rmse": fedavg_global_rmse,
        "best_mae": train_results.get("best_mae", float("nan")),
        "num_rounds_actual": len(train_results.get("history", [])),
        "time_seconds": fedavg_time,
        "per_client_mae": [m.get("mae", float("nan")) for m in fedavg_per_client],
        "fairness": fedavg_fairness,
    }
    print(f"    FedAvg done: MAE={fedavg_global_mae:.6f}, "
          f"time={fedavg_time:.0f}s")

    # ================================================================
    # 2. Local-only Training
    # ================================================================
    t0 = time.time()
    local_epochs_per_round = cfg.get("saits", {}).get("local_epochs", 5)
    total_local_epochs = rounds * local_epochs_per_round
    print(f"  [Seed {seed}] Local-only training ({total_local_epochs} epochs)...")

    local_per_client = []
    for i, ds in enumerate(client_datasets):
        client = SAITSClient(
            client_id=i, train_data=ds, val_data=None,
            config=cfg, device=device,
        )
        client.model.set_training_params(
            epochs=total_local_epochs,
            patience=min(20, max(total_local_epochs - 1, 1)),
        )
        client.local_train()

        # Evaluate on global test set
        if test_dataset is not None:
            test_m = client.model.evaluate(
                observed=test_dataset.data.numpy(),
                masks=test_dataset.masks.numpy(),
                ground_truth=test_dataset.ground_truth.numpy(),
                eval_masks=test_dataset.eval_masks.numpy(),
            )
        else:
            test_m = {}

        local_per_client.append({
            "client_id": i,
            "test_mae": test_m.get("mae", float("nan")),
            "test_rmse": test_m.get("rmse", float("nan")),
        })
        print(f"    Client {i}: test_MAE={test_m.get('mae', 0):.6f}")

    local_time = time.time() - t0
    local_maes = [r["test_mae"] for r in local_per_client]
    result["local"] = {
        "avg_test_mae": float(np.mean(local_maes)),
        "std_test_mae": float(np.std(local_maes)),
        "best_test_mae": float(np.min(local_maes)),
        "worst_test_mae": float(np.max(local_maes)),
        "per_client_mae": local_maes,
        "time_seconds": local_time,
    }
    print(f"    Local done: avg_MAE={np.mean(local_maes):.6f}, "
          f"time={local_time:.0f}s")

    # ================================================================
    # Improvement calculation
    # ================================================================
    improvement = (
        (result["local"]["avg_test_mae"] - result["fedavg"]["global_mae"])
        / result["local"]["avg_test_mae"] * 100
    )
    result["improvement_pct"] = improvement
    print(f"  [Seed {seed}] FL improvement: {improvement:+.1f}%")

    return result


# ================================================================
# Results I/O
# ================================================================

def load_completed_seeds(results_dir: str) -> Dict[int, dict]:
    """Load all completed seed results from the results directory."""
    completed = {}
    json_dir = os.path.join(results_dir, "per_seed")
    if os.path.isdir(json_dir):
        for fname in os.listdir(json_dir):
            if fname.startswith("seed_") and fname.endswith(".json"):
                path = os.path.join(json_dir, fname)
                with open(path) as f:
                    data = json.load(f)
                completed[data["seed"]] = data
    return completed


def save_seed_result(result: dict, results_dir: str):
    """Save a single seed result to JSON."""
    json_dir = os.path.join(results_dir, "per_seed")
    os.makedirs(json_dir, exist_ok=True)
    path = os.path.join(json_dir, f"seed_{result['seed']:04d}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def write_csv_summary(completed: Dict[int, dict], results_dir: str):
    """Write a CSV summary of all completed seeds."""
    csv_path = os.path.join(results_dir, "random100_results.csv")

    rows = []
    for seed in sorted(completed.keys()):
        r = completed[seed]
        row = {
            "seed": seed,
            "num_clients": r.get("num_clients", 5),
            "fedavg_global_mae": r.get("fedavg", {}).get("global_mae"),
            "fedavg_global_rmse": r.get("fedavg", {}).get("global_rmse"),
            "fedavg_best_mae": r.get("fedavg", {}).get("best_mae"),
            "fedavg_rounds": r.get("fedavg", {}).get("num_rounds_actual"),
            "fedavg_time_s": r.get("fedavg", {}).get("time_seconds"),
            "local_avg_mae": r.get("local", {}).get("avg_test_mae"),
            "local_std_mae": r.get("local", {}).get("std_test_mae"),
            "local_best_mae": r.get("local", {}).get("best_test_mae"),
            "local_worst_mae": r.get("local", {}).get("worst_test_mae"),
            "local_time_s": r.get("local", {}).get("time_seconds"),
            "improvement_pct": r.get("improvement_pct"),
        }
        # Per-client MAEs
        fedavg_pc = r.get("fedavg", {}).get("per_client_mae", [])
        local_pc = r.get("local", {}).get("per_client_mae", [])
        for i in range(5):
            row[f"fedavg_client{i}_mae"] = fedavg_pc[i] if i < len(fedavg_pc) else None
            row[f"local_client{i}_mae"] = local_pc[i] if i < len(local_pc) else None

        # Fairness
        fairness = r.get("fedavg", {}).get("fairness", {})
        row["fedavg_mae_std"] = fairness.get("mae_std")
        row["fedavg_mae_gap"] = fairness.get("mae_gap")
        row["fedavg_mae_cv"] = fairness.get("mae_cv")

        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV summary saved to {csv_path} ({len(rows)} seeds)")


def print_analysis(completed: Dict[int, dict]):
    """Print statistical analysis of completed results."""
    if not completed:
        print("  No completed results to analyze.")
        return

    n = len(completed)
    fedavg_maes = [r["fedavg"]["global_mae"] for r in completed.values()]
    local_maes = [r["local"]["avg_test_mae"] for r in completed.values()]
    improvements = [r["improvement_pct"] for r in completed.values()]

    print(f"\n{'='*70}")
    print(f"  Statistical Analysis — {n} Random Seeds")
    print(f"{'='*70}")

    print(f"\n  FedAvg Global MAE:")
    print(f"    Mean:   {np.mean(fedavg_maes):.6f}")
    print(f"    Std:    {np.std(fedavg_maes):.6f}")
    print(f"    Min:    {np.min(fedavg_maes):.6f}")
    print(f"    Max:    {np.max(fedavg_maes):.6f}")
    print(f"    Median: {np.median(fedavg_maes):.6f}")

    print(f"\n  Local-only Avg MAE:")
    print(f"    Mean:   {np.mean(local_maes):.6f}")
    print(f"    Std:    {np.std(local_maes):.6f}")
    print(f"    Min:    {np.min(local_maes):.6f}")
    print(f"    Max:    {np.max(local_maes):.6f}")
    print(f"    Median: {np.median(local_maes):.6f}")

    print(f"\n  FL Improvement (%):")
    print(f"    Mean:   {np.mean(improvements):+.2f}%")
    print(f"    Std:    {np.std(improvements):.2f}%")
    print(f"    Min:    {np.min(improvements):+.2f}%")
    print(f"    Max:    {np.max(improvements):+.2f}%")
    print(f"    Median: {np.median(improvements):+.2f}%")

    # Win rate
    wins = sum(1 for imp in improvements if imp > 0)
    print(f"\n  FL wins: {wins}/{n} ({wins/n*100:.0f}%)")

    # Statistical tests (if scipy available)
    try:
        from scipy import stats

        # Paired t-test
        t_stat, p_value_t = stats.ttest_rel(local_maes, fedavg_maes)
        print(f"\n  Paired t-test (Local vs FedAvg MAE):")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    p-value:     {p_value_t:.2e}")
        print(f"    Significant: {'Yes (p<0.05)' if p_value_t < 0.05 else 'No'}")

        # Wilcoxon signed-rank test
        w_stat, p_value_w = stats.wilcoxon(local_maes, fedavg_maes)
        print(f"\n  Wilcoxon signed-rank test:")
        print(f"    W-statistic: {w_stat:.4f}")
        print(f"    p-value:     {p_value_w:.2e}")
        print(f"    Significant: {'Yes (p<0.05)' if p_value_w < 0.05 else 'No'}")

        # Effect size (Cohen's d)
        diff = np.array(local_maes) - np.array(fedavg_maes)
        d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        print(f"\n  Effect size (Cohen's d): {d:.4f}")
        if abs(d) < 0.2:
            print(f"    Interpretation: Negligible")
        elif abs(d) < 0.5:
            print(f"    Interpretation: Small")
        elif abs(d) < 0.8:
            print(f"    Interpretation: Medium")
        else:
            print(f"    Interpretation: Large")

    except ImportError:
        print("\n  [scipy not available — skipping statistical tests]")
        print("  Install: pip install scipy")

    print(f"\n{'='*70}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="100× Random Seed Experiment: FedAvg vs Local-only"
    )
    parser.add_argument("--start", type=int, default=0,
                        help="Starting seed index (default: 0)")
    parser.add_argument("--end", type=int, default=100,
                        help="Ending seed index exclusive (default: 100)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu | cuda | mps | auto")
    parser.add_argument("--rounds", type=int, default=50,
                        help="FL rounds per seed (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode: 3 seeds, 5 rounds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run, don't execute")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from completed seeds (default: True)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results")
    parser.add_argument("--results-dir", type=str,
                        default="results/random100",
                        help="Directory for results output")
    args = parser.parse_args()

    # Resolve device
    import torch
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Quick mode
    if args.quick:
        args.start = 0
        args.end = 3
        args.rounds = 5

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing results
    completed = load_completed_seeds(results_dir)

    if args.analyze_only:
        print_analysis(completed)
        write_csv_summary(completed, results_dir)
        return

    # Determine seeds to run
    all_seeds = list(range(args.start, args.end))
    if args.resume:
        seeds_to_run = [s for s in all_seeds if s not in completed]
    else:
        seeds_to_run = all_seeds

    total = len(seeds_to_run)

    print("=" * 70)
    print("  100× Random Seed Experiment — FedAvg vs Local-only")
    print(f"  Seeds:      {args.start}..{args.end-1} ({len(all_seeds)} total)")
    print(f"  To run:     {total} (skipping {len(all_seeds)-total} completed)")
    print(f"  Rounds:     {args.rounds}")
    print(f"  Clients:    5")
    print(f"  Device:     {device}")
    print(f"  Results:    {results_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n  [DRY RUN] Would run seeds:", seeds_to_run)
        return

    if total == 0:
        print("\n  All seeds already completed!")
        print_analysis(completed)
        write_csv_summary(completed, results_dir)
        return

    # ---- Main Loop ----
    times = []
    start_time = time.time()

    for idx, seed in enumerate(seeds_to_run):
        seed_start = time.time()
        print(f"\n{'='*70}")
        print(f"  Seed {seed} ({idx+1}/{total})")
        if times:
            avg_time = np.mean(times)
            remaining = avg_time * (total - idx)
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"  ETA: {eta.strftime('%Y-%m-%d %H:%M')} "
                  f"(~{remaining/3600:.1f}h remaining)")
        print(f"{'='*70}")

        try:
            result = run_single_seed(
                seed=seed,
                config=DEFAULT_CONFIG,
                device=device,
                rounds=args.rounds,
            )

            # Save immediately (resume-friendly)
            save_seed_result(result, results_dir)
            completed[seed] = result

            # Update CSV after each seed
            write_csv_summary(completed, results_dir)

        except Exception as e:
            print(f"\n  *** ERROR at seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            # Save error record
            error_result = {"seed": seed, "error": str(e)}
            save_seed_result(error_result, results_dir)
            continue

        seed_time = time.time() - seed_start
        times.append(seed_time)
        print(f"\n  Seed {seed} completed in {seed_time/60:.1f} min")

        # Running stats every 5 seeds
        if (idx + 1) % 5 == 0:
            valid_completed = {
                k: v for k, v in completed.items()
                if "fedavg" in v and "local" in v
            }
            if valid_completed:
                imps = [v["improvement_pct"] for v in valid_completed.values()]
                print(f"\n  --- Running stats ({len(valid_completed)} seeds) ---")
                print(f"    Mean improvement: {np.mean(imps):+.2f}%")
                print(f"    Win rate: {sum(1 for i in imps if i > 0)}/{len(imps)}")

    # ---- Final Analysis ----
    total_time = time.time() - start_time
    print(f"\n\n  Total time: {total_time/3600:.1f} hours")

    valid_completed = {
        k: v for k, v in completed.items()
        if "fedavg" in v and "local" in v
    }
    print_analysis(valid_completed)
    write_csv_summary(valid_completed, results_dir)


if __name__ == "__main__":
    main()
