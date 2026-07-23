#!/usr/bin/env python3
"""
Experiment Runner — Systematic evaluation of FL under client heterogeneity.

Research Question:
    FL assumes client data is roughly equivalent (IID). In clinical settings,
    hospitals differ in equipment, patient populations, and data quality.
    How does this heterogeneity affect federated time-series imputation?

Experiment Matrix:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Phase 1: Baselines                                                 │
    │   E0  centralized       —           Upper bound (no FL)            │
    │   E1  local-only        equipment   Lower bound (no collaboration) │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Phase 2: IID vs Non-IID  (all FedAvg)                              │
    │   E2  federated         random      IID baseline                   │
    │   E3  federated         equipment   Equipment heterogeneity        │
    │   E4  federated         missing_rate Data quality heterogeneity    │
    │   E5  federated         acuity      Patient population heterog.    │
    │   E6  federated         duration    Surgery complexity heterog.    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Phase 3: Aggregation strategies (equipment split)                   │
    │   E7  federated+FedProx  equipment  Can FedProx help?             │
    │   E8  federated+FedAdam  equipment  Can FedAdam help?             │
    │   E9  federated+FedProx  missing_rate Robustness check            │
    │   E10 federated+FedAdam  missing_rate Robustness check            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Phase 4: Ablation (equipment split, FedAvg)                        │
    │   E11 federated         equipment   3 clients                      │
    │   E12 federated         equipment   10 clients                     │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # Run all experiments
    python experiments/run_all.py

    # Run a specific phase
    python experiments/run_all.py --phase 1

    # Run specific experiments by ID
    python experiments/run_all.py --experiments E0 E2 E3

    # Dry-run: print commands only
    python experiments/run_all.py --dry-run

    # Quick mode: reduce rounds for testing
    python experiments/run_all.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Experiment:
    """Definition of a single experiment run."""
    exp_id: str
    name: str
    mode: str                     # centralized | local | federated
    split_strategy: str           # random | equipment | missing_rate | ...
    aggregation: str = "fedavg"   # fedavg | fedprox | fedadam
    num_clients: int = 5
    rounds: int = 50              # 50 rounds sufficient per convergence analysis
    phase: int = 1
    description: str = ""


# ================================================================
# Experiment Definitions
# ================================================================

EXPERIMENTS = [
    # --- Phase 1: Baselines ---
    Experiment(
        exp_id="E0", name="centralized_baseline",
        mode="centralized", split_strategy="random",
        phase=1, description="Upper bound: all data trained together",
    ),
    Experiment(
        exp_id="E1", name="local_equipment",
        mode="local", split_strategy="equipment",
        phase=1, description="Lower bound: local training only, equipment split",
    ),

    # --- Phase 2: IID vs Non-IID (all FedAvg) ---
    Experiment(
        exp_id="E2", name="fedavg_random",
        mode="federated", split_strategy="random", aggregation="fedavg",
        phase=2, description="FL-IID baseline: random equal split",
    ),
    Experiment(
        exp_id="E3", name="fedavg_equipment",
        mode="federated", split_strategy="equipment", aggregation="fedavg",
        phase=2, description="FL-NonIID: equipment-based split",
    ),
    Experiment(
        exp_id="E4", name="fedavg_missing_rate",
        mode="federated", split_strategy="missing_rate", aggregation="fedavg",
        phase=2, description="FL-NonIID: missing-rate-based split",
    ),
    Experiment(
        exp_id="E5", name="fedavg_acuity",
        mode="federated", split_strategy="acuity", aggregation="fedavg",
        phase=2, description="FL-NonIID: patient acuity-based split",
    ),
    Experiment(
        exp_id="E6", name="fedavg_duration",
        mode="federated", split_strategy="duration", aggregation="fedavg",
        phase=2, description="FL-NonIID: surgery duration-based split",
    ),

    # --- Phase 3: Aggregation strategies ---
    Experiment(
        exp_id="E7", name="fedprox_equipment",
        mode="federated", split_strategy="equipment", aggregation="fedprox",
        phase=3, description="FedProx with equipment split",
    ),
    Experiment(
        exp_id="E8", name="fedadam_equipment",
        mode="federated", split_strategy="equipment", aggregation="fedadam",
        phase=3, description="FedAdam with equipment split",
    ),
    Experiment(
        exp_id="E9", name="fedprox_missing_rate",
        mode="federated", split_strategy="missing_rate", aggregation="fedprox",
        phase=3, description="FedProx with missing-rate split",
    ),
    Experiment(
        exp_id="E10", name="fedadam_missing_rate",
        mode="federated", split_strategy="missing_rate", aggregation="fedadam",
        phase=3, description="FedAdam with missing-rate split",
    ),

    # --- Phase 4: Ablation — number of clients ---
    Experiment(
        exp_id="E11", name="fedavg_equipment_3clients",
        mode="federated", split_strategy="equipment", aggregation="fedavg",
        num_clients=3, phase=4,
        description="Equipment split with 3 clients",
    ),
    Experiment(
        exp_id="E12", name="fedavg_equipment_10clients",
        mode="federated", split_strategy="equipment", aggregation="fedavg",
        num_clients=10, phase=4,
        description="Equipment split with 10 clients",
    ),
]


def build_command(exp: Experiment, config_path: str, quick: bool = False) -> List[str]:
    """Build the CLI command for a single experiment."""
    cmd = [
        sys.executable, "main_saits.py",
        "--config", config_path,
        "--mode", exp.mode,
        "--split-strategy", exp.split_strategy,
        "--aggregation", exp.aggregation,
        "--num-clients", str(exp.num_clients),
        "--experiment-name", exp.name,
    ]
    if quick:
        cmd.extend(["--rounds", "5"])
    else:
        cmd.extend(["--rounds", str(exp.rounds)])
    return cmd


def run_experiment(
    exp: Experiment,
    config_path: str,
    results_dir: str,
    quick: bool = False,
    dry_run: bool = False,
) -> Optional[dict]:
    """Run a single experiment and collect results."""

    cmd = build_command(exp, config_path, quick=quick)
    cmd_str = " ".join(cmd)

    print(f"\n{'='*70}")
    print(f"  [{exp.exp_id}] {exp.name}")
    print(f"  {exp.description}")
    print(f"  mode={exp.mode}, split={exp.split_strategy}, "
          f"agg={exp.aggregation}, clients={exp.num_clients}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return None

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        elapsed = time.time() - start

        # Save stdout/stderr
        log_path = os.path.join(results_dir, f"{exp.exp_id}_{exp.name}.log")
        with open(log_path, "w") as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n")

        success = result.returncode == 0
        print(f"  {'✅ SUCCESS' if success else '❌ FAILED'} "
              f"({elapsed:.0f}s)")

        if not success:
            # Print last few lines of error
            err_lines = result.stderr.strip().split("\n")[-5:]
            for line in err_lines:
                print(f"    {line}")

        return {
            "exp_id": exp.exp_id,
            "name": exp.name,
            "success": success,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏰ TIMEOUT after {elapsed:.0f}s")
        return {
            "exp_id": exp.exp_id,
            "name": exp.name,
            "success": False,
            "elapsed_seconds": elapsed,
            "returncode": -1,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  💥 ERROR: {e}")
        return {
            "exp_id": exp.exp_id,
            "name": exp.name,
            "success": False,
            "elapsed_seconds": elapsed,
            "returncode": -2,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic FL experiments"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/saits_config.yaml",
        help="Base config YAML",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run only experiments from this phase (1-4)",
    )
    parser.add_argument(
        "--experiments", nargs="+", type=str, default=None,
        help="Run specific experiments by ID (e.g., E0 E2 E3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 rounds only (for testing pipeline)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="experiments/results",
        help="Directory for experiment logs",
    )
    args = parser.parse_args()

    # Filter experiments
    exps = EXPERIMENTS
    if args.phase is not None:
        exps = [e for e in exps if e.phase == args.phase]
    if args.experiments is not None:
        selected = set(args.experiments)
        exps = [e for e in exps if e.exp_id in selected]

    if not exps:
        print("No experiments selected!")
        return

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  FL Heterogeneity Experiment Suite")
    print(f"  {len(exps)} experiments to run")
    print(f"  Results: {results_dir}")
    print(f"  Quick mode: {args.quick}")
    print(f"{'='*70}")

    # Print experiment table
    print(f"\n{'ID':>4s}  {'Name':<30s}  {'Mode':<12s}  {'Split':<15s}  {'Agg':<8s}  {'K':>3s}")
    print("-" * 80)
    for e in exps:
        print(f"{e.exp_id:>4s}  {e.name:<30s}  {e.mode:<12s}  "
              f"{e.split_strategy:<15s}  {e.aggregation:<8s}  {e.num_clients:>3d}")

    # Run experiments
    all_results = []
    total_start = time.time()

    for idx, exp in enumerate(exps):
        result = run_experiment(
            exp,
            config_path=args.config,
            results_dir=results_dir,
            quick=args.quick,
            dry_run=args.dry_run,
        )
        if result is not None:
            all_results.append(result)

        # Cooling pause between experiments (important for M2 Air thermals)
        if not args.dry_run and idx < len(exps) - 1:
            cool_secs = 90
            print(f"\n  ❄️  Cooling pause: {cool_secs}s before next experiment...")
            time.sleep(cool_secs)

    total_time = time.time() - total_start

    # Save summary
    if all_results:
        summary = {
            "timestamp": timestamp,
            "total_time_seconds": total_time,
            "num_experiments": len(all_results),
            "num_success": sum(1 for r in all_results if r["success"]),
            "experiments": all_results,
        }
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  Experiment suite complete")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Success: {summary['num_success']}/{summary['num_experiments']}")
        print(f"  Summary: {summary_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
