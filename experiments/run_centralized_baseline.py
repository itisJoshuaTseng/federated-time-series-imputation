"""
Centralized Baseline — the "ceiling" reference for Fed-SAITS-CA.

Trains a single SAITS model on ALL 2500 VitalDB samples pooled together
(completely ignoring federation / data privacy), using the SAME per-client
MNAR masking used by run_mnar_experiment.py.  The goal is to produce an
upper bound: what SAITS could achieve if all hospital data were freely
shareable.

Evaluation is identical to run_mnar_experiment.py: MAE/RMSE computed only
on the MNAR holes (positions that were originally observed but were
artificially masked by the MNAR mechanism).  Per-client metrics are
reported so the comparison against federated per-client results is direct.

Usage:
    python experiments/run_centralized_baseline.py \
        --scenario S1 --mnar-method quantile --missing-rate 0.5 \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.vitaldb_loader import load_from_local_tensor
from src.models.saits_model import FederatedSAITS

# Reuse the MNAR data preparation from the federated experiment so the
# training data and eval masks are IDENTICAL to the federated runs.
from experiments.run_mnar_experiment import (
    DEFAULT_CONFIG,
    compute_mnar_metrics,
    prepare_client_data,
    resolve_device,
    set_seed,
    atomic_write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Centralized SAITS baseline (ceiling)")
    p.add_argument("--tensor-dir", type=str,
                   default="../2026_vitalDB/vitaldb_14feats_tensor_T300")
    p.add_argument(
        "--scenario",
        type=str,
        default="S1",
        choices=["S1", "S2", "S3", "S4"],
    )
    p.add_argument("--mnar-method", type=str, default="quantile",
                   choices=["quantile", "logit"])
    p.add_argument("--missing-rate", type=float, default=0.5)
    p.add_argument("--target-features", type=int, nargs="+",
                   default=[0, 2, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-path", type=str, default=None)
    # Matches federated total compute: rounds * local_epochs
    p.add_argument("--epochs", type=int, default=None,
                   help="Total training epochs. Default = fed_rounds * local_epochs.")
    return p.parse_args()


def run_centralized_single(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list,
    scenario: str,
    mnar_method: str,
    target_features: list,
    missing_rate: float,
    seed: int,
    device: str,
    total_epochs: int,
) -> dict:
    """
    Pool all 5 clients' (observed, train_masks) into one dataset, train a
    single SAITS on it, then evaluate on each client's MNAR holes.
    """
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"  CENTRALIZED baseline (ceiling)")
    print(f"  scenario={scenario}, method={mnar_method}, rho={missing_rate}, "
          f"seed={seed}")
    print(f"{'='*60}")

    # --- Same per-client MNAR allocation as federated runs ---
    client_list = prepare_client_data(
        ground_truth=ground_truth,
        masks=masks,
        feature_names=feature_names,
        scenario=scenario,
        mnar_method=mnar_method,
        target_features=target_features,
        missing_rate=missing_rate,
        seed=seed,
    )

    # --- Pool everything into one big training set ---
    all_observed = np.concatenate([cd["observed_data"] for cd in client_list], axis=0)
    all_masks = np.concatenate([cd["train_masks"] for cd in client_list], axis=0)
    all_gt = np.concatenate([cd["ground_truth"] for cd in client_list], axis=0)

    N, T, D = all_observed.shape
    print(f"  Pooled training set: N={N}, T={T}, D={D}")

    # --- Build a single SAITS model.  Train budget matches federated total:
    #     rounds * local_epochs.  This keeps compute fair. ---
    saits_cfg = DEFAULT_CONFIG["saits"]
    model = FederatedSAITS(
        num_features=D,
        seq_length=T,
        n_layers=saits_cfg["n_layers"],
        d_model=saits_cfg["d_model"],
        n_heads=saits_cfg["n_heads"],
        d_ffn=saits_cfg["d_ffn"],
        d_k=saits_cfg["d_k"],
        d_v=saits_cfg["d_v"],
        dropout=saits_cfg["dropout"],
        attn_dropout=saits_cfg["attn_dropout"],
        diagonal_attention_mask=saits_cfg["diagonal_attention_mask"],
        ORT_weight=saits_cfg["ORT_weight"],
        MIT_weight=saits_cfg["MIT_weight"],
        epochs=total_epochs,
        batch_size=saits_cfg["batch_size"],
        learning_rate=saits_cfg["learning_rate"],
        patience=min(20, max(total_epochs - 1, 1)),
        device=device,
    )

    # Small held-out slice for PyPOTS early-stopping (10%)
    val_size = max(1, int(N * 0.1))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    print(f"  Training SAITS on {len(tr_idx)} samples "
          f"(val={len(val_idx)}), epochs={total_epochs}...")
    t0 = time.time()
    model.fit(
        observed=all_observed[tr_idx],
        masks=all_masks[tr_idx],
        val_observed=all_observed[val_idx],
        val_masks=all_masks[val_idx],
        val_ground_truth=all_gt[val_idx],
    )
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s.")

    # --- Per-client imputation + MNAR-hole metrics ---
    client_metrics = []
    for cd in client_list:
        imputed = model.impute(
            observed=cd["observed_data"],
            masks=cd["train_masks"],
        )
        m = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=cd["ground_truth"],
            eval_mask=cd["eval_mask"],
        )
        client_metrics.append({
            "client_id": cd["client_id"],
            "num_eval_points": m["num_eval_points"],
            "mae": m["mae"],
            "rmse": m["rmse"],
        })
        print(f"    Client {cd['client_id']}: "
              f"MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, "
              f"eval_pts={m['num_eval_points']}")

    maes = [m["mae"] for m in client_metrics]
    rmses = [m["rmse"] for m in client_metrics]
    mean_mae = float(np.nanmean(maes))
    mean_rmse = float(np.nanmean(rmses))
    print(f"  Centralized mean MAE={mean_mae:.6f}, mean RMSE={mean_rmse:.6f}")

    return {
        "seed": seed,
        "scenario": scenario,
        "mnar_method": mnar_method,
        "missing_rate": missing_rate,
        "method": "centralized",
        "num_clients": 5,
        "samples_per_client": N // 5,
        "total_samples": N,
        "epochs": total_epochs,
        "train_time_sec": train_time,
        "client_metrics": client_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)

    seeds = args.seeds if args.seeds else [args.seed]

    # Default compute budget = federated total steps
    fed_cfg = DEFAULT_CONFIG["federation"]
    saits_cfg = DEFAULT_CONFIG["saits"]
    total_epochs = args.epochs or (
        fed_cfg["rounds"] * saits_cfg["local_epochs"]
    )

    print("[1/2] Loading data...")
    ground_truth, _, masks, feature_names, _ = \
        load_from_local_tensor(args.tensor_dir, normalize=True)
    print(f"  Loaded: {ground_truth.shape} (N, T, D)")

    all_results = []
    t_start = time.time()
    for seed in seeds:
        r = run_centralized_single(
            ground_truth=ground_truth,
            masks=masks,
            feature_names=feature_names,
            scenario=args.scenario,
            mnar_method=args.mnar_method,
            target_features=args.target_features,
            missing_rate=args.missing_rate,
            seed=seed,
            device=device,
            total_epochs=total_epochs,
        )
        all_results.append(r)

    total_time = time.time() - t_start

    # --- Aggregate over seeds ---
    mean_over_seeds = float(np.mean([r["mean_mae"] for r in all_results]))
    std_over_seeds = float(np.std([r["mean_mae"] for r in all_results]))

    payload = {
        "experiment": "centralized_ceiling_baseline",
        "scenario": args.scenario,
        "mnar_method": args.mnar_method,
        "missing_rate": args.missing_rate,
        "target_features": args.target_features,
        "seeds": seeds,
        "epochs": total_epochs,
        "total_time_sec": total_time,
        "evaluation_note": (
            "Single SAITS trained on all 2500 pooled samples (no federation). "
            "Metrics computed ONLY on MNAR holes, identical to "
            "run_mnar_experiment.py evaluation."
        ),
        "mean_mae_over_seeds": mean_over_seeds,
        "std_mae_over_seeds": std_over_seeds,
        "results": all_results,
    }

    # --- Output path ---
    if args.output_path:
        out_path = args.output_path
    else:
        log_dir = "logs/saits_mnar"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rho_tag = f"rho{str(args.missing_rate).replace('.', 'p')}"
        mm_tag = "q" if args.mnar_method == "quantile" else "l"
        seed_tag = f"seeds_{min(seeds)}-{max(seeds)}"
        filename = (
            f"centralized_ceiling_{args.scenario}_{mm_tag}_"
            f"{rho_tag}_{seed_tag}.json"
        )
        out_path = os.path.join(log_dir, filename)

    atomic_write_json(out_path, payload)
    print(f"\nSaved: {out_path}")
    print(f"Ceiling MAE over {len(seeds)} seeds: "
          f"{mean_over_seeds:.4f} ± {std_over_seeds:.4f}")


if __name__ == "__main__":
    main()
