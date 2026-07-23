"""
Client-local held-out MNAR experiment.

This runner is a stricter validation variant of run_mnar_experiment.py:

    1. Split the 2500 VitalDB cases into 5 clients.
    2. Within each client, split cases into train/test.
    3. Apply the same client-specific MNAR mechanism separately to train
       and test cases.
    4. Train Local/FedAvg/FedProx/Fed-CA using only client train cases.
    5. Evaluate each final model on that client's held-out test MNAR holes.

The goal is not to replace the main reconstruction protocol. It adds a
case-level hold-out check while preserving the personalized-FL objective:
each personalized model is tested on its own client's distribution.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_mnar_experiment import (
    DEFAULT_CONFIG,
    atomic_write_json,
    compute_mnar_metrics,
    resolve_device,
    set_seed,
)
from src.data.dataset import TimeSeriesDataset
from src.data.mnar_masking import apply_mnar_logit, apply_mnar_quantile
from src.data.vitaldb_loader import load_from_local_tensor
from src.federation.saits_client import SAITSClient
from src.federation.saits_server import SAITSFederatedServer
from src.models.saits_model import FederatedSAITS


def parse_args():
    p = argparse.ArgumentParser(
        description="Client-local held-out MNAR evaluation for Fed-SAITS-CA"
    )
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
                   default=[0, 2, 6],
                   help="Default: 0=HR, 2=NIBP_SBP, 6=SpO2")
    p.add_argument("--test-size", type=float, default=0.1,
                   help="Per-client held-out case fraction")
    p.add_argument("--corr-fraction", type=float, default=0.25,
                   help="Logit MNAR correlated-feature fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--ca-scale-factor", type=float, default=0.5,
                   help="Default is our final beta=0.5 setting")
    p.add_argument("--ca-tau", type=float, default=1.0)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--fail-if-output-exists", action="store_true")

    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--skip-centralized", action="store_true")
    p.add_argument("--skip-fedavg", action="store_true")
    p.add_argument("--skip-fedprox", action="store_true")
    p.add_argument("--skip-fed-ca", action="store_true")
    p.add_argument("--only-fed-ca", action="store_true")
    p.add_argument("--only-core", action="store_true",
                   help="Run only Local, FedAvg, and Fed-CA")
    return p.parse_args()


def _rho_tag(rho: float) -> str:
    return str(rho).replace(".", "p")


def resolve_output_path(args) -> str:
    if args.output_path:
        return args.output_path
    os.makedirs("logs/saits_mnar", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    m = "q" if args.mnar_method == "quantile" else "l"
    fname = (
        f"heldout_sf{_rho_tag(args.ca_scale_factor)}_"
        f"{args.scenario}_{m}_rho{_rho_tag(args.missing_rate)}_"
        f"{timestamp}.json"
    )
    return os.path.join("logs/saits_mnar", fname)


def _apply_mnar_for_client(
    X: np.ndarray,
    masks: np.ndarray,
    client_id: int,
    scenario: str,
    mnar_method: str,
    target_features: list[int],
    missing_rate: float,
    corr_fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, str, float]]]:
    """Apply the same S1/S4/S2/S3 MNAR rule used in the main runner."""
    new_masks = masks.copy()
    configs = []

    for f in target_features:
        if scenario == "S1":
            direction = "left" if client_id == 0 else "right"
            rho_k = missing_rate
            logit_seed = seed + client_id * 1000 + f
            quantile_seed = seed + client_id
        elif scenario == "S2":
            if client_id == 0:
                direction = "left"
                rho_k = missing_rate
            else:
                direction = "right"
                rho_k = _sample_grid_rate(
                    seed=seed,
                    feature_idx=f,
                    client_id=client_id,
                    exclude=1.0 - missing_rate,
                )
            logit_seed = seed + client_id * 1000 + f
            quantile_seed = seed + client_id
        elif scenario == "S3":
            direction = "left"
            if client_id == 0:
                rho_k = missing_rate
            else:
                rho_k = _sample_grid_rate(
                    seed=seed + 17,
                    feature_idx=f,
                    client_id=client_id,
                    include=missing_rate,
                )
            logit_seed = seed + client_id * 1000 + f
            quantile_seed = seed + client_id
        else:
            direction = "left"
            rho_k = missing_rate
            # S4 logit intentionally uses the same seed across clients.
            logit_seed = seed + f
            quantile_seed = seed + client_id

        configs.append((f, direction, rho_k))

        if mnar_method == "quantile":
            new_masks = apply_mnar_quantile(
                X,
                new_masks,
                feature_idx=f,
                missing_rate=rho_k,
                direction=direction,
                seed=quantile_seed,
            )
        else:
            new_masks = apply_mnar_logit(
                X,
                new_masks,
                feature_idx=f,
                missing_rate=rho_k,
                corr_fraction=corr_fraction,
                seed=logit_seed,
            )

    return new_masks, configs


def _sample_grid_rate(
    seed: int,
    feature_idx: int,
    client_id: int,
    exclude: float | None = None,
    include: float | None = None,
) -> float:
    grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
    if include is not None and not np.any(np.isclose(grid, include)):
        grid = np.append(grid, float(include))
    if exclude is not None:
        keep = ~np.isclose(grid, float(exclude), atol=1e-8)
        if keep.any():
            grid = grid[keep]
    rng = np.random.RandomState(seed + feature_idx * 1009 + client_id * 9176)
    return float(rng.choice(grid))


def _build_split_record(
    X: np.ndarray,
    orig_masks: np.ndarray,
    client_id: int,
    scenario: str,
    mnar_method: str,
    target_features: list[int],
    missing_rate: float,
    corr_fraction: float,
    seed: int,
    feature_names: list[str],
    make_dataset: bool,
) -> dict:
    final_masks, configs = _apply_mnar_for_client(
        X=X,
        masks=orig_masks,
        client_id=client_id,
        scenario=scenario,
        mnar_method=mnar_method,
        target_features=target_features,
        missing_rate=missing_rate,
        corr_fraction=corr_fraction,
        seed=seed,
    )

    observed = X.copy()
    observed[np.isnan(observed)] = 0.0
    observed = observed * final_masks

    eval_mask = (
        (orig_masks > 0.5) & (final_masks < 0.5)
    ).astype(np.float32)

    record = {
        "ground_truth": X,
        "observed_data": observed,
        "orig_masks": orig_masks,
        "masks": final_masks,
        "eval_mask": eval_mask,
        "num_eval_points": int(eval_mask.sum()),
        "mnar_config": {
            "scenario": scenario,
            "method": mnar_method,
            "feature_configs": configs,
            "missing_rate": missing_rate,
            "corr_fraction": corr_fraction,
        },
    }

    if make_dataset:
        record["dataset"] = TimeSeriesDataset(
            data=observed,
            ground_truth=X,
            masks=final_masks,
            eval_masks=np.zeros_like(final_masks),
            feature_names=feature_names,
        )

    return record


def prepare_client_heldout_data(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    scenario: str,
    mnar_method: str,
    target_features: list[int],
    missing_rate: float,
    test_size: float,
    corr_fraction: float,
    seed: int,
    num_clients: int = 5,
) -> list[dict]:
    """
    Split global cases into clients, then each client into train/test.

    MNAR masks are generated separately for train and test. This prevents
    test cases from influencing the training masks or CA fingerprints.
    """
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(ground_truth.shape[0])
    client_splits = np.array_split(all_indices, num_clients)

    client_list = []
    for cid, client_idx in enumerate(client_splits):
        local_rng = np.random.RandomState(seed + 7919 + cid)
        local_perm = local_rng.permutation(client_idx)
        n_test = max(1, int(round(len(local_perm) * test_size)))
        n_test = min(n_test, len(local_perm) - 1)
        test_idx = np.sort(local_perm[:n_test])
        train_idx = np.sort(local_perm[n_test:])

        train = _build_split_record(
            X=ground_truth[train_idx].copy(),
            orig_masks=masks[train_idx].copy(),
            client_id=cid,
            scenario=scenario,
            mnar_method=mnar_method,
            target_features=target_features,
            missing_rate=missing_rate,
            corr_fraction=corr_fraction,
            seed=seed,
            feature_names=feature_names,
            make_dataset=True,
        )
        test = _build_split_record(
            X=ground_truth[test_idx].copy(),
            orig_masks=masks[test_idx].copy(),
            client_id=cid,
            scenario=scenario,
            mnar_method=mnar_method,
            target_features=target_features,
            missing_rate=missing_rate,
            corr_fraction=corr_fraction,
            seed=seed,
            feature_names=feature_names,
            make_dataset=False,
        )

        print(
            f"  Client {cid}: train={len(train_idx)}, test={len(test_idx)}, "
            f"train_eval_pts={train['num_eval_points']}, "
            f"test_eval_pts={test['num_eval_points']}"
        )

        client_list.append({
            "client_id": cid,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "dataset": train["dataset"],
            "train": train,
            "test": test,
        })

    return client_list


def _evaluate_client_model_on_heldout(client, client_record: dict) -> dict:
    test = client_record["test"]
    imputed = client.impute(
        observed=test["observed_data"],
        masks=test["masks"],
    )
    metrics = compute_mnar_metrics(
        imputed_data=imputed,
        ground_truth=test["ground_truth"],
        eval_mask=test["eval_mask"],
    )
    return {
        "client_id": client_record["client_id"],
        "num_train_samples": len(client_record["train_indices"]),
        "num_test_samples": len(client_record["test_indices"]),
        "num_eval_points": metrics["num_eval_points"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }


def _summarize_client_metrics(client_metrics: list[dict]) -> dict:
    maes = [m["mae"] for m in client_metrics]
    rmses = [m["rmse"] for m in client_metrics]
    return {
        "client_metrics": client_metrics,
        "mean_mae": float(np.nanmean(maes)),
        "std_mae_clients": float(np.nanstd(maes)),
        "mean_rmse": float(np.nanmean(rmses)),
        "std_rmse_clients": float(np.nanstd(rmses)),
    }


def run_federated_heldout(
    client_list: list[dict],
    config: dict,
    device: str,
    method: str,
) -> dict:
    clients = []
    for cd in client_list:
        clients.append(SAITSClient(
            client_id=cd["client_id"],
            train_data=cd["dataset"],
            val_data=None,
            config=config,
            device=device,
        ))

    server = SAITSFederatedServer(
        clients=clients,
        test_data=None,
        config=config,
        device=device,
    )

    print(f"  Training {method} on client train splits...")
    train_results = server.train()

    print("  Evaluating each model on its own held-out client test split...")
    client_metrics = []
    for i, cd in enumerate(client_list):
        cid = cd["client_id"]
        if server._use_pd:
            if server._use_ca and cid in server._personalized_params:
                clients[i].download_global_layers(server._personalized_params[cid])
            else:
                clients[i].download_global_layers(server.global_params)
        elif hasattr(server, "_personalized_params") and cid in server._personalized_params:
            clients[i].download_global_model(server._personalized_params[cid])
        elif server.global_params is not None:
            clients[i].download_global_model(server.global_params)

        metrics = _evaluate_client_model_on_heldout(clients[i], cd)
        client_metrics.append(metrics)
        print(
            f"    Client {cid}: MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, eval_pts={metrics['num_eval_points']}"
        )

    summary = _summarize_client_metrics(client_metrics)
    summary["num_rounds"] = len(train_results.get("history", []))
    summary["history"] = train_results.get("history", [])
    return summary


def run_local_heldout(client_list: list[dict], config: dict, device: str) -> dict:
    fed_cfg = config.get("federation", {})
    saits_cfg = config.get("saits", {})
    total_epochs = fed_cfg.get("rounds", 50) * saits_cfg.get("local_epochs", 5)

    client_metrics = []
    for cd in client_list:
        cid = cd["client_id"]
        print(f"  Local client {cid}: train={len(cd['train_indices'])}, "
              f"test={len(cd['test_indices'])}, epochs={total_epochs}")

        client = SAITSClient(
            client_id=cid,
            train_data=cd["dataset"],
            val_data=None,
            config=config,
            device=device,
        )
        client.model.set_training_params(
            epochs=total_epochs,
            patience=min(20, max(total_epochs - 1, 1)),
        )
        client.local_train()

        metrics = _evaluate_client_model_on_heldout(client, cd)
        client_metrics.append(metrics)
        print(
            f"    Client {cid}: MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, eval_pts={metrics['num_eval_points']}"
        )

    return _summarize_client_metrics(client_metrics)


def run_centralized_heldout(
    client_list: list[dict],
    config: dict,
    device: str,
) -> dict:
    fed_cfg = config.get("federation", {})
    saits_cfg = config.get("saits", {})
    total_epochs = fed_cfg.get("rounds", 50) * saits_cfg.get("local_epochs", 5)

    all_observed = np.concatenate(
        [cd["train"]["observed_data"] for cd in client_list], axis=0
    )
    all_masks = np.concatenate(
        [cd["train"]["masks"] for cd in client_list], axis=0
    )
    all_gt = np.concatenate(
        [cd["train"]["ground_truth"] for cd in client_list], axis=0
    )
    N, T, D = all_observed.shape

    print(f"  Centralized train pool: N={N}, T={T}, D={D}, epochs={total_epochs}")
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

    val_size = max(1, int(N * 0.1))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    model.fit(
        observed=all_observed[tr_idx],
        masks=all_masks[tr_idx],
        val_observed=all_observed[val_idx],
        val_masks=all_masks[val_idx],
        val_ground_truth=all_gt[val_idx],
    )

    client_metrics = []
    for cd in client_list:
        test = cd["test"]
        imputed = model.impute(
            observed=test["observed_data"],
            masks=test["masks"],
        )
        metrics = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=test["ground_truth"],
            eval_mask=test["eval_mask"],
        )
        item = {
            "client_id": cd["client_id"],
            "num_train_samples": len(cd["train_indices"]),
            "num_test_samples": len(cd["test_indices"]),
            "num_eval_points": metrics["num_eval_points"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        }
        client_metrics.append(item)
        print(
            f"    Client {cd['client_id']}: MAE={item['mae']:.6f}, "
            f"RMSE={item['rmse']:.6f}, eval_pts={item['num_eval_points']}"
        )

    return _summarize_client_metrics(client_metrics)


def run_single(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    args,
    seed: int,
    device: str,
    checkpoint_root: str,
) -> list[dict]:
    set_seed(seed)
    print("\n" + "=" * 60)
    print("  Client-local held-out MNAR experiment")
    print(f"  scenario={args.scenario}, method={args.mnar_method}, "
          f"rho={args.missing_rate}, seed={seed}")
    print(f"  test_size={args.test_size}, ca_scale_factor={args.ca_scale_factor}")
    print("=" * 60)

    print("\n[Step 1] Client split + per-client train/test split + MNAR...")
    client_list = prepare_client_heldout_data(
        ground_truth=ground_truth,
        masks=masks,
        feature_names=feature_names,
        scenario=args.scenario,
        mnar_method=args.mnar_method,
        target_features=args.target_features,
        missing_rate=args.missing_rate,
        test_size=args.test_size,
        corr_fraction=args.corr_fraction,
        seed=seed,
        num_clients=5,
    )

    results = []

    only = args.only_fed_ca
    skip_local = args.skip_local or only
    skip_centralized = args.skip_centralized or only or args.only_core
    skip_fedavg = args.skip_fedavg or only
    skip_fedprox = args.skip_fedprox or only or args.only_core
    skip_fed_ca = args.skip_fed_ca

    base_meta = {
        "seed": seed,
        "scenario": args.scenario,
        "mnar_method": args.mnar_method,
        "missing_rate": args.missing_rate,
        "target_features": args.target_features,
        "test_size": args.test_size,
        "corr_fraction": args.corr_fraction,
        "num_clients": 5,
        "train_samples_per_client": [len(cd["train_indices"]) for cd in client_list],
        "test_samples_per_client": [len(cd["test_indices"]) for cd in client_list],
    }

    if not skip_local:
        set_seed(seed)
        print(f"\n[Step 2a] Local-only held-out test")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "local"
        )
        out = run_local_heldout(client_list, cfg, device)
        results.append({**base_meta, "method": "local", **out})

    if not skip_centralized:
        set_seed(seed)
        print(f"\n[Step 2b] Centralized pooled-train held-out test")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "centralized"
        )
        out = run_centralized_heldout(client_list, cfg, device)
        results.append({**base_meta, "method": "centralized", **out})

    if not skip_fedavg:
        set_seed(seed)
        print(f"\n[Step 2c] FedAvg held-out test")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["federation"]["aggregation"] = "fedavg"
        cfg["federation"]["mu"] = 0.0
        cfg["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fedavg"
        )
        out = run_federated_heldout(client_list, cfg, device, method="fedavg")
        results.append({**base_meta, "method": "fedavg", **out})

    if not skip_fedprox:
        set_seed(seed)
        print(f"\n[Step 2d] FedProx held-out test (mu={args.mu})")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["federation"]["aggregation"] = "fedprox"
        cfg["federation"]["mu"] = args.mu
        cfg["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fedprox"
        )
        out = run_federated_heldout(client_list, cfg, device, method="fedprox")
        results.append({**base_meta, "method": "fedprox", "mu": args.mu, **out})

    if not skip_fed_ca:
        set_seed(seed)
        print(f"\n[Step 2e] Fed-CA held-out test "
              f"(scale_factor={args.ca_scale_factor})")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["federation"]["aggregation"] = "fed_ca"
        cfg["federation"]["ca_tau"] = args.ca_tau
        cfg["federation"]["ca_scale_factor"] = args.ca_scale_factor
        cfg["federation"]["mu"] = 0.0
        cfg["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fed_ca"
        )
        out = run_federated_heldout(client_list, cfg, device, method="fed_ca")
        results.append({
            **base_meta,
            "method": "fed_ca",
            "ca_tau": args.ca_tau,
            "ca_scale_factor": args.ca_scale_factor,
            **out,
        })

    return results


def print_summary(results: list[dict], seeds: list[int]):
    methods = sorted(set(r["method"] for r in results))
    print("\n--- Held-out Client-Local Summary ---")
    print(f"{'Seed':>6} | ", end="")
    for m in methods:
        print(f"{m+' MAE':>14} | {m+' RMSE':>14} | ", end="")
    print()
    print("-" * (8 + len(methods) * 34))

    for seed in seeds:
        print(f"{seed:>6} | ", end="")
        for m in methods:
            hits = [r for r in results if r["seed"] == seed and r["method"] == m]
            if hits:
                print(f"{hits[0]['mean_mae']:>14.6f} | "
                      f"{hits[0]['mean_rmse']:>14.6f} | ", end="")
            else:
                print(f"{'N/A':>14} | {'N/A':>14} | ", end="")
        print()

    if len(seeds) > 1:
        print("-" * (8 + len(methods) * 34))
        print(f"{'Mean':>6} | ", end="")
        for m in methods:
            vals = [r for r in results if r["method"] == m]
            if vals:
                print(f"{np.nanmean([r['mean_mae'] for r in vals]):>14.6f} | "
                      f"{np.nanmean([r['mean_rmse'] for r in vals]):>14.6f} | ", end="")
            else:
                print(f"{'N/A':>14} | {'N/A':>14} | ", end="")
        print()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    result_path = resolve_output_path(args)
    result_stem = os.path.splitext(os.path.basename(result_path))[0]
    checkpoint_root = os.path.join("checkpoints", "saits_mnar_heldout", result_stem)

    print("[1/2] Loading VitalDB tensors...")
    ground_truth, _, masks, feature_names, _ = load_from_local_tensor(
        args.tensor_dir,
        normalize=True,
    )
    N, T, D = ground_truth.shape
    print(f"  Loaded: N={N}, T={T}, D={D}")
    print(f"  Target MNAR features: {[feature_names[f] for f in args.target_features]}")
    print(f"  Checkpoints: {checkpoint_root}")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    all_results = []
    start = time.time()

    print(f"\n[2/2] Running {len(seeds)} seed(s)...")
    for seed in seeds:
        all_results.extend(run_single(
            ground_truth=ground_truth,
            masks=masks,
            feature_names=feature_names,
            args=args,
            seed=seed,
            device=device,
            checkpoint_root=checkpoint_root,
        ))

    total_time = time.time() - start
    print_summary(all_results, seeds)

    atomic_write_json(
        path=result_path,
        payload={
            "experiment": "mnar_client_local_heldout",
            "scenario": args.scenario,
            "mnar_method": args.mnar_method,
            "missing_rate": args.missing_rate,
            "target_features": args.target_features,
            "test_size": args.test_size,
            "corr_fraction": args.corr_fraction,
            "ca_scale_factor": args.ca_scale_factor,
            "mu": args.mu,
            "num_clients": 5,
            "total_samples": N,
            "seeds": seeds,
            "total_time_sec": total_time,
            "evaluation_note": (
                "Each client is split into train/test cases first. MNAR holes "
                "are generated separately on train and test. Training and CA "
                "fingerprints use train cases only. Metrics are computed only "
                "on held-out test positions that were originally observed and "
                "then artificially MNAR-masked."
            ),
            "results": all_results,
        },
        fail_if_exists=args.fail_if_output_exists,
    )
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
