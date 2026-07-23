"""
MNAR Heterogeneity Experiment Runner — Imputation Reconstruction Evaluation.

This script evaluates whether federated models can accurately reconstruct
values that were artificially removed by MNAR mechanisms on each client's
local data. The evaluation is ONLY on the newly added MNAR holes, NOT on
a separate test set or MCAR eval masking.

Key design:
    - No train/test split: all 2500 samples are distributed to 5 clients
    - Each client gets 500 samples
    - MNAR is applied per-client via HeterogeneousDataAllocator (S1-S4)
    - No additional MCAR eval masking (eval_masks = zeros)
    - Post-training evaluation: impute → compare only at MNAR hole positions
    - eval_mask = (observed_mask_original == 1) & (mnar_mask_added == 1)

Usage:
    # Single seed, S1 scenario
    python experiments/run_mnar_experiment.py --scenario S1 --seed 42

    # S2/S3/S4 scenarios
    python experiments/run_mnar_experiment.py --scenario S2 --seed 42
    python experiments/run_mnar_experiment.py --scenario S3 --seed 42
    python experiments/run_mnar_experiment.py --scenario S4 --seed 42

    # Multiple seeds
    python experiments/run_mnar_experiment.py --scenario S1 --seeds 0 1 2 3 4

    # Logit-based MNAR
    python experiments/run_mnar_experiment.py --mnar-method logit

    # Skip FedAvg or Local
    python experiments/run_mnar_experiment.py --skip-fedavg
    python experiments/run_mnar_experiment.py --skip-local

    # Fast FedICE-only check
    python experiments/run_mnar_experiment.py --scenario S2 --only-fedice --fedice-rounds 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.vitaldb_loader import load_from_local_tensor
from src.data.heterogeneous_allocator import HeterogeneousDataAllocator
from src.baselines import FedICEImputer


# ================================================================
# Default config
# ================================================================

DEFAULT_CONFIG = {
    "data": {
        "type": "vitaldb",
        "seq_length": 300,
        "num_features": 14,
        "missing_mechanism": "mnar",
        # No eval masking — we evaluate on MNAR holes only
        "eval_missing_rate": 0.0,
        "train_ratio": 1.0,  # use all data, no train/test split
        "normalize": True,
    },
    "federation": {
        "num_clients": 5,
        "rounds": 50,
        "frac_clients": 1.0,
        "aggregation": "fedavg",
        "mu": 0.01,
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
        "checkpoint_dir": "checkpoints/saits_mnar",
    },
    "logging": {
        "log_dir": "logs/saits_mnar",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="MNAR Imputation Reconstruction Experiment"
    )
    p.add_argument("--tensor-dir", type=str,
                    default="../2026_vitalDB/vitaldb_14feats_tensor_T300",
                    help="Path to pre-processed tensor directory")
    p.add_argument("--scenario", type=str, default="S1",
                    choices=["S1", "S2", "S3", "S4"],
                    help=("S1=Perfect, S2=Imperfect, "
                          "S3=One-sided, S4=None"))
    p.add_argument("--mnar-method", type=str, default="quantile",
                    choices=["quantile", "logit"],
                    help="MNAR masking method")
    p.add_argument("--missing-rate", type=float, default=0.5,
                    help="MNAR missing rate rho (0.3~0.7)")
    p.add_argument("--target-features", type=int, nargs="+",
                    default=[0, 2, 6],
                    help="Feature indices for MNAR (default: 0=HR, 2=NIBP_SBP, 6=SpO2)")
    p.add_argument("--num-clients", type=int, default=5,
                    help="Number of federated clients (default: 5)")
    p.add_argument("--client-ids-path", type=str, default=None,
                    help=("Optional .npy file assigning each sample to a fixed "
                          "client id. Useful for hospital-cluster splits."))
    p.add_argument("--seed", type=int, default=42,
                    help="Single seed (ignored if --seeds is used)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Multiple seeds for batch run")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mu", type=float, default=0.01,
                    help="FedProx proximal term weight")
    p.add_argument("--rounds", type=int, default=None,
                    help="Override federated training rounds")
    p.add_argument("--output-path", type=str, default=None,
                    help="Optional explicit output path for the result JSON")
    p.add_argument("--fail-if-output-exists", action="store_true",
                    help="Abort instead of overwriting if the output path already exists")
    p.add_argument("--skip-fedavg", action="store_true")
    p.add_argument("--run-fedadam", action="store_true",
                    help="Also run FedAdam in multi-method runs")
    p.add_argument("--skip-fedadam", action="store_true",
                    help="Skip FedAdam server-side adaptive aggregation")
    p.add_argument("--skip-fedprox", action="store_true")
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--skip-fedice", action="store_true",
                    help="Skip FedICE linear chained-equations baseline")
    p.add_argument("--skip-fedice-ca", action="store_true",
                    help="Skip FedICE-CA linear CAFE-style baseline")
    p.add_argument("--skip-local-ice", action="store_true",
                    help="Skip Local-ICE linear chained-equations baseline")
    p.add_argument("--skip-fed-ca", action="store_true",
                    help="Skip Fed-SAITS-CA")
    p.add_argument("--skip-fed-pd", action="store_true",
                    help="Skip Fed-SAITS-PD")
    p.add_argument("--skip-fed-ca-pd", action="store_true",
                    help="Skip Fed-SAITS-CA+PD")
    p.add_argument("--only-fed-ca", action="store_true",
                    help="Only run Fed-SAITS-CA")
    p.add_argument("--only-local", action="store_true",
                    help="Only run Local-SAITS")
    p.add_argument("--only-fedadam", action="store_true",
                    help="Only run FedAdam")
    p.add_argument("--only-fedice", action="store_true",
                    help="Only run FedICE")
    p.add_argument("--only-fedice-ca", action="store_true",
                    help="Only run FedICE-CA")
    p.add_argument("--only-local-ice", action="store_true",
                    help="Only run Local-ICE")
    p.add_argument("--only-fed-pd", action="store_true",
                    help="Only run Fed-SAITS-PD")
    p.add_argument("--only-fed-ca-pd", action="store_true",
                    help="Only run Fed-SAITS-CA+PD")
    p.add_argument("--ca-tau", type=float, default=1.0,
                    help="Temperature for CA softmax (default: 1.0)")
    p.add_argument("--ca-scale-factor", type=float, default=4.0,
                    help="Power-law exponent for CA weighting (default: 4, "
                         "CAFE original). scale_factor=1 disables sharpening; "
                         "fractional values (e.g. 0.5) also accepted.")
    p.add_argument("--fedadam-server-lr", type=float, default=0.01,
                    help="FedAdam server learning rate")
    p.add_argument("--fedadam-beta1", type=float, default=0.9,
                    help="FedAdam first-moment coefficient")
    p.add_argument("--fedadam-beta2", type=float, default=0.999,
                    help="FedAdam second-moment coefficient")
    p.add_argument("--fedadam-tau", type=float, default=1e-3,
                    help="FedAdam numerical stabilizer")
    p.add_argument("--fedice-rounds", type=int, default=20,
                    help="ICE imputation rounds for FedICE baselines")
    p.add_argument("--fedice-ridge-alpha", type=float, default=1.0,
                    help="Ridge alpha for FedICE per-feature regressors")
    return p.parse_args()


def resolve_output_path(args) -> str:
    if args.output_path:
        return args.output_path

    log_dir = "logs/saits_mnar"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnar_recon_{args.scenario}_{args.mnar_method}_{timestamp}.json"
    return os.path.join(log_dir, filename)


def atomic_write_json(path: str, payload: dict, fail_if_exists: bool = False):
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if fail_if_exists and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")

    fd, tmp_path = tempfile.mkstemp(
        prefix="tmp_mnar_recon_",
        suffix=".json",
        dir=output_dir,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        replace_error = None
        for _ in range(5):
            try:
                os.replace(tmp_path, path)
                replace_error = None
                break
            except PermissionError as exc:
                replace_error = exc
                time.sleep(1)

        if replace_error is not None:
            shutil.copyfile(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass
        raise
    else:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass


def resolve_device(device_str: str) -> str:
    if device_str in ("auto", None):
        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_saits_components():
    try:
        from src.federation.saits_client import SAITSClient
        from src.federation.saits_server import SAITSFederatedServer
    except ImportError as exc:
        raise ImportError(
            "SAITS methods require torch and the full training environment. "
            "Activate the project environment, or use --only-fedice / "
            "--only-fedice-ca for the sklearn baseline."
        ) from exc

    return SAITSClient, SAITSFederatedServer


# ================================================================
# Evaluation on MNAR holes only
# ================================================================

def compute_mnar_metrics(
    imputed_data: np.ndarray,
    ground_truth: np.ndarray,
    eval_mask: np.ndarray,
) -> dict:
    """
    Compute MAE and RMSE only on positions where:
        eval_mask == 1  (i.e., originally observed AND newly MNAR-masked)

    Args:
        imputed_data:  (N, T, D) model output (fully filled)
        ground_truth:  (N, T, D) original values before MNAR masking
        eval_mask:     (N, T, D) binary mask, 1 = evaluate here

    Returns:
        dict with 'mae', 'rmse', 'num_eval_points'
    """
    eval_bool = eval_mask > 0.5
    num_eval = int(eval_bool.sum())

    if num_eval == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "num_eval_points": 0}

    pred = imputed_data[eval_bool]
    true = ground_truth[eval_bool]

    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {"mae": mae, "rmse": rmse, "num_eval_points": num_eval}


# ================================================================
# Build per-client data structures
# ================================================================

def prepare_client_data(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list,
    scenario: str,
    mnar_method: str,
    target_features: list,
    missing_rate: float,
    seed: int,
    num_clients: int = 5,
    client_ids: np.ndarray | None = None,
    make_dataset: bool = True,
) -> list:
    """
    Distribute all data to 5 clients and apply MNAR masking.

    Returns a list of dicts, one per client, each containing:
        - "dataset": TimeSeriesDataset for training if make_dataset=True
        - "ground_truth": (n_k, T, D) original values (for metric computation)
        - "observed_mask_original": (n_k, T, D) original masks before MNAR
        - "mnar_mask_added": (n_k, T, D) 1 where this experiment added MNAR holes
        - "eval_mask": (n_k, T, D) = observed_mask_original & mnar_mask_added
        - "client_id": int
        - "mnar_config": dict
    """
    # --- Use HeterogeneousDataAllocator to split & apply MNAR ---
    allocator = HeterogeneousDataAllocator(
        X=ground_truth,
        masks=masks,
        num_clients=num_clients,
        feature_names=feature_names,
        client_ids=client_ids,
    )

    client_raw = allocator.allocate(
        scenario=scenario,
        mnar_method=mnar_method,
        target_features=target_features,
        missing_rate=missing_rate,
        seed=seed,
    )

    # --- Build per-client structures ---
    client_list = []
    for cd in client_raw:
        k = cd["client_id"]
        n_k = cd["X"].shape[0]

        # Ground truth for this client (original values before any MNAR)
        client_ground_truth = cd["X"]  # (n_k, T, D), the raw data for this client

        # observed_mask_original: original mask before MNAR was applied
        observed_mask_original = cd["orig_masks"]  # (n_k, T, D)

        # masks after MNAR: the final training mask
        train_missing_mask_final = cd["masks"]  # (n_k, T, D), 1=observed, 0=missing

        # mnar_mask_added: 1 where this experiment newly created MNAR holes
        # These are positions that WERE observed (orig_masks==1) but are NOW missing (masks==0)
        mnar_mask_added = ((observed_mask_original > 0.5) &
                           (train_missing_mask_final < 0.5)).astype(np.float32)

        # eval_mask: only evaluate where we have ground truth AND it was MNAR-masked
        # Since mnar_mask_added already requires observed_mask_original==1,
        # eval_mask = mnar_mask_added
        eval_mask = mnar_mask_added

        num_mnar_holes = int(eval_mask.sum())
        total_observed_orig = int(observed_mask_original.sum())
        print(f"  Client {k}: {n_k} samples, "
              f"MNAR holes={num_mnar_holes} "
              f"({num_mnar_holes/max(total_observed_orig,1)*100:.1f}% of originally observed)")

        dataset = None
        if make_dataset:
            from src.data.dataset import TimeSeriesDataset

            # eval_masks = zeros → no additional MCAR eval masking during training
            zero_eval_masks = np.zeros_like(train_missing_mask_final)

            dataset = TimeSeriesDataset(
                data=cd["observed"],           # (n_k, T, D) with 0 where missing
                ground_truth=client_ground_truth,
                masks=train_missing_mask_final,
                eval_masks=zero_eval_masks,    # NO extra eval masking
                feature_names=feature_names,
            )

        client_list.append({
            "dataset": dataset,
            "indices": cd["indices"],
            "ground_truth": client_ground_truth,
            "observed_mask_original": observed_mask_original,
            "mnar_mask_added": mnar_mask_added,
            "eval_mask": eval_mask,
            "observed_data": cd["observed"],
            "train_masks": train_missing_mask_final,
            "client_id": k,
            "mnar_config": cd["mnar_config"],
        })

    return client_list


# ================================================================
# FedAvg / FedProx Training + MNAR Evaluation
# ================================================================

def run_federated(
    client_list: list,
    config: dict,
    device: str,
    method: str = "fedavg",
) -> dict:
    """
    Run federated training (FedAvg or FedProx), then evaluate
    on each client's MNAR holes.

    Returns dict with per-client metrics and aggregated metrics.
    """
    SAITSClient, SAITSFederatedServer = _require_saits_components()

    # --- Create SAITSClient objects ---
    clients = []
    for cd in client_list:
        client = SAITSClient(
            client_id=cd["client_id"],
            train_data=cd["dataset"],
            val_data=None,
            config=config,
            device=device,
        )
        clients.append(client)

    # --- Create server with NO test_data (we evaluate on MNAR holes instead) ---
    server = SAITSFederatedServer(
        clients=clients,
        test_data=None,  # No global test set evaluation
        config=config,
        device=device,
    )

    # --- Federated training ---
    print(f"  Training {method} for {config['federation']['rounds']} rounds...")
    train_results = server.train()

    # --- Post-training: impute on each client's local data ---
    print(f"  Evaluating on MNAR holes...")
    client_metrics = []

    for i, cd in enumerate(client_list):
        cid = cd['client_id']
        if server._use_pd:
            # PD: each client already has local layers; load global layers
            if server._use_ca and cid in server._personalized_params:
                clients[i].download_global_layers(
                    server._personalized_params[cid]
                )
            else:
                clients[i].download_global_layers(server.global_params)
        elif (hasattr(server, '_personalized_params')
                and cid in server._personalized_params):
            clients[i].download_global_model(
                server._personalized_params[cid]
            )
        elif server.global_params is not None:
            clients[i].download_global_model(server.global_params)

        # Impute: feed the MNAR-masked observed data through the model
        imputed = clients[i].impute(
            observed=cd["observed_data"],
            masks=cd["train_masks"],
        )  # (n_k, T, D)

        # Compute metrics only on MNAR holes
        metrics = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=cd["ground_truth"],
            eval_mask=cd["eval_mask"],
        )

        client_metrics.append({
            "client_id": cd["client_id"],
            "num_eval_points": metrics["num_eval_points"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        })
        print(f"    Client {cd['client_id']}: "
              f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
              f"eval_pts={metrics['num_eval_points']}")

    # --- Aggregate across clients (nanmean for safety) ---
    maes = [m["mae"] for m in client_metrics]
    rmses = [m["rmse"] for m in client_metrics]
    mean_mae = float(np.nanmean(maes))
    mean_rmse = float(np.nanmean(rmses))

    print(f"  {method} mean MAE={mean_mae:.6f}, mean RMSE={mean_rmse:.6f}")

    return {
        "client_metrics": client_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "num_rounds": len(train_results.get("history", [])),
        "history": train_results.get("history", []),
    }


# ================================================================
# Local-only Training + MNAR Evaluation
# ================================================================

def run_local(
    client_list: list,
    config: dict,
    device: str,
) -> dict:
    """
    Train each client independently (no federation), then evaluate
    on each client's own MNAR holes.
    """
    SAITSClient, _ = _require_saits_components()

    fed_cfg = config.get("federation", {})
    saits_cfg = config.get("saits", {})
    total_epochs = fed_cfg.get("rounds", 50) * saits_cfg.get("local_epochs", 5)

    client_metrics = []

    for cd in client_list:
        k = cd["client_id"]
        print(f"  Local client {k} ({len(cd['dataset'])} samples, "
              f"{total_epochs} epochs)...")

        local_config = copy.deepcopy(config)
        local_config["federation"]["aggregation"] = "fedavg"  # doesn't matter

        client = SAITSClient(
            client_id=k,
            train_data=cd["dataset"],
            val_data=None,
            config=local_config,
            device=device,
        )
        client.model.set_training_params(
            epochs=total_epochs,
            patience=min(20, max(total_epochs - 1, 1)),
        )
        client.local_train()

        # Impute on this client's own data
        imputed = client.impute(
            observed=cd["observed_data"],
            masks=cd["train_masks"],
        )

        # Evaluate only on MNAR holes
        metrics = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=cd["ground_truth"],
            eval_mask=cd["eval_mask"],
        )

        client_metrics.append({
            "client_id": k,
            "num_eval_points": metrics["num_eval_points"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        })
        print(f"    Client {k}: "
              f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
              f"eval_pts={metrics['num_eval_points']}")

    maes = [m["mae"] for m in client_metrics]
    rmses = [m["rmse"] for m in client_metrics]
    mean_mae = float(np.nanmean(maes))
    mean_rmse = float(np.nanmean(rmses))

    print(f"  Local mean MAE={mean_mae:.6f}, mean RMSE={mean_rmse:.6f}")

    return {
        "client_metrics": client_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
    }


def run_fedice_baseline(
    client_list: list,
    seed: int,
    n_rounds: int,
    ridge_alpha: float,
    use_ca: bool,
    ca_scale_factor: float,
    ca_tau: float,
) -> dict:
    """
    Run the linear ICE baseline on the same MNAR-masked client tensors.

    FedICE flattens each client's (N, T, D) tensor into tabular rows and
    federates one chained linear imputation model per feature. With
    ``use_ca=True`` it uses CAFE-style complementarity weighting, giving the
    linear + CA comparison point for Fed-SAITS-CA.
    """
    method = "fedice_ca" if use_ca else "fedice"
    print(f"  Training {method} for {n_rounds} ICE rounds...")

    imputer = FedICEImputer(
        n_rounds=n_rounds,
        ridge_alpha=ridge_alpha,
        use_ca=use_ca,
        ca_scale_factor=ca_scale_factor,
        seed=seed,
    )
    imputed_clients = imputer.fit_transform(
        client_ground_truths=[cd["ground_truth"] for cd in client_list],
        client_masks=[cd["train_masks"] for cd in client_list],
    )

    client_metrics = []
    for cd, imputed in zip(client_list, imputed_clients):
        metrics = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=cd["ground_truth"],
            eval_mask=cd["eval_mask"],
        )
        client_metrics.append({
            "client_id": cd["client_id"],
            "num_eval_points": metrics["num_eval_points"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        })
        print(f"    Client {cd['client_id']}: "
              f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
              f"eval_pts={metrics['num_eval_points']}")

    maes = [m["mae"] for m in client_metrics]
    rmses = [m["rmse"] for m in client_metrics]
    mean_mae = float(np.nanmean(maes))
    mean_rmse = float(np.nanmean(rmses))

    print(f"  {method} mean MAE={mean_mae:.6f}, mean RMSE={mean_rmse:.6f}")

    ret = {
        "client_metrics": client_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "ice_rounds": n_rounds,
        "ridge_alpha": ridge_alpha,
    }
    if use_ca:
        ret["ca_tau"] = ca_tau
        ret["ca_scale_factor"] = ca_scale_factor
        if imputer.ca_weights_:
            ca_weights = np.asarray(imputer.ca_weights_, dtype=float)
            ret["ca_weights_last_round"] = ca_weights[-1].tolist()
            ret["ca_weights_mean"] = ca_weights.mean(axis=(0, 1)).tolist()
    return ret


def run_local_ice_baseline(
    client_list: list,
    seed: int,
    n_rounds: int,
    ridge_alpha: float,
) -> dict:
    """
    Run independent ICE/MICE-style chained imputation on each client.

    This is the linear counterpart to Local-SAITS: no model coefficients,
    fingerprints, initial values, or clipping ranges are shared across clients.
    """
    print(f"  Training local_ice for {n_rounds} ICE rounds...")

    client_metrics = []
    for cd in client_list:
        k = cd["client_id"]
        imputer = FedICEImputer(
            n_rounds=n_rounds,
            ridge_alpha=ridge_alpha,
            use_ca=False,
            seed=seed,
        )
        imputed = imputer.fit_transform(
            client_ground_truths=[cd["ground_truth"]],
            client_masks=[cd["train_masks"]],
        )[0]

        metrics = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=cd["ground_truth"],
            eval_mask=cd["eval_mask"],
        )
        client_metrics.append({
            "client_id": k,
            "num_eval_points": metrics["num_eval_points"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        })
        print(f"    Client {k}: "
              f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
              f"eval_pts={metrics['num_eval_points']}")

    maes = [m["mae"] for m in client_metrics]
    rmses = [m["rmse"] for m in client_metrics]
    mean_mae = float(np.nanmean(maes))
    mean_rmse = float(np.nanmean(rmses))

    print(f"  local_ice mean MAE={mean_mae:.6f}, mean RMSE={mean_rmse:.6f}")

    return {
        "client_metrics": client_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "ice_rounds": n_rounds,
        "ridge_alpha": ridge_alpha,
    }


# ================================================================
# Single Experiment Run (one seed)
# ================================================================

def run_single(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list,
    scenario: str,
    mnar_method: str,
    target_features: list,
    missing_rate: float,
    mu: float,
    seed: int,
    device: str,
    checkpoint_root: str,
    num_clients: int = 5,
    client_ids: np.ndarray | None = None,
    rounds: int | None = None,
    skip_fedavg: bool = False,
    skip_fedadam: bool = False,
    skip_fedprox: bool = False,
    skip_local: bool = False,
    skip_fedice: bool = False,
    skip_fedice_ca: bool = False,
    skip_local_ice: bool = False,
    skip_fed_ca: bool = False,
    skip_fed_pd: bool = False,
    skip_fed_ca_pd: bool = False,
    ca_tau: float = 1.0,
    ca_scale_factor: float = 4.0,
    fedadam_server_lr: float = 0.01,
    fedadam_beta1: float = 0.9,
    fedadam_beta2: float = 0.999,
    fedadam_tau: float = 1e-3,
    fedice_rounds: int = 20,
    fedice_ridge_alpha: float = 1.0,
) -> list:
    """
    Run one seed: prepare data, train FedAvg/FedProx/Local, evaluate on MNAR holes.

    Returns a list of result dicts (one per method), each in the specified JSON format.
    """
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"  MNAR Imputation Reconstruction Experiment")
    print(f"  scenario={scenario}, method={mnar_method}, seed={seed}")
    print(f"  rho={missing_rate}, device={device}")
    print(f"  Total samples={ground_truth.shape[0]}, clients={num_clients}, "
          f"per_client~={ground_truth.shape[0] // num_clients}")
    print(f"{'='*60}")

    def fresh_config() -> dict:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["data"]["seq_length"] = int(ground_truth.shape[1])
        cfg["data"]["num_features"] = int(ground_truth.shape[2])
        cfg["federation"]["num_clients"] = int(num_clients)
        if rounds is not None:
            cfg["federation"]["rounds"] = int(rounds)
        return cfg

    # --- Prepare client data (all 2500 samples, no train/test split) ---
    print(f"\n[Step 1] Distributing data to {num_clients} clients + applying MNAR...")
    needs_saits_dataset = not (
        skip_fedavg and skip_fedprox and skip_local
        and skip_fedadam and skip_fed_ca and skip_fed_pd and skip_fed_ca_pd
    )
    client_list = prepare_client_data(
        ground_truth=ground_truth,
        masks=masks,
        feature_names=feature_names,
        scenario=scenario,
        mnar_method=mnar_method,
        target_features=target_features,
        missing_rate=missing_rate,
        seed=seed,
        num_clients=num_clients,
        client_ids=client_ids,
        make_dataset=needs_saits_dataset,
    )

    results = []

    # --- FedAvg ---
    if not skip_fedavg:
        set_seed(seed)  # fair comparison: each method starts from the same RNG state
        print(f"\n[Step 2a] FedAvg (seed={seed})")
        config_fa = fresh_config()
        config_fa["federation"]["aggregation"] = "fedavg"
        config_fa["federation"]["mu"] = 0.0
        config_fa["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fedavg"
        )

        fa_result = run_federated(client_list, config_fa, device, method="fedavg")
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fedavg",
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_fa["federation"]["rounds"],
            "client_metrics": fa_result["client_metrics"],
            "mean_mae": fa_result["mean_mae"],
            "mean_rmse": fa_result["mean_rmse"],
        })

    # --- FedProx ---
    if not skip_fedprox:
        set_seed(seed)
        print(f"\n[Step 2b] FedProx (seed={seed}, mu={mu})")
        config_fp = fresh_config()
        config_fp["federation"]["aggregation"] = "fedprox"
        config_fp["federation"]["mu"] = mu
        config_fp["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fedprox"
        )

        fp_result = run_federated(client_list, config_fp, device, method="fedprox")
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fedprox",
            "mu": mu,
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_fp["federation"]["rounds"],
            "client_metrics": fp_result["client_metrics"],
            "mean_mae": fp_result["mean_mae"],
            "mean_rmse": fp_result["mean_rmse"],
        })

    # --- FedAdam ---
    if not skip_fedadam:
        set_seed(seed)
        print(f"\n[Step 2b-adam] FedAdam (seed={seed}, "
              f"server_lr={fedadam_server_lr})")
        config_adam = fresh_config()
        config_adam["federation"]["aggregation"] = "fedadam"
        config_adam["federation"]["server_lr"] = fedadam_server_lr
        config_adam["federation"]["beta1"] = fedadam_beta1
        config_adam["federation"]["beta2"] = fedadam_beta2
        config_adam["federation"]["tau"] = fedadam_tau
        config_adam["federation"]["mu"] = 0.0
        config_adam["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fedadam"
        )

        adam_result = run_federated(
            client_list, config_adam, device, method="fedadam"
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fedadam",
            "fedadam_server_lr": fedadam_server_lr,
            "fedadam_beta1": fedadam_beta1,
            "fedadam_beta2": fedadam_beta2,
            "fedadam_tau": fedadam_tau,
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_adam["federation"]["rounds"],
            "client_metrics": adam_result["client_metrics"],
            "mean_mae": adam_result["mean_mae"],
            "mean_rmse": adam_result["mean_rmse"],
        })

    # --- FedICE (linear chained equations + FedAvg coefficient aggregation) ---
    if not skip_fedice:
        set_seed(seed)
        print(f"\n[Step 2c] FedICE (seed={seed})")
        fedice_result = run_fedice_baseline(
            client_list=client_list,
            seed=seed,
            n_rounds=fedice_rounds,
            ridge_alpha=fedice_ridge_alpha,
            use_ca=False,
            ca_scale_factor=ca_scale_factor,
            ca_tau=ca_tau,
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fedice",
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            **fedice_result,
        })

    # --- FedICE-CA (linear chained equations + CAFE-style CA) ---
    if not skip_fedice_ca:
        set_seed(seed)
        print(f"\n[Step 2d] FedICE-CA (seed={seed}, "
              f"scale_factor={ca_scale_factor})")
        fedice_ca_result = run_fedice_baseline(
            client_list=client_list,
            seed=seed,
            n_rounds=fedice_rounds,
            ridge_alpha=fedice_ridge_alpha,
            use_ca=True,
            ca_scale_factor=ca_scale_factor,
            ca_tau=ca_tau,
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fedice_ca",
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            **fedice_ca_result,
        })

    # --- Local-ICE (independent linear chained equations, no sharing) ---
    if not skip_local_ice:
        set_seed(seed)
        print(f"\n[Step 2d-local] Local-ICE (seed={seed})")
        local_ice_result = run_local_ice_baseline(
            client_list=client_list,
            seed=seed,
            n_rounds=fedice_rounds,
            ridge_alpha=fedice_ridge_alpha,
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "local_ice",
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            **local_ice_result,
        })

    # --- Local-only ---
    if not skip_local:
        set_seed(seed)
        print(f"\n[Step 2e] Local-only (seed={seed})")
        config_lo = fresh_config()
        config_lo["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "local"
        )

        lo_result = run_local(client_list, config_lo, device)
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "local",
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_lo["federation"]["rounds"],
            "client_metrics": lo_result["client_metrics"],
            "mean_mae": lo_result["mean_mae"],
            "mean_rmse": lo_result["mean_rmse"],
        })

    # --- Fed-SAITS-CA (Complementarity-Aware Aggregation) ---
    if not skip_fed_ca:
        set_seed(seed)
        print(f"\n[Step 2f] Fed-SAITS-CA (seed={seed}, tau={ca_tau}, "
              f"scale_factor={ca_scale_factor})")
        config_ca = fresh_config()
        config_ca["federation"]["aggregation"] = "fed_ca"
        config_ca["federation"]["ca_tau"] = ca_tau
        config_ca["federation"]["ca_scale_factor"] = ca_scale_factor
        config_ca["federation"]["mu"] = 0.0  # no proximal term for CA-only
        config_ca["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fed_ca"
        )

        ca_result = run_federated(
            client_list, config_ca, device, method="fed_ca"
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fed_ca",
            "ca_tau": ca_tau,
            "ca_scale_factor": ca_scale_factor,
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_ca["federation"]["rounds"],
            "client_metrics": ca_result["client_metrics"],
            "mean_mae": ca_result["mean_mae"],
            "mean_rmse": ca_result["mean_rmse"],
        })

    # --- Fed-SAITS-PD (Partial Decoupling only) ---
    if not skip_fed_pd:
        set_seed(seed)
        print(f"\n[Step 2g] Fed-SAITS-PD (seed={seed})")
        config_pd = fresh_config()
        config_pd["federation"]["aggregation"] = "fed_pd"
        config_pd["federation"]["mu"] = 0.0
        config_pd["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fed_pd"
        )

        pd_result = run_federated(
            client_list, config_pd, device, method="fed_pd"
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fed_pd",
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_pd["federation"]["rounds"],
            "client_metrics": pd_result["client_metrics"],
            "mean_mae": pd_result["mean_mae"],
            "mean_rmse": pd_result["mean_rmse"],
        })

    # --- Fed-SAITS-CA+PD (full minus adaptive mu) ---
    if not skip_fed_ca_pd:
        set_seed(seed)
        print(f"\n[Step 2h] Fed-SAITS-CA+PD (seed={seed}, tau={ca_tau}, "
              f"scale_factor={ca_scale_factor})")
        config_capd = fresh_config()
        config_capd["federation"]["aggregation"] = "fed_ca_pd"
        config_capd["federation"]["ca_tau"] = ca_tau
        config_capd["federation"]["ca_scale_factor"] = ca_scale_factor
        config_capd["federation"]["mu"] = 0.0
        config_capd["training"]["checkpoint_dir"] = os.path.join(
            checkpoint_root, f"seed_{seed}", "fed_ca_pd"
        )

        capd_result = run_federated(
            client_list, config_capd, device, method="fed_ca_pd"
        )
        results.append({
            "seed": seed,
            "scenario": scenario,
            "mnar_method": mnar_method,
            "missing_rate": missing_rate,
            "method": "fed_ca_pd",
            "ca_tau": ca_tau,
            "ca_scale_factor": ca_scale_factor,
            "num_clients": num_clients,
            "samples_per_client": ground_truth.shape[0] // num_clients,
            "rounds": config_capd["federation"]["rounds"],
            "client_metrics": capd_result["client_metrics"],
            "mean_mae": capd_result["mean_mae"],
            "mean_rmse": capd_result["mean_rmse"],
        })

    return results


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    device = resolve_device(args.device)
    result_path = resolve_output_path(args)
    result_stem = os.path.splitext(os.path.basename(result_path))[0]
    checkpoint_root = os.path.join("checkpoints", "saits_mnar", result_stem)

    # --- Load data (all 2500 samples, no split) ---
    print("[1/2] Loading data...")
    ground_truth, _, masks, feature_names, _ = \
        load_from_local_tensor(args.tensor_dir, normalize=True)

    N, T, D = ground_truth.shape
    bad_features = [f for f in args.target_features if f < 0 or f >= D]
    if bad_features:
        raise ValueError(
            f"Invalid --target-features for tensor with D={D}: {bad_features}. "
            f"Valid feature indices are 0..{D - 1}."
        )
    if args.num_clients < 2:
        raise ValueError("--num-clients must be at least 2.")
    if args.num_clients > N:
        raise ValueError(
            f"--num-clients={args.num_clients} exceeds N={N} samples."
        )
    client_ids = None
    if args.client_ids_path:
        client_ids = np.load(args.client_ids_path).astype(int)
        if client_ids.shape[0] != N:
            raise ValueError(
                f"--client-ids-path length {client_ids.shape[0]} does not "
                f"match N={N}"
            )
        observed_client_ids = sorted(np.unique(client_ids).tolist())
        expected_client_ids = list(range(args.num_clients))
        if observed_client_ids != expected_client_ids:
            raise ValueError(
                f"client ids must be {expected_client_ids}, got "
                f"{observed_client_ids}"
            )

    print(f"  Loaded: {ground_truth.shape} (N={N}, T={T}, D={D})")
    print(f"  Features: {feature_names}")
    print(f"  Original missing rate: {1.0 - masks.mean():.1%}")
    print(f"  Target MNAR features: "
          f"{[feature_names[f] for f in args.target_features]}")
    print(f"  Num clients: {args.num_clients}")
    if client_ids is not None:
        client_counts = [
            int((client_ids == k).sum()) for k in range(args.num_clients)
        ]
        print(f"  Fixed client split: {client_counts}")
    print(f"  Checkpoints: {checkpoint_root}")

    # --- Determine seeds ---
    seeds = args.seeds if args.seeds is not None else [args.seed]

    # --- Run experiments ---
    print(f"\n[2/2] Running {len(seeds)} seed(s)...")
    all_results = []
    start_time = time.time()

    for seed in seeds:
        # Handle --only-* flags: skip everything except the selected method
        only = (
            args.only_fed_ca or args.only_local or args.only_fedadam
            or args.only_fedice or args.only_fedice_ca
            or args.only_local_ice or args.only_fed_pd or args.only_fed_ca_pd
        )
        skip_fa = args.skip_fedavg or only
        skip_adam = (
            args.skip_fedadam
            or (not args.run_fedadam and not args.only_fedadam)
            or (only and not args.only_fedadam)
        )
        skip_fp = args.skip_fedprox or only
        skip_lo = args.skip_local or (only and not args.only_local)
        skip_fi = args.skip_fedice or (only and not args.only_fedice)
        skip_fica = args.skip_fedice_ca or (only and not args.only_fedice_ca)
        skip_li = args.skip_local_ice or (only and not args.only_local_ice)
        skip_ca = args.skip_fed_ca or (only and not args.only_fed_ca)
        skip_pd = args.skip_fed_pd or (only and not args.only_fed_pd)
        skip_capd = args.skip_fed_ca_pd or (only and not args.only_fed_ca_pd)

        seed_results = run_single(
            ground_truth=ground_truth,
            masks=masks,
            feature_names=feature_names,
            scenario=args.scenario,
            mnar_method=args.mnar_method,
            target_features=args.target_features,
            missing_rate=args.missing_rate,
            mu=args.mu,
            seed=seed,
            device=device,
            checkpoint_root=checkpoint_root,
            num_clients=args.num_clients,
            client_ids=client_ids,
            rounds=args.rounds,
            skip_fedavg=skip_fa,
            skip_fedadam=skip_adam,
            skip_fedprox=skip_fp,
            skip_local=skip_lo,
            skip_fedice=skip_fi,
            skip_fedice_ca=skip_fica,
            skip_local_ice=skip_li,
            skip_fed_ca=skip_ca,
            skip_fed_pd=skip_pd,
            skip_fed_ca_pd=skip_capd,
            ca_tau=args.ca_tau,
            ca_scale_factor=args.ca_scale_factor,
            fedadam_server_lr=args.fedadam_server_lr,
            fedadam_beta1=args.fedadam_beta1,
            fedadam_beta2=args.fedadam_beta2,
            fedadam_tau=args.fedadam_tau,
            fedice_rounds=args.fedice_rounds,
            fedice_ridge_alpha=args.fedice_ridge_alpha,
        )
        all_results.extend(seed_results)

    total_time = time.time() - start_time

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Scenario: {args.scenario}, MNAR method: {args.mnar_method}")
    print(f"  Seeds: {seeds}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*60}")

    _print_summary(all_results, seeds)

    # --- Save results safely ---
    atomic_write_json(
        path=result_path,
        payload={
            "experiment": "mnar_imputation_reconstruction",
            "scenario": args.scenario,
            "mnar_method": args.mnar_method,
            "missing_rate": args.missing_rate,
            "target_features": args.target_features,
            "client_ids_path": args.client_ids_path,
            "mu": args.mu,
            "rounds_override": args.rounds,
            "fedadam_server_lr": args.fedadam_server_lr,
            "fedadam_beta1": args.fedadam_beta1,
            "fedadam_beta2": args.fedadam_beta2,
            "fedadam_tau": args.fedadam_tau,
            "fedice_rounds": args.fedice_rounds,
            "fedice_ridge_alpha": args.fedice_ridge_alpha,
            "num_clients": args.num_clients,
            "total_samples": N,
            "samples_per_client": N // args.num_clients,
            "seeds": seeds,
            "total_time_sec": total_time,
            "evaluation_note": (
                "Metrics computed ONLY on positions where "
                "observed_mask_original==1 AND mnar_mask_added==1. "
                "No test set, no MCAR eval masking."
            ),
            "results": all_results,
        },
        fail_if_exists=args.fail_if_output_exists,
    )

    print(f"\nResults saved to {result_path}")


def _print_summary(results: list, seeds: list):
    """Print summary table grouped by method."""
    methods = sorted(set(r["method"] for r in results))

    print(f"\n--- Summary (evaluation on MNAR holes only) ---")
    print(f"{'Seed':>6} | ", end="")
    for m in methods:
        print(f"{m+' MAE':>14} | {m+' RMSE':>14} | ", end="")
    print()
    print("-" * (8 + len(methods) * 34))

    for seed in seeds:
        print(f"{seed:>6} | ", end="")
        for m in methods:
            matching = [r for r in results if r["seed"] == seed and r["method"] == m]
            if matching:
                r = matching[0]
                print(f"{r['mean_mae']:>14.6f} | {r['mean_rmse']:>14.6f} | ", end="")
            else:
                print(f"{'N/A':>14} | {'N/A':>14} | ", end="")
        print()

    if len(seeds) > 1:
        print("-" * (8 + len(methods) * 34))
        print(f"{'Mean':>6} | ", end="")
        for m in methods:
            method_results = [r for r in results if r["method"] == m]
            if method_results:
                avg_mae = np.nanmean([r["mean_mae"] for r in method_results])
                avg_rmse = np.nanmean([r["mean_rmse"] for r in method_results])
                print(f"{avg_mae:>14.6f} | {avg_rmse:>14.6f} | ", end="")
            else:
                print(f"{'N/A':>14} | {'N/A':>14} | ", end="")
        print()

        print(f"{'Std':>6} | ", end="")
        for m in methods:
            method_results = [r for r in results if r["method"] == m]
            if method_results:
                std_mae = np.nanstd([r["mean_mae"] for r in method_results])
                std_rmse = np.nanstd([r["mean_rmse"] for r in method_results])
                print(f"{std_mae:>14.6f} | {std_rmse:>14.6f} | ", end="")
            else:
                print(f"{'N/A':>14} | {'N/A':>14} | ", end="")
        print()

    # --- FL improvement over local ---
    for seed in seeds:
        local = [r for r in results if r["seed"] == seed and r["method"] == "local"]
        if not local:
            continue
        lo_mae = local[0]["mean_mae"]
        for m in [
            "fedavg", "fedprox", "fedice", "fedice_ca",
            "fed_ca", "fed_pd", "fed_ca_pd",
        ]:
            fed = [r for r in results if r["seed"] == seed and r["method"] == m]
            if fed and lo_mae > 0:
                imp = (lo_mae - fed[0]["mean_mae"]) / lo_mae * 100
                print(f"  Seed {seed}: {m} improvement over local = {imp:.1f}%")


if __name__ == "__main__":
    main()
