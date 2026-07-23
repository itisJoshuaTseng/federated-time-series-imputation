"""
Downstream utility evaluation on eICU-demo.

This script evaluates whether different imputation methods lead to different
performance on a simple clinical downstream task:

    ICU length-of-stay >= 48 hours

The intended minimal setting for the project is:

    eICU-demo, quantile MNAR S1/S4, rho=0.5,
    FedSAITS+CA vs FedICE+CA, XGBoost classifier.

The imputation stage is unsupervised and uses the same MNAR reconstruction
setup as run_mnar_experiment.py.  After imputation, each multivariate time
series is converted into tabular summary features and evaluated with a fixed
train/test split for each seed.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project imports
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import FedICEImputer
from src.data.vitaldb_loader import load_from_local_tensor


def _load_mnar_runner():
    """Load the sibling run_mnar_experiment.py by exact file path."""
    path = EXPERIMENT_DIR / "run_mnar_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "_fed_tsi_run_mnar_experiment",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MNAR runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mnar_runner = _load_mnar_runner()
DEFAULT_CONFIG = _mnar_runner.DEFAULT_CONFIG
_require_saits_components = _mnar_runner._require_saits_components
compute_mnar_metrics = _mnar_runner.compute_mnar_metrics
prepare_client_data = _mnar_runner.prepare_client_data
resolve_device = _mnar_runner.resolve_device
set_seed = _mnar_runner.set_seed


METHOD_LABELS = {
    "local_saits": "Local-SAITS",
    "fedsaits": "FedSAITS",
    "fedsaits_ca": "FedSAITS+CA",
    "local_ice": "Local-ICE",
    "fedice": "FedICE",
    "fedice_ca": "FedICE+CA",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eICU downstream utility after federated imputation"
    )
    p.add_argument(
        "--tensor-dir",
        type=str,
        default="data/eicu_demo_tensor_T288_D6",
        help="Path to the prepared eICU-demo tensor directory.",
    )
    p.add_argument(
        "--patient-csv",
        type=str,
        default="data/eicu_demo/patient.csv.gz",
        help="eICU patient.csv.gz path for LOS/mortality labels.",
    )
    p.add_argument(
        "--client-ids-path",
        type=str,
        default="data/eicu_demo_tensor_T288_D6/client_ids_5hospital_clusters.npy",
        help="Optional fixed client id assignment for hospital-cluster clients.",
    )
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["S1", "S4"],
        choices=["S1", "S2", "S3", "S4"],
    )
    p.add_argument("--mnar-method", type=str, default="quantile",
                   choices=["quantile", "logit"])
    p.add_argument("--missing-rate", type=float, default=0.5)
    p.add_argument(
        "--target-features",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Default eICU target features: HR, RR, SpO2.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["fedsaits_ca", "fedice_ca"],
        choices=[
            "local_saits",
            "fedsaits",
            "fedsaits_ca",
            "local_ice",
            "fedice",
            "fedice_ca",
        ],
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--fedice-rounds", type=int, default=20)
    p.add_argument("--fedice-ridge-alpha", type=float, default=1.0)
    p.add_argument("--saits-ca-scale-factor", type=float, default=0.5)
    p.add_argument("--ice-ca-scale-factor", type=float, default=4.0)
    p.add_argument("--ca-tau", type=float, default=1.0)
    p.add_argument(
        "--label",
        type=str,
        default="los48h",
        choices=["los48h", "unit_mortality", "hospital_mortality"],
    )
    p.add_argument(
        "--los-hours",
        type=float,
        default=48.0,
        help="Threshold for los48h label. Default is 48 hours.",
    )
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument(
        "--classifier",
        type=str,
        default="xgboost",
        choices=["xgboost", "histgb", "logistic"],
        help="Downstream classifier. If xgboost is unavailable, histgb is used.",
    )
    p.add_argument(
        "--feature-mode",
        type=str,
        default="flatten",
        choices=["flatten", "summary"],
        help=(
            "How to convert imputed time series to tabular classifier inputs. "
            "'flatten' follows the xgboost.ipynb style more closely; "
            "'summary' uses mean/std/min/max/first/last/slope."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="experiments/figures/downstream_eicu_los",
    )
    p.add_argument(
        "--save-imputed",
        action="store_true",
        help="Save imputed tensors as compressed npz files. Can be large.",
    )
    return p.parse_args()


def load_case_ids(tensor_dir: Path) -> np.ndarray:
    case_path = tensor_dir / "case_ids.npy"
    if not case_path.exists():
        raise FileNotFoundError(f"Missing case ids: {case_path}")
    return np.load(case_path)


def load_labels(
    case_ids: np.ndarray,
    patient_csv: Path,
    label: str,
    los_hours: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return y and valid mask aligned to case_ids."""
    patient = pd.read_csv(patient_csv)
    aligned = (
        pd.DataFrame({"patientunitstayid": case_ids})
        .merge(patient, on="patientunitstayid", how="left")
    )

    if label == "los48h":
        offsets = pd.to_numeric(aligned["unitdischargeoffset"], errors="coerce")
        valid = offsets.notna().to_numpy()
        y = (offsets.to_numpy() >= los_hours * 60.0).astype(int)
        label_name = f"ICU LOS >= {los_hours:g}h"
    elif label == "unit_mortality":
        status = aligned["unitdischargestatus"]
        valid = status.notna().to_numpy()
        y = (status.astype(str).str.lower() == "expired").to_numpy().astype(int)
        label_name = "Unit mortality"
    else:
        status = aligned["hospitaldischargestatus"]
        valid = status.notna().to_numpy()
        y = (status.astype(str).str.lower() == "expired").to_numpy().astype(int)
        label_name = "Hospital mortality"

    y = y.astype(int)
    valid = valid.astype(bool)
    info = {
        "label": label,
        "label_name": label_name,
        "n_total": int(len(y)),
        "n_valid": int(valid.sum()),
        "n_positive": int(y[valid].sum()),
        "positive_rate": float(y[valid].mean()) if valid.any() else float("nan"),
    }
    return y, valid, info


def build_saits_config(
    seq_length: int,
    num_features: int,
    num_clients: int,
    rounds: int,
    checkpoint_dir: Path,
    aggregation: str,
    ca_scale_factor: float | None = None,
    ca_tau: float = 1.0,
) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data"]["seq_length"] = int(seq_length)
    cfg["data"]["num_features"] = int(num_features)
    cfg["federation"]["num_clients"] = int(num_clients)
    cfg["federation"]["rounds"] = int(rounds)
    cfg["federation"]["aggregation"] = aggregation
    cfg["federation"]["mu"] = 0.0
    if aggregation == "fed_ca":
        cfg["federation"]["ca_scale_factor"] = float(ca_scale_factor)
        cfg["federation"]["ca_tau"] = float(ca_tau)
    cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
    return cfg


def assemble_by_global_index(
    client_list: list[dict[str, Any]],
    imputed_clients: list[np.ndarray],
    shape: tuple[int, int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for cd, imputed in zip(client_list, imputed_clients):
        out[np.asarray(cd["indices"], dtype=int)] = imputed.astype(np.float32)
    return out


def mnar_metrics_for_clients(
    client_list: list[dict[str, Any]],
    imputed_clients: list[np.ndarray],
) -> dict[str, float]:
    maes, rmses = [], []
    for cd, imputed in zip(client_list, imputed_clients):
        metrics = compute_mnar_metrics(
            imputed_data=imputed,
            ground_truth=cd["ground_truth"],
            eval_mask=cd["eval_mask"],
        )
        maes.append(metrics["mae"])
        rmses.append(metrics["rmse"])
    return {
        "imputation_mae": float(np.nanmean(maes)),
        "imputation_rmse": float(np.nanmean(rmses)),
    }


def run_saits_imputation(
    client_list: list[dict[str, Any]],
    method: str,
    seed: int,
    device: str,
    rounds: int,
    checkpoint_dir: Path,
    ca_scale_factor: float,
    ca_tau: float,
) -> tuple[list[np.ndarray], dict[str, float]]:
    SAITSClient, SAITSFederatedServer = _require_saits_components()
    set_seed(seed)

    first = client_list[0]["ground_truth"]
    seq_length, num_features = first.shape[1], first.shape[2]
    num_clients = len(client_list)

    if method == "local_saits":
        cfg = build_saits_config(
            seq_length,
            num_features,
            num_clients,
            rounds,
            checkpoint_dir,
            aggregation="fedavg",
        )
        fed_cfg = cfg.get("federation", {})
        saits_cfg = cfg.get("saits", {})
        total_epochs = int(fed_cfg.get("rounds", 50)) * int(
            saits_cfg.get("local_epochs", 5)
        )
        imputed_clients = []
        for cd in client_list:
            client = SAITSClient(
                client_id=cd["client_id"],
                train_data=cd["dataset"],
                val_data=None,
                config=cfg,
                device=device,
            )
            client.model.set_training_params(
                epochs=total_epochs,
                patience=min(20, max(total_epochs - 1, 1)),
            )
            client.local_train()
            imputed_clients.append(
                client.impute(cd["observed_data"], cd["train_masks"])
            )
        return imputed_clients, mnar_metrics_for_clients(client_list, imputed_clients)

    aggregation = "fed_ca" if method == "fedsaits_ca" else "fedavg"
    cfg = build_saits_config(
        seq_length,
        num_features,
        num_clients,
        rounds,
        checkpoint_dir,
        aggregation=aggregation,
        ca_scale_factor=ca_scale_factor,
        ca_tau=ca_tau,
    )

    clients = [
        SAITSClient(
            client_id=cd["client_id"],
            train_data=cd["dataset"],
            val_data=None,
            config=cfg,
            device=device,
        )
        for cd in client_list
    ]
    server = SAITSFederatedServer(
        clients=clients,
        test_data=None,
        config=cfg,
        device=device,
    )
    server.train()

    imputed_clients = []
    for client, cd in zip(clients, client_list):
        cid = cd["client_id"]
        if hasattr(server, "_personalized_params") and cid in server._personalized_params:
            client.download_global_model(server._personalized_params[cid])
        elif server.global_params is not None:
            client.download_global_model(server.global_params)
        imputed_clients.append(
            client.impute(cd["observed_data"], cd["train_masks"])
        )

    return imputed_clients, mnar_metrics_for_clients(client_list, imputed_clients)


def run_ice_imputation(
    client_list: list[dict[str, Any]],
    method: str,
    seed: int,
    n_rounds: int,
    ridge_alpha: float,
    ca_scale_factor: float,
) -> tuple[list[np.ndarray], dict[str, float]]:
    set_seed(seed)

    if method == "local_ice":
        imputed_clients = []
        for cd in client_list:
            imputer = FedICEImputer(
                n_rounds=n_rounds,
                ridge_alpha=ridge_alpha,
                use_ca=False,
                seed=seed,
            )
            imputed_clients.append(
                imputer.fit_transform(
                    client_ground_truths=[cd["ground_truth"]],
                    client_masks=[cd["train_masks"]],
                )[0]
            )
    else:
        imputer = FedICEImputer(
            n_rounds=n_rounds,
            ridge_alpha=ridge_alpha,
            use_ca=(method == "fedice_ca"),
            ca_scale_factor=ca_scale_factor,
            seed=seed,
        )
        imputed_clients = imputer.fit_transform(
            client_ground_truths=[cd["ground_truth"] for cd in client_list],
            client_masks=[cd["train_masks"] for cd in client_list],
        )

    return imputed_clients, mnar_metrics_for_clients(client_list, imputed_clients)


def extract_tabular_features(
    x: np.ndarray,
    feature_names: list[str],
    mode: str = "flatten",
) -> tuple[np.ndarray, list[str]]:
    """
    Convert (N, T, D) time series into tabular features for XGBoost.

    ``flatten`` drops time metadata and flattens a fixed-length time-series
    segment into one tabular vector, matching the XGBoost baseline setup.

    ``summary`` uses interpretable statistics per vital sign:
    mean, std, min, max, first, last, and linear slope.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0)
    n, t, d = x.shape

    if mode == "flatten":
        feats = x.reshape(n, t * d)
        names = [
            f"t{time_idx:03d}_{feature_names[feature_idx]}"
            for time_idx in range(t)
            for feature_idx in range(d)
        ]
        return feats.astype(np.float32), names

    if mode != "summary":
        raise ValueError(f"Unknown feature mode: {mode}")

    time = np.arange(t, dtype=np.float32)
    time_centered = time - time.mean()
    denom = float((time_centered ** 2).sum()) or 1.0
    x_centered = x - x.mean(axis=1, keepdims=True)
    slope = (x_centered * time_centered[None, :, None]).sum(axis=1) / denom

    stats = {
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "min": x.min(axis=1),
        "max": x.max(axis=1),
        "first": x[:, 0, :],
        "last": x[:, -1, :],
        "slope": slope,
    }
    feats = np.concatenate(list(stats.values()), axis=1)
    names = [
        f"{fname}_{stat}"
        for stat in stats
        for fname in feature_names
    ]
    assert feats.shape == (n, len(names))
    return feats.astype(np.float32), names


def make_classifier(kind: str, seed: int):
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier

            # Match the simple XGBoost setup used in xgboost.ipynb.
            clf = XGBClassifier(
                objective="binary:logistic",
                max_depth=4,
                alpha=10,
                learning_rate=1.0,
                n_estimators=100,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
            )
            return clf, "xgboost"
        except ImportError:
            print("[warn] xgboost is not installed; falling back to HistGradientBoosting.")
            return HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                random_state=seed,
            ), "histgb"

    if kind == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=seed,
        ), "histgb"

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    return clf, "logistic"


def classifier_scores(clf, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = clf.predict(x_test).astype(int)
    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(x_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(x_test)
    else:
        score = pred.astype(float)
    return pred, score


def evaluate_downstream(
    x_imputed: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    classifier_kind: str,
    feature_names: list[str],
    feature_mode: str,
) -> dict[str, Any]:
    x_tab, tab_names = extract_tabular_features(
        x_imputed,
        feature_names,
        mode=feature_mode,
    )
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    clf, actual_classifier = make_classifier(classifier_kind, seed)
    clf.fit(x_tab[train_idx], y[train_idx])
    pred, score = classifier_scores(clf, x_tab[test_idx])

    metrics = {
        "classifier": actual_classifier,
        "feature_mode": feature_mode,
        "n_features": int(len(tab_names)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_positive_rate": float(y[test_idx].mean()),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y[test_idx], score))
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(y[test_idx], score))
    except ValueError:
        metrics["auprc"] = float("nan")
    return metrics


def summarize_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "imputation_mae",
        "imputation_rmse",
        "auroc",
        "auprc",
        "f1",
        "balanced_accuracy",
        "accuracy",
    ]
    grouped = (
        df.groupby(["scenario", "method", "method_label"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    grouped.columns = [
        "_".join([c for c in col if c]).rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading eICU tensor and labels...")
    ground_truth, _, masks, feature_names, _ = load_from_local_tensor(
        args.tensor_dir,
        normalize=True,
    )
    tensor_dir = Path(args.tensor_dir)
    case_ids = load_case_ids(tensor_dir)
    y, valid, label_info = load_labels(
        case_ids=case_ids,
        patient_csv=Path(args.patient_csv),
        label=args.label,
        los_hours=args.los_hours,
    )
    client_ids = None
    if args.client_ids_path:
        client_ids = np.load(args.client_ids_path).astype(int)

    # Drop records without labels.
    ground_truth = ground_truth[valid]
    masks = masks[valid]
    y = y[valid]
    case_ids = case_ids[valid]
    if client_ids is not None:
        client_ids = client_ids[valid]

    print(f"  Data: {ground_truth.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Label: {label_info['label_name']}")
    print(
        f"  Valid labels: {len(y)}, positives: {int(y.sum())} "
        f"({y.mean():.1%})"
    )

    rows: list[dict[str, Any]] = []
    metadata = {
        "args": vars(args),
        "label_info": label_info,
        "feature_names": feature_names,
        "case_ids_n": int(len(case_ids)),
    }

    saits_methods = {"local_saits", "fedsaits", "fedsaits_ca"}
    for scenario in args.scenarios:
        for seed in args.seeds:
            print("\n" + "=" * 72)
            print(f"[2/4] Preparing MNAR data: scenario={scenario}, seed={seed}")
            print("=" * 72)
            make_dataset = any(m in saits_methods for m in args.methods)
            set_seed(seed)
            client_list = prepare_client_data(
                ground_truth=ground_truth,
                masks=masks,
                feature_names=feature_names,
                scenario=scenario,
                mnar_method=args.mnar_method,
                target_features=args.target_features,
                missing_rate=args.missing_rate,
                seed=seed,
                num_clients=args.num_clients,
                client_ids=client_ids,
                make_dataset=make_dataset,
            )

            for method in args.methods:
                method_label = METHOD_LABELS[method]
                print("\n" + "-" * 72)
                print(f"[3/4] Imputation: {method_label}, scenario={scenario}, seed={seed}")
                print("-" * 72)

                ckpt_dir = (
                    out_dir
                    / "checkpoints"
                    / f"{scenario}_{args.mnar_method}_rho{args.missing_rate:g}"
                    / f"seed_{seed}"
                    / method
                )

                if method in saits_methods:
                    imputed_clients, imp_metrics = run_saits_imputation(
                        client_list=client_list,
                        method=method,
                        seed=seed,
                        device=device,
                        rounds=args.rounds,
                        checkpoint_dir=ckpt_dir,
                        ca_scale_factor=args.saits_ca_scale_factor,
                        ca_tau=args.ca_tau,
                    )
                else:
                    imputed_clients, imp_metrics = run_ice_imputation(
                        client_list=client_list,
                        method=method,
                        seed=seed,
                        n_rounds=args.fedice_rounds,
                        ridge_alpha=args.fedice_ridge_alpha,
                        ca_scale_factor=args.ice_ca_scale_factor,
                    )

                x_imputed = assemble_by_global_index(
                    client_list,
                    imputed_clients,
                    ground_truth.shape,
                )
                if args.save_imputed:
                    np.savez_compressed(
                        out_dir / f"imputed_{scenario}_seed{seed}_{method}.npz",
                        x_imputed=x_imputed,
                        y=y,
                        case_ids=case_ids,
                    )

                print(f"[4/4] Downstream classification: {args.label}")
                cls_metrics = evaluate_downstream(
                    x_imputed=x_imputed,
                    y=y,
                    seed=seed,
                    test_size=args.test_size,
                    classifier_kind=args.classifier,
                    feature_names=feature_names,
                    feature_mode=args.feature_mode,
                )

                row = {
                    "dataset": "eICU-demo",
                    "label": args.label,
                    "label_name": label_info["label_name"],
                    "scenario": scenario,
                    "mnar_method": args.mnar_method,
                    "missing_rate": args.missing_rate,
                    "seed": seed,
                    "method": method,
                    "method_label": method_label,
                    **imp_metrics,
                    **cls_metrics,
                }
                rows.append(row)
                print(
                    f"  {method_label}: MAE={row['imputation_mae']:.4f}, "
                    f"AUROC={row['auroc']:.4f}, AUPRC={row['auprc']:.4f}, "
                    f"F1={row['f1']:.4f}, BAcc={row['balanced_accuracy']:.4f}"
                )

                pd.DataFrame(rows).to_csv(
                    out_dir / "eicu_downstream_los_per_seed.csv",
                    index=False,
                )
                summarize_results(rows).to_csv(
                    out_dir / "eicu_downstream_los_summary.csv",
                    index=False,
                )
                with (out_dir / "eicu_downstream_los_metadata.json").open("w") as f:
                    json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Per-seed results: {out_dir / 'eicu_downstream_los_per_seed.csv'}")
    print(f"Summary results:  {out_dir / 'eicu_downstream_los_summary.csv'}")


if __name__ == "__main__":
    main()
