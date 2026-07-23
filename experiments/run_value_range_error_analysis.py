#!/usr/bin/env python3
"""
Targeted value-range error analysis for the MNAR crossover story.

This script reruns selected settings and computes MAE by fixed ground-truth
value ranges (low/mid/high) immediately after each method finishes.  The ranges
are defined from the original observed distribution, not from the induced MNAR
holes, so the bins are comparable across quantile/logit mechanisms.  It is
needed because the existing result JSONs only store aggregate MAE, and
Fed-SAITS-CA's personalized parameters are not fully recoverable from the saved
checkpoints.

Recommended server run:
  python experiments/run_value_range_error_analysis.py \
    --settings quantile:S4:0.5 logit:S4:0.5 \
    --seeds 0 1 2 \
    --methods fedavg fed_ca fedice fedice_ca
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_mnar_experiment import (  # noqa: E402
    DEFAULT_CONFIG,
    _require_saits_components,
    prepare_client_data,
    set_seed,
)
from src.baselines import FedICEImputer  # noqa: E402
from src.data.vitaldb_loader import load_from_local_tensor  # noqa: E402


METHOD_LABELS = {
    "fedavg": "FedSAITS",
    "fed_ca": "FedSAITS+CA",
    "fedice": "FedICE",
    "fedice_ca": "FedICE+CA",
}

COLORS = {
    "FedSAITS": "#1f77b4",
    "FedSAITS+CA": "#2ca02c",
    "FedICE": "#9467bd",
    "FedICE+CA": "#d62728",
}

BIN_LABELS = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tensor-dir",
        default="../2026_vitalDB/vitaldb_14feats_tensor_T300",
    )
    p.add_argument(
        "--settings",
        nargs="+",
        default=["quantile:S4:0.5", "logit:S4:0.5"],
        help="Each setting is mnar_method:scenario:rho",
    )
    p.add_argument("--target-features", type=int, nargs="+", default=[0, 2, 6])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["fedavg", "fed_ca", "fedice", "fedice_ca"],
        choices=["fedavg", "fed_ca", "fedice", "fedice_ca"],
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--ca-scale-factor", type=float, default=0.5)
    p.add_argument("--fedice-ca-scale-factor", type=float, default=4.0)
    p.add_argument("--fedice-rounds", type=int, default=20)
    p.add_argument("--fedice-ridge-alpha", type=float, default=1.0)
    p.add_argument(
        "--output-dir",
        default="experiments/figures/value_range_error",
    )
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_setting(setting: str) -> tuple[str, str, float]:
    parts = setting.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad setting {setting!r}; expected method:scenario:rho")
    mnar_method, scenario, rho = parts
    return mnar_method, scenario, float(rho)


def collect_reference_values(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    target_features: list[int],
) -> np.ndarray:
    vals = []
    observed = masks.astype(bool)
    for f in target_features:
        vals.append(ground_truth[..., f][observed[..., f]])
    vals = np.concatenate(vals)
    return vals[np.isfinite(vals)]


def bin_edges(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.nanquantile(values, [1 / 3, 2 / 3])
    return np.array([-np.inf, q1, q2, np.inf], dtype=float)


def range_error_rows(
    imputed_clients: list[np.ndarray],
    client_list: list[dict],
    range_specs: list[dict],
) -> list[dict]:
    rows = []
    per_spec_errors = {
        spec["range_group"]: {label: [] for label in BIN_LABELS}
        for spec in range_specs
    }
    per_spec_counts = {
        spec["range_group"]: {label: 0 for label in BIN_LABELS}
        for spec in range_specs
    }

    for imputed, cd in zip(imputed_clients, client_list):
        gt = cd["ground_truth"]
        mask = cd["eval_mask"].astype(bool)

        for spec in range_specs:
            edges = spec["edges"]
            for f in spec["target_features"]:
                valid = mask[..., f] & np.isfinite(gt[..., f])
                if not np.any(valid):
                    continue
                y = gt[..., f][valid]
                err = np.abs(imputed[..., f][valid] - y)
                for idx, label in enumerate(BIN_LABELS):
                    in_bin = (y >= edges[idx]) & (y < edges[idx + 1])
                    n_hidden = int(in_bin.sum())
                    if n_hidden > 0:
                        per_spec_errors[spec["range_group"]][label].append(err[in_bin])
                        per_spec_counts[spec["range_group"]][label] += n_hidden

    for spec in range_specs:
        group = spec["range_group"]
        edges = spec["edges"]
        for idx, label in enumerate(BIN_LABELS):
            chunks = per_spec_errors[group][label]
            rows.append({
                "range_group": group,
                "feature_idx": spec["feature_idx"],
                "feature_name": spec["feature_name"],
                "bin_source": spec["bin_source"],
                "value_bin": label,
                "mae": float(np.concatenate(chunks).mean()) if chunks else float("nan"),
                "n_hidden": per_spec_counts[group][label],
                "edge_low": edges[idx],
                "edge_high": edges[idx + 1],
            })
    return rows


def build_range_specs(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    target_features: list[int],
) -> list[dict]:
    specs = [{
        "range_group": "all_features_global_bins",
        "feature_idx": -1,
        "feature_name": "All target features",
        "target_features": list(target_features),
        "bin_source": "original_observed_all_target_features",
        "edges": bin_edges(collect_reference_values(ground_truth, masks, target_features)),
    }]

    for f in target_features:
        name = feature_names[f] if f < len(feature_names) else f"feature_{f}"
        specs.append({
            "range_group": f"feature_{f}",
            "feature_idx": f,
            "feature_name": name,
            "target_features": [f],
            "bin_source": "original_observed_per_feature",
            "edges": bin_edges(collect_reference_values(ground_truth, masks, [f])),
        })
    return specs


def run_saits_method(
    client_list: list[dict],
    method: str,
    device: str,
    checkpoint_dir: str,
    ca_scale_factor: float,
) -> list[np.ndarray]:
    SAITSClient, SAITSFederatedServer = _require_saits_components()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["training"]["checkpoint_dir"] = checkpoint_dir
    cfg["federation"]["aggregation"] = method
    cfg["federation"]["mu"] = 0.0
    if method == "fed_ca":
        cfg["federation"]["ca_tau"] = 1.0
        cfg["federation"]["ca_scale_factor"] = ca_scale_factor

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
    server = SAITSFederatedServer(clients=clients, test_data=None, config=cfg, device=device)
    server.train()

    imputed_clients = []
    for i, cd in enumerate(client_list):
        cid = cd["client_id"]
        if hasattr(server, "_personalized_params") and cid in server._personalized_params:
            clients[i].download_global_model(server._personalized_params[cid])
        elif server.global_params is not None:
            clients[i].download_global_model(server.global_params)
        imputed_clients.append(
            clients[i].impute(cd["observed_data"], cd["train_masks"])
        )
    return imputed_clients


def run_ice_method(
    client_list: list[dict],
    method: str,
    seed: int,
    n_rounds: int,
    ridge_alpha: float,
    fedice_ca_scale_factor: float,
) -> list[np.ndarray]:
    imputer = FedICEImputer(
        n_rounds=n_rounds,
        ridge_alpha=ridge_alpha,
        use_ca=(method == "fedice_ca"),
        ca_scale_factor=fedice_ca_scale_factor,
        seed=seed,
    )
    return imputer.fit_transform(
        client_ground_truths=[cd["ground_truth"] for cd in client_list],
        client_masks=[cd["train_masks"] for cd in client_list],
    )


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    summary = summary[summary["range_group"] == "all_features_global_bins"]
    settings = list(summary["setting"].drop_duplicates())
    methods = [
        METHOD_LABELS[m]
        for m in ["fedavg", "fed_ca", "fedice", "fedice_ca"]
        if METHOD_LABELS[m] in set(summary["method_label"])
    ]
    x = np.arange(len(BIN_LABELS))
    width = 0.18

    fig, axes = plt.subplots(1, len(settings), figsize=(6.2 * len(settings), 4.2), sharey=True)
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        data = summary[summary["setting"] == setting]
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
        for i, method_label in enumerate(methods):
            rows = data[data["method_label"] == method_label].set_index("value_bin")
            vals = [rows.loc[b, "mean_mae"] if b in rows.index else np.nan for b in BIN_LABELS]
            errs = [rows.loc[b, "std_mae"] if b in rows.index else 0.0 for b in BIN_LABELS]
            ax.bar(
                x + offsets[i],
                vals,
                width=width,
                yerr=errs,
                capsize=2,
                color=COLORS[method_label],
                label=method_label,
            )
        ax.set_title(setting)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel("Original observed value range")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("MAE on induced MNAR holes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("C. Error by fixed original-observed value range", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=len(methods), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(output_dir / "figC_error_by_value_range.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figC_error_by_value_range.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_per_feature_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    per_feature = summary[summary["range_group"] != "all_features_global_bins"].copy()
    if per_feature.empty:
        return

    settings = list(per_feature["setting"].drop_duplicates())
    features = (
        per_feature[["feature_idx", "feature_name"]]
        .drop_duplicates()
        .sort_values("feature_idx")
        .to_dict("records")
    )
    methods = [
        METHOD_LABELS[m]
        for m in ["fedavg", "fed_ca", "fedice", "fedice_ca"]
        if METHOD_LABELS[m] in set(per_feature["method_label"])
    ]

    x = np.arange(len(BIN_LABELS))
    width = 0.18
    fig, axes = plt.subplots(
        len(settings),
        len(features),
        figsize=(5.0 * len(features), 3.5 * len(settings)),
        sharey=True,
        squeeze=False,
    )

    for r, setting in enumerate(settings):
        for c, feature in enumerate(features):
            ax = axes[r][c]
            data = per_feature[
                (per_feature["setting"] == setting)
                & (per_feature["feature_idx"] == feature["feature_idx"])
            ]
            offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
            for i, method_label in enumerate(methods):
                rows = data[data["method_label"] == method_label].set_index("value_bin")
                vals = [rows.loc[b, "mean_mae"] if b in rows.index else np.nan for b in BIN_LABELS]
                errs = [rows.loc[b, "std_mae"] if b in rows.index else 0.0 for b in BIN_LABELS]
                ax.bar(
                    x + offsets[i],
                    vals,
                    width=width,
                    yerr=errs,
                    capsize=2,
                    color=COLORS[method_label],
                    label=method_label,
                )
            ax.set_title(f"{setting} | {feature['feature_name']}")
            ax.set_xticks(x)
            ax.set_xticklabels(BIN_LABELS)
            ax.grid(axis="y", alpha=0.25)
            if c == 0:
                ax.set_ylabel("MAE on induced MNAR holes")
            if r == len(settings) - 1:
                ax.set_xlabel("Per-feature original observed range")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("C. Per-feature error by fixed original-observed value range", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(methods), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_dir / "figC_error_by_value_range_per_feature.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figC_error_by_value_range_per_feature.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth, _, masks, feature_names, _ = load_from_local_tensor(args.tensor_dir, normalize=True)
    range_specs = build_range_specs(
        ground_truth=ground_truth,
        masks=masks,
        feature_names=feature_names,
        target_features=args.target_features,
    )

    raw_rows = []
    start = time.time()
    for setting_str in args.settings:
        mnar_method, scenario, rho = parse_setting(setting_str)
        setting_label = f"{mnar_method}-{scenario}-rho{rho:.1f}"
        for seed in args.seeds:
            set_seed(seed)
            make_dataset = any(m in args.methods for m in ["fedavg", "fed_ca"])
            client_list = prepare_client_data(
                ground_truth=ground_truth,
                masks=masks,
                feature_names=feature_names,
                scenario=scenario,
                mnar_method=mnar_method,
                target_features=args.target_features,
                missing_rate=rho,
                seed=seed,
                make_dataset=make_dataset,
            )
            for method in args.methods:
                print(f"\n== {setting_label}, seed={seed}, method={method} ==")
                if method in {"fedavg", "fed_ca"}:
                    ckpt_dir = str(output_dir / "checkpoints" / setting_label / f"seed_{seed}" / method)
                    imputed_clients = run_saits_method(
                        client_list,
                        method=method,
                        device=device,
                        checkpoint_dir=ckpt_dir,
                        ca_scale_factor=args.ca_scale_factor,
                    )
                else:
                    imputed_clients = run_ice_method(
                        client_list,
                        method=method,
                        seed=seed,
                        n_rounds=args.fedice_rounds,
                        ridge_alpha=args.fedice_ridge_alpha,
                        fedice_ca_scale_factor=args.fedice_ca_scale_factor,
                    )

                metrics = range_error_rows(
                    imputed_clients,
                    client_list,
                    range_specs,
                )
                for metric_row in metrics:
                    raw_rows.append({
                        "setting": setting_label,
                        "mnar_method": mnar_method,
                        "scenario": scenario,
                        "rho": rho,
                        "seed": seed,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        **metric_row,
                    })

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output_dir / "figC_error_by_value_range_raw.csv", index=False)
    summary = (
        raw.groupby([
            "setting",
            "mnar_method",
            "scenario",
            "rho",
            "method",
            "method_label",
            "range_group",
            "feature_idx",
            "feature_name",
            "bin_source",
            "value_bin",
            "edge_low",
            "edge_high",
        ], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            n_seeds=("seed", "nunique"),
            mean_n_hidden=("n_hidden", "mean"),
            total_n_hidden=("n_hidden", "sum"),
        )
    )
    summary.to_csv(output_dir / "figC_error_by_value_range_summary.csv", index=False)
    plot_summary(summary, output_dir)
    plot_per_feature_summary(summary, output_dir)

    payload = {
        "settings": args.settings,
        "seeds": args.seeds,
        "methods": args.methods,
        "target_features": args.target_features,
        "binning": {
            "aggregate": "tertiles of original observed values across all target features",
            "per_feature": "tertiles of original observed values within each target feature",
        },
        "ca_scale_factor": args.ca_scale_factor,
        "fedice_ca_scale_factor": args.fedice_ca_scale_factor,
        "elapsed_sec": time.time() - start,
    }
    with (output_dir / "figC_error_by_value_range_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved value-range error analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
