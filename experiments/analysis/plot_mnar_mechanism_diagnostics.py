#!/usr/bin/env python3
"""
Mechanism diagnostics for quantile-vs-logit MNAR behavior.

This script makes two diagnostic figures that do not require trained model
checkpoints:

  A. Observed-vs-hidden value distributions.
  B. CAFE-style client complementarity heatmaps from logistic missingness
     fingerprints.

The default rho=0.5 is intentional: it is the clearest explanatory setting and
aligns with the local-ablation figure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import warnings


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.heterogeneous_allocator import HeterogeneousDataAllocator
from src.data.vitaldb_loader import load_from_local_tensor


SCENARIOS = ["S1", "S2", "S3", "S4"]
MNAR_METHODS = ["quantile", "logit"]
DEFAULT_TARGET_FEATURES = [0, 2, 6]  # HR, NIBP_SBP, SpO2

COLORS = {
    "observed": "#4c78a8",
    "hidden": "#e45756",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tensor-dir",
        default="../2026_vitalDB/vitaldb_14feats_tensor_T300",
        help="Path to preprocessed VitalDB tensor directory.",
    )
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-features",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_FEATURES,
    )
    p.add_argument("--max-hist-points", type=int, default=200_000)
    p.add_argument("--max-lr-rows", type=int, default=30_000)
    p.add_argument(
        "--output-dir",
        default="experiments/figures/mechanism_diagnostics",
    )
    return p.parse_args()


def sample_values(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= max_points:
        return values
    rng = np.random.RandomState(seed)
    idx = rng.choice(values.size, size=max_points, replace=False)
    return values[idx]


def collect_setting(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    scenario: str,
    mnar_method: str,
    rho: float,
    seed: int,
    target_features: list[int],
) -> list[dict]:
    allocator = HeterogeneousDataAllocator(
        X=ground_truth,
        masks=masks,
        num_clients=5,
        feature_names=feature_names,
    )
    return allocator.allocate(
        scenario=scenario,
        mnar_method=mnar_method,
        target_features=target_features,
        missing_rate=rho,
        seed=seed,
    )


def values_for_distribution(client_data: list[dict], target_features: list[int]) -> tuple[np.ndarray, np.ndarray]:
    observed_values = []
    hidden_values = []
    for cd in client_data:
        x = cd["X"]
        orig = cd["orig_masks"].astype(bool)
        after = cd["masks"].astype(bool)
        hidden = orig & ~after
        for f in target_features:
            observed_values.append(x[..., f][orig[..., f]])
            hidden_values.append(x[..., f][hidden[..., f]])
    return np.concatenate(observed_values), np.concatenate(hidden_values)


def plot_distribution_grid(
    settings: dict[tuple[str, str], list[dict]],
    target_features: list[int],
    rho: float,
    seed: int,
    max_hist_points: int,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True, sharey=True)

    all_vals = []
    for key, client_data in settings.items():
        obs, hid = values_for_distribution(client_data, target_features)
        all_vals.append(sample_values(obs, max_hist_points, seed))
        all_vals.append(sample_values(hid, max_hist_points, seed + 1))
    all_vals = np.concatenate(all_vals)
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)

    for r, mnar_method in enumerate(MNAR_METHODS):
        for c, scenario in enumerate(SCENARIOS):
            ax = axes[r, c]
            client_data = settings[(mnar_method, scenario)]
            obs, hid = values_for_distribution(client_data, target_features)
            obs_s = sample_values(obs, max_hist_points, seed)
            hid_s = sample_values(hid, max_hist_points, seed + 17)

            ax.hist(
                obs_s,
                bins=bins,
                density=True,
                alpha=0.35,
                color=COLORS["observed"],
                label="Originally observed",
            )
            ax.hist(
                hid_s,
                bins=bins,
                density=True,
                alpha=0.55,
                color=COLORS["hidden"],
                label="Hidden by MNAR",
            )
            ax.set_title(f"{mnar_method}, {scenario}", fontsize=11)
            ax.grid(axis="y", alpha=0.2)
            if c == 0:
                ax.set_ylabel("Density")
            if r == 1:
                ax.set_xlabel("Normalized value")

            rows.append({
                "mnar_method": mnar_method,
                "scenario": scenario,
                "rho": rho,
                "observed_n": int(obs.size),
                "hidden_n": int(hid.size),
                "observed_mean": float(np.nanmean(obs)),
                "hidden_mean": float(np.nanmean(hid)),
                "observed_q10": float(np.nanpercentile(obs, 10)),
                "hidden_q10": float(np.nanpercentile(hid, 10)),
                "observed_q50": float(np.nanpercentile(obs, 50)),
                "hidden_q50": float(np.nanpercentile(hid, 50)),
                "observed_q90": float(np.nanpercentile(obs, 90)),
                "hidden_q90": float(np.nanpercentile(hid, 90)),
            })

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"A. Observed vs MNAR-hidden value distributions (rho={rho:.1f})", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figA_observed_vs_hidden_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figA_observed_vs_hidden_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "figA_distribution_stats.csv", index=False)
    return stats


def estimate_fingerprint(
    x: np.ndarray,
    mask: np.ndarray,
    target_features: list[int],
    max_rows: int,
    seed: int,
) -> np.ndarray:
    n, t, d = x.shape
    flat_x = np.nan_to_num(x.reshape(-1, d), nan=0.0, posinf=0.0, neginf=0.0)
    flat_m = mask.reshape(-1, d).astype(bool)

    rng = np.random.RandomState(seed)
    if flat_x.shape[0] > max_rows:
        idx = rng.choice(flat_x.shape[0], size=max_rows, replace=False)
        flat_x = flat_x[idx]
        flat_m = flat_m[idx]

    seg_len = d + 1
    fingerprint = np.zeros(len(target_features) * seg_len, dtype=float)
    for out_idx, f in enumerate(target_features):
        y = (~flat_m[:, f]).astype(int)
        if np.unique(y).size < 2:
            continue
        try:
            lr = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                lr.fit(flat_x, y)
            segment = np.concatenate([lr.coef_.reshape(-1), lr.intercept_])
            fingerprint[out_idx * seg_len : (out_idx + 1) * seg_len] = segment
        except Exception:
            continue
    return fingerprint


def complementarity_matrix(fingerprints: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8
    p = fingerprints / norms
    cos = p @ p.T
    return (1.0 - cos) / 2.0


def plot_complementarity_grid(
    settings: dict[tuple[str, str], list[dict]],
    target_features: list[int],
    rho: float,
    seed: int,
    max_lr_rows: int,
    output_dir: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.2))
    score_rows = []

    for r, mnar_method in enumerate(MNAR_METHODS):
        for c, scenario in enumerate(SCENARIOS):
            ax = axes[r, c]
            fps = []
            for cd in settings[(mnar_method, scenario)]:
                fps.append(
                    estimate_fingerprint(
                        cd["X"],
                        cd["masks"],
                        target_features,
                        max_rows=max_lr_rows,
                        seed=seed + cd["client_id"] * 31,
                    )
                )
            comp = complementarity_matrix(np.stack(fps, axis=0))
            off_diag = comp[~np.eye(comp.shape[0], dtype=bool)]
            mean_comp = float(off_diag.mean())

            im = ax.imshow(comp, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(f"{mnar_method}, {scenario}\nmean={mean_comp:.2f}", fontsize=10)
            ax.set_xticks(range(comp.shape[0]))
            ax.set_yticks(range(comp.shape[0]))
            ax.set_xticklabels([f"C{i}" for i in range(comp.shape[0])], fontsize=8)
            ax.set_yticklabels([f"C{i}" for i in range(comp.shape[0])], fontsize=8)
            for i in range(comp.shape[0]):
                for j in range(comp.shape[1]):
                    ax.text(j, i, f"{comp[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if comp[i, j] > 0.55 else "black")

            for i in range(comp.shape[0]):
                for j in range(comp.shape[1]):
                    score_rows.append({
                        "mnar_method": mnar_method,
                        "scenario": scenario,
                        "rho": rho,
                        "client_i": i,
                        "client_j": j,
                        "complementarity": float(comp[i, j]),
                    })

    fig.suptitle(f"B. CAFE-style client complementarity heatmaps (rho={rho:.1f})", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.96, 0.93])
    cbar_ax = fig.add_axes([0.965, 0.18, 0.012, 0.68])
    fig.colorbar(im, cax=cbar_ax, label="Complementarity score")
    fig.savefig(output_dir / "figB_complementarity_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figB_complementarity_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(output_dir / "figB_complementarity_scores.csv", index=False)
    return scores


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth, _, masks, feature_names, _ = load_from_local_tensor(args.tensor_dir, normalize=True)
    settings = {}
    for mnar_method in MNAR_METHODS:
        for scenario in SCENARIOS:
            settings[(mnar_method, scenario)] = collect_setting(
                ground_truth,
                masks,
                feature_names,
                scenario,
                mnar_method,
                args.rho,
                args.seed,
                args.target_features,
            )

    plot_distribution_grid(
        settings,
        args.target_features,
        args.rho,
        args.seed,
        args.max_hist_points,
        output_dir,
    )
    plot_complementarity_grid(
        settings,
        args.target_features,
        args.rho,
        args.seed,
        args.max_lr_rows,
        output_dir,
    )

    print(f"Saved MNAR mechanism diagnostics to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
