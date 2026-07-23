"""
Figure H: Pairwise Complementarity Score Heatmap
Corresponds to CAFE Figure 4 — verifies that our CA mechanism correctly
detects complementarity between clients.

Run from: federated learning/
  python experiments/plot_complementarity_heatmap.py
"""

import sys
import os
sys.path.insert(0, ".")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

matplotlib.rcParams.update({
    "font.family":      "Microsoft YaHei",
    "font.size":        11,
    "axes.titlesize":   12,
    "figure.dpi":       150,
})

from src.data.vitaldb_loader import load_from_local_tensor
from src.data.heterogeneous_allocator import HeterogeneousDataAllocator
from src.federation.aggregation import compute_complementarity_matrix
from pathlib import Path

FIG = Path("experiments/figures")
FIG.mkdir(exist_ok=True)

TENSOR_DIR = ("../2026_vitalDB/tensor-file-for-4feature-20260304T112438Z-3-001"
              "/tensor-file-for-4feature/vitaldb_14feats_tensor_T300")


def compute_mechanism_coefs(X_gt, masks, num_features=14):
    """
    Fit per-feature logistic regression on (gt, mask) to get mechanism fingerprint.
    Same logic as SAITSClient.get_mechanism_coefs().
    """
    N, T, D = X_gt.shape
    X_flat = X_gt.reshape(-1, D).astype(float)
    M_flat = masks.reshape(-1, D).astype(int)

    seg_len = D + 1
    fingerprint = np.zeros(D * seg_len, dtype=np.float64)

    for d in range(D):
        y = (1 - M_flat[:, d]).astype(int)  # 1=missing, 0=observed (matches SAITSClient)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos < 5 or n_neg < 5:
            continue
        try:
            lr = LogisticRegressionCV(
                Cs=[1.0],
                cv=StratifiedKFold(3),
                random_state=0,
                max_iter=500,
                n_jobs=1,
                class_weight="balanced",   # critical: VitalDB has high natural
                                           # missing rates → severe class imbalance
            )
            lr.fit(X_flat, y)
            start = d * seg_len
            fingerprint[start:start + D] = lr.coef_[0]
            fingerprint[start + D] = lr.intercept_[0]
        except Exception:
            pass

    return fingerprint


def get_client_coefs(scenario, mnar_method, missing_rate=0.3, seed=42,
                     gt=None, masks=None, feature_names=None):
    """Allocate data to clients and compute mechanism coefs for each."""
    allocator = HeterogeneousDataAllocator(
        X=gt, masks=masks, num_clients=5, feature_names=feature_names
    )
    client_data = allocator.allocate(
        scenario=scenario,
        mnar_method=mnar_method,
        missing_rate=missing_rate,
        seed=seed,
    )
    coefs = []
    for cd in client_data:
        coef = compute_mechanism_coefs(cd["X"], cd["masks"])
        coefs.append(coef)
    return coefs, client_data


def plot_heatmap(ax, comp_matrix, title, client_directions=None):
    """Plot a single complementarity matrix as heatmap."""
    n = comp_matrix.shape[0]
    im = ax.imshow(comp_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = comp_matrix[i, j]
            color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if client_directions:
        xlabels = [f"C{i}\n({d})" for i, d in enumerate(client_directions)]
        ylabels = [f"C{i} ({d})" for i, d in enumerate(client_directions)]
    else:
        xlabels = [f"Client {i}" for i in range(n)]
        ylabels = [f"Client {i}" for i in range(n)]

    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    return im


def main():
    print("Loading VitalDB data...")
    gt, _, masks, feature_names, _ = load_from_local_tensor(
        TENSOR_DIR, normalize=True
    )
    print(f"  Loaded: {gt.shape}")

    # ── 4 panels: S1 quantile / S1 logit / S4 quantile / S4 logit ──────────
    configs = [
        ("S1", "quantile", 0.5,
         ["L", "R", "R", "R", "R"],   # L=MNAR-Left, R=MNAR-Right
         "S1 (quantile, ρ=0.5)\n完美互補：C0 Left, C1–4 Right"),
        ("S1", "logit",    0.3,
         ["L", "R", "R", "R", "R"],
         "S1 (logit, ρ=0.3)\n完美互補：C0 Left, C1–4 Right"),
        ("S4", "quantile", 0.5,
         ["L", "L", "L", "L", "L"],
         "S4 (quantile, ρ=0.5)\n無互補：所有 Client Left"),
        ("S4", "logit",    0.3,
         ["L*", "L*", "L*", "L*", "L*"],
         "S4 (logit, ρ=0.3)\n名義無互補：各 Client logit weights 不同"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    fig.suptitle(
        "Figure H：Pairwise Complementarity Score Heatmap\n"
        "（值域 0–1；0=完全相同機制，1=完全相反機制）\n"
        "對應 CAFE Fig. 4 — 驗證 CA 機制正確偵測互補性",
        fontsize=12, fontweight="bold", y=1.05
    )

    im_last = None
    for ax, (scenario, method, rho, directions, title) in zip(axes, configs):
        print(f"\nComputing: {scenario} {method} ρ={rho}...")
        coefs, _ = get_client_coefs(
            scenario, method, missing_rate=rho, seed=42,
            gt=gt, masks=masks, feature_names=feature_names
        )
        comp = compute_complementarity_matrix(coefs)
        im_last = plot_heatmap(ax, comp, title, client_directions=directions)
        mean_off_diag = comp[~np.eye(5, dtype=bool)].mean()
        print(f"  Mean off-diagonal complementarity: {mean_off_diag:.3f}")

    # Shared colorbar
    cbar = fig.colorbar(im_last, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Complementarity Score\n(0=相同機制, 1=完全互補)", fontsize=10)

    plt.tight_layout()
    out = FIG / "figH_complementarity_heatmap.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
