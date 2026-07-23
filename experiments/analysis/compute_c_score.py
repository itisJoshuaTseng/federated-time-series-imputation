"""
Phase 1 Step 1 — Phase A: compute C_score for existing S1/S4 settings.

Purpose: before running any new experiments at S2/S3, verify that the
CAFE-style fingerprint complementarity gives sensible numeric values
on the settings we already know (S1, S4). This tells us:

  - What C_score S1 and S4 actually have (we think S4 ≈ 0, S1 higher)
  - How stable C_score is across seeds
  - What range of C_score the new S2/S3 runs need to fill

No training; just allocation + logistic-regression fingerprint +
(1 - cos)/2 matrix, averaged over off-diagonal.

Run from federated learning/:
  python experiments/compute_c_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.heterogeneous_allocator import HeterogeneousDataAllocator
from src.data.vitaldb_loader import load_from_local_tensor


TENSOR_DIR = str(
    REPO.parent / "2026_vitalDB" / "vitaldb_14feats_tensor_T300"
)

SETTINGS = [
    ("S1", "quantile", 0.3), ("S1", "quantile", 0.5),
    ("S1", "quantile", 0.7), ("S1", "logit", 0.3),
    ("S4", "quantile", 0.3), ("S4", "quantile", 0.5),
    ("S4", "quantile", 0.7), ("S4", "logit", 0.3),
]
SEEDS = [0, 1, 2, 3, 4]


def fit_fingerprint(
    ground_truth: np.ndarray,
    masks: np.ndarray,
    subsample: int = 20000,
    rng_seed: int = 0,
) -> np.ndarray:
    """CAFE-style fingerprint, faster variant.

    Logistic regression on a random subsample (liblinear, fixed C=1,
    balanced). The full dataset is 150K rows × 14 features; a 20K
    subsample gives >3 orders of magnitude more data than LR needs
    to converge, and matches the SAITS client's fingerprint well
    enough for our analysis (we've checked: cosine similarity between
    full and 20K-subsampled fingerprints is > 0.99 across clients).
    """
    from sklearn.linear_model import LogisticRegression

    N, T, D = ground_truth.shape
    X_full = ground_truth.reshape(-1, D).astype(float)
    M_full = masks.reshape(-1, D).astype(int)

    if subsample and len(X_full) > subsample:
        rng = np.random.RandomState(rng_seed)
        idx = rng.choice(len(X_full), subsample, replace=False)
        X = X_full[idx]
        M = M_full[idx]
    else:
        X = X_full
        M = M_full

    seg_len = D + 1
    fp = np.zeros(D * seg_len, dtype=np.float64)
    for d in range(D):
        y_d = (1 - M[:, d]).astype(int)
        n_pos = int(y_d.sum())
        n_neg = len(y_d) - n_pos
        if n_pos < 5 or n_neg < 5:
            continue
        try:
            lr = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=300,
                class_weight="balanced",
            )
            lr.fit(X, y_d)
            seg = np.concatenate([lr.coef_[0], lr.intercept_])
            fp[d * seg_len : (d + 1) * seg_len] = seg
        except Exception:
            pass
    return fp


def complementarity_matrix(fingerprints: list) -> np.ndarray:
    P = np.stack(fingerprints)
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    Pn = P / norms
    cos_sim = Pn @ Pn.T
    return (1.0 - cos_sim) / 2.0


def scalar_c_score(comp_matrix: np.ndarray) -> float:
    """Mean of upper-triangular off-diagonal entries."""
    N = comp_matrix.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(comp_matrix[iu].mean())


def main():
    print("Loading VitalDB tensor...")
    gt, obs, masks, feat_names, _ = load_from_local_tensor(
        tensor_dir=TENSOR_DIR, normalize=True,
    )
    print(f"  ground_truth={gt.shape}, masks={masks.shape}")

    # Use training portion as the allocator does (full data, no split here
    # since that matches experiment's allocator call).
    allocator = HeterogeneousDataAllocator(
        X=gt, masks=masks, num_clients=5, feature_names=feat_names,
    )

    print("\nComputing C_score per (scenario, mnar_method, rho, seed)...")
    print(f"{'setting':<22} {'seeds mean ± std':<22} {'range':<22}")
    print("-" * 66)

    all_results = {}
    for scen, mnar, rho in SETTINGS:
        c_scores = []
        for seed in SEEDS:
            client_data = allocator.allocate(
                scenario=scen, mnar_method=mnar,
                target_features=[0, 2, 6],
                missing_rate=rho, seed=seed,
            )
            fps = []
            for cd in client_data:
                fp = fit_fingerprint(cd["X"], cd["masks"])
                fps.append(fp)
            M = complementarity_matrix(fps)
            c_scores.append(scalar_c_score(M))

        m = mean(c_scores)
        s = stdev(c_scores) if len(c_scores) > 1 else 0.0
        tag = f"{scen} {mnar[:1]}  rho={rho}"
        rng = f"[{min(c_scores):.3f}, {max(c_scores):.3f}]"
        print(f"{tag:<22} {m:.3f} ± {s:.3f}          {rng:<22}")
        all_results[(scen, mnar, rho)] = c_scores

    return all_results


if __name__ == "__main__":
    main()
