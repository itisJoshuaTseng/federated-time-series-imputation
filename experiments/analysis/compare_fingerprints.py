"""
Phase 1 Step 4 — Fingerprint quality comparison.

Hypothesis: S1 logit ρ=0.3 looks "mild" (Δ_CA maj ≈ −4%) not because the
underlying MNAR structure lacks complementarity, but because the
CAFE-style LR fingerprint fails to capture it. A nonlinear fingerprint
should reveal sharper complementarity (bigger C_score, more skewed CA
weight matrix).

For two settings — S1 q ρ=0.3 (LR baseline sees it, ends in disaster)
and S1 l ρ=0.3 (LR baseline misses it) — compute fingerprints with:
  1. LR  : logistic regression on mask (current baseline)
  2. MI  : mutual information mask ↔ feature values (nonparametric,
           captures nonlinear dependence)
  3. RF  : random-forest feature importance (tree-based, nonlinear)

Each fingerprint type produces a 5-client fingerprint array, a 5×5
complementarity matrix, and a 5×5 CA weight matrix (at sf=4). Compare
across (method, setting) grid.

Output:
  - figures/phase1_step4_fingerprint_compare.png (3 rows × 2 cols)
  - stdout summary table (C_score, avg maj→C0)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.heterogeneous_allocator import HeterogeneousDataAllocator
from src.data.vitaldb_loader import load_from_local_tensor
from src.federation.aggregation import compute_complementarity_matrix

matplotlib.rcParams.update({
    "font.family":      ["Heiti TC", "Hiragino Sans", "Arial Unicode MS"],
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})

TENSOR_DIR = str(REPO.parent / "2026_vitalDB" / "vitaldb_14feats_tensor_T300")
FIG_DIR = REPO / "experiments" / "figures"

SETTINGS = [
    ("S1", "quantile", 0.3, "S1 q ρ=0.3 (reference: hard MNAR)"),
    ("S1", "logit",    0.3, "S1 l ρ=0.3 (soft MNAR, weak)"),
    ("S1", "logit",    0.5, "S1 l ρ=0.5 (soft MNAR, mid)"),
    ("S1", "logit",    0.7, "S1 l ρ=0.7 (soft MNAR, high)"),
]

METHODS = ["LR", "MI", "RF"]

ALPHA = 0.95
SEED = 0
SCALE_FACTOR = 4
SUBSAMPLE = 10000   # keep runtime sane for MI/RF


# ------------- Fingerprint backends ------------------------------------


def fit_fingerprint_LR(X, M, rng_seed=0):
    """Logistic regression on mask — mirrors compute_c_score.py."""
    from sklearn.linear_model import LogisticRegression
    D = X.shape[1]
    seg_len = D + 1
    fp = np.zeros(D * seg_len, dtype=np.float64)
    for d in range(D):
        y_d = (1 - M[:, d]).astype(int)
        if y_d.sum() < 5 or (len(y_d) - y_d.sum()) < 5:
            continue
        lr = LogisticRegression(C=1.0, solver="liblinear",
                                max_iter=300, class_weight="balanced")
        lr.fit(X, y_d)
        fp[d * seg_len : (d + 1) * seg_len] = np.concatenate(
            [lr.coef_[0], lr.intercept_]
        )
    return fp


def fit_fingerprint_MI(X, M, rng_seed=0):
    """Mutual-information fingerprint: (D, D) matrix of MI(X_j ; M_d)."""
    from sklearn.feature_selection import mutual_info_classif
    D = X.shape[1]
    fp = np.zeros(D * D, dtype=np.float64)
    for d in range(D):
        y_d = (1 - M[:, d]).astype(int)
        if y_d.sum() < 5 or (len(y_d) - y_d.sum()) < 5:
            continue
        mi = mutual_info_classif(X, y_d, random_state=rng_seed,
                                 n_neighbors=3)
        fp[d * D : (d + 1) * D] = mi
    return fp


def fit_fingerprint_RF(X, M, rng_seed=0):
    """Random-forest feature-importance fingerprint (D × D)."""
    from sklearn.ensemble import RandomForestClassifier
    D = X.shape[1]
    fp = np.zeros(D * D, dtype=np.float64)
    for d in range(D):
        y_d = (1 - M[:, d]).astype(int)
        if y_d.sum() < 5 or (len(y_d) - y_d.sum()) < 5:
            continue
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1,
            random_state=rng_seed, class_weight="balanced",
        )
        rf.fit(X, y_d)
        fp[d * D : (d + 1) * D] = rf.feature_importances_
    return fp


FINGERPRINT_FN = {
    "LR": fit_fingerprint_LR,
    "MI": fit_fingerprint_MI,
    "RF": fit_fingerprint_RF,
}


def prep_data(gt, masks, subsample, rng_seed):
    N, T, D = gt.shape
    X_full = gt.reshape(-1, D).astype(float)
    M_full = masks.reshape(-1, D).astype(int)
    if subsample and len(X_full) > subsample:
        rng = np.random.RandomState(rng_seed)
        idx = rng.choice(len(X_full), subsample, replace=False)
        return X_full[idx], M_full[idx]
    return X_full, M_full


def compute_ca_weight_matrix(fingerprints, scale_factor=SCALE_FACTOR):
    N = len(fingerprints)
    comp = compute_complementarity_matrix(fingerprints)
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        other = [j for j in range(N) if j != i]
        mech_w = np.array([comp[i, j] for j in other]) + 1e-5
        size_w = np.ones(len(other))
        raw_w = (ALPHA * mech_w + (1 - ALPHA) * size_w) ** scale_factor
        final_w = raw_w / raw_w.sum()
        for k, j in enumerate(other):
            W[i, j] = final_w[k]
    return W, comp


def c_score(comp):
    iu = np.triu_indices(comp.shape[0], k=1)
    return float(comp[iu].mean())


# ------------- Main driver --------------------------------------------


def main():
    print("Loading VitalDB tensor…")
    gt, _, masks, feat_names, _ = load_from_local_tensor(
        tensor_dir=TENSOR_DIR, normalize=True,
    )
    allocator = HeterogeneousDataAllocator(
        X=gt, masks=masks, num_clients=5, feature_names=feat_names,
    )

    # Allocate once per setting
    alloc_by_setting = {}
    for scen, mnar, rho, note in SETTINGS:
        cds = allocator.allocate(
            scenario=scen, mnar_method=mnar,
            target_features=[0, 2, 6],
            missing_rate=rho, seed=SEED,
        )
        alloc_by_setting[(scen, mnar, rho)] = (cds, note)

    # Compute fingerprints for each (method, setting)
    results = {}  # (method, idx) -> dict
    for method in METHODS:
        fn = FINGERPRINT_FN[method]
        print(f"\n=== Fingerprint method: {method} ===")
        for idx, (scen, mnar, rho, note) in enumerate(SETTINGS):
            cds, _ = alloc_by_setting[(scen, mnar, rho)]
            fps = []
            t0 = time.time()
            for c_idx, cd in enumerate(cds):
                X, M = prep_data(cd["X"], cd["masks"], SUBSAMPLE, SEED)
                fps.append(fn(X, M, rng_seed=SEED))
            W, comp = compute_ca_weight_matrix(fps)
            maj_to_c0 = float(np.mean([W[i, 0] for i in range(1, 5)]))
            cs = c_score(comp)
            results[(method, idx)] = {
                "scen": scen, "mnar": mnar, "rho": rho, "note": note,
                "W": W, "comp": comp, "c_score": cs,
                "maj_to_c0": maj_to_c0,
            }
            print(f"  [{method}] {scen} {mnar[:1]} ρ={rho}  "
                  f"C_score={cs:.3f}  avg maj→C0={maj_to_c0:.3f}  "
                  f"({time.time() - t0:.1f}s)")

    # ---- Figure: rows = method, cols = setting ----
    n_rows, n_cols = len(METHODS), len(SETTINGS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
    )
    axes_list = list(axes.flat)

    for row, method in enumerate(METHODS):
        for col in range(n_cols):
            ax = axes[row, col]
            r = results[(method, col)]
            W = r["W"]
            im = ax.imshow(W, vmin=0, vmax=1, cmap="Blues", aspect="equal")
            for i in range(5):
                for j in range(5):
                    if i == j:
                        ax.text(j, i, "—", ha="center", va="center",
                                color="#aaa", fontsize=9)
                        continue
                    v = W[i, j]
                    color = "white" if v > 0.5 else "#333"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color=color, fontsize=8.5)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels([f"C{c}" for c in range(5)])
            ax.set_yticklabels([f"C{c}" for c in range(5)])
            if row == n_rows - 1:
                ax.set_xlabel("peer j")
            if col == 0:
                ax.set_ylabel(f"{method}\nclient i")
            ax.set_title(
                f"{r['scen']} {r['mnar']}  ρ={r['rho']}\n"
                f"C={r['c_score']:.3f}, avg maj→C0={r['maj_to_c0']:.2f}",
                fontsize=10,
            )
            ax.add_patch(plt.Rectangle(
                (-0.5, -0.5), 1, 5, fill=False,
                edgecolor="#d62728", linewidth=2.0,
            ))

    fig.suptitle(
        f"Fingerprint method comparison  (scale_factor = {SCALE_FACTOR})\n"
        "Does a nonlinear fingerprint recover the S1 logit complementarity "
        "that LR missed?",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes_list, fraction=0.02, pad=0.02)
    cbar.set_label("peer weight")

    out = FIG_DIR / "phase1_step4_fingerprint_compare_logit_sweep.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")

    # ---- Summary table ----
    print("\n=== Summary ===")
    header = f"{'method':<6}"
    for scen, mnar, rho, _ in SETTINGS:
        header += f"  {scen} {mnar[:1]} ρ={rho}"
    print(header)
    print(f"{'':6}", end="")
    for _ in SETTINGS:
        print(f"  {'C_score':>8} {'maj→C0':>8}", end="")
    print()
    print("-" * (6 + 20 * len(SETTINGS)))
    for method in METHODS:
        print(f"{method:<6}", end="")
        for col in range(len(SETTINGS)):
            r = results[(method, col)]
            print(f"  {r['c_score']:>8.3f} {r['maj_to_c0']:>8.3f}", end="")
        print()


if __name__ == "__main__":
    main()
