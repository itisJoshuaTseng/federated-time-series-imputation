"""
Phase 1 Step 2 sanity check — dump CA aggregation weight matrix.

Goal: verify the hypothesis that in S1 quantile ρ=0.3 (the "majority
degradation" case, Δ_CA majority ≈ −20%), the power-law weighting
(scale_factor=4) severely overweights minority C0 from majority clients'
point of view, causing systematic drift toward C0's solution.

For each setting:
  1. Allocate data with HeterogeneousDataAllocator (same seed as experiments).
  2. Fit CAFE-style fingerprint per client (logistic-regression on masks).
  3. Compute complementarity matrix C[i,j] = (1 - cos(fp_i, fp_j))/2.
  4. Apply CAFE power-law:  raw_w = (α·mech_w + (1-α)·size_w)^scale_factor
     with α=0.95, scale_factor=4, equal sample sizes.
  5. Row-normalize → W[i,j] = weight client i assigns to peer j.

Output:
  - figures/phase1_step2_ca_weights.png : 6-panel heatmap
  - stdout summary: W[1, 0] across all settings (majority → minority pull)
"""

from __future__ import annotations

import sys
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

# Settings to inspect — focus on S1 quantile ρ-sweep for sf=4 vs sf=1 comparison.
SETTINGS = [
    ("S1", "quantile", 0.3, "ρ=0.3 (low signal)"),
    ("S1", "quantile", 0.5, "ρ=0.5 (mid signal)"),
    ("S1", "quantile", 0.7, "ρ=0.7 (high signal)"),
]

ALPHA = 0.95
SEED = 0  # use one seed; fingerprint is stable

# Scale factors to compare side-by-side.
SCALE_FACTORS = [4, 1]


def fit_fingerprint(gt, masks, subsample=20000, rng_seed=0):
    """Match compute_c_score.py's fast fingerprint."""
    from sklearn.linear_model import LogisticRegression

    N, T, D = gt.shape
    X_full = gt.reshape(-1, D).astype(float)
    M_full = masks.reshape(-1, D).astype(int)
    if subsample and len(X_full) > subsample:
        rng = np.random.RandomState(rng_seed)
        idx = rng.choice(len(X_full), subsample, replace=False)
        X, M = X_full[idx], M_full[idx]
    else:
        X, M = X_full, M_full

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


def compute_ca_weight_matrix(fingerprints, scale_factor):
    """Return 5x5 matrix W where W[i,j] = weight client i assigns to peer j.

    W[i, i] = 0 (self handled separately by γ blending).
    Rows (excluding diagonal) sum to 1.
    """
    N = len(fingerprints)
    comp = compute_complementarity_matrix(fingerprints)  # (N, N) in [0, 1]

    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        other = [j for j in range(N) if j != i]
        mech_w = np.array([comp[i, j] for j in other]) + 1e-5
        size_w = np.ones(len(other))                    # equal sizes
        raw_w = (ALPHA * mech_w + (1 - ALPHA) * size_w) ** scale_factor
        final_w = raw_w / raw_w.sum()
        for k, j in enumerate(other):
            W[i, j] = final_w[k]
    return W, comp


def c_score(comp):
    iu = np.triu_indices(comp.shape[0], k=1)
    return float(comp[iu].mean())


def main():
    print("Loading VitalDB tensor…")
    gt, _, masks, feat_names, _ = load_from_local_tensor(
        tensor_dir=TENSOR_DIR, normalize=True,
    )
    allocator = HeterogeneousDataAllocator(
        X=gt, masks=masks, num_clients=5, feature_names=feat_names,
    )

    # Compute fingerprints once per setting (scale_factor doesn't affect them)
    per_setting = []
    print("\nFitting fingerprints…")
    for scen, mnar, rho, note in SETTINGS:
        cds = allocator.allocate(
            scenario=scen, mnar_method=mnar,
            target_features=[0, 2, 6],
            missing_rate=rho, seed=SEED,
        )
        fps = [fit_fingerprint(cd["X"], cd["masks"]) for cd in cds]
        per_setting.append({
            "scen": scen, "mnar": mnar, "rho": rho, "note": note,
            "fps": fps,
        })
        print(f"  {scen} {mnar[:1]} ρ={rho}  ({note})")

    # Compute W for each (scale_factor, setting) pair
    results = {}      # (sf, idx) -> {W, comp, c, ...}
    for sf in SCALE_FACTORS:
        for idx, s in enumerate(per_setting):
            W, comp = compute_ca_weight_matrix(s["fps"], sf)
            results[(sf, idx)] = {
                **s, "W": W, "comp": comp, "c": c_score(comp), "sf": sf,
            }

    # ---- Figure: 2 rows (sf=4 top, sf=1 bottom) x len(SETTINGS) cols ----
    n_cols = len(SETTINGS)
    fig, axes = plt.subplots(
        len(SCALE_FACTORS), n_cols,
        figsize=(4.5 * n_cols, 4.2 * len(SCALE_FACTORS)),
    )
    if len(SCALE_FACTORS) == 1:
        axes = np.array([axes])
    axes_list = list(axes.flat)

    for row, sf in enumerate(SCALE_FACTORS):
        for col in range(n_cols):
            ax = axes[row, col]
            r = results[(sf, col)]
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
            if row == len(SCALE_FACTORS) - 1:
                ax.set_xlabel("peer j")
            if col == 0:
                ax.set_ylabel(f"scale_factor = {sf}\nclient i")
            ax.set_title(
                f"{r['scen']} {r['mnar']}  ρ={r['rho']}  "
                f"(C={r['c']:.3f})",
                fontsize=10,
            )
            # Red box around C0 column to highlight minority-pull
            ax.add_patch(plt.Rectangle(
                (-0.5, -0.5), 1, 5, fill=False,
                edgecolor="#d62728", linewidth=2.0,
            ))
            # Side annotation: avg maj → C0
            majs = [W[i, 0] for i in range(1, 5)]
            ax.text(
                5.2, 2, f"avg maj→C0\n= {np.mean(majs):.2f}",
                ha="left", va="center", fontsize=9,
                color="#d62728", fontweight="bold",
            )

    fig.suptitle(
        "CA aggregation weight matrix, scale_factor sweep\n"
        "(top: CAFE default sf=4;  bottom: sf=1, no power-law sharpening)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes_list, fraction=0.02, pad=0.02)
    cbar.set_label("peer weight")

    out = FIG_DIR / "phase1_step3_ca_weights_sf_sweep.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")

    # ---- Summary: avg maj → C0 for each (sf, rho) ----
    print("\n=== Majority → C0 pull (avg over C1..C4) ===")
    print(f"{'rho':<8}", end="")
    for sf in SCALE_FACTORS:
        print(f"  sf={sf:<4}", end="")
    print()
    print("-" * (8 + 8 * len(SCALE_FACTORS)))
    for idx, s in enumerate(per_setting):
        print(f"ρ={s['rho']:<6}", end="")
        for sf in SCALE_FACTORS:
            W = results[(sf, idx)]["W"]
            avg_maj = float(np.mean([W[i, 0] for i in range(1, 5)]))
            print(f"  {avg_maj:<6.3f}", end="")
        print()


if __name__ == "__main__":
    main()
