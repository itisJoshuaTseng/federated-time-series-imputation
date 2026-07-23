"""
Phase 1 Step 2b — test the "clone C0 propagates weak signal" hypothesis.

Step 2 found that S1 quantile gives avg(majority → C0 weight) ≈ 0.98 across
ρ=0.3/0.5/0.7, yet outcomes flip from −20% (ρ=0.3) to +14% (ρ=0.7) on majority.
Since the CA weight structure is ~identical, the difference must come from
what C0's *local parameters* look like at each ρ.

Zero-cost check: from existing logs, read per-client MAE for
  (Local / FedAvg / Fed-CA) × (C0 / C1–C4)
and compute:

  - Local_C0_MAE      : how well does C0's local model do on C0's own data?
  - CA_majority_MAE   : what does "clone C0 model" achieve on majority data?
  - FedAvg_majority   : baseline (democratic average) on majority data
  - Local_majority    : majority's own local solutions

If the hypothesis holds, at ρ=0.3 we expect:
  - Local_C0 is weak/similar to Local_majority (signal too weak to learn a
    distinctively useful MNAR-Left model)
  - CA_majority ≫ FedAvg_majority  (cloning that weak model hurts majority)

At ρ=0.7 we expect:
  - Local_C0 is strong (strong MNAR signal)
  - CA_majority < FedAvg_majority  (cloning that strong model spreads the
    benefit)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "saits_mnar"

SETTINGS = [
    ("S1", "q", 0.3),
    ("S1", "q", 0.5),
    ("S1", "q", 0.7),
    ("S1", "l", 0.3),
    ("S4", "q", 0.3),
    ("S4", "q", 0.7),
]


def load(scen, mnar, rho):
    rho_tag = f"rho0p{int(rho * 10)}"
    path = LOG_DIR / f"cafe_fix_v2_{scen}_{mnar}_{rho_tag}_seeds_0-4.json"
    with path.open() as f:
        return json.load(f)


def per_client_mean(data):
    bucket = {}
    for r in data["results"]:
        bucket.setdefault(r["method"], []).append(
            [cm["mae"] for cm in r["client_metrics"]]
        )
    return {m: np.mean(v, axis=0) for m, v in bucket.items()}


def main():
    print(
        f"{'setting':<14} "
        f"{'Local_C0':>10} {'Local_maj':>10} {'C0-gap':>8}  "
        f"{'FedAvg_maj':>11} {'CA_maj':>10} {'CA−FA maj':>10}  "
        f"{'maj clone':>10}"
    )
    print("-" * 100)

    for scen, mnar, rho in SETTINGS:
        data = load(scen, mnar, rho)
        pc = per_client_mean(data)
        lc = pc["local"]
        fa = pc["fedavg"]
        ca = pc["fed_ca"]

        local_c0 = lc[0]
        local_maj = float(np.mean(lc[1:]))
        c0_gap = local_c0 - local_maj          # negative → C0 strongly better
        fedavg_maj = float(np.mean(fa[1:]))
        ca_maj = float(np.mean(ca[1:]))
        ca_vs_fa_maj = (fedavg_maj - ca_maj) / fedavg_maj * 100

        # "clone indicator": how close is CA_maj to Local_C0?
        # If majority ≈ clones C0, CA_maj should be ≈ Local_C0 (same model,
        # different data). Ratio = CA_maj / Local_C0.
        clone_ratio = ca_maj / local_c0 if local_c0 > 0 else float("nan")

        tag = f"{scen} {mnar} ρ={rho}"
        print(
            f"{tag:<14} "
            f"{local_c0:>10.3f} {local_maj:>10.3f} {c0_gap:>+8.3f}  "
            f"{fedavg_maj:>11.3f} {ca_maj:>10.3f} {ca_vs_fa_maj:>+9.2f}%  "
            f"{clone_ratio:>10.3f}"
        )

    print()
    print("Legend:")
    print("  C0-gap     = Local_C0 − Local_majority")
    print("               (<0 → C0 trains a better local model than majority;")
    print("                 ≈0 → C0 is not distinctly better; ")
    print("                >0 → C0 is actively worse than majority)")
    print("  maj clone  = CA_majority / Local_C0  (≈1 means majority ≈ C0's model)")


if __name__ == "__main__":
    main()
