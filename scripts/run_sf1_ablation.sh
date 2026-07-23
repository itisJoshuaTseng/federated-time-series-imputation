#!/usr/bin/env bash
#
# Phase 1 Step 2 — H1''-C causal test:
# run CA with scale_factor=1 (no power-law sharpening) on the two
# pivotal settings, to verify that flipping the weighting scheme flips
# the outcome:
#   - S1 q ρ=0.3 : expect Δ_CA majority −20% → ~0% (failure → neutral)
#   - S1 q ρ=0.7 : expect Δ_CA majority +14% → ~0% (success   → neutral)
#
# Uses --only-fed-ca so Local/FedAvg baselines are reused from existing
# cafe_fix_v2_S1_q_rho0p*_seeds_0-4.json logs.
#
# Usage (on lab server):
#   bash scripts/run_sf1_ablation.sh
#
# Env overrides:
#   CONDA_SH, CONDA_ENV, DEVICE, TENSOR_DIR, PYTHON_BIN

set -u

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fed-marl}"
DEVICE="${DEVICE:-cuda}"
TENSOR_DIR="${TENSOR_DIR:-../2026_vitalDB/vitaldb_14feats_tensor_T300}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

mkdir -p logs/saits_mnar
mkdir -p logs/overnight_runs
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="logs/overnight_runs/sf1_ablation_${TIMESTAMP}.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "scale_factor=1 ablation"
echo "started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "device  : $DEVICE"
echo "tensor  : $TENSOR_DIR"
echo "log     : $MASTER_LOG"
echo "============================================================"

SEEDS=(0 1 2 3 4)

run_setting() {
    local rho="$1"
    local rho_tag="$2"
    local out="logs/saits_mnar/ablation_sf1_S1_q_${rho_tag}_seeds_0-4.json"

    echo
    echo "------------------------------------------------------------"
    echo "S1 quantile  ρ=${rho}  scale_factor=1   (only fed_ca)"
    echo "output: $out"
    echo "------------------------------------------------------------"

    "$PYTHON_BIN" experiments/run_mnar_experiment.py \
        --scenario S1 \
        --mnar-method quantile \
        --missing-rate "$rho" \
        --seeds "${SEEDS[@]}" \
        --device "$DEVICE" \
        --tensor-dir "$TENSOR_DIR" \
        --only-fed-ca \
        --ca-scale-factor 1 \
        --output-path "$out"
}

run_setting "0.3" "rho0p3"
run_setting "0.7" "rho0p7"

echo
echo "============================================================"
echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "outputs:"
echo "  logs/saits_mnar/ablation_sf1_S1_q_rho0p3_seeds_0-4.json"
echo "  logs/saits_mnar/ablation_sf1_S1_q_rho0p7_seeds_0-4.json"
echo "============================================================"
