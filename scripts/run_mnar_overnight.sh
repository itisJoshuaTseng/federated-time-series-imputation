#!/usr/bin/env bash

set -u

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fed-marl}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${DEVICE:-cuda}"
TENSOR_DIR="${TENSOR_DIR:-../2026_vitalDB/tensor-file-for-4feature-20260304T112438Z-3-001/tensor-file-for-4feature/vitaldb_14feats_tensor_T300}"
RUN_LOGIT="${RUN_LOGIT:-0}"
RUN_MISSING="${RUN_MISSING:-0}"
RUN_S23="${RUN_S23:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -f "$CONDA_SH" ]; then
    echo "[FATAL] conda init script not found: $CONDA_SH"
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH" || {
    echo "[FATAL] Failed to source conda init script: $CONDA_SH"
    exit 1
}

conda activate "$CONDA_ENV" || {
    echo "[FATAL] Failed to activate conda environment: $CONDA_ENV"
    exit 1
}

cd "$WORKDIR" || {
    echo "[FATAL] Failed to cd into workdir: $WORKDIR"
    exit 1
}

mkdir -p logs/overnight_runs
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="logs/overnight_runs/mnar_overnight_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "MNAR overnight batch started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "workdir      : $(pwd)"
echo "conda env    : $CONDA_ENV"
echo "device       : $DEVICE"
echo "tensor dir   : $TENSOR_DIR"
echo "run logit    : $RUN_LOGIT"
echo "run missing  : $RUN_MISSING"
echo "run S2/S3    : $RUN_S23"
echo "log file     : $LOG_FILE"
echo "============================================================"

declare -a METHODS
METHODS=("quantile")
if [ "$RUN_LOGIT" = "1" ]; then
    METHODS+=("logit")
fi

declare -a RATES
RATES=("0.3")
if [ "$RUN_MISSING" = "1" ]; then
    RATES+=("0.5" "0.7")
fi

SEEDS=(0 1 2 3 4)
SCENARIOS=("S1" "S4")
if [ "$RUN_S23" = "1" ]; then
    SCENARIOS=("S1" "S2" "S3" "S4")
fi

TOTAL_JOBS=0
SUCCESS_JOBS=0
FAILED_JOBS=0
declare -a FAILED_COMMANDS

run_job() {
    local scenario="$1"
    local method="$2"
    local missing_rate="$3"
    shift 3
    local seeds=("$@")

    local cmd=(
        "$PYTHON_BIN" experiments/run_mnar_experiment.py
        --scenario "$scenario"
        --seeds
    )

    local seed
    for seed in "${seeds[@]}"; do
        cmd+=("$seed")
    done

    cmd+=(
        --mnar-method "$method"
        --missing-rate "$missing_rate"
        --device "$DEVICE"
        --tensor-dir "$TENSOR_DIR"
    )

    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    echo
    echo "------------------------------------------------------------"
    echo "Job #$TOTAL_JOBS"
    echo "scenario     : $scenario"
    echo "seeds        : ${seeds[*]}"
    echo "method       : $method"
    echo "missing-rate : $missing_rate"
    echo "command      : ${cmd[*]}"
    echo "started at   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    "${cmd[@]}"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        SUCCESS_JOBS=$((SUCCESS_JOBS + 1))
        echo "[OK] Job #$TOTAL_JOBS finished successfully"
    else
        FAILED_JOBS=$((FAILED_JOBS + 1))
        FAILED_COMMANDS+=("${cmd[*]}")
        echo "[ERROR] Job #$TOTAL_JOBS failed with exit code $exit_code"
    fi
}

for method in "${METHODS[@]}"; do
    for missing_rate in "${RATES[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
            run_job "$scenario" "$method" "$missing_rate" "${SEEDS[@]}"
        done
    done
done

echo
echo "============================================================"
echo "MNAR overnight batch finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "total jobs     : $TOTAL_JOBS"
echo "success        : $SUCCESS_JOBS"
echo "failed         : $FAILED_JOBS"
if [ $FAILED_JOBS -gt 0 ]; then
    echo "failed commands:"
    for failed_cmd in "${FAILED_COMMANDS[@]}"; do
        echo "  - $failed_cmd"
    done
else
    echo "failed commands: none"
fi
echo "log file       : $LOG_FILE"
echo "============================================================"

if [ $FAILED_JOBS -gt 0 ]; then
    exit 1
fi

exit 0
