#!/usr/bin/env bash

set -u

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fed-marl}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${DEVICE:-cuda}"
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
LOG_FILE="logs/overnight_runs/mnar_smoke_test_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "MNAR smoke test started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "workdir      : $(pwd)"
echo "conda env    : $CONDA_ENV"
echo "device       : $DEVICE"
echo "log file     : $LOG_FILE"
echo "============================================================"

CMD=(
    "$PYTHON_BIN" experiments/run_mnar_experiment.py
    --scenario S1
    --seed 42
    --device "$DEVICE"
    --skip-fedavg
    --skip-fedprox
)

echo "command      : ${CMD[*]}"
"${CMD[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Smoke test failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "[OK] Smoke test completed successfully"
exit 0
