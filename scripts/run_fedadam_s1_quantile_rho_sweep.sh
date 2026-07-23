#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
ROUNDS="${ROUNDS:-50}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

OUT_DIR="logs/saits_mnar"

run_rho() {
  local rho="$1"
  local tag="$2"
  local out="${OUT_DIR}/S1_q_${tag}_fedadam_seeds_0-2.json"

  if [[ -f "${out}" && "${OVERWRITE_EXISTING}" != "1" ]]; then
    echo "== Skip existing ${out} =="
    return
  fi

  echo
  echo "== VitalDB S1 quantile rho=${rho}, FedAdam, seeds 0-2 =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --scenario S1 \
    --mnar-method quantile \
    --missing-rate "${rho}" \
    --target-features 0 2 6 \
    --seeds 0 1 2 \
    --device "${DEVICE}" \
    --rounds "${ROUNDS}" \
    --only-fedadam \
    --output-path "${out}"
}

run_rho 0.3 rho0p3
run_rho 0.5 rho0p5
run_rho 0.7 rho0p7

echo
echo "All FedAdam S1 quantile rho sweep experiments finished."
