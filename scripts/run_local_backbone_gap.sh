#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(0 1 2 3 4)
RHOS=(0.3 0.5 0.7)
PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

rho_tag() {
  local rho="$1"
  echo "${rho/./p}"
}

method_tag() {
  local method="$1"
  if [[ "${method}" == "quantile" ]]; then
    echo "q"
  else
    echo "${method}"
  fi
}

skip_if_done() {
  local out="$1"
  if [[ -f "${out}" && "${OVERWRITE_EXISTING}" != "1" ]]; then
    echo "== Skip existing ${out} =="
    return 0
  fi
  return 1
}

run_local_saits() {
  local scenario="$1"
  local rho="$2"
  local mnar_method="${3:-quantile}"
  local tag
  tag="$(rho_tag "$rho")"
  local mtag
  mtag="$(method_tag "${mnar_method}")"
  local out="logs/saits_mnar/local_saits_${scenario}_${mtag}_rho${tag}_seeds_0-4.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo "== Local-SAITS ${scenario} ${mnar_method} rho=${rho} =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --scenario "${scenario}" \
    --mnar-method "${mnar_method}" \
    --missing-rate "${rho}" \
    --seeds "${SEEDS[@]}" \
    --only-local \
    --output-path "${out}"
}

run_local_ice() {
  local scenario="$1"
  local rho="$2"
  local mnar_method="${3:-quantile}"
  local tag
  tag="$(rho_tag "$rho")"
  local mtag
  mtag="$(method_tag "${mnar_method}")"
  local out="logs/saits_mnar/local_ice_${scenario}_${mtag}_rho${tag}_seeds_0-4.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo "== Local-ICE ${scenario} ${mnar_method} rho=${rho} =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --scenario "${scenario}" \
    --mnar-method "${mnar_method}" \
    --missing-rate "${rho}" \
    --seeds "${SEEDS[@]}" \
    --only-local-ice \
    --fedice-rounds 20 \
    --fedice-ridge-alpha 1.0 \
    --output-path "${out}"
}

run_fedice() {
  local scenario="$1"
  local rho="$2"
  local mnar_method="${3:-quantile}"
  local tag
  tag="$(rho_tag "$rho")"
  local mtag
  mtag="$(method_tag "${mnar_method}")"
  local out="logs/saits_mnar/${scenario}_${mtag}_rho${tag}_fedice.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo "== FedICE ${scenario} ${mnar_method} rho=${rho} =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --scenario "${scenario}" \
    --mnar-method "${mnar_method}" \
    --missing-rate "${rho}" \
    --seeds "${SEEDS[@]}" \
    --only-fedice \
    --fedice-rounds 20 \
    --fedice-ridge-alpha 1.0 \
    --output-path "${out}"
}

run_fedice_ca() {
  local scenario="$1"
  local rho="$2"
  local mnar_method="${3:-quantile}"
  local tag
  tag="$(rho_tag "$rho")"
  local mtag
  mtag="$(method_tag "${mnar_method}")"
  local out="logs/saits_mnar/${scenario}_${mtag}_rho${tag}_fedice_ca_b4.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo "== FedICE-CA b=4 ${scenario} ${mnar_method} rho=${rho} =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --scenario "${scenario}" \
    --mnar-method "${mnar_method}" \
    --missing-rate "${rho}" \
    --seeds "${SEEDS[@]}" \
    --only-fedice-ca \
    --fedice-rounds 20 \
    --fedice-ridge-alpha 1.0 \
    --ca-scale-factor 4 \
    --output-path "${out}"
}

# Missing Local-SAITS cells in the current S1-S4 quantile table.
for scenario in S2 S3; do
  for rho in "${RHOS[@]}"; do
    run_local_saits "${scenario}" "${rho}" "quantile"
  done
done

# Local linear baseline for all four scenarios.
for scenario in S1 S2 S3 S4; do
  for rho in "${RHOS[@]}"; do
    run_local_ice "${scenario}" "${rho}" "quantile"
  done
done

# Logit MNAR additions requested for S1/S4:
# Local-SAITS is already available, so only run Local-ICE, FedICE,
# and FedICE-CA b=4.
for scenario in S1 S4; do
  for rho in "${RHOS[@]}"; do
    run_local_ice "${scenario}" "${rho}" "logit"
    run_fedice "${scenario}" "${rho}" "logit"
    run_fedice_ca "${scenario}" "${rho}" "logit"
  done
done

echo "All requested local backbone-gap experiments finished."
