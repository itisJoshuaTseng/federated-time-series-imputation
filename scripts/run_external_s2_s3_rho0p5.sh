#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

skip_if_done() {
  local out="$1"
  if [[ -f "${out}" && "${OVERWRITE_EXISTING}" != "1" ]]; then
    echo "== Skip existing ${out} =="
    return 0
  fi
  return 1
}

run_open_data() {
  local scenario="$1"
  local out="logs/saits_mnar/open_data_${scenario}_q_rho0p5_3clients_allfeat_fedsaits_fedice_seeds_0-2.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo "== open-data ${scenario} quantile rho=0.5, 3 clients, all 4 features =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --tensor-dir data/open_data_tensor_T24 \
    --scenario "${scenario}" \
    --mnar-method quantile \
    --missing-rate 0.5 \
    --target-features 0 1 2 3 \
    --num-clients 3 \
    --seeds 0 1 2 \
    --skip-fedprox \
    --skip-local \
    --skip-fedice-ca \
    --skip-local-ice \
    --skip-fed-ca \
    --skip-fed-pd \
    --skip-fed-ca-pd \
    --output-path "${out}"
}

run_eicu_demo() {
  local scenario="$1"
  local out="logs/saits_mnar/eicu_demo_${scenario}_q_rho0p5_5clients_fedsaits_fedice_seeds_0-2.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo "== eICU-demo ${scenario} quantile rho=0.5, 5 hospital-cluster clients =="
  "${PYTHON_BIN}" experiments/run_mnar_experiment.py \
    --tensor-dir data/eicu_demo_tensor_T288_D6 \
    --client-ids-path data/eicu_demo_tensor_T288_D6/client_ids_5hospital_clusters.npy \
    --num-clients 5 \
    --scenario "${scenario}" \
    --mnar-method quantile \
    --missing-rate 0.5 \
    --target-features 0 1 2 \
    --seeds 0 1 2 \
    --skip-fedprox \
    --skip-local \
    --skip-fedice-ca \
    --skip-local-ice \
    --skip-fed-ca \
    --skip-fed-pd \
    --skip-fed-ca-pd \
    --output-path "${out}"
}

for scenario in S2 S3; do
  run_open_data "${scenario}"
done

for scenario in S2 S3; do
  run_eicu_demo "${scenario}"
done

echo "All external S2/S3 rho=0.5 experiments finished."
