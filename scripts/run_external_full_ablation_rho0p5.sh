#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"
DEVICE="${DEVICE:-auto}"
ROUNDS="${ROUNDS:-50}"
FEDICE_ROUNDS="${FEDICE_ROUNDS:-20}"

OUT_DIR="logs/saits_mnar/external_full_ablation_rho0p5"
mkdir -p "${OUT_DIR}"

skip_if_done() {
  local out="$1"
  if [[ -f "${out}" && "${OVERWRITE_EXISTING}" != "1" ]]; then
    echo "== Skip existing ${out} =="
    return 0
  fi
  return 1
}

run_one() {
  local dataset="$1"
  local scenario="$2"
  local method="$3"
  local tensor_dir="$4"
  local num_clients="$5"
  local target_features="$6"
  local client_ids_path="${7:-}"

  local method_flag=""
  local ca_scale_factor="4"
  local method_tag="${method}"

  case "${method}" in
    local_saits)
      method_flag="--only-local"
      ;;
    local_ice)
      method_flag="--only-local-ice"
      ;;
    fedsaits_ca_b0p5)
      method_flag="--only-fed-ca"
      ca_scale_factor="0.5"
      ;;
    fedice_ca_b4)
      method_flag="--only-fedice-ca"
      ca_scale_factor="4"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  local out="${OUT_DIR}/${dataset}_${scenario}_q_rho0p5_${method_tag}_seeds_0-2.json"
  if skip_if_done "${out}"; then
    return
  fi

  echo
  echo "== ${dataset} ${scenario} quantile rho=0.5 ${method_tag} =="

  local args=(
    experiments/run_mnar_experiment.py
    --tensor-dir "${tensor_dir}"
    --scenario "${scenario}"
    --mnar-method quantile
    --missing-rate 0.5
    --target-features ${target_features}
    --num-clients "${num_clients}"
    --seeds 0 1 2
    --device "${DEVICE}"
    --rounds "${ROUNDS}"
    --fedice-rounds "${FEDICE_ROUNDS}"
    --ca-scale-factor "${ca_scale_factor}"
    ${method_flag}
    --output-path "${out}"
  )

  if [[ -n "${client_ids_path}" ]]; then
    args+=(--client-ids-path "${client_ids_path}")
  fi

  "${PYTHON_BIN}" "${args[@]}"
}

run_open_data() {
  local scenario="$1"
  local method="$2"
  run_one \
    "open_data_3clients_allfeat" \
    "${scenario}" \
    "${method}" \
    "data/open_data_tensor_T24" \
    "3" \
    "0 1 2 3"
}

run_eicu_demo() {
  local scenario="$1"
  local method="$2"
  run_one \
    "eicu_demo_5clients_hr_rr_spo2" \
    "${scenario}" \
    "${method}" \
    "data/eicu_demo_tensor_T288_D6" \
    "5" \
    "0 1 2" \
    "data/eicu_demo_tensor_T288_D6/client_ids_5hospital_clusters.npy"
}

for dataset_fn in run_open_data run_eicu_demo; do
  for scenario in S1 S2 S3 S4; do
    for method in local_saits local_ice fedsaits_ca_b0p5 fedice_ca_b4; do
      "${dataset_fn}" "${scenario}" "${method}"
    done
  done
done

echo
echo "All external full ablation rho=0.5 experiments finished."
