#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
ROUNDS="${ROUNDS:-50}"
CLASSIFIER="${CLASSIFIER:-xgboost}"

"${PYTHON_BIN}" experiments/run_eicu_downstream_los.py \
  --tensor-dir data/eicu_demo_tensor_T288_D6 \
  --patient-csv data/eicu_demo/patient.csv.gz \
  --client-ids-path data/eicu_demo_tensor_T288_D6/client_ids_5hospital_clusters.npy \
  --num-clients 5 \
  --scenarios S1 S4 \
  --mnar-method quantile \
  --missing-rate 0.5 \
  --target-features 0 1 2 \
  --seeds 0 1 2 \
  --methods fedsaits_ca fedice_ca \
  --device "${DEVICE}" \
  --rounds "${ROUNDS}" \
  --fedice-rounds 20 \
  --saits-ca-scale-factor 0.5 \
  --ice-ca-scale-factor 4 \
  --classifier "${CLASSIFIER}" \
  --feature-mode flatten \
  --label los48h \
  --los-hours 48 \
  --output-dir experiments/figures/downstream_eicu_los
