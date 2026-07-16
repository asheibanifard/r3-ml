#!/usr/bin/env bash
set -uo pipefail

# Validation test for config_v11.yml (minimal-Gaussian-count: corrected
# densify_grad_thresh/densify_max_scale, real max_gaussians=5000 ceiling
# instead of v10's 1.5M safety valve, and the four count-reducing loss
# terms -- lambda_aniso/count/L1/coverage -- turned on for the first time)
# on the same 2 representative blocks used for the v2-v10 growth-trajectory
# diagnosis.
#
# Run from the repo root:
#   bash fafb_pilot/scripts/train_pilot_blocks_v11_test.sh

CONFIG="fafb_pilot/config_v11.yml"
OUT_ROOT="fafb_pilot/models/blocks_v11_test"
LOG_ROOT="fafb_pilot/results/block_logs_v11_test"
mkdir -p "$LOG_ROOT"

BLOCKS="30 30 30 b_000
31 33 33 b_133"

pids=()
while read -r z y x bname; do
  vol="data/fafb/blocks/image_z${z}_y${y}_x${x}.tif"
  out="${OUT_ROOT}/${bname}"

  /venv/r3-ml/bin/python3 scripts/_3dgs/_3dgs.py \
    --config "$CONFIG" \
    --volume "$vol" \
    --out "$out" \
    > "${LOG_ROOT}/${bname}.log" 2>&1 &

  pids+=($!)
  echo "[launch] ${bname}  (z=$z y=$y x=$x)  pid=$!"
done <<< "$BLOCKS"

wait "${pids[@]}"
echo "V11 TEST DONE"
