#!/usr/bin/env bash
set -uo pipefail

# Validation test for config_v9.yml (aggressive prune-only: prune_inten_thresh
# 10x default, prune_interval 4x more frequent than growth, growth trigger
# left at defaults) on the same 2 representative blocks used for the
# v2-v8 growth-trajectory diagnosis.
#
# Run from the repo root:
#   bash fafb_pilot/scripts/train_pilot_blocks_v9_test.sh

CONFIG="fafb_pilot/config_v9.yml"
OUT_ROOT="fafb_pilot/models/blocks_v9_test"
LOG_ROOT="fafb_pilot/results/block_logs_v9_test"
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
echo "V9 TEST DONE"
