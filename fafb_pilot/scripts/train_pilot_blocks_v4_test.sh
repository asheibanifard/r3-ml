#!/usr/bin/env bash
set -uo pipefail

# Validation test for config_v4.yml (rebalanced densification, 5000 epochs)
# on 2 representative blocks only, before committing to all 64 -- config_v4.yml
# uses untuned, first-attempt threshold values and a 10x longer epoch budget
# than v2/v3, so per-block cost is unknown and could be substantial.
#
# Run from the repo root:
#   bash fafb_pilot/scripts/train_pilot_blocks_v4_test.sh

CONFIG="fafb_pilot/config_v4.yml"
OUT_ROOT="fafb_pilot/models/blocks_v4_test"
LOG_ROOT="fafb_pilot/results/block_logs_v4_test"
mkdir -p "$LOG_ROOT"

# b_000 (z=30,y=30,x=30) and b_133 (z=31,y=33,x=33) -- same two blocks
# already used to verify the v2/v3 growth-trajectory diagnosis.
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
echo "V4 TEST DONE"
