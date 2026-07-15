#!/usr/bin/env bash
set -uo pipefail

# Retrains the same 64 blocks as train_pilot_blocks.sh / retrain_pilot_blocks_v2.sh
# with the mid-budget config_v3.yml (max_gaussians=20000, epochs=500,
# densify from epoch 10 to epoch 400 -- same schedule as v2). Output goes to
# models/blocks_v3/ so v1's blocks/ and v2's blocks_v2/ are preserved untouched.
#
# Run from the repo root:
#   bash fafb_pilot/scripts/train_pilot_blocks_v3.sh

Z0=30
Y0=30
X0=30
N=4
CONCURRENCY=16

CONFIG="fafb_pilot/config_v3.yml"
OUT_ROOT="fafb_pilot/models/blocks_v3"
LOG_ROOT="fafb_pilot/results/block_logs_v3"
mkdir -p "$LOG_ROOT"

pids=()
count=0
total=$((N*N*N))

for iz in $(seq 0 $((N-1))); do
  for iy in $(seq 0 $((N-1))); do
    for ix in $(seq 0 $((N-1))); do
      z=$((Z0+iz))
      y=$((Y0+iy))
      x=$((X0+ix))
      bname=$(printf "b_%d%d%d" "$iz" "$iy" "$ix")
      vol="data/fafb/blocks/image_z${z}_y${y}_x${x}.tif"
      out="${OUT_ROOT}/${bname}"

      if [ -f "${out}/last.pth" ]; then
        echo "[skip] ${bname} already has last.pth"
        continue
      fi

      /venv/r3-ml/bin/python3 scripts/_3dgs/_3dgs.py \
        --config "$CONFIG" \
        --volume "$vol" \
        --out "$out" \
        > "${LOG_ROOT}/${bname}.log" 2>&1 &

      pids+=($!)
      count=$((count+1))
      echo "[launch $count/$total] ${bname}  (z=$z y=$y x=$x)  pid=$!"

      if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
        wait "${pids[@]}"
        pids=()
      fi
    done
  done
done

if [ "${#pids[@]}" -gt 0 ]; then
  wait "${pids[@]}"
fi

echo "ALL BLOCKS DONE"
