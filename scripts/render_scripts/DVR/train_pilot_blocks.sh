#!/usr/bin/env bash
set -uo pipefail

Z0=30
Y0=30
X0=30
N=4
CONCURRENCY=16
EPOCHS=200
STEPS_PER_EPOCH=50
N_INIT=1000
MAX_GAUSS=5000

OUT_ROOT="models_fafb_pilot/blocks"
LOG_ROOT="/tmp/claude-0/-root-project/a0c2bf74-b908-4bbd-97ee-70b6409019f0/scratchpad/block_logs"
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
        --volume "$vol" \
        --use_kernel \
        --flat_out \
        --no_swc_init \
        --no_wandb \
        --out "$out" \
        --n_init $N_INIT --max_gaussians $MAX_GAUSS \
        --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH \
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
