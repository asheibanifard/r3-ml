#!/usr/bin/env bash
set -uo pipefail

# Retrains the same 64 blocks as retrain_pilot_blocks_v2.sh (config_v2.yml /
# models/blocks_v2/) with config_v18.yml -- the structural growth throttle
# (densify_thresh_population_exponent=0.8) plus a bare L = 0.7*L1 + 0.3*(1-SSIM)
# reconstruction loss with every other regularizer zeroed. Validated on b_211
# alone first (fafb_pilot/models/blocks_v18_test/b_211): N=30,872 vs the
# original config_v2 result of N=50,000, and vol_PSNR=35.49 dB vs 40.47 dB --
# ~38% fewer Gaussians at a real but modest ~5 dB fidelity cost, and notably
# BETTER than every earlier fewer-Gaussian attempt (v13's hard cap, v14-v17's
# loss-term tuning) at a similar or smaller budget. Output goes to
# models/blocks_v18/ so blocks_v2/ is preserved untouched for comparison.
#
# Run from the repo root:
#   bash fafb_pilot/scripts/retrain_pilot_blocks_v18.sh

Z0=30
Y0=30
X0=30
N=4
CONCURRENCY=16

CONFIG="fafb_pilot/config_v18.yml"
OUT_ROOT="fafb_pilot/models/blocks_v18"
LOG_ROOT="fafb_pilot/results/block_logs_v18"
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
