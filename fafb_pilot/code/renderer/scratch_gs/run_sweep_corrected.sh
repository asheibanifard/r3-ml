#!/usr/bin/env bash
# Runs gaussian_splat_scratch_corrected + render_outputs_corrected.py for each
# screen size in the sweep, reusing the same trained checkpoint every time
# (screen size only affects rendering, not training).
#
# USAGE
#   fafb_pilot/code/renderer/scratch_gs/run_sweep_corrected.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BIN="./gaussian_splat_scratch_corrected"
PY_SCRIPT="render_outputs_corrected.py"
CHECKPOINT="checkpoint.bin"
SIZES=(64 128 256 512 1024 2048)

for size in "${SIZES[@]}"; do
  echo "=== screen size ${size} ==="
  frames_dir="frames_${size}"
  out_dir="results_${size}"

  rm -rf "${frames_dir}"
  mkdir -p "${frames_dir}"

  "${BIN}" \
    "${frames_dir}" \
    "${size}" "${size}" \
    "${CHECKPOINT}"

  /venv/r3-ml/bin/python3 "${PY_SCRIPT}" \
    --frames_dir "${frames_dir}" \
    --out_dir "${out_dir}" \
    --video_fps 24 \
    --lpips_device auto
done

echo ""
echo "=== Done: all screen sizes (${SIZES[*]}) complete ==="
