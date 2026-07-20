#!/usr/bin/env bash
# Bake-once, render-many pipeline for a stitched, hard-gated scene.
#
# WHY THIS SCRIPT EXISTS
# -----------------------
# Rendering the raw, continuous Gaussian-sum field directly (either
# ray-marched -- pretrained_gaussian_hard_gated -- or rasterized --
# pretrained_gaussian_rasterized) samples the model at arbitrary points
# between its training voxel centres. Between those points, each
# independently-trained block drifts from true tissue brightness by a
# different, uncorrelated amount -- measured directly against real ground
# truth in fidelity_comparison_size*.png: only 22-28 dB PSNR for both
# continuous paths, visible as a distinct brightness bias per block, versus
# 46-49 dB PSNR (near-invisible error) for the BOUNDED path, which evaluates
# the model exactly at its training voxel centres and clamps there before
# compositing. Clamping timing elsewhere (e.g. per depth-bin instead of only
# on the final pixel) cannot fix this -- verified empirically to be a
# mathematical no-op (min(x,1) is monotonic, so max and clamp commute
# regardless of where the clamp is applied) -- the fix has to be evaluating
# the model where it was actually trained, not smoothing after the fact.
#
# So: BAKE the stitched scene to a bounded per-voxel grid ONCE (the same
# rec_<name>.tif stitch_blocks.py already produces -- hard-gated by
# construction, clamped [0,1] per voxel), then RENDER that baked grid many
# times through the renderer's dense_voxel path, which is pure texture
# sampling (no Gaussian math at all) and measured at ~300,000 FPS. The bake
# is the only expensive step, and only needs to happen again if the
# underlying Gaussians change -- not per frame, not per camera angle.
#
# STEPS
# -----
# 1. Compile the renderer (same nvcc invocation as run.sh).
# 2. BAKE: stitch_blocks.py evaluates every block on its own training voxel
#    grid, hard-gated, clamped [0,1] -- rec_<name>.tif. Skipped if that file
#    already exists (pass --force-rebake to redo it, e.g. after retraining).
# 3. Export the baked grid to the renderer's dense_voxel binary format
#    (export_renderer_bin.py -- no Gaussian export needed at all here).
# 4. RENDER --yaw-steps camera angles from the SAME baked grid -- no
#    re-baking between frames, which is what makes this fast.
#
# USAGE
# -----
#   fafb_pilot/code/renderer/bake_and_render.sh --size 128 --yaw-steps 8
#   fafb_pilot/code/renderer/bake_and_render.sh --size 256 --yaw-steps 4 --force-rebake
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CUDA_SRC="${SCRIPT_DIR}/Mip_Render_Inside_Volume.cu"
CUDA_EXE="${SCRIPT_DIR}/Mip_Render_Inside_Volume"
STITCH_SCRIPT="${PROJECT_ROOT}/fafb_pilot/code/data/stitch_blocks.py"
EXPORTER="${SCRIPT_DIR}/export_renderer_bin.py"
PFM_CONVERTER="${SCRIPT_DIR}/pfm_to_png.py"
GAUSSIAN_JSON="${PROJECT_ROOT}/fafb_pilot/code/data/gaussians.json"
BIN_DIR="${SCRIPT_DIR}/bins"
RESULT_DIR="${SCRIPT_DIR}/results"
STITCH_OUT_DIR="${PROJECT_ROOT}/fafb_pilot/results/data"

SIZE=128
NAME=""
YAW_STEPS=8
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
RAY_SAMPLES=64
BENCHMARK_FRAMES=200
FORCE_REBAKE=0

usage() {
  echo "Usage: $0 --size {64|128|256} [--name NAME] [--yaw-steps N] [--force-rebake]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size) SIZE="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --yaw-steps) YAW_STEPS="$2"; shift 2;;
    --force-rebake) FORCE_REBAKE=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown argument: $1"; usage;;
  esac
done

case "${SIZE}" in
  64) N_PER_AXIS=1;;
  128) N_PER_AXIS=2;;
  256) N_PER_AXIS=4;;
  *) echo "--size must be 64, 128, or 256 (got '${SIZE}')"; exit 1;;
esac

NAME="${NAME:-size${SIZE}}"
REC_TIF="${STITCH_OUT_DIR}/rec_${NAME}.tif"

mkdir -p "${BIN_DIR}" "${RESULT_DIR}" "${STITCH_OUT_DIR}"

echo "=== Step 1: compiling renderer ==="
nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  "${CUDA_SRC}" \
  -o "${CUDA_EXE}"

if [[ -f "${REC_TIF}" && "${FORCE_REBAKE}" -eq 0 ]]; then
  echo "=== Step 2: BAKE skipped -- ${REC_TIF} already exists (pass --force-rebake to redo) ==="
else
  echo "=== Step 2: BAKE -- evaluating ${N_PER_AXIS}x${N_PER_AXIS}x${N_PER_AXIS} blocks on their own training voxel grid, hard-gated, clamped [0,1] ==="
  /venv/r3-ml/bin/python3 "${STITCH_SCRIPT}" \
    --gaussian-json "${GAUSSIAN_JSON}" \
    --n-per-axis "${N_PER_AXIS}" \
    --output-dir "${STITCH_OUT_DIR}" \
    --name "${NAME}"
fi

echo "=== Step 3: exporting the baked grid to the renderer's dense_voxel binary format ==="
/venv/r3-ml/bin/python3 "${EXPORTER}" \
  dense_voxel \
  "${REC_TIF}" \
  "${BIN_DIR}/baked_${NAME}.bin" \
  --normalise none

echo "=== Step 4: RENDER -- ${YAW_STEPS} camera angles from the SAME baked grid (no re-baking between frames) ==="
FPS_LOG="$(mktemp)"
trap 'rm -f "${FPS_LOG}"' EXIT

for ((i = 0; i < YAW_STEPS; i++)); do
  YAW=$(/venv/r3-ml/bin/python3 -c "print(f'{360.0 * ${i} / ${YAW_STEPS}:.2f}')")
  FRAME_NAME=$(printf "baked_%s_frame%03d" "${NAME}" "${i}")

  echo "--- frame ${i}: yaw=${YAW} ---"
  "${CUDA_EXE}" \
    dense_voxel \
    "${BIN_DIR}/baked_${NAME}.bin" \
    "${RESULT_DIR}/${FRAME_NAME}.pfm" \
    "${IMAGE_WIDTH}" "${IMAGE_HEIGHT}" \
    "${RAY_SAMPLES}" "${BENCHMARK_FRAMES}" \
    "${YAW}" 0 0 90 \
    -1 -1 -1 1 1 1 \
    | tee -a "${FPS_LOG}" | grep -E "FPS|Dense volume"

  /venv/r3-ml/bin/python3 "${PFM_CONVERTER}" \
    "${RESULT_DIR}/${FRAME_NAME}.pfm" \
    "${RESULT_DIR}/${FRAME_NAME}.png" \
    --vmin 0 --vmax 1 > /dev/null
done

echo ""
echo "=== Done ==="
echo "Baked grid (reusable for any number of future renders without re-baking):"
echo "  ${BIN_DIR}/baked_${NAME}.bin"
echo "Frames:"
echo "  ${RESULT_DIR}/baked_${NAME}_frame*.png"
echo ""
awk '/FPS:/{sum+=$2; n++} END{if (n>0) printf "Average FPS across %d camera angles: %.1f\n", n, sum/n}' "${FPS_LOG}"
