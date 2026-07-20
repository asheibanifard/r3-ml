#!/usr/bin/env bash
# Render the real CUDA splatting/MIP renderer (Mip_Render_Inside_Volume.cu,
# pretrained_gaussian_hard_gated mode) against the SAME 3 hard-gated stitched
# scenes used by stitch_quality_report.py / sliding_cube_eval.py, but at all
# 3 combined-scene sizes: 64 (1 block, n_per_axis=1 -- gating trivially does
# nothing, sanity baseline), 128 (2x2x2=8 blocks, n_per_axis=2 -- the octant
# case already verified in test.ipynb), and 256 (4x4x4=64 blocks,
# n_per_axis=4 -- the full blocks_v18 pilot grid).
#
# n_per_axis=4 rendering requires Mip_Render_Inside_Volume.cu's hard-gating
# to distinguish 4 blocks per axis, not just 2 -- same_octant() (sign-only,
# only ever correct for exactly 2 partitions) was generalised to same_cell()
# (an n-way cell-index test, exact generalisation of stitch_blocks.py's
# combined_range() convention) to make this possible; n_per_axis=2 recovers
# the original sign-based behaviour exactly.
#
# STEPS
# -----
# 1. Compile the renderer (same nvcc invocation as run.sh).
# 2. For each size: stitch_blocks.py --n-per-axis {1,2,4} produces the
#    combined, globally-remapped Gaussians as gaussian_size<N>.pth (a single
#    flat Gaussian list covering the whole stitched scene -- see
#    stitch_blocks.py's own module docstring for why remapping is needed
#    here specifically, unlike the hard-gated VOXEL reconstruction path) --
#    AND, as a side effect of the same run, the hard-gated voxel
#    reconstruction rec_size<N>.tif (each voxel from exactly one block's own
#    clamped-to-[0,1] reconstruction -- no re-gating needed at render time).
# 3. export_renderer_bin.py's existing pretrained_gaussian mode packs the
#    combined .pth straight into the renderer's binary format (same flags
#    run.sh already uses for a single block's best.pth -- the combined
#    checkpoint has the identical key layout).
# 4. Mip_Render_Inside_Volume renders it in pretrained_gaussian_hard_gated
#    mode, camera at the box centre (box is always [-1,1]^3 in the combined
#    frame regardless of n_per_axis -- only voxel RESOLUTION grows with
#    n_per_axis, not physical extent, so the same camera/box works for all
#    3 sizes unchanged), passing n_per_axis as the new trailing CLI argument.
#    NOTE: this path ray-marches the CONTINUOUS, unbounded Gaussian-sum field
#    and MIP-keeps the max along each ray -- summing many overlapping
#    Gaussians can push that max near/above 1 at almost every pixel once
#    enough blocks are stitched together, so the image looks washed-out/
#    oversaturated even though it's clamped to [0,1] (this is the same
#    "sparse/oversaturated MIP" issue already documented for the single-block
#    case -- see export_renderer_bin.py's voxelized_gaussian docstring). It is
#    NOT a hard-gating bug -- it is the expected behaviour of this specific
#    representation mode.
# 5. ALSO render the bounded/clamped rec_size<N>.tif from step 2 through the
#    dense_voxel path (export_dense_voxel + dense_voxel render mode -- no
#    hard_gate needed, since gating already happened when rec_size<N>.tif was
#    built) -- a correct, non-oversaturated visualisation of the exact same
#    stitched scene, for comparison against the raw splat above.
# 6. pfm_to_png.py converts every render for viewing.
#
# USAGE
# -----
#   fafb_pilot/code/renderer/render_stitched_sizes.sh
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

mkdir -p "${BIN_DIR}" "${RESULT_DIR}" "${STITCH_OUT_DIR}"

echo "=== Step 1: compiling renderer ==="
nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  "${CUDA_SRC}" \
  -o "${CUDA_EXE}"

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
RAY_SAMPLES=16
BENCHMARK_FRAMES=200
YAW=0
PITCH=0
ROLL=0
FOV=90

# size -> n_per_axis (blocks per axis; size = n_per_axis * 64 native voxels)
SIZES=(64 128 256)
N_PER_AXIS=(1 2 4)

for i in "${!SIZES[@]}"; do
  SIZE="${SIZES[$i]}"
  N="${N_PER_AXIS[$i]}"
  NAME="size${SIZE}"

  echo ""
  echo "=== size=${SIZE} (n_per_axis=${N}) ==="

  echo "--- Step 2: stitching + remapping Gaussians ---"
  /venv/r3-ml/bin/python3 "${STITCH_SCRIPT}" \
    --gaussian-json "${GAUSSIAN_JSON}" \
    --n-per-axis "${N}" \
    --output-dir "${STITCH_OUT_DIR}" \
    --name "${NAME}"

  echo "--- Step 3: exporting combined Gaussians to renderer .bin ---"
  /venv/r3-ml/bin/python3 "${EXPORTER}" \
    pretrained_gaussian \
    "${STITCH_OUT_DIR}/gaussian_${NAME}.pth" \
    "${BIN_DIR}/gaussians_${NAME}.bin" \
    --means-key means \
    --scales-key log_scales \
    --quaternions-key quats \
    --intensity-key intensities \
    --scale-activation exp \
    --intensity-activation softplus \
    --quaternion-order wxyz

  echo "--- Step 4: rendering (pretrained_gaussian_hard_gated, camera at centre) ---"
  "${CUDA_EXE}" \
    pretrained_gaussian_hard_gated \
    "${BIN_DIR}/gaussians_${NAME}.bin" \
    "${RESULT_DIR}/stitched_${NAME}.pfm" \
    "${IMAGE_WIDTH}" "${IMAGE_HEIGHT}" \
    "${RAY_SAMPLES}" \
    "${BENCHMARK_FRAMES}" \
    "${YAW}" "${PITCH}" "${ROLL}" \
    "${FOV}" \
    -1 -1 -1 \
     1  1  1 \
    "${N}"

  echo "--- Step 5b: exporting + rendering the bounded rec_${NAME}.tif (dense_voxel path) ---"
  /venv/r3-ml/bin/python3 "${EXPORTER}" \
    dense_voxel \
    "${STITCH_OUT_DIR}/rec_${NAME}.tif" \
    "${BIN_DIR}/rec_${NAME}.bin" \
    --normalise none

  "${CUDA_EXE}" \
    dense_voxel \
    "${BIN_DIR}/rec_${NAME}.bin" \
    "${RESULT_DIR}/stitched_voxelized_${NAME}.pfm" \
    "${IMAGE_WIDTH}" "${IMAGE_HEIGHT}" \
    "${RAY_SAMPLES}" \
    "${BENCHMARK_FRAMES}" \
    "${YAW}" "${PITCH}" "${ROLL}" \
    "${FOV}" \
    -1 -1 -1 \
     1  1  1

  echo "--- Step 6: converting .pfm to .png ---"
  /venv/r3-ml/bin/python3 "${PFM_CONVERTER}" \
    "${RESULT_DIR}/stitched_${NAME}.pfm" \
    "${RESULT_DIR}/stitched_${NAME}.png" \
    --vmin 0 --vmax 1

  /venv/r3-ml/bin/python3 "${PFM_CONVERTER}" \
    "${RESULT_DIR}/stitched_voxelized_${NAME}.pfm" \
    "${RESULT_DIR}/stitched_voxelized_${NAME}.png" \
    --vmin 0 --vmax 1
done

echo ""
echo "=== Done. ==="
echo "  Raw continuous Gaussian-splat renders (oversaturated, see step 4 note): ${RESULT_DIR}/stitched_size{64,128,256}.png"
echo "  Bounded, correctly hard-gated renders (use these for a real look):      ${RESULT_DIR}/stitched_voxelized_size{64,128,256}.png"
