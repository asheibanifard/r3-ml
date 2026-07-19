#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_SRC="${SCRIPT_DIR}/Mip_Render_Inside_Volume.cu"
CUDA_EXE="${SCRIPT_DIR}/Mip_Render_Inside_Volume"
EXPORTER="${SCRIPT_DIR}/export_renderer_bin.py"
PFM_CONVERTER="${SCRIPT_DIR}/pfm_to_png.py"
BIN_DIR="${SCRIPT_DIR}/bins"
RESULT_DIR="${SCRIPT_DIR}/results"

mkdir -p "${BIN_DIR}" "${RESULT_DIR}"

nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  "${CUDA_SRC}" \
  -o "${CUDA_EXE}"


# ----------------------------------------------------------------------
# Dense voxel ground truth
# ----------------------------------------------------------------------

/venv/r3-ml/bin/python3 "${EXPORTER}" \
  dense_voxel \
  /root/project/data/fafb/blocks/image_z32_y31_x31.tif \
  "${BIN_DIR}/gt_volume.bin" \
  --dataset raw \
  --normalise minmax

"${CUDA_EXE}" \
  dense_voxel \
  "${BIN_DIR}/gt_volume.bin" \
  "${RESULT_DIR}/gt_block211_yaw0_pitch0.pfm" \
  128 128 \
  64 \
  200 \
  0 0 0 \
  90 \
  -1 -1 -1 \
   1  1  1

/venv/r3-ml/bin/python3 "${PFM_CONVERTER}" \
  "${RESULT_DIR}/gt_block211_yaw0_pitch0.pfm" \
  "${RESULT_DIR}/gt_block211_yaw0_pitch0.png" \
  --vmin 0 --vmax 1


# ----------------------------------------------------------------------
# Pretrained Gaussian reconstruction
# ----------------------------------------------------------------------

/venv/r3-ml/bin/python3 "${EXPORTER}" \
  pretrained_gaussian \
  /root/project/fafb_pilot/models/blocks_v18/b_211/best.pth \
  "${BIN_DIR}/gaussians.bin" \
  --means-key means \
  --scales-key log_scales \
  --quaternions-key quats \
  --intensity-key intensities \
  --scale-activation exp \
  --intensity-activation softplus \
  --quaternion-order wxyz

"${CUDA_EXE}" \
  pretrained_gaussian \
  "${BIN_DIR}/gaussians.bin" \
  "${RESULT_DIR}/rec_block211_yaw0_pitch0.pfm" \
  128 128 \
  32 \
  200 \
  0 0 0 \
  90 \
  -1 -1 -1 \
   1  1  1

/venv/r3-ml/bin/python3 "${PFM_CONVERTER}" \
  "${RESULT_DIR}/rec_block211_yaw0_pitch0.pfm" \
  "${RESULT_DIR}/rec_block211_yaw0_pitch0.png" \
   --vmin 0 --vmax 1


# ----------------------------------------------------------------------
# Voxelized Gaussian reconstruction (fair comparison against GT)
#
# The raw pretrained_gaussian path above ray-marches the continuous,
# Unbounded Gaussian-sum field, which can expose overfitting between
# Training-grid points (sparse/oversaturated MIP, poor correlation with GT).
# This path instead evaluates the same checkpoint on the exact training
# Voxel grid, clamps to [0,1], and renders it through the SAME bounded
# Dense_voxel MIP path used for GT above -- an apples-to-apples comparison.
# ----------------------------------------------------------------------

/venv/r3-ml/bin/python3 "${EXPORTER}" \
  voxelized_gaussian \
  /root/project/fafb_pilot/models/blocks_v18/b_211/best.pth \
  "${BIN_DIR}/voxelized_gaussian.bin" \
  --means-key means \
  --scales-key log_scales \
  --quaternions-key quats \
  --intensity-key intensities \
  --scale-activation exp \
  --intensity-activation softplus \
  --quaternion-order wxyz \
  --grid-size 64

"${CUDA_EXE}" \
  dense_voxel \
  "${BIN_DIR}/voxelized_gaussian.bin" \
  "${RESULT_DIR}/voxelized_rec_block211_yaw0_pitch0.pfm" \
  128 128 \
  64 \
  200 \
  0 0 0 \
  90 \
  -1 -1 -1 \
   1  1  1

/venv/r3-ml/bin/python3 "${PFM_CONVERTER}" \
  "${RESULT_DIR}/voxelized_rec_block211_yaw0_pitch0.pfm" \
  "${RESULT_DIR}/voxelized_rec_block211_yaw0_pitch0.png" \
  --vmin 0 --vmax 1
