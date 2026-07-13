#!/usr/bin/env bash
set -euo pipefail

CUDA_SRC="scripts/render_scripts/DVR/Mip_Render_Inside_Volume.cu"
CUDA_EXE="scripts/render_scripts/DVR/Mip_Render_Inside_Volume"
EXPORTER="scripts/render_scripts/DVR/export_renderer_bin.py"
PFM_CONVERTER="scripts/render_scripts/DVR/pfm_to_png.py"
RESULT_DIR="scripts/render_scripts/DVR/results/dvr"

mkdir -p "${RESULT_DIR}"

nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  "${CUDA_SRC}" \
  -o "${CUDA_EXE}"


# ----------------------------------------------------------------------
# Dense voxel ground truth
# ----------------------------------------------------------------------

python "${EXPORTER}" \
  dense_voxel \
  data/smoke_data/blocks/block_z0_y1_x6.h5 \
  scripts/render_scripts/DVR/gt_volume_016.bin \
  --dataset raw \
  --normalise minmax

"${CUDA_EXE}" \
  dense_voxel \
  scripts/render_scripts/DVR/gt_volume_016.bin \
  "${RESULT_DIR}/gt_block016_yaw0_pitch0.pfm" \
  128 128 \
  64 \
  200 \
  0 0 0 \
  90 \
  -1 -1 -1 \
   1  1  1

python "${PFM_CONVERTER}" \
  "${RESULT_DIR}/gt_block016_yaw0_pitch0.pfm" \
  "${RESULT_DIR}/gt_block016_yaw0_pitch0.png" \
  --vmin 0 --vmax 1


# ----------------------------------------------------------------------
# Pretrained Gaussian reconstruction
# ----------------------------------------------------------------------

python "${EXPORTER}" \
  pretrained_gaussian \
  models_smoke/block_z000_y001_x006/best.pth \
  scripts/render_scripts/DVR/gaussians_016.bin \
  --means-key means \
  --scales-key log_scales \
  --quaternions-key quats \
  --intensity-key intensities \
  --scale-activation exp \
  --intensity-activation softplus \
  --quaternion-order wxyz

"${CUDA_EXE}" \
  pretrained_gaussian \
  scripts/render_scripts/DVR/gaussians_016.bin \
  "${RESULT_DIR}/rec_block016_yaw0_pitch0.pfm" \
  128 128 \
  64 \
  200 \
  0 0 0 \
  90 \
  -1 -1 -1 \
   1  1  1

python "${PFM_CONVERTER}" \
  "${RESULT_DIR}/rec_block016_yaw0_pitch0.pfm" \
  "${RESULT_DIR}/rec_block016_yaw0_pitch0.png" \
  --vmin 0 --vmax 1