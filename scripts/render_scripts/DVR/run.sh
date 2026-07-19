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
  /root/project/data/fafb/blocks/image_z32_y31_x31.tif \
  /root/project/fafb_pilot/code/visualisation/DVR/export_renderer_bin.bin \
  --dataset raw \
  --normalise minmax

"${CUDA_EXE}" \
  dense_voxel \
  /root/project/fafb_pilot/code/visualisation/DVR/export_renderer_bin.bin \
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
  /root/project/fafb_pilot/models/blocks_v2/b_211/best.pth \
  /root/project/fafb_pilot/code/visualisation/DVR/export_renderer_bin.bin \
  --means-key means \
  --scales-key log_scales \
  --quaternions-key quats \
  --intensity-key intensities \
  --scale-activation exp \
  --intensity-activation softplus \
  --quaternion-order wxyz

"${CUDA_EXE}" \
  pretrained_gaussian \
  /root/project/fafb_pilot/code/visualisation/DVR/export_renderer_bin.bin \
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
  # --vmin 0 --vmax 1