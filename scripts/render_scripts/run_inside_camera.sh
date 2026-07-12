#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_SOURCE="${CUDA_SOURCE:-${SCRIPT_DIR}/gaussian_mip_inside_camera.cu}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-${SCRIPT_DIR}/export_gaussians.py}"
CHECKPOINT="${CHECKPOINT:-../../models_smoke/block_z000_y001_x006/best.pth}"

GAUSSIAN_BIN="${GAUSSIAN_BIN:-${SCRIPT_DIR}/gaussians.bin}"
EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/gaussian_mip_inside_camera}"
OUTPUT_PFM="${OUTPUT_PFM:-${SCRIPT_DIR}/inside_view.pfm}"

WIDTH="${WIDTH:-128}"
HEIGHT="${HEIGHT:-128}"
DEPTH_SAMPLES="${DEPTH_SAMPLES:-64}"
BENCHMARK_FRAMES="${BENCHMARK_FRAMES:-200}"

# Fixed camera location at the center of a normalized [-1,1]^3 block.
CAM_X="${CAM_X:-0.0}"
CAM_Y="${CAM_Y:-0.0}"
CAM_Z="${CAM_Z:-0.0}"

# Rotate the camera around its own fixed position.
YAW="${YAW:-0.0}"
PITCH="${PITCH:-0.0}"
ROLL="${ROLL:-0.0}"
FOV_Y="${FOV_Y:-90.0}"

BOX_MIN_X="${BOX_MIN_X:--1.0}"
BOX_MIN_Y="${BOX_MIN_Y:--1.0}"
BOX_MIN_Z="${BOX_MIN_Z:--1.0}"
BOX_MAX_X="${BOX_MAX_X:-1.0}"
BOX_MAX_Y="${BOX_MAX_Y:-1.0}"
BOX_MAX_Z="${BOX_MAX_Z:-1.0}"

INTENSITY_MODE="${INTENSITY_MODE:-softplus}"
QUAT_ORDER="${QUAT_ORDER:-wxyz}"
CUDA_ARCH="${CUDA_ARCH:-89}"

nvcc \
    -O3 \
    -std=c++17 \
    --use_fast_math \
    -lineinfo \
    -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
    "${CUDA_SOURCE}" \
    -o "${EXECUTABLE}"

echo "Camera: (${CAM_X}, ${CAM_Y}, ${CAM_Z})"
echo "Block: [${BOX_MIN_X}, ${BOX_MIN_Y}, ${BOX_MIN_Z}] to [${BOX_MAX_X}, ${BOX_MAX_Y}, ${BOX_MAX_Z}]"
echo "Intensity activation: ${INTENSITY_MODE}"
echo

python "${EXPORT_SCRIPT}" \
    "${CHECKPOINT}" \
    "${GAUSSIAN_BIN}" \
    --intensity-mode "${INTENSITY_MODE}" \
    --quat-order "${QUAT_ORDER}"

"${EXECUTABLE}" \
    "${GAUSSIAN_BIN}" \
    "${OUTPUT_PFM}" \
    "${WIDTH}" \
    "${HEIGHT}" \
    "${DEPTH_SAMPLES}" \
    "${BENCHMARK_FRAMES}" \
    "${CAM_X}" "${CAM_Y}" "${CAM_Z}" \
    "${YAW}" "${PITCH}" "${ROLL}" \
    "${FOV_Y}" \
    "${BOX_MIN_X}" "${BOX_MIN_Y}" "${BOX_MIN_Z}" \
    "${BOX_MAX_X}" "${BOX_MAX_Y}" "${BOX_MAX_Z}"
