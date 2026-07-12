#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Build, export, run, and benchmark the standalone CUDA Gaussian-mixture
# MIP renderer.
#
# Usage:
#   chmod +x run_gaussian_mip.sh
#   ./run_gaussian_mip.sh
#
# Optional environment overrides:
#   CHECKPOINT=../models_smoke/block_z000_y001_x006/best.pth
#   WIDTH=128
#   HEIGHT=128
#   DEPTH_SAMPLES=50
#   BENCHMARK_FRAMES=200
#   INTENSITY_MODE=direct
#   QUAT_ORDER=wxyz
#   CUDA_ARCH=89
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_SOURCE="${CUDA_SOURCE:-${SCRIPT_DIR}/gaussian_mip_realtime_fixed.cu}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-${SCRIPT_DIR}/export_gaussians.py}"

CHECKPOINT="${CHECKPOINT:-../../models_smoke/block_z000_y001_x006/best.pth}"
GAUSSIAN_BIN="${GAUSSIAN_BIN:-${SCRIPT_DIR}/gaussians.bin}"
EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/gaussian_mip_realtime}"
OUTPUT_PFM="${OUTPUT_PFM:-${SCRIPT_DIR}/output.pfm}"

WIDTH="${WIDTH:-128}"
HEIGHT="${HEIGHT:-128}"
DEPTH_SAMPLES="${DEPTH_SAMPLES:-50}"
BENCHMARK_FRAMES="${BENCHMARK_FRAMES:-200}"

# Use "direct" when the checkpoint already stores activated intensities.
# Use "softplus" only for raw unconstrained intensity parameters.
INTENSITY_MODE="${INTENSITY_MODE:-direct}"
QUAT_ORDER="${QUAT_ORDER:-wxyz}"

# RTX 4060 = compute capability 8.9.
CUDA_ARCH="${CUDA_ARCH:-89}"

command -v nvcc >/dev/null 2>&1 || {
    echo "Error: nvcc was not found in PATH." >&2
    exit 1
}

command -v python >/dev/null 2>&1 || {
    echo "Error: python was not found in PATH." >&2
    exit 1
}

[[ -f "${CUDA_SOURCE}" ]] || {
    echo "Error: CUDA source not found: ${CUDA_SOURCE}" >&2
    exit 1
}

[[ -f "${EXPORT_SCRIPT}" ]] || {
    echo "Error: exporter not found: ${EXPORT_SCRIPT}" >&2
    exit 1
}

[[ -f "${CHECKPOINT}" ]] || {
    echo "Error: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
}

echo "============================================================"
echo "1. Compiling CUDA renderer"
echo "============================================================"

nvcc \
    -O3 \
    -std=c++17 \
    --use_fast_math \
    -lineinfo \
    -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
    "${CUDA_SOURCE}" \
    -o "${EXECUTABLE}"

echo
echo "============================================================"
echo "2. Exporting Gaussian checkpoint"
echo "============================================================"

python "${EXPORT_SCRIPT}" \
    "${CHECKPOINT}" \
    "${GAUSSIAN_BIN}" \
    --intensity-mode "${INTENSITY_MODE}" \
    --quat-order "${QUAT_ORDER}"

echo
echo "============================================================"
echo "3. Rendering and benchmarking"
echo "============================================================"

"${EXECUTABLE}" \
    "${GAUSSIAN_BIN}" \
    "${OUTPUT_PFM}" \
    "${WIDTH}" \
    "${HEIGHT}" \
    "${DEPTH_SAMPLES}" \
    "${BENCHMARK_FRAMES}"

echo
echo "============================================================"
echo "Completed"
echo "============================================================"
echo "Executable:  ${EXECUTABLE}"
echo "Gaussian bin: ${GAUSSIAN_BIN}"
echo "Output image: ${OUTPUT_PFM}"
