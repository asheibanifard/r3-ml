#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_SOURCE="${CUDA_SOURCE:-${DIR}/gaussian_volume_reconstruct.cu}"
CUDA_EXE="${CUDA_EXE:-${DIR}/gaussian_volume_reconstruct}"

CHECKPOINT="${CHECKPOINT:-models_smoke/block_z000_y001_x006/best.pth}"
OUTPUT="${OUTPUT:-${DIR}/reconstructed_volume.raw}"

NX="${NX:-50}"
NY="${NY:-50}"
NZ="${NZ:-50}"

MIN_X="${MIN_X:--1}"
MIN_Y="${MIN_Y:--1}"
MIN_Z="${MIN_Z:--1}"
MAX_X="${MAX_X:-1}"
MAX_Y="${MAX_Y:-1}"
MAX_Z="${MAX_Z:-1}"

CUTOFF="${CUTOFF:-20}"
INTENSITY_MODE="${INTENSITY_MODE:-softplus}"
SCALE_MODE="${SCALE_MODE:-direct}"
QUAT_ORDER="${QUAT_ORDER:-wxyz}"
CUDA_ARCH="${CUDA_ARCH:-89}"

command -v nvcc >/dev/null 2>&1 || {
    echo "Error: nvcc not found." >&2
    exit 1
}

command -v python >/dev/null 2>&1 || {
    echo "Error: python not found." >&2
    exit 1
}

[[ -f "${CHECKPOINT}" ]] || {
    echo "Error: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
}

echo "Compiling CUDA reconstructor..."
nvcc \
    -O3 \
    -std=c++17 \
    --use_fast_math \
    -lineinfo \
    -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
    "${CUDA_SOURCE}" \
    -o "${CUDA_EXE}"

echo "Loading best.pth and reconstructing..."
python "${DIR}/reconstruct_from_pth.py" \
    "${CHECKPOINT}" \
    "${OUTPUT}" \
    --cuda-exe "${CUDA_EXE}" \
    --nx "${NX}" \
    --ny "${NY}" \
    --nz "${NZ}" \
    --bounds \
        "${MIN_X}" "${MIN_Y}" "${MIN_Z}" \
        "${MAX_X}" "${MAX_Y}" "${MAX_Z}" \
    --cutoff "${CUTOFF}" \
    --intensity-mode "${INTENSITY_MODE}" \
    --scale-mode "${SCALE_MODE}" \
    --quat-order "${QUAT_ORDER}"

echo "Saved volume: ${OUTPUT}"
echo "Saved metadata: ${OUTPUT}.json"