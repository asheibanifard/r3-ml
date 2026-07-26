#!/usr/bin/env bash
# Ground-truth "DVR" (ray-marched Maximum Intensity Projection — see
# gaussian_volume.renderer.rasterisation.dvr_ground_truth's docstring for
# why this is a max reduction, not Beer-Lambert compositing, despite the
# name) for every trained checkpoint under ckpt/ (written by
# train_processed_data.sh): one ray per output pixel, marched straight
# through the raw voxel grid with trilinear interpolation — no Gaussians,
# no CUDA kernel. Rendered at native volume resolution, matching
# mip_{xy,xz,yz}.png (the model's own normalized reconstruction's MIP —
# see pipeline.py), so the two are directly comparable. Run this on its
# own (e.g. after re-tuning the ray-march step size) without redoing the
# CUDA Gaussian rendering.
#
# Writes into the SAME per-block folder rasterise_processed_data.sh uses,
# outputs/rasterised/<block_name>/:
#
#   outputs/rasterised/<block_name>/gt_dvr_xy.png
#   outputs/rasterised/<block_name>/gt_dvr_xz.png
#   outputs/rasterised/<block_name>/gt_dvr_yz.png
#
# (the underlying pipeline.py call also (re)writes mip_*.png alongside,
# since both are computed together in one pass — harmless, just
# recomputed with the same values if it already existed.)
#
# Uses each block's best.pth (falls back to last.pth if best.pth is
# missing). Skips blocks that already have a gt_dvr_xy.png.
#
# Every parameter is overridable via environment variable:
#   DVR_STEP_SIZE           Ray-march sample spacing in voxels. 1.0 (default)
#                           samples once per voxel; smaller values sample
#                           more densely between voxel centres.
#   CUTOFF_SIGMA            Gaussian Mahalanobis cutoff radius, needed to
#                           build the model even though gt_dvr itself
#                           doesn't use it. Default: 4.0.
#   GT_VOLUME               Explicit ground-truth volume path, applied to
#                           every block (normally left unset — each block's
#                           own input path is auto-found from its checkpoint).
#
# Usage:
#   ./dvr_processed_data.sh
#   DVR_STEP_SIZE=0.5 ./dvr_processed_data.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/config.toml"
CKPT_DIR="ckpt"
OUTPUT_DIR="outputs/rasterised"
CUTOFF_SIGMA="${CUTOFF_SIGMA:-4.0}"
DVR_STEP_SIZE="${DVR_STEP_SIZE:-1.0}"

for block_dir in "$CKPT_DIR"/*/; do
    [[ -d "$block_dir" ]] || continue

    block_name="$(basename "$block_dir")"

    checkpoint="${block_dir}best.pth"
    if [[ ! -f "$checkpoint" ]]; then
        checkpoint="${block_dir}last.pth"
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "[${block_name}] skipping: no best.pth or last.pth found in ${block_dir}"
        continue
    fi

    block_output_dir="${OUTPUT_DIR}/${block_name}"

    if [[ -f "${block_output_dir}/gt_dvr_xy.png" ]]; then
        echo "[${block_name}] skipping: ${block_output_dir}/gt_dvr_xy.png already exists"
        continue
    fi

    echo "[${block_name}] DVR-rendering ${checkpoint} -> ${block_output_dir}"

    # GT_VOLUME, if set, overrides auto-discovery of each block's ground-truth volume.
    extra_args=()
    [[ -n "${GT_VOLUME:-}" ]] && extra_args+=(--gt-volume "$GT_VOLUME")

    "$PYTHON" scripts/gaussian/rendering/pipeline.py \
        --config "$CONFIG" \
        "$checkpoint" \
        --output-dir "$block_output_dir" \
        --mode mips \
        --cutoff-sigma "$CUTOFF_SIGMA" \
        --dvr-step-size "$DVR_STEP_SIZE" \
        "${extra_args[@]}"

    echo "[${block_name}] done"
done
