#!/usr/bin/env bash
# Ground-truth direct volume rendering (DVR) for every trained checkpoint
# under ckpt/ (written by train_processed_data.sh): front-to-back
# Beer-Lambert compositing along each of the three orthogonal axes,
# marching parallel rays straight through the raw voxel grid — no
# Gaussians, no CUDA kernel (gaussian_volume.rasterisation.dvr_ground_truth
# is pure PyTorch). This is the DVR counterpart to rasterise_processed_data.sh's
# plain-MIP ground truth; run this on its own (e.g. after re-tuning DVR
# parameters) without redoing the CUDA Gaussian MIP splatting.
#
# Writes into the SAME per-block folder rasterise_processed_data.sh uses,
# outputs/rasterised/<block_name>/:
#
#   outputs/rasterised/<block_name>/gt_dvr_xy.png
#   outputs/rasterised/<block_name>/gt_dvr_xz.png
#   outputs/rasterised/<block_name>/gt_dvr_yz.png
#
# (the underlying rasterise.py call also (re)writes mip_*.png/gt_mip_*.png
# alongside, since all three are computed together in one pass — harmless,
# just recomputed with the same values if they already existed.)
#
# Uses each block's best.pth (falls back to last.pth if best.pth is
# missing). Skips blocks that already have a gt_dvr_xy.png.
#
# Every DVR parameter is overridable via environment variable:
#   DVR_DENSITY_SCALE       Beer-Lambert opacity scale. Unset (default) ->
#                           auto-exposed per view so its brightest pixel
#                           reaches TARGET_OPACITY. Set a number to disable
#                           auto-exposure and use it directly.
#   DVR_STEP_SIZE           Ray-march step size in voxels. Default: 1.0.
#   TARGET_OPACITY          Auto-exposure target (shared with the MIPs'
#                           own auto-exposure). Default: 0.9.
#   REFERENCE_DENSITY_SCALE density_scale used for the single auto-exposure
#                           probe. Default: 1e-4.
#   CUTOFF_SIGMA            Gaussian Mahalanobis cutoff radius, needed to
#                           build the model even though DVR itself doesn't
#                           use it. Default: 4.0.
#   GT_VOLUME               Explicit ground-truth volume path, applied to
#                           every block (normally left unset — each block's
#                           own input path is auto-found from its checkpoint).
#
# Usage:
#   ./dvr_processed_data.sh
#   DVR_DENSITY_SCALE=6.0 DVR_STEP_SIZE=0.5 ./dvr_processed_data.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/config.toml"
CKPT_DIR="ckpt/gaussian_volume"
OUTPUT_DIR="outputs/rasterised"
CUTOFF_SIGMA="${CUTOFF_SIGMA:-4.0}"
DVR_STEP_SIZE="${DVR_STEP_SIZE:-1.0}"
TARGET_OPACITY="${TARGET_OPACITY:-0.9}"
REFERENCE_DENSITY_SCALE="${REFERENCE_DENSITY_SCALE:-1e-4}"

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

    # DVR_DENSITY_SCALE/GT_VOLUME, if set, override auto-exposure/auto-discovery.
    extra_args=()
    [[ -n "${DVR_DENSITY_SCALE:-}" ]] && extra_args+=(--dvr-density-scale "$DVR_DENSITY_SCALE")
    [[ -n "${GT_VOLUME:-}" ]] && extra_args+=(--gt-volume "$GT_VOLUME")

    "$PYTHON" scripts/gaussian/rendering/rasterise.py \
        --config "$CONFIG" \
        "$checkpoint" \
        --output-dir "$block_output_dir" \
        --mode mips \
        --cutoff-sigma "$CUTOFF_SIGMA" \
        --dvr-step-size "$DVR_STEP_SIZE" \
        --target-opacity "$TARGET_OPACITY" \
        --reference-density-scale "$REFERENCE_DENSITY_SCALE" \
        "${extra_args[@]}"

    echo "[${block_name}] done"
done
