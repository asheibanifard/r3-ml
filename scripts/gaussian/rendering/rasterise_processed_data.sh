#!/usr/bin/env bash
# Rasterise every trained checkpoint under ckpt/ (written by
# train_processed_data.sh) via the CUDA rasterisation kernels
# (gaussian_volume.rasterisation — csrc/eval/{reconstruct_volume,
# splat_mip}.cu), writing each block's outputs under
# outputs/rasterised/<block_name>/:
#
#   outputs/rasterised/<block_name>/volume.tif      dense per-voxel rasterisation
#   outputs/rasterised/<block_name>/mip_xy.png       orthogonal MIP splats
#   outputs/rasterised/<block_name>/mip_xz.png
#   outputs/rasterised/<block_name>/mip_yz.png
#   outputs/rasterised/<block_name>/gt_mip_{xy,xz,yz}.png  plain max-projection
#                                                     of the raw ground-truth
#                                                     voxel grid (auto-found
#                                                     from the checkpoint's
#                                                     stored input path)
#
# Uses each block's best.pth (falls back to last.pth if best.pth is
# missing). Skips blocks whose rasterisation already exists.
#
# Passes --config configs/config.toml, so MIP output resolution follows
# that file's [rasterisation] screen_width/screen_height (independent of
# each block's own 64x64x64 shape) instead of defaulting to it.
#
# Usage:
#   ./rasterise_processed_data.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/config.toml"
CKPT_DIR="ckpt/gaussian_volume"
OUTPUT_DIR="outputs/rasterised"
MODE="${MODE:-both}"
CUTOFF_SIGMA="${CUTOFF_SIGMA:-4.0}"

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

    if [[ -f "${block_output_dir}/volume.tif" || -f "${block_output_dir}/mip_xy.png" ]]; then
        echo "[${block_name}] skipping: ${block_output_dir} already has rasterised output"
        continue
    fi

    echo "[${block_name}] rasterising ${checkpoint} -> ${block_output_dir}"

    "$PYTHON" scripts/gaussian/rendering/rasterise.py \
        --config "$CONFIG" \
        "$checkpoint" \
        --output-dir "$block_output_dir" \
        --mode "$MODE" \
        --cutoff-sigma "$CUTOFF_SIGMA"

    echo "[${block_name}] done"
done
