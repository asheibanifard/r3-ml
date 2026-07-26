#!/usr/bin/env bash
# Rasterise every trained checkpoint under ckpt/ (written by
# train_processed_data.sh), writing each block's outputs under
# outputs/rasterised/<block_name>/:
#
#   outputs/rasterised/<block_name>/volume.tif      dense per-voxel
#                                                    rasterisation (CUDA,
#                                                    gaussian_volume.renderer
#                                                    — reconstruct_volume.cu)
#   outputs/rasterised/<block_name>/mip_xy.png       true MIP (amax) of the
#   outputs/rasterised/<block_name>/mip_xz.png       model's own normalized
#   outputs/rasterised/<block_name>/mip_yz.png       reconstruction (the
#                                                    CUDA training extension
#                                                    — same formula training
#                                                    optimizes against)
#   outputs/rasterised/<block_name>/gt_dvr_{xy,xz,yz}.png  ray-marched
#                                                     max-projection of the
#                                                     raw ground-truth voxel
#                                                     grid (auto-found from
#                                                     the checkpoint's
#                                                     stored input path)
#
# Uses each block's best.pth (falls back to last.pth if best.pth is
# missing). Always re-rasterises every block (no skip-if-exists check),
# so re-running after a rasterisation-kernel change reflects the update.
#
# Usage:
#   ./rasterise_processed_data.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/config.toml"
CKPT_DIR="ckpt"
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

    echo "[${block_name}] rasterising ${checkpoint} -> ${block_output_dir}"

    "$PYTHON" scripts/gaussian/rendering/pipeline.py \
        --config "$CONFIG" \
        "$checkpoint" \
        --output-dir "$block_output_dir" \
        --mode "$MODE" \
        --cutoff-sigma "$CUTOFF_SIGMA"

    echo "[${block_name}] done"
done
