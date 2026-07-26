#!/usr/bin/env bash
# Train one SIREN voxel field per block under processed_data/, saving each
# block's checkpoints under ckpt/voxel/<block_name>/ (named after
# the input volume's own filename stem, since processed_data/ holds flat
# .tif files — e.g. processed_data/complex_image_z14_y11_x17.tif ->
# ckpt/voxel/complex_image_z14_y11_x17/):
#
#   ckpt/voxel/<block_name>/init.pth        pre-training snapshot
#   ckpt/voxel/<block_name>/best.pth        highest full-grid PSNR so far
#   ckpt/voxel/<block_name>/last.pth        final checkpoint
#   ckpt/voxel/<block_name>/<step>.pth      one snapshot every --checkpoint-every iters
#   ckpt/voxel/<block_name>/*.pdf           axial/coronal/sagittal
#                                                   reconstruction-vs-GT-vs-diff
#                                                   grid, one per *.pth above
#   ckpt/voxel/<block_name>/reconstruction.tif  final reconstructed volume
#
# grid_size is inferred from each block's own shape (real EM blocks are
# 64^3), and orbit-frame/benchmark rendering is kept minimal here — this
# script is about fitting every block, not producing per-block demo videos
# (see train.py directly, or src/voxel/README.md, for that; rendering a
# saved checkpoint afterwards is scripts/voxel/rendering/reconstruct.py).
#
# Backgrounds itself under nohup on first invocation, logging batch-level
# progress to logs/train_processed_data_voxel.log, so the batch
# survives a terminal disconnect (same convention as this project's other
# batch launch commands, e.g. logs/train_all_blocks.log).
#
# Usage:
#   ./train_processed_data.sh
set -euo pipefail

if [[ -z "${VOXEL_TRAIN_NOHUP_WRAPPED:-}" ]]; then
    mkdir -p logs
    LOG_FILE="logs/train_processed_data_voxel.log"
    echo "Backgrounding under nohup, logging to ${LOG_FILE}"
    VOXEL_TRAIN_NOHUP_WRAPPED=1 nohup "$0" "$@" >> "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/siren.yml"
DATA_DIR="processed_data"
OUTPUT_DIR="ckpt/voxel"

for input_file in "$DATA_DIR"/*.tif; do
    [[ -e "$input_file" ]] || continue

    block_name="$(basename "$input_file")"
    block_name="${block_name%.*}"

    block_output_dir="${OUTPUT_DIR}/${block_name}"

    if [[ -f "${block_output_dir}/last.pth" ]]; then
        echo "[${block_name}] skipping: ${block_output_dir}/last.pth already exists"
        continue
    fi

    echo "[${block_name}] training from ${input_file} -> ${block_output_dir}"

    # ITERATIONS/LEARNING_RATE env vars, if set, override configs/siren.yml
    # (CLI flags always win over YAML — see train.py's parse_args).
    extra_args=()
    [[ -n "${ITERATIONS:-}" ]] && extra_args+=(--iterations "$ITERATIONS")
    [[ -n "${LEARNING_RATE:-}" ]] && extra_args+=(--learning-rate "$LEARNING_RATE")

    "$PYTHON" scripts/voxel/representation/train.py \
        --config "$CONFIG" \
        --volume "$input_file" \
        --out "$block_output_dir" \
        --save-volume "${block_output_dir}/reconstruction.tif" \
        "${extra_args[@]}"

    echo "[${block_name}] done"
done
