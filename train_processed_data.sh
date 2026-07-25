#!/usr/bin/env bash
# Train one 3D Gaussian model per block under processed_data/, saving each
# block's checkpoints under outputs/<block_name>/ (named after the input
# volume's own filename stem, since processed_data/ holds flat .tif files —
# e.g. processed_data/complex_image_z14_y11_x17.tif -> outputs/complex_image_z14_y11_x17/):
#
#   outputs/<block_name>/init.pth        initial state, before training
#   outputs/<block_name>/best.pth        highest full-grid PSNR so far
#   outputs/<block_name>/last.pth        final checkpoint
#   outputs/<block_name>/<step>.pth      one snapshot every logging.checkpoint_every iters
#   outputs/<block_name>/reconstruction.tif  final reconstructed volume
#
# pipeline.py writes its own per-block log file (logging.log_dir in
# configs/config.toml), so this script doesn't need to capture/tee output
# itself.
#
# Usage:
#   ./train_processed_data.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/config.toml"
DATA_DIR="processed_data"
OUTPUT_DIR="ckpt"

for input_file in "$DATA_DIR"/*.tif; do
    [[ -e "$input_file" ]] || continue

    block_name="$(basename "$input_file")"
    block_name="${block_name%.*}"

    block_output_dir="${OUTPUT_DIR}/${block_name}"

    if [[ -f "${block_output_dir}/last.pth" ]]; then
        echo "[${block_name}] skipping: ${block_output_dir}/last.pth already exists"
        continue
    fi

    echo "[${block_name}] training from ${input_file}"

    "$PYTHON" scripts/gaussian/pipeline.py \
        --config "$CONFIG" \
        "$input_file" \
        --output "$block_output_dir" \
        --save-volume "${block_output_dir}/reconstruction.tif"

    echo "[${block_name}] done"
done
