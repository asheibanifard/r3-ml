#!/usr/bin/env bash
# Train one 3D Gaussian field (gaussian_volume/_3dgs.py, the GaussianCloud
# pipeline) per block under processed_data/, using configs/config_v18.yml.
#
# --out is left unset so _3dgs.py auto-derives it from the input volume's
# own path: processed_data/<block_name>/<file>.tif -> outputs/<block_name>/<file>/
# receiving (flat_out=true in config_v18.yml):
#
#   outputs/<block_name>/<file>/init.pth        initial state, before training
#   outputs/<block_name>/<file>/best.pth        highest PSNR checkpoint so far
#   outputs/<block_name>/<file>/last.pth        final checkpoint
#   outputs/<block_name>/<file>/epoch_NNNN.pth  periodic snapshots
#
# Usage:
#   ./train_all_blocks_3dgs.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/config_v18.yml"
DATA_DIR="processed_data"
OUTPUT_DIR="outputs"

for block_dir in "$DATA_DIR"/*/; do
    block_name="$(basename "$block_dir")"

    input_file="$(find "$block_dir" -maxdepth 1 -name '*.tif' | head -n 1)"

    if [[ -z "$input_file" ]]; then
        echo "[${block_name}] skipping: no .tif input found in ${block_dir}"
        continue
    fi

    file_stem="$(basename "$input_file")"
    file_stem="${file_stem%.*}"

    block_output_dir="${OUTPUT_DIR}/${block_name}/${file_stem}"

    if [[ -f "${block_output_dir}/last.pth" ]]; then
        echo "[${block_name}] skipping: ${block_output_dir}/last.pth already exists"
        continue
    fi

    echo "[${block_name}] training from ${input_file} -> ${block_output_dir}"

    "$PYTHON" gaussian_volume/_3dgs.py \
        --config "$CONFIG" \
        --volume "$input_file"

    echo "[${block_name}] done"
done
