#!/usr/bin/env bash
# Direct volume rendering (DVR) for every trained SIREN checkpoint under
# ckpt/voxel/ (written by representation/train_processed_data.sh): the
# ground-truth render uses the voxel package's own JIT-compiled CUDA DVR
# kernel (render_direct_volume, front-to-back Beer-Lambert ray marching —
# src/voxel/extension/render.cu), and the predicted render uses the
# trained field's fast rasterizer (render_voxel_rasterizer). Both are
# already wired into rendering/reconstruct.py's single-checkpoint call;
# this script just batches that call over every block, on its own (e.g.
# after re-tuning DVR parameters) without redoing training.
#
# Each block's own ground_truth_volume.tif (written next to its checkpoints
# by representation/pipeline.py's --volume run) is passed back in via
# --volume, so the comparison is against the real trained EM block, not the
# synthetic sphere/torus/blob demo volume reconstruct.py falls back to when
# --volume is omitted.
#
# Writes into OUTPUT_DIR/<block_name>/<checkpoint-stem>/:
#
#   OUTPUT_DIR/<block_name>/best/slices.pdf        axial/coronal/sagittal
#                                                   reconstruction-vs-GT-vs-diff
#   OUTPUT_DIR/<block_name>/best/rendered.pdf      GT | pred | diff DVR
#                                                   render, annotated with
#                                                   PSNR/SSIM/LPIPS/FPS
#   OUTPUT_DIR/<block_name>/best/reconstructed_volume.tif
#   OUTPUT_DIR/<block_name>/best/gt_projection.pgm, pred_projection.pgm,
#                                diff.pgm, comparison.ppm
#
# Uses each block's best.pth (falls back to last.pth if best.pth is
# missing). Skips blocks that already have a rendered.pdf. Orbit frames are
# disabled by default (batch runs are about covering every block, not
# producing per-block demo videos — see reconstruct.py directly, or
# src/voxel/README.md, for that).
#
# Every DVR/rendering parameter is overridable via environment variable:
#   DENSITY_SCALE      Beer-Lambert opacity scale, shared by GT DVR and the
#                       pred rasterizer. Default: from configs/siren.yml.
#   DVR_STEPS          Ray-march step count for the GT DVR kernel. Default:
#                       from configs/siren.yml.
#   IMAGE_WIDTH/HEIGHT Render resolution. Default: from configs/siren.yml.
#   ORBIT_FRAMES        Number of saved orbit views; 0 (default) disables
#                       orbit output for the batch run.
#   BENCHMARK_FRAMES    Number of timed GPU frames per renderer. Default:
#                       from configs/siren.yml.
#   LPIPS_NET           Pretrained LPIPS backbone (alex/vgg/squeeze).
#                       Default: alex.
#
# Usage:
#   ./dvr_processed_data.sh
#   DENSITY_SCALE=6.0 DVR_STEPS=512 ./dvr_processed_data.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/r3-ml/bin/python3}"
CONFIG="configs/siren.yml"
CKPT_DIR="ckpt/voxel"
OUTPUT_DIR="outputs/voxel_rendered"
ORBIT_FRAMES="${ORBIT_FRAMES:-0}"
LPIPS_NET="${LPIPS_NET:-alex}"

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

    volume="${block_dir}ground_truth_volume.tif"
    if [[ ! -f "$volume" ]]; then
        echo "[${block_name}] skipping: no ground_truth_volume.tif found in ${block_dir}"
        continue
    fi

    block_output_dir="${OUTPUT_DIR}/${block_name}"
    checkpoint_stem="$(basename "$checkpoint" .pth)"

    if [[ -f "${block_output_dir}/${checkpoint_stem}/rendered.pdf" ]]; then
        echo "[${block_name}] skipping: ${block_output_dir}/${checkpoint_stem}/rendered.pdf already exists"
        continue
    fi

    echo "[${block_name}] DVR-rendering ${checkpoint} -> ${block_output_dir}"

    # DENSITY_SCALE/DVR_STEPS/IMAGE_WIDTH/IMAGE_HEIGHT/BENCHMARK_FRAMES, if
    # set, override configs/siren.yml (CLI flags always win over YAML — see
    # reconstruct.py's parse_args).
    extra_args=()
    [[ -n "${DENSITY_SCALE:-}" ]] && extra_args+=(--density-scale "$DENSITY_SCALE")
    [[ -n "${DVR_STEPS:-}" ]] && extra_args+=(--dvr-steps "$DVR_STEPS")
    [[ -n "${IMAGE_WIDTH:-}" ]] && extra_args+=(--image-width "$IMAGE_WIDTH")
    [[ -n "${IMAGE_HEIGHT:-}" ]] && extra_args+=(--image-height "$IMAGE_HEIGHT")
    [[ -n "${BENCHMARK_FRAMES:-}" ]] && extra_args+=(--benchmark-frames "$BENCHMARK_FRAMES")

    "$PYTHON" scripts/voxel/rendering/reconstruct.py \
        --config "$CONFIG" \
        --checkpoint "$checkpoint" \
        --volume "$volume" \
        --out "$block_output_dir" \
        --orbit-frames "$ORBIT_FRAMES" \
        --lpips-net "$LPIPS_NET" \
        "${extra_args[@]}"

    echo "[${block_name}] done"
done
