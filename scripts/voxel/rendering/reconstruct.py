#!/usr/bin/env python3
"""Render and evaluate a saved SIREN checkpoint, without retraining.

Loads a `<iteration>.pth`/`best.pth`/`last.pth` parameter checkpoint written
by representation/train.py, queries the SIREN field at every voxel centre,
and renders it against a synthetic ground-truth volume of the same grid size
(regenerated deterministically from the same seed, since the volume itself
isn't part of the checkpoint). This is the rendering/benchmarking counterpart
to representation/train.py — fitting and rendering are separate steps.

Every checkpoint gets its own self-contained folder,
`<out>/<checkpoint-stem>/` (e.g. `--checkpoint ckpt/best.pth` ->
`<out>/best/`):

    <out>/<checkpoint-stem>/<checkpoint-stem>.pth   copy of the checkpoint
    <out>/<checkpoint-stem>/slices.pdf               axial/coronal/sagittal
                                                      reconstruction-vs-GT-vs-diff
                                                      grid (3D, no camera)
    <out>/<checkpoint-stem>/rendered.pdf              GT | pred | diff
                                                      *rendered* comparison,
                                                      annotated with
                                                      PSNR/SSIM/LPIPS/FPS
    <out>/<checkpoint-stem>/reconstructed_volume.tif
    <out>/<checkpoint-stem>/gt_projection.pgm, pred_projection.pgm, diff.pgm,
                             comparison.ppm            (for visualize.py/
                                                      lpips_metric.py)
    <out>/<checkpoint-stem>/orbit_{gt,pred,diff,comparison}/*  (if
                                                      --orbit-frames > 0)

Example:
    python3 reconstruct.py --checkpoint output/best.pth --out output
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import torch

# scripts/voxel/ (this file's parent's parent) — added to sys.path so the
# `from _cli_config import ...`/`from _render_helpers import ...` below
# resolve their siblings one directory up (those helpers are shared with
# representation/pipeline.py too, so they don't live under rendering/).
VOXEL_SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(VOXEL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(VOXEL_SCRIPTS_DIR))

from _cli_config import parse_args_with_config
from _render_helpers import (
    benchmark_renderer,
    compute_lpips_image,
    render_pair,
    save_rendered_pdf,
)
from voxel import (
    VoxelFieldConfig,
    evaluate_images,
    load_siren_checkpoint,
    load_volume,
    make_camera,
    render_direct_volume,
    render_voxel_rasterizer,
    save_comparison_ppm,
    save_pgm,
    save_reconstruction_pdf,
    save_volume,
)
from voxel.extension.loader import load_voxel_extension
from voxel.model import SirenVoxelField
from voxel.training import CHECKPOINT_FIGURE_DPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "YAML config file (e.g. configs/siren.yml); CLI flags override YAML "
            "values. Should be the same config the checkpoint was trained with — "
            "hidden_size/hidden_layers/seed/first_omega_0/hidden_omega_0 must "
            "match whatever initialized it."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="SIREN parameter .pth file")
    parser.add_argument(
        "--volume", type=Path, default=None,
        help=(
            "Real cubic ground-truth volume the checkpoint was trained on "
            "(e.g. <ckpt-dir>/ground_truth_volume.tif, written by "
            "representation/pipeline.py's --volume run). grid_size is "
            "inferred from this file and --grid-size is ignored. Omit to "
            "compare against the synthetic sphere/torus/blob demo volume "
            "instead (only correct if the checkpoint was itself trained "
            "without --volume)."
        ),
    )
    parser.add_argument("--grid-size", type=int, default=512, help="Ignored if --volume is given")
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument(
        "--first-omega-0", type=float, default=30.0,
        help="Must match the first_omega_0 the checkpoint was trained with",
    )
    parser.add_argument(
        "--hidden-omega-0", type=float, default=1.0,
        help="Must match the hidden_omega_0 the checkpoint was trained with",
    )
    parser.add_argument("--dvr-steps", type=int, default=384)
    parser.add_argument("--density-scale", type=float, default=8.0)
    parser.add_argument(
        "--orbit-frames", type=int, default=120,
        help="Number of saved orbit views; 0 disables orbit output",
    )
    parser.add_argument(
        "--benchmark-frames", type=int, default=1000,
        help="Number of timed GPU frames per renderer",
    )
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex",
        help="Pretrained LPIPS backbone",
    )
    parser.add_argument("--train-batch", type=int, default=4096, help="Unused here; kept for a shared config")
    parser.add_argument("--seed", type=int, default=1234, help="Must match the seed train.py used")
    parser.add_argument("--out", type=Path, default=Path("output"))

    # Training-only keys accepted (but unused) purely so the same
    # configs/siren.yml that trained a checkpoint can be passed here too
    # without --config failing on "unknown keys".
    parser.add_argument("--iterations", type=int, default=2000, help="Unused here; kept for a shared config")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Unused here; kept for a shared config")
    parser.add_argument("--lr-min-ratio", type=float, default=1.0, help="Unused here; kept for a shared config")
    parser.add_argument("--lr-warmup-steps", type=int, default=0, help="Unused here; kept for a shared config")
    parser.add_argument(
        "--lr-warmup-init-factor", type=float, default=0.1, help="Unused here; kept for a shared config"
    )
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Unused here; kept for a shared config")

    return parse_args_with_config(parser, path_dests=("checkpoint", "volume", "out"))


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the voxel rasteriser requires an NVIDIA GPU.")

    device = torch.device("cuda")

    if args.volume is not None:
        ground_truth_volume = load_volume(args.volume, device=device)
        grid_size = ground_truth_volume.shape[0]
    else:
        ground_truth_volume = None
        grid_size = args.grid_size

    config = VoxelFieldConfig(
        grid_size=grid_size,
        image_width=args.image_width,
        image_height=args.image_height,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        train_batch=args.train_batch,
        dvr_steps=args.dvr_steps,
        density_scale=args.density_scale,
        seed=args.seed,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
    )

    extension = load_voxel_extension(config.hidden_size, config.hidden_layers)

    checkpoint_dir = args.out / args.checkpoint.stem
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.orbit_frames > 0:
        for name in ("orbit_gt", "orbit_pred", "orbit_diff", "orbit_comparison"):
            (checkpoint_dir / name).mkdir(parents=True, exist_ok=True)

    if ground_truth_volume is None:
        ground_truth_volume = extension.create_ground_truth_volume(config.grid_size, device)

    field = SirenVoxelField(config, device=device)
    field.load_parameters(load_siren_checkpoint(args.checkpoint, device=device))
    reconstructed_volume = field.reconstruct()

    shutil.copy2(args.checkpoint, checkpoint_dir / f"{args.checkpoint.stem}.pth")
    save_volume(checkpoint_dir / "reconstructed_volume.tif", reconstructed_volume)

    # 3D axial/coronal/sagittal slice comparison — no camera involved.
    save_reconstruction_pdf(
        reconstructed_volume, ground_truth_volume,
        checkpoint_dir / "slices.pdf", dpi=CHECKPOINT_FIGURE_DPI,
    )

    evaluation_camera = make_camera(0.72, 0.57, 4.15, 45.0, device=device)
    gt_image, pred_image = render_pair(
        ground_truth_volume, reconstructed_volume, evaluation_camera, config
    )

    mse, psnr, ssim = evaluate_images(
        gt_image, pred_image, hidden_size=config.hidden_size, hidden_layers=config.hidden_layers
    )
    lpips_value = compute_lpips_image(gt_image, pred_image, network=args.lpips_net, device=device)

    gt_host = gt_image.cpu().numpy()
    pred_host = pred_image.cpu().numpy()
    diff_host = abs(gt_host - pred_host)

    save_pgm(checkpoint_dir / "gt_projection.pgm", gt_host)
    save_pgm(checkpoint_dir / "pred_projection.pgm", pred_host)
    save_pgm(checkpoint_dir / "diff.pgm", diff_host)
    save_comparison_ppm(checkpoint_dir / "comparison.ppm", gt_host, pred_host, diff_host)

    print(f"Reconstructed from {args.checkpoint}")
    print(f"MSE   = {mse:.8g}")
    print(f"PSNR  = {psnr:.4f} dB")
    print(f"SSIM  = {ssim:.6f}")
    print(f"LPIPS = {lpips_value:.6f}")

    print(f"\nGPU rendering performance at {config.image_width}x{config.image_height}")
    warmup_frames = 100
    gt_fps = benchmark_renderer(
        "GT direct volume renderer",
        lambda: render_direct_volume(
            ground_truth_volume, evaluation_camera,
            image_width=config.image_width, image_height=config.image_height,
            dvr_steps=config.dvr_steps, density_scale=config.density_scale,
            hidden_size=config.hidden_size, hidden_layers=config.hidden_layers,
        ),
        warmup_frames, args.benchmark_frames,
    )
    pred_fps = benchmark_renderer(
        "Pred voxel rasterizer",
        lambda: render_voxel_rasterizer(
            reconstructed_volume, evaluation_camera,
            image_width=config.image_width, image_height=config.image_height,
            density_scale=config.density_scale,
            hidden_size=config.hidden_size, hidden_layers=config.hidden_layers,
        ),
        warmup_frames, args.benchmark_frames,
    )

    save_rendered_pdf(
        checkpoint_dir / "rendered.pdf",
        gt_host, pred_host, diff_host,
        metrics={"psnr": psnr, "ssim": ssim, "lpips": lpips_value, "gt_fps": gt_fps, "pred_fps": pred_fps},
    )

    if args.orbit_frames > 0:
        print(f"\nSaving {args.orbit_frames} orbit views...")
        for frame in range(args.orbit_frames):
            azimuth = 2.0 * math.pi * frame / args.orbit_frames
            elevation = 0.34 + 0.10 * math.sin(2.0 * azimuth)
            orbit_camera = make_camera(azimuth, elevation, 4.15, 45.0, device=device)

            gt_frame, pred_frame = render_pair(
                ground_truth_volume, reconstructed_volume, orbit_camera, config
            )
            gt_frame_host = gt_frame.cpu().numpy()
            pred_frame_host = pred_frame.cpu().numpy()
            diff_frame_host = abs(gt_frame_host - pred_frame_host)

            frame_name = f"{frame:04d}"
            save_pgm(checkpoint_dir / "orbit_gt" / f"gt_{frame_name}.pgm", gt_frame_host)
            save_pgm(checkpoint_dir / "orbit_pred" / f"pred_{frame_name}.pgm", pred_frame_host)
            save_pgm(checkpoint_dir / "orbit_diff" / f"diff_{frame_name}.pgm", diff_frame_host)
            save_comparison_ppm(
                checkpoint_dir / "orbit_comparison" / f"comparison_{frame_name}.ppm",
                gt_frame_host, pred_frame_host, diff_frame_host,
            )

            if (frame + 1) % 20 == 0 or frame + 1 == args.orbit_frames:
                print(f"  saved {frame + 1}/{args.orbit_frames} views")

    print(f"\nOutputs (in {checkpoint_dir}/)")
    print(f"  {args.checkpoint.stem}.pth")
    print("  slices.pdf")
    print("  rendered.pdf")
    print("  reconstructed_volume.tif")
    print("  gt_projection.pgm, pred_projection.pgm, diff.pgm, comparison.ppm")
    if args.orbit_frames > 0:
        print("  orbit_gt/, orbit_pred/, orbit_diff/, orbit_comparison/")
    print(
        f"Run: python3 visualize.py --output-dir {checkpoint_dir} --make-video --show\n"
        f"Run LPIPS standalone: python3 lpips_metric.py "
        f"{checkpoint_dir}/gt_projection.pgm {checkpoint_dir}/pred_projection.pgm"
    )


if __name__ == "__main__":
    main()
