# scripts/gaussian/rendering/rasterise.py
"""
Rasterise a trained Gaussian-volume checkpoint: dense per-voxel evaluation
and/or orthogonal MIP splatting via the CUDA rasterisation kernels
(gaussian_volume.rasterisation — csrc/eval/{reconstruct_volume,splat_mip}.cu).

Separate from scripts/gaussian/reconstruct.py: that script reconstructs via
the model's own normalized training-time formula (the CUDA training
extension); this one rasterises via the unnormalized, Beer-Lambert-style
kernels used for visualization (see gaussian_volume/rasterisation.py's
module docstring for why the two won't produce identical images).

Also optionally renders two ground-truth views directly on the raw voxel
grid (no Gaussians, no CUDA kernel — pure PyTorch) alongside the
rasterised MIPs, for direct side-by-side comparison. Auto-discovered from
the checkpoint's stored input path unless --gt-volume/--no-gt-volume says
otherwise:

    gt_mip_{xy,xz,yz}.png   plain Maximum Intensity Projection (max-reduction)
    gt_dvr_{xy,xz,yz}.png   front-to-back Beer-Lambert direct volume
                            rendering (DVR), marching parallel rays along
                            each axis — see --dvr-density-scale/--dvr-step-size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# scripts/gaussian/ (this file's parent's parent) — added to sys.path so
# the `from reconstruct import ...` below resolves reconstruct.py, its
# sibling one directory up (reconstruct.py itself doesn't live under
# rendering/, since it isn't rendering-specific).
GAUSSIAN_SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(GAUSSIAN_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_SCRIPTS_DIR))

import numpy as np
import torch
from PIL import Image

from gaussian_volume import (
    load_volume,
    render_orthogonal_dvr_ground_truth,
    render_orthogonal_mips,
    render_orthogonal_mips_ground_truth,
    render_volume,
    save_volume,
)
from gaussian_volume.config import config_defaults, load_config

# Reuse reconstruct.py's checkpoint-loading/model-building helpers instead
# of duplicating them.
from reconstruct import (
    build_model,
    get_gaussian_data,
    load_checkpoint,
    resolve_volume_shape,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    A --config TOML file (e.g. configs/config.toml's [rasterisation]
    section), if given, supplies defaults for the parameters listed in
    gaussian_volume.config._CONFIG_TO_ARG; any of those parameters also
    passed explicitly on the command line override the config value.
    """
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, default=None, help="Path to a TOML experiment config file.",
    )
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description=(
            "Rasterise a trained Gaussian-volume checkpoint: dense "
            "per-voxel evaluation and/or orthogonal MIP splatting."
        ),
    )

    parser.add_argument("checkpoint", type=Path, help="Path to the trained Gaussian checkpoint.")

    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rasterised"),
        help="Directory to write volume.tif / mip_{xy,xz,yz}.png into. Default: outputs/rasterised.",
    )

    parser.add_argument(
        "--mode", choices=("volume", "mips", "both"), default="both",
        help="Which rasterisation(s) to produce. Default: both.",
    )

    parser.add_argument(
        "--gt-volume", type=Path, default=None,
        help=(
            "Ground-truth voxel volume (e.g. a processed_data/*.tif block) to also MIP-project "
            "(plain max-projection, no Gaussians involved) for side-by-side comparison against "
            "the rasterised MIPs, saved as gt_mip_{xy,xz,yz}.png. Defaults to the input volume "
            "path stored in the checkpoint (extra.input_path), if present. Pass --no-gt-volume "
            "to disable even when the checkpoint has one."
        ),
    )
    parser.add_argument(
        "--no-gt-volume", action="store_true",
        help="Skip ground-truth MIP rendering even if the checkpoint has a stored input path.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="Rasterisation device. Default: cuda.")

    parser.add_argument("--depth", type=int, default=None, help="Output depth (required if not in checkpoint).")
    parser.add_argument("--height", type=int, default=None, help="Output height (required if not in checkpoint).")
    parser.add_argument("--width", type=int, default=None, help="Output width (required if not in checkpoint).")

    parser.add_argument(
        "--cutoff-sigma", type=float, default=4.0,
        help="Gaussian Mahalanobis cutoff radius. Default: 4.0.",
    )
    parser.add_argument(
        "--depth-samples", type=int, default=32,
        help="Accepted for API parity with the legacy CPU MIP fallback; unused by the CUDA kernel. Default: 32.",
    )
    parser.add_argument(
        "--density-scale", type=float, default=None,
        help=(
            "MIP opacity mapping scale (1 - exp(-density_scale * accumulated_density)). "
            "Default: auto-exposed per view so its brightest pixel reaches --target-opacity "
            "(see --target-opacity/--reference-density-scale). Pass a value here to disable "
            "auto-exposure and use it directly."
        ),
    )
    parser.add_argument(
        "--target-opacity", type=float, default=0.9,
        help="Auto-exposure target: brightest pixel's opacity after rescaling. Default: 0.9.",
    )
    parser.add_argument(
        "--reference-density-scale", type=float, default=1.0e-4,
        help="density_scale used for the single auto-exposure probe launch. Default: 1e-4.",
    )
    parser.add_argument(
        "--max-gauss-per-tile", type=int, default=0,
        help="Cap Gaussians considered per screen tile (0 = unlimited). Default: 0.",
    )
    parser.add_argument(
        "--screen-width", type=int, default=None,
        help=(
            "Independent MIP-projection output width, decoupled from the volume's own "
            "voxel-grid dimensions (like a camera's screen resolution). Default: each of the "
            "three orthogonal views matches the volume's own shape exactly (no downsampling). "
            "Must be given together with --screen-height."
        ),
    )
    parser.add_argument(
        "--screen-height", type=int, default=None,
        help="Independent MIP-projection output height; see --screen-width. Default: unset.",
    )

    parser.add_argument(
        "--dvr-density-scale", type=float, default=None,
        help=(
            "Direct volume rendering (DVR) opacity mapping scale for the ground-truth "
            "gt_dvr_{xy,xz,yz}.png (1 - exp(-dvr_density_scale * accumulated_density * "
            "dvr_step_size)), front-to-back Beer-Lambert compositing along each axis. "
            "Default: auto-exposed so its brightest pixel reaches --target-opacity, same as "
            "--density-scale for the rasterised MIPs. Pass a value here to disable auto-exposure."
        ),
    )
    parser.add_argument(
        "--dvr-step-size", type=float, default=1.0,
        help="DVR ray-march step size in voxels. Default: 1.0.",
    )

    config: dict[str, Any] = {}
    if config_args.config is not None:
        config = load_config(config_args.config)
        parser.set_defaults(**config_defaults(config))

    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    if args.gt_volume is not None:
        if args.no_gt_volume:
            raise ValueError("--gt-volume and --no-gt-volume are mutually exclusive.")
        if not args.gt_volume.exists():
            raise FileNotFoundError(f"--gt-volume does not exist: {args.gt_volume}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if args.cutoff_sigma <= 0:
        raise ValueError(f"--cutoff-sigma must be positive, got {args.cutoff_sigma}.")

    if args.density_scale is not None and args.density_scale <= 0:
        raise ValueError(f"--density-scale must be positive, got {args.density_scale}.")

    if not (0.0 < args.target_opacity < 1.0):
        raise ValueError(f"--target-opacity must be in (0, 1), got {args.target_opacity}.")

    if args.reference_density_scale <= 0:
        raise ValueError(
            f"--reference-density-scale must be positive, got {args.reference_density_scale}."
        )

    if args.dvr_density_scale is not None and args.dvr_density_scale <= 0:
        raise ValueError(f"--dvr-density-scale must be positive, got {args.dvr_density_scale}.")

    if args.dvr_step_size <= 0:
        raise ValueError(f"--dvr-step-size must be positive, got {args.dvr_step_size}.")

    if args.max_gauss_per_tile < 0:
        raise ValueError(f"--max-gauss-per-tile cannot be negative, got {args.max_gauss_per_tile}.")

    supplied = (args.depth, args.height, args.width)
    supplied_count = sum(value is not None for value in supplied)
    if supplied_count not in (0, 3):
        raise ValueError("--depth, --height, and --width must either all be provided or all omitted.")
    if supplied_count == 3 and any(value <= 0 for value in supplied):
        raise ValueError("Output dimensions must all be positive.")

    screen_supplied = (args.screen_width, args.screen_height)
    screen_supplied_count = sum(value is not None for value in screen_supplied)
    if screen_supplied_count not in (0, 2):
        raise ValueError("--screen-width and --screen-height must either both be provided or both omitted.")
    if screen_supplied_count == 2 and any(value <= 0 for value in screen_supplied):
        raise ValueError("--screen-width and --screen-height must both be positive.")


def find_input_path(checkpoint: dict[str, Any]) -> Path | None:
    """
    Look up the source volume path pipeline.py stored in the checkpoint
    (extra.input_path), if any — used to auto-locate a ground-truth
    volume for gt_mip_{xy,xz,yz}.png without requiring --gt-volume.
    """
    extra = checkpoint.get("extra", {})
    input_path = extra.get("input_path") if isinstance(extra, dict) else None
    return Path(input_path) if input_path else None


def save_mip_png(path: Path, image: torch.Tensor) -> None:
    array = image.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array_uint8 = np.round(array * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_uint8, mode="L").save(path)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint")
    print(f"  Path:   {args.checkpoint}")
    print(f"  Device: {device}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    volume_shape = resolve_volume_shape(checkpoint, args)
    gaussians = get_gaussian_data(checkpoint, device)
    model = build_model(gaussians, checkpoint, device)

    print("Gaussian model")
    print(f"  Number of Gaussians: {model.num_gaussians}")
    print(f"  Volume shape:        {volume_shape}")
    print(f"  Cutoff sigma:        {args.cutoff_sigma}")

    if args.mode in ("volume", "both"):
        volume = render_volume(model, volume_shape, cutoff_sigma=args.cutoff_sigma)
        volume_path = args.output_dir / "volume.tif"
        save_volume(
            volume, volume_path, dataset="rasterised",
            metadata={
                "checkpoint": str(args.checkpoint.resolve()),
                "volume_shape": volume_shape,
                "num_gaussians": model.num_gaussians,
                "cutoff_sigma": args.cutoff_sigma,
            },
        )
        print(f"  Wrote {volume_path}  (min={volume.min().item():.4f} max={volume.max().item():.4f})")

    if args.mode in ("mips", "both"):
        screen_size = (
            (args.screen_height, args.screen_width)
            if args.screen_width is not None
            else None
        )
        if screen_size is not None:
            print(f"  Screen size:         {screen_size[1]}x{screen_size[0]} (independent of volume shape)")

        mips = render_orthogonal_mips(
            model, volume_shape,
            cutoff_sigma=args.cutoff_sigma,
            screen_size=screen_size,
            depth_samples=args.depth_samples,
            density_scale=args.density_scale,
            target_opacity=args.target_opacity,
            reference_density_scale=args.reference_density_scale,
            max_gauss_per_tile=args.max_gauss_per_tile,
        )
        for name, image in mips.items():
            mip_path = args.output_dir / f"mip_{name}.png"
            save_mip_png(mip_path, image)
            print(f"  Wrote {mip_path}  (min={image.min().item():.6f} max={image.max().item():.6f})")

        gt_volume_path = args.gt_volume
        if gt_volume_path is None and not args.no_gt_volume:
            gt_volume_path = find_input_path(checkpoint)
            if gt_volume_path is not None and not gt_volume_path.exists():
                print(f"  Checkpoint's stored input_path no longer exists, skipping ground truth: {gt_volume_path}")
                gt_volume_path = None

        if gt_volume_path is not None:
            print(f"  Ground-truth volume: {gt_volume_path}")
            gt_volume, _ = load_volume(
                gt_volume_path, dataset=None, device=device, normalize=True,
                normalization_minimum=None, normalization_maximum=None,
            )
            if tuple(gt_volume.shape) != tuple(volume_shape):
                raise ValueError(
                    f"--gt-volume shape {tuple(gt_volume.shape)} does not match the "
                    f"checkpoint's volume_shape {volume_shape}."
                )

            # Always at the ground-truth volume's own native resolution — a
            # plain max-projection, not resized to --screen-width/-height,
            # so it stays a direct, unaltered look at the raw data.
            gt_mips = render_orthogonal_mips_ground_truth(gt_volume)
            for name, image in gt_mips.items():
                gt_mip_path = args.output_dir / f"gt_mip_{name}.png"
                save_mip_png(gt_mip_path, image)
                print(
                    f"  Wrote {gt_mip_path}  "
                    f"(min={image.min().item():.6f} max={image.max().item():.6f})"
                )

            # Same native-resolution rule as gt_mip above.
            gt_dvrs = render_orthogonal_dvr_ground_truth(
                gt_volume,
                density_scale=args.dvr_density_scale,
                step_size=args.dvr_step_size,
                target_opacity=args.target_opacity,
                reference_density_scale=args.reference_density_scale,
            )
            for name, image in gt_dvrs.items():
                gt_dvr_path = args.output_dir / f"gt_dvr_{name}.png"
                save_mip_png(gt_dvr_path, image)
                print(
                    f"  Wrote {gt_dvr_path}  "
                    f"(min={image.min().item():.6f} max={image.max().item():.6f})"
                )

    print("Rasterisation complete")


if __name__ == "__main__":
    main()
