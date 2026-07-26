# scripts/gaussian/rendering/pipeline.py
"""
Rasterise a trained Gaussian-volume checkpoint: dense per-voxel evaluation
(--mode volume, via the CUDA rasterisation kernel reconstruct_volume.cu)
and/or a true Maximum Intensity Projection of the model's own normalized
reconstruction (--mode mips):

    mip_{xy,xz,yz}.png   volume.amax(dim=axis) on the dense volume from
                         reconstruct.reconstruct() -- the model's actual
                         normalized training-time formula (V(x) = sum
                         w_i f_i / (sum w_i + eps), via the CUDA training
                         extension), the same one scripts/gaussian/
                         reconstruct.py itself uses. This blends
                         overlapping Gaussians the way training actually
                         does; a per-Gaussian analytic max (as
                         gaussian_volume.renderer.render_mip/splat_mip.cu
                         computes) can never blend them, so it shows
                         individual unblended Gaussians ("rice grains")
                         instead of the smooth trained anatomy.

Also optionally renders a ground-truth view directly on the raw voxel
grid (no Gaussians — pure PyTorch) alongside the MIPs, for direct
side-by-side comparison. Auto-discovered from the checkpoint's stored
input path unless --gt-volume/--no-gt-volume says otherwise:

    gt_dvr_{xy,xz,yz}.png   ray-marched Maximum Intensity Projection (NOT
                            Beer-Lambert DVR, despite the name — kept for
                            continuity with the existing CLI/config
                            surface) — see --dvr-step-size
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
    render_volume,
    save_volume,
)
from gaussian_volume.representation.config import config_defaults, load_config

# Reuse reconstruct.py's checkpoint-loading/model-building/reconstruction
# helpers instead of duplicating them. `reconstruct` calls the model's own
# normalized training-time formula (V(x) = sum w_i f_i / (sum w_i + eps)) —
# used here (not the CUDA rasterisation kernels' per-Gaussian analytic MIP)
# so mip_{xy,xz,yz}.png shows the blended, trained reconstruction rather
# than individual unblended Gaussians ("rice grains"): a per-Gaussian max
# can never blend overlapping Gaussians the way the normalized sum does.
from reconstruct import (
    build_model,
    get_gaussian_data,
    load_checkpoint,
    reconstruct,
    resolve_volume_shape,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    A --config TOML file (e.g. configs/config.toml's [rasterisation]
    section), if given, supplies defaults for the parameters listed in
    gaussian_volume.representation.config._CONFIG_TO_ARG; any of those
    parameters also passed explicitly on the command line override the
    config value.
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
            "(ray-marched max-projection, no Gaussians involved) for side-by-side comparison "
            "against the rasterised MIPs, saved as gt_dvr_{xy,xz,yz}.png. Defaults to the input "
            "volume path stored in the checkpoint (extra.input_path), if present. Pass "
            "--no-gt-volume to disable even when the checkpoint has one."
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
        "--epsilon", type=float, default=1.0e-8,
        help=(
            "Stability term in the normalized reconstruction's denominator "
            "(V(x) = sum w_i f_i / (sum w_i + epsilon)), used for mip_{xy,xz,yz}.png. "
            "Should match the value the checkpoint was trained with. Default: 1e-8."
        ),
    )

    parser.add_argument(
        "--dvr-step-size", type=float, default=1.0,
        help=(
            "Ray-march sample spacing in voxels for the ground-truth gt_dvr_{xy,xz,yz}.png "
            "(a ray-marched, interpolated max-intensity projection — see "
            "gaussian_volume.renderer.rasterisation.dvr_ground_truth). 1.0 samples once per "
            "voxel; smaller values sample more densely between voxel centres. Default: 1.0."
        ),
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

    if args.epsilon <= 0:
        raise ValueError(f"--epsilon must be positive, got {args.epsilon}.")

    if args.dvr_step_size <= 0:
        raise ValueError(f"--dvr-step-size must be positive, got {args.dvr_step_size}.")

    supplied = (args.depth, args.height, args.width)
    supplied_count = sum(value is not None for value in supplied)
    if supplied_count not in (0, 3):
        raise ValueError("--depth, --height, and --width must either all be provided or all omitted.")
    if supplied_count == 3 and any(value <= 0 for value in supplied):
        raise ValueError("Output dimensions must all be positive.")


def find_input_path(checkpoint: dict[str, Any]) -> Path | None:
    """
    Look up the source volume path pipeline.py stored in the checkpoint
    (extra.input_path), if any — used to auto-locate a ground-truth
    volume for gt_dvr_{xy,xz,yz}.png without requiring --gt-volume.
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
        # The model's own normalized reconstruction (same formula/CUDA
        # extension training optimizes against), not the rasterisation
        # kernels' per-Gaussian analytic max — blending overlapping
        # Gaussians the way training actually does, instead of showing
        # individual unblended Gaussians. Native volume resolution only
        # (reconstruct.reconstruct has no independent screen-size concept).
        normalized_volume = reconstruct(model, volume_shape, args.cutoff_sigma, args.epsilon)
        mips = {
            "xy": normalized_volume.amax(dim=0),
            "xz": normalized_volume.amax(dim=1),
            "yz": normalized_volume.amax(dim=2),
        }
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

            # A ray-marched, interpolated max reduction (see
            # gaussian_volume.renderer.rasterisation.dvr_ground_truth) —
            # native resolution, matching mip_{xy,xz,yz}.png above.
            gt_dvrs = render_orthogonal_dvr_ground_truth(
                gt_volume,
                step_size=args.dvr_step_size,
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
