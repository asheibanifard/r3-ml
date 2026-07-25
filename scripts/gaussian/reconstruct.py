# scripts/reconstruction.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from gaussian_volume import (
    GaussianVolumeModel,
    denormalize_volume,
    save_volume,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a dense 3D voxel volume from a trained "
            "Gaussian-volume checkpoint."
        )
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the trained Gaussian checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstruction.tif"),
        help=(
            "Output volume path. Supported formats: "
            ".tif, .tiff, .npy, .npz, .h5, .hdf5, .pt, .pth."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Reconstruction device. Default: cuda.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "Output depth. Required only when volume_shape is not "
            "stored in the checkpoint."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Output height. Required only when volume_shape is not "
            "stored in the checkpoint."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Output width. Required only when volume_shape is not "
            "stored in the checkpoint."
        ),
    )

    parser.add_argument(
        "--cutoff-sigma",
        type=float,
        default=4.0,
        help="Gaussian Mahalanobis cutoff radius. Default: 4.0.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help=(
            "Numerical denominator stability constant. "
            "Default: 1e-8."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="reconstruction",
        help=(
            "Dataset name when saving HDF5 or NPZ output. "
            "Default: reconstruction."
        ),
    )

    parser.add_argument(
        "--denormalize",
        action="store_true",
        help=(
            "Convert the reconstructed [0,1] volume back to the "
            "original target intensity range using checkpoint metadata."
        ),
    )

    parser.add_argument(
        "--clamp",
        action="store_true",
        help="Clamp the normalized reconstruction to [0,1].",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.checkpoint}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    if args.cutoff_sigma <= 0:
        raise ValueError(
            "--cutoff-sigma must be positive, "
            f"got {args.cutoff_sigma}."
        )

    if args.epsilon <= 0:
        raise ValueError(
            f"--epsilon must be positive, got {args.epsilon}."
        )

    supplied_dimensions = (
        args.depth,
        args.height,
        args.width,
    )

    supplied_count = sum(
        value is not None
        for value in supplied_dimensions
    )

    if supplied_count not in {0, 3}:
        raise ValueError(
            "--depth, --height, and --width must either all be "
            "provided or all omitted."
        )

    if supplied_count == 3:
        if any(value <= 0 for value in supplied_dimensions):
            raise ValueError(
                "Output dimensions must all be positive."
            )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must contain a dictionary."
        )

    return checkpoint


def resolve_volume_shape(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    if (
        args.depth is not None
        and args.height is not None
        and args.width is not None
    ):
        return (
            int(args.depth),
            int(args.height),
            int(args.width),
        )

    shape = checkpoint.get("volume_shape")

    if shape is None:
        extra = checkpoint.get("extra", {})

        if isinstance(extra, dict):
            shape = extra.get("volume_shape")

    if shape is None:
        raise ValueError(
            "The checkpoint does not contain volume_shape. "
            "Provide --depth, --height, and --width."
        )

    if len(shape) != 3:
        raise ValueError(
            "Stored volume_shape must contain [D,H,W], "
            f"got {shape}."
        )

    depth, height, width = map(int, shape)

    if depth <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid stored volume shape: {shape}."
        )

    return depth, height, width


def get_gaussian_data(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    gaussians = checkpoint.get("gaussians")

    if gaussians is None:
        raise KeyError(
            "The checkpoint does not contain exported physical Gaussian "
            "parameters under the key 'gaussians'."
        )

    if not isinstance(gaussians, dict):
        raise ValueError(
            "checkpoint['gaussians'] must be a dictionary."
        )

    required = {
        "means",
        "scales",
        "quaternions",
        "confidences",
        "features",
    }

    missing = required.difference(gaussians)

    if missing:
        raise KeyError(
            "The checkpoint is missing Gaussian parameters: "
            f"{sorted(missing)}"
        )

    result: dict[str, torch.Tensor] = {}

    for name in required:
        tensor = gaussians[name]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Gaussian parameter {name!r} must be a tensor."
            )

        result[name] = tensor.to(
            device=device,
            dtype=torch.float32,
        ).contiguous()

    return result


def build_model(
    gaussians: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> GaussianVolumeModel:
    features = gaussians["features"]

    feature_min = 0.0
    feature_max = 1.0
    constrain_features = True
    minimum_scale = 1e-3
    minimum_confidence = 1e-6

    extra = checkpoint.get("extra", {})

    if isinstance(extra, dict):
        model_config = extra.get("model_config", {})

        if isinstance(model_config, dict):
            feature_min = float(
                model_config.get("feature_min", feature_min)
            )
            feature_max = float(
                model_config.get("feature_max", feature_max)
            )
            constrain_features = bool(
                model_config.get(
                    "constrain_features",
                    constrain_features,
                )
            )
            minimum_scale = float(
                model_config.get(
                    "minimum_scale",
                    minimum_scale,
                )
            )
            minimum_confidence = float(
                model_config.get(
                    "minimum_confidence",
                    minimum_confidence,
                )
            )

    model = GaussianVolumeModel(
        means=gaussians["means"],
        scales=gaussians["scales"],
        quaternions=gaussians["quaternions"],
        confidences=gaussians["confidences"],
        features=features,
        minimum_scale=minimum_scale,
        minimum_confidence=minimum_confidence,
        constrain_features=constrain_features,
        feature_min=feature_min,
        feature_max=feature_max,
    )

    return model.to(device)


def find_normalization(
    checkpoint: dict[str, Any],
) -> dict[str, float] | None:
    normalization = checkpoint.get("normalization")

    if normalization is None:
        extra = checkpoint.get("extra", {})

        if isinstance(extra, dict):
            normalization = extra.get("normalization")

    if not isinstance(normalization, dict):
        return None

    minimum = normalization.get("minimum")
    maximum = normalization.get("maximum")

    if minimum is None or maximum is None:
        return None

    return {
        "minimum": float(minimum),
        "maximum": float(maximum),
    }


@torch.inference_mode()
def reconstruct(
    model: GaussianVolumeModel,
    shape: tuple[int, int, int],
    cutoff_sigma: float,
    epsilon: float,
) -> torch.Tensor:
    model.eval()

    reconstructed = model(
        shape=shape,
        cutoff_sigma=cutoff_sigma,
        epsilon=epsilon,
    )

    if reconstructed.shape != shape:
        raise RuntimeError(
            "The model returned an incorrect reconstruction shape. "
            f"Expected {shape}, got {tuple(reconstructed.shape)}."
        )

    if not torch.isfinite(reconstructed).all():
        raise RuntimeError(
            "The reconstructed volume contains NaN or infinite values."
        )

    return reconstructed


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    device = torch.device(args.device)

    print("Loading checkpoint")
    print(f"  Path:   {args.checkpoint}")
    print(f"  Device: {device}")

    checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    volume_shape = resolve_volume_shape(
        checkpoint,
        args,
    )

    gaussians = get_gaussian_data(
        checkpoint,
        device,
    )

    model = build_model(
        gaussians,
        checkpoint,
        device,
    )

    print("Gaussian model")
    print(f"  Number of Gaussians: {model.num_gaussians}")
    print(f"  Volume shape:        {volume_shape}")
    print(f"  Cutoff sigma:        {args.cutoff_sigma}")
    print(f"  Epsilon:             {args.epsilon}")

    reconstructed = reconstruct(
        model=model,
        shape=volume_shape,
        cutoff_sigma=args.cutoff_sigma,
        epsilon=args.epsilon,
    )

    if args.clamp:
        reconstructed = reconstructed.clamp(0.0, 1.0)

    normalization = find_normalization(checkpoint)

    output_metadata: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "volume_shape": volume_shape,
        "num_gaussians": model.num_gaussians,
        "cutoff_sigma": args.cutoff_sigma,
        "epsilon": args.epsilon,
        "normalized": not args.denormalize,
    }

    if args.denormalize:
        if normalization is None:
            raise ValueError(
                "--denormalize was requested, but the checkpoint "
                "does not contain normalization minimum and maximum."
            )

        reconstructed = denormalize_volume(
            reconstructed,
            minimum=normalization["minimum"],
            maximum=normalization["maximum"],
        )

        output_metadata["normalization"] = normalization
        output_metadata["normalized"] = False

    output_path = save_volume(
        reconstructed,
        args.output,
        dataset=args.dataset,
        metadata=output_metadata,
    )

    print("Reconstruction complete")
    print(f"  Minimum: {reconstructed.min().item():.6f}")
    print(f"  Maximum: {reconstructed.max().item():.6f}")
    print(f"  Mean:    {reconstructed.mean().item():.6f}")
    print(f"  Saved:   {output_path}")


if __name__ == "__main__":
    main()