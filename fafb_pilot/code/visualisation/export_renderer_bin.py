#!/usr/bin/env python3
"""
Unified binary exporter for the adaptive CUDA inside-camera renderer.

Supported modes
---------------
1. dense_voxel
   Input: .npy, .h5, or .hdf5 dense voxel block
   Output magic: VOXL (0x564F584C)

2. pretrained_gaussian
   Input: PyTorch checkpoint containing Gaussian parameter tensors
   Output magic: GSMP (0x47534D50)

Dense example
-------------
python export_renderer_bin.py dense_voxel \
    block.h5 volume.bin \
    --dataset raw \
    --normalise none

Gaussian example
----------------
python export_renderer_bin.py pretrained_gaussian \
    checkpoint.pt gaussians.bin \
    --means-key means \
    --scales-key log_scales \
    --quaternions-key quaternions \
    --intensity-key intensity_logits \
    --scale-activation exp \
    --intensity-activation softplus \
    --quaternion-order wxyz

The Gaussian checkpoint may contain the tensors directly or inside a nested
mapping such as `state_dict`. Use --checkpoint-root state_dict in that case.

Gaussian binary record
----------------------
mean_x mean_y mean_z
scale_x scale_y scale_z
quat_w quat_x quat_y quat_z
intensity

All values are float32. Each Gaussian occupies 44 bytes.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

VOXEL_MAGIC = 0x564F584C
GAUSSIAN_MAGIC = 0x47534D50
VERSION = 1


def load_dense_volume(path: Path, dataset: str) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        volume = np.load(path)
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "h5py is required for HDF5 input: pip install h5py"
            ) from error

        with h5py.File(path, "r") as file:
            if dataset not in file:
                raise KeyError(
                    f"Dataset {dataset!r} not found. "
                    f"Available datasets: {list(file.keys())}"
                )
            volume = file[dataset][...]
    else:
        raise ValueError(
            "Dense voxel input must be .npy, .h5, or .hdf5."
        )

    volume = np.asarray(volume)
    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D dense volume after squeeze, got {volume.shape}."
        )

    return volume


def export_dense_voxel(
    input_path: Path,
    output_path: Path,
    dataset: str,
    normalise: str,
) -> None:
    volume = load_dense_volume(input_path, dataset).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(volume).all():
        raise ValueError("Dense volume contains NaN or Inf.")

    if normalise == "minmax":
        minimum = float(volume.min())
        maximum = float(volume.max())

        if maximum > minimum:
            volume = (volume - minimum) / (maximum - minimum)
        else:
            volume = np.zeros_like(volume)

    depth, height, width = volume.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        file.write(
            struct.pack(
                "<IIIII",
                VOXEL_MAGIC,
                VERSION,
                depth,
                height,
                width,
            )
        )
        file.write(
            volume.astype("<f4", copy=False).tobytes(order="C")
        )

    expected_size = 20 + depth * height * width * 4
    actual_size = output_path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"Dense binary size mismatch: expected {expected_size}, "
            f"got {actual_size}."
        )

    print("Representation: dense_voxel")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Shape [D,H,W]: {volume.shape}")
    print(
        f"Range: [{float(volume.min())}, {float(volume.max())}]"
    )
    print(f"Bytes: {actual_size}")


def load_torch_checkpoint(path: Path) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for pretrained_gaussian mode."
        ) from error

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def resolve_root(checkpoint: Any, root: str | None) -> Any:
    if not root:
        return checkpoint

    current = checkpoint

    for part in root.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(
                    f"Checkpoint root component {part!r} was not found."
                )
            current = current[part]
        else:
            if not hasattr(current, part):
                raise AttributeError(
                    f"Checkpoint object has no attribute {part!r}."
                )
            current = getattr(current, part)

    return current


def resolve_value(container: Any, key: str) -> Any:
    current = container

    for part in key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(
                    f"Gaussian key component {part!r} was not found "
                    f"while resolving {key!r}."
                )
            current = current[part]
        else:
            if not hasattr(current, part):
                raise AttributeError(
                    f"Object has no attribute {part!r} "
                    f"while resolving {key!r}."
                )
            current = getattr(current, part)

        if callable(current):
            current = current()

    return current


def to_tensor(value: Any):
    import torch

    if isinstance(value, torch.nn.Parameter):
        value = value.data

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    return value.detach().cpu()


def to_nxk(tensor, width: int, label: str):
    tensor = tensor.squeeze()

    if tensor.ndim == 1 and width == 1:
        tensor = tensor[:, None]

    if tensor.ndim != 2:
        raise ValueError(
            f"{label} must be 2D after squeeze, got {tuple(tensor.shape)}."
        )

    if tensor.shape[1] == width:
        return tensor

    if tensor.shape[0] == width:
        return tensor.transpose(0, 1)

    raise ValueError(
        f"{label} must contain {width} values per Gaussian, "
        f"got {tuple(tensor.shape)}."
    )


def apply_activation(tensor, activation: str, label: str):
    import torch

    if activation == "none":
        return tensor
    if activation == "exp":
        return torch.exp(tensor)
    if activation == "softplus":
        return torch.nn.functional.softplus(tensor)
    if activation == "sigmoid":
        return torch.sigmoid(tensor)

    raise ValueError(
        f"Unsupported {label} activation: {activation}"
    )


def export_pretrained_gaussian(
    input_path: Path,
    output_path: Path,
    checkpoint_root: str | None,
    means_key: str,
    scales_key: str,
    quaternions_key: str,
    intensity_key: str,
    scale_activation: str,
    intensity_activation: str,
    quaternion_order: str,
) -> None:
    import torch

    checkpoint = load_torch_checkpoint(input_path)
    container = resolve_root(checkpoint, checkpoint_root)

    means = to_nxk(
        to_tensor(resolve_value(container, means_key)),
        3,
        "means",
    )
    scales = to_nxk(
        to_tensor(resolve_value(container, scales_key)),
        3,
        "scales",
    )
    quaternions = to_nxk(
        to_tensor(resolve_value(container, quaternions_key)),
        4,
        "quaternions",
    )
    intensity = to_nxk(
        to_tensor(resolve_value(container, intensity_key)),
        1,
        "intensity",
    )

    scales = apply_activation(
        scales,
        scale_activation,
        "scale",
    )
    intensity = apply_activation(
        intensity,
        intensity_activation,
        "intensity",
    )
    quaternions = torch.nn.functional.normalize(
        quaternions,
        dim=-1,
    )

    if quaternion_order == "xyzw":
        quaternions = quaternions[:, [3, 0, 1, 2]]
    elif quaternion_order != "wxyz":
        raise ValueError(
            "quaternion-order must be wxyz or xyzw."
        )

    gaussian_count = means.shape[0]

    for label, tensor in (
        ("scales", scales),
        ("quaternions", quaternions),
        ("intensity", intensity),
    ):
        if tensor.shape[0] != gaussian_count:
            raise ValueError(
                f"{label} contains {tensor.shape[0]} Gaussians, "
                f"but means contains {gaussian_count}."
            )

    if not torch.isfinite(means).all():
        raise ValueError("Means contain NaN or Inf.")
    if not torch.isfinite(scales).all():
        raise ValueError("Scales contain NaN or Inf.")
    if not torch.all(scales > 0):
        raise ValueError(
            "Exported scales must be strictly positive. "
            "Check --scale-activation."
        )
    if not torch.isfinite(quaternions).all():
        raise ValueError("Quaternions contain NaN or Inf.")
    if not torch.isfinite(intensity).all():
        raise ValueError("Intensity contains NaN or Inf.")
    if torch.any(intensity < 0):
        raise ValueError(
            "Exported intensity must be non-negative. "
            "Check --intensity-activation."
        )

    records = torch.cat(
        [means, scales, quaternions, intensity],
        dim=1,
    ).to(torch.float32).contiguous().numpy()

    if records.shape != (gaussian_count, 11):
        raise RuntimeError(
            f"Unexpected Gaussian record shape: {records.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        file.write(
            struct.pack(
                "<IIQ",
                GAUSSIAN_MAGIC,
                VERSION,
                gaussian_count,
            )
        )
        file.write(
            records.astype("<f4", copy=False).tobytes(order="C")
        )

    expected_size = 16 + gaussian_count * 44
    actual_size = output_path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"Gaussian binary size mismatch: expected {expected_size}, "
            f"got {actual_size}."
        )

    print("Representation: pretrained_gaussian")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Gaussians: {gaussian_count}")
    print("Bytes per Gaussian: 44")
    print(
        f"Scale range: [{float(scales.min())}, "
        f"{float(scales.max())}]"
    )
    print(
        f"Intensity range: [{float(intensity.min())}, "
        f"{float(intensity.max())}]"
    )
    print(f"Bytes: {actual_size}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export either dense voxel data or pretrained Gaussian "
            "parameters for adaptive_inside_camera_renderer."
        )
    )

    parser.add_argument(
        "representation",
        choices=("dense_voxel", "pretrained_gaussian"),
    )
    parser.add_argument("input")
    parser.add_argument("output")

    parser.add_argument(
        "--dataset",
        default="raw",
        help="HDF5 dataset for dense_voxel mode.",
    )
    parser.add_argument(
        "--normalise",
        choices=("none", "minmax"),
        default="none",
        help="Dense voxel normalisation.",
    )

    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help=(
            "Optional nested checkpoint root, e.g. state_dict or model."
        ),
    )
    parser.add_argument(
        "--means-key",
        default="means",
    )
    parser.add_argument(
        "--scales-key",
        default="scales",
    )
    parser.add_argument(
        "--quaternions-key",
        default="quaternions",
    )
    parser.add_argument(
        "--intensity-key",
        default="intensity",
    )
    parser.add_argument(
        "--scale-activation",
        choices=("none", "exp", "softplus"),
        default="none",
    )
    parser.add_argument(
        "--intensity-activation",
        choices=("none", "exp", "softplus", "sigmoid"),
        default="none",
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input does not exist: {input_path}"
        )

    if args.representation == "dense_voxel":
        export_dense_voxel(
            input_path=input_path,
            output_path=output_path,
            dataset=args.dataset,
            normalise=args.normalise,
        )
    else:
        export_pretrained_gaussian(
            input_path=input_path,
            output_path=output_path,
            checkpoint_root=args.checkpoint_root,
            means_key=args.means_key,
            scales_key=args.scales_key,
            quaternions_key=args.quaternions_key,
            intensity_key=args.intensity_key,
            scale_activation=args.scale_activation,
            intensity_activation=args.intensity_activation,
            quaternion_order=args.quaternion_order,
        )


if __name__ == "__main__":
    main()
