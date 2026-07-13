#!/usr/bin/env python3
"""
Reconstruct a dense volume directly from a PyTorch best.pth checkpoint.

The CUDA executable still consumes a compact binary internally because native
CUDA/C++ cannot safely parse a Python pickle-based .pth file without linking
against LibTorch. This wrapper loads best.pth with PyTorch, converts the
Gaussian tensors to the binary layout expected by gaussian_volume_reconstruct,
runs the CUDA program, and removes the temporary binary automatically.

Expected checkpoint tensors (common aliases are supported):
    means / xyz / positions
    scales / scale
    rotations / quaternions / quats
    intensities / intensity / opacity / densities

Usage:
    python reconstruct_from_pth.py \
        best.pth \
        reconstructed_volume.raw \
        --nx 128 --ny 128 --nz 128 \
        --bounds -1 -1 -1 1 1 1 \
        --intensity-mode softplus \
        --quat-order wxyz \
        --cuda-exe ./gaussian_volume_reconstruct
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


MAGIC = 0x47534D50
VERSION = 1
HEADER = struct.Struct("<IIQ")


def find_tensor(obj: Any, aliases: tuple[str, ...]) -> torch.Tensor | None:
    if isinstance(obj, dict):
        for key in aliases:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key]
        for value in obj.values():
            found = find_tensor(value, aliases)
            if found is not None:
                return found
    return None


def require_tensor(
    checkpoint: Any,
    aliases: tuple[str, ...],
    label: str,
) -> torch.Tensor:
    tensor = find_tensor(checkpoint, aliases)
    if tensor is None:
        raise KeyError(
            f"Could not find {label}. Tried aliases: {', '.join(aliases)}"
        )
    return tensor.detach().cpu().float()


def normalize_shape(tensor: torch.Tensor, cols: int, name: str) -> torch.Tensor:
    if tensor.ndim == 1 and cols == 1:
        return tensor[:, None]
    if tensor.ndim == 2 and tensor.shape[1] == cols:
        return tensor
    if tensor.ndim == 2 and tensor.shape[0] == cols:
        return tensor.T.contiguous()
    raise ValueError(
        f"{name} must have shape [N,{cols}] or [{cols},N], got {tuple(tensor.shape)}"
    )


def activate_scales(scales: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "direct":
        return scales.abs()
    if mode == "exp":
        return torch.exp(scales)
    if mode == "softplus":
        return F.softplus(scales)
    raise ValueError(mode)


def activate_intensities(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "direct":
        return values
    if mode == "softplus":
        return F.softplus(values)
    if mode == "sigmoid":
        return torch.sigmoid(values)
    if mode == "exp":
        return torch.exp(values)
    raise ValueError(mode)


def reorder_quaternions(quats: torch.Tensor, order: str) -> torch.Tensor:
    if order == "wxyz":
        return quats
    if order == "xyzw":
        return quats[:, [3, 0, 1, 2]]
    raise ValueError(order)


def export_temp_binary(
    checkpoint_path: Path,
    output_path: Path,
    intensity_mode: str,
    scale_mode: str,
    quat_order: str,
) -> tuple[int, dict[str, tuple[float, float]]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    means = normalize_shape(
        require_tensor(
            checkpoint,
            ("means", "xyz", "positions", "centers", "mu"),
            "Gaussian means",
        ),
        3,
        "means",
    )

    scales = normalize_shape(
        require_tensor(
            checkpoint,
            ("log_scales", "log_s", "scales", "scale", "scaling", "stds", "sigmas"),
            "Gaussian scales",
        ),
        3,
        "scales",
    )

    quats = normalize_shape(
        require_tensor(
            checkpoint,
            ("rotations", "rotation", "quaternions", "quats", "quat"),
            "Gaussian quaternions",
        ),
        4,
        "quaternions",
    )

    intensities = normalize_shape(
        require_tensor(
            checkpoint,
            (
                "intensities",
                "intensity",
                "opacity",
                "opacities",
                "density",
                "densities",
                "amplitudes",
                "features",
            ),
            "Gaussian intensities",
        ),
        1,
        "intensities",
    )

    count = means.shape[0]
    for name, tensor in (
        ("scales", scales),
        ("quaternions", quats),
        ("intensities", intensities),
    ):
        if tensor.shape[0] != count:
            raise ValueError(
                f"{name} count {tensor.shape[0]} does not match means count {count}"
            )

    scales = activate_scales(scales, scale_mode)
    intensities = activate_intensities(intensities, intensity_mode)
    quats = reorder_quaternions(quats, quat_order)
    quats = F.normalize(quats, dim=1, eps=1e-12)

    records = torch.cat(
        [means, scales, quats, intensities],
        dim=1,
    ).contiguous().numpy().astype("<f4", copy=False)

    finite = np.isfinite(records).all(axis=1)
    if not finite.all():
        dropped = int((~finite).sum())
        print(f"Warning: dropping {dropped} non-finite Gaussian records.")
        records = records[finite]

    with output_path.open("wb") as file:
        file.write(HEADER.pack(MAGIC, VERSION, int(records.shape[0])))
        file.write(records.tobytes(order="C"))

    stats = {
        "means": (float(records[:, 0:3].min()), float(records[:, 0:3].max())),
        "scales": (float(records[:, 3:6].min()), float(records[:, 3:6].max())),
        "intensities": (float(records[:, 10].min()), float(records[:, 10].max())),
    }
    return int(records.shape[0]), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)

    parser.add_argument("--cuda-exe", type=Path, required=True)

    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz", type=int, default=50)

    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        default=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )

    parser.add_argument("--cutoff", type=float, default=20.0)

    parser.add_argument(
        "--intensity-mode",
        choices=("direct", "softplus", "sigmoid", "exp"),
        default="softplus",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("direct", "exp", "softplus"),
        default="exp",
    )
    parser.add_argument(
        "--quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
    )

    parser.add_argument(
        "--keep-temp-bin",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.cuda_exe.is_file():
        raise FileNotFoundError(args.cuda_exe)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_temp_bin:
        temp_bin = args.output.with_suffix(args.output.suffix + ".gaussians.bin")
        count, stats = export_temp_binary(
            args.checkpoint,
            temp_bin,
            args.intensity_mode,
            args.scale_mode,
            args.quat_order,
        )

        print(f"Exported Gaussians: {count}")
        print(f"Means range: {stats['means']}")
        print(f"Scales range: {stats['scales']}")
        print(f"Intensity range: {stats['intensities']}")

        subprocess.run(
            [
                str(args.cuda_exe.resolve()),
                str(temp_bin),
                str(args.output),
                str(args.nx),
                str(args.ny),
                str(args.nz),
                *(str(v) for v in args.bounds),
                str(args.cutoff),
            ],
            check=True,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="gaussian_reconstruct_") as tmp:
            temp_bin = Path(tmp) / "gaussians.bin"

            count, stats = export_temp_binary(
                args.checkpoint,
                temp_bin,
                args.intensity_mode,
                args.scale_mode,
                args.quat_order,
            )

            print(f"Exported Gaussians: {count}")
            print(f"Means range: {stats['means']}")
            print(f"Scales range: {stats['scales']}")
            print(f"Intensity range: {stats['intensities']}")

            subprocess.run(
                [
                    str(args.cuda_exe.resolve()),
                    str(temp_bin),
                    str(args.output),
                    str(args.nx),
                    str(args.ny),
                    str(args.nz),
                    *(str(v) for v in args.bounds),
                    str(args.cutoff),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()