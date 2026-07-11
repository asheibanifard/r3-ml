#!/usr/bin/env python3
"""
Export a PyTorch Gaussian checkpoint into the compact binary format consumed by
gaussian_mip_realtime.cu.

This is a one-time conversion utility. The CUDA renderer itself has no Python
or PyTorch dependency.

Examples:
    python export_gaussians.py \
        ../models_smoke/block_z000_y001_x006/best.pth \
        gaussians.bin

The exporter recognizes common field names:
    means
    log_s or log_scales
    quats
    inten or intensities

Intensity handling:
    --intensity-mode softplus  Use for raw unconstrained intensity parameters.
    --intensity-mode direct    Use when the checkpoint already stores positive
                               activated intensity values.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MAGIC = 0x47534D50
VERSION = 1


def find_tensor(checkpoint: dict, names: tuple[str, ...]) -> torch.Tensor:
    for name in names:
        if name in checkpoint:
            value = checkpoint[name]
            if isinstance(value, torch.Tensor):
                return value
    raise KeyError(f"None of these tensor fields were found: {names}")


def unwrap_checkpoint(obj):
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint root must be a dictionary.")

    for key in ("state_dict", "model", "gaussians", "params"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            keys = set(nested)
            if keys & {"means", "log_s", "log_scales", "quats"}:
                return nested
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--intensity-mode",
        choices=("softplus", "direct"),
        default="softplus",
    )
    parser.add_argument(
        "--quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
    )
    args = parser.parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint = unwrap_checkpoint(raw)

    means = find_tensor(checkpoint, ("means",)).float()
    log_scales = find_tensor(checkpoint, ("log_s", "log_scales")).float()
    quats = find_tensor(checkpoint, ("quats", "quaternions")).float()
    intensity_parameter = find_tensor(
        checkpoint,
        ("inten", "intensities", "intensity"),
    ).float().reshape(-1)

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must be [P,3], received {tuple(means.shape)}")
    if log_scales.shape != means.shape:
        raise ValueError(
            f"log scales must match means, received {tuple(log_scales.shape)}"
        )
    if quats.shape != (means.shape[0], 4):
        raise ValueError(f"quats must be [P,4], received {tuple(quats.shape)}")
    if intensity_parameter.shape != (means.shape[0],):
        raise ValueError(
            f"intensity must be [P], received {tuple(intensity_parameter.shape)}"
        )

    scales = torch.exp(log_scales)

    if args.intensity_mode == "softplus":
        intensities = F.softplus(intensity_parameter)
    else:
        intensities = intensity_parameter

    if args.quat_order == "xyzw":
        quats = quats[:, [3, 0, 1, 2]]

    quats = F.normalize(quats, dim=-1, eps=1e-8)
    intensities = intensities.clamp_min(0.0)

    tensors = (means, scales, quats, intensities)
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("Checkpoint contains NaN or infinite values.")

    P = means.shape[0]

    packed = np.concatenate(
        (
            means.numpy(),
            scales.numpy(),
            quats.numpy(),
            intensities[:, None].numpy(),
        ),
        axis=1,
    ).astype("<f4", copy=False)

    if packed.shape != (P, 11):
        raise RuntimeError(f"Unexpected packed shape: {packed.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("wb") as file:
        file.write(struct.pack("<IIQ", MAGIC, VERSION, P))
        file.write(packed.tobytes(order="C"))

    print(f"Exported {P} Gaussians")
    print(f"Means range: {means.amin(dim=0).tolist()} to {means.amax(dim=0).tolist()}")
    print(f"Scale range: {float(scales.min())} to {float(scales.max())}")
    print(
        f"Intensity range: {float(intensities.min())} "
        f"to {float(intensities.max())}"
    )
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
