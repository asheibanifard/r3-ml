#!/usr/bin/env python3
"""
Plot results from the inside-camera CUDA Gaussian MIP renderer.

Usage:
    python plot_inside_camera_results.py inside_view.pfm

Optional:
    python plot_inside_camera_results.py inside_view.pfm \
        --output-dir plots \
        --clip-min 0.0 \
        --clip-max 1.0

Outputs:
    mip_raw.png
    mip_normalized.png
    mip_histogram.png
    mip_profiles.png
    mip_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = file.readline().decode("ascii").strip()

        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header: {header!r}")

        channels = 1 if header == "Pf" else 3

        dimensions = file.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = file.readline().decode("ascii").strip()

        width, height = map(int, dimensions.split())

        scale = float(file.readline().decode("ascii").strip())
        little_endian = scale < 0

        dtype = "<f4" if little_endian else ">f4"
        data = np.fromfile(file, dtype=dtype)

    expected = width * height * channels
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} float values, found {data.size}."
        )

    if channels == 1:
        image = data.reshape(height, width)
    else:
        image = data.reshape(height, width, channels)

    # PFM stores scanlines from bottom to top.
    return np.flipud(image).astype(np.float32, copy=False)


def robust_normalize(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    low = float(np.percentile(image, low_percentile))
    high = float(np.percentile(image, high_percentile))

    if high <= low:
        high = low + 1e-6

    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return normalized, low, high


def save_raw_image(
    image: np.ndarray,
    output_path: Path,
    clip_min: float | None,
    clip_max: float | None,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(
        image,
        cmap="gray",
        vmin=clip_min,
        vmax=clip_max,
    )
    plt.title("Inside-camera Gaussian MIP")
    plt.xlabel("Pixel x")
    plt.ylabel("Pixel y")
    plt.colorbar(label="Accumulated Gaussian density")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_normalized_image(
    normalized: np.ndarray,
    low: float,
    high: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(normalized, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(
        f"Contrast-normalized MIP\n"
        f"1st–99th percentile: {low:.4f}–{high:.4f}"
    )
    plt.xlabel("Pixel x")
    plt.ylabel("Pixel y")
    plt.colorbar(label="Normalized intensity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_histogram(image: np.ndarray, output_path: Path) -> None:
    values = image[np.isfinite(image)].ravel()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=100)
    plt.axvline(float(values.mean()), linestyle="--", label="Mean")
    plt.axvline(float(np.median(values)), linestyle=":", label="Median")
    plt.title("MIP intensity distribution")
    plt.xlabel("Accumulated Gaussian density")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_profiles(image: np.ndarray, output_path: Path) -> None:
    height, width = image.shape
    middle_y = height // 2
    middle_x = width // 2

    horizontal = image[middle_y, :]
    vertical = image[:, middle_x]

    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(width), horizontal, label=f"Horizontal y={middle_y}")
    plt.plot(np.arange(height), vertical, label=f"Vertical x={middle_x}")
    plt.title("Central MIP intensity profiles")
    plt.xlabel("Pixel coordinate")
    plt.ylabel("Accumulated Gaussian density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary(
    image: np.ndarray,
    normalized: np.ndarray,
    output_path: Path,
) -> None:
    height, width = image.shape
    middle_y = height // 2
    middle_x = width // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    raw_view = axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Raw MIP")
    axes[0, 0].set_xlabel("Pixel x")
    axes[0, 0].set_ylabel("Pixel y")
    fig.colorbar(raw_view, ax=axes[0, 0], fraction=0.046)

    normalized_view = axes[0, 1].imshow(
        normalized,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("Robust normalized MIP")
    axes[0, 1].set_xlabel("Pixel x")
    axes[0, 1].set_ylabel("Pixel y")
    fig.colorbar(normalized_view, ax=axes[0, 1], fraction=0.046)

    values = image[np.isfinite(image)].ravel()
    axes[1, 0].hist(values, bins=100)
    axes[1, 0].set_title("Intensity histogram")
    axes[1, 0].set_xlabel("Density")
    axes[1, 0].set_ylabel("Pixel count")

    axes[1, 1].plot(
        np.arange(width),
        image[middle_y, :],
        label=f"Horizontal y={middle_y}",
    )
    axes[1, 1].plot(
        np.arange(height),
        image[:, middle_x],
        label=f"Vertical x={middle_x}",
    )
    axes[1, 1].set_title("Central profiles")
    axes[1, 1].set_xlabel("Pixel coordinate")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    fig.suptitle(
        "Inside-camera Gaussian MIP summary\n"
        f"shape={image.shape}, "
        f"min={float(np.nanmin(image)):.6f}, "
        f"max={float(np.nanmax(image)):.6f}, "
        f"mean={float(np.nanmean(image)):.6f}",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pfm", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
    )
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--clip-max", type=float, default=None)
    args = parser.parse_args()

    image = read_pfm(args.pfm)

    if image.ndim == 3:
        image = image.mean(axis=-1)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, received {image.shape}")

    if not np.isfinite(image).all():
        print("Warning: replacing NaN/inf pixels with zero.")
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalized, low, high = robust_normalize(image)

    save_raw_image(
        image,
        args.output_dir / "mip_raw.png",
        args.clip_min,
        args.clip_max,
    )
    save_normalized_image(
        normalized,
        low,
        high,
        args.output_dir / "mip_normalized.png",
    )
    save_histogram(
        image,
        args.output_dir / "mip_histogram.png",
    )
    save_profiles(
        image,
        args.output_dir / "mip_profiles.png",
    )
    save_summary(
        image,
        normalized,
        args.output_dir / "mip_summary.png",
    )

    print("Image shape:", image.shape)
    print("Minimum:", float(image.min()))
    print("Maximum:", float(image.max()))
    print("Mean:", float(image.mean()))
    print("Median:", float(np.median(image)))
    print("Standard deviation:", float(image.std()))
    print("1st percentile:", low)
    print("99th percentile:", high)
    print("Saved plots to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
