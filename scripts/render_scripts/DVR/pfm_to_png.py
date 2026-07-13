#!/usr/bin/env python3
"""
Convert one or more grayscale PFM renderings to PNG.

Examples
--------
python pfm_to_png.py render.pfm render.png

python pfm_to_png.py \
    --input-dir results/dvr \
    --output-dir results/dvr/png
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
            raise ValueError(f"{path}: unsupported PFM header {header!r}")

        dimensions = file.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = file.readline().decode("ascii").strip()

        width, height = map(int, dimensions.split())
        scale = float(file.readline().decode("ascii").strip())

        endian = "<" if scale < 0 else ">"
        channels = 1 if header == "Pf" else 3

        data = np.fromfile(file, dtype=endian + "f4")
        expected = width * height * channels

        if data.size != expected:
            raise ValueError(
                f"{path}: expected {expected} floats, found {data.size}"
            )

        if channels == 1:
            image = data.reshape(height, width)
        else:
            image = data.reshape(height, width, channels)

        # PFM rows are stored bottom-to-top.
        return np.flipud(image)


def save_png(
    pfm_path: Path,
    png_path: Path,
    *,
    vmin: float | None,
    vmax: float | None,
    percentile_low: float,
    percentile_high: float,
) -> None:
    image = read_pfm(pfm_path)

    if image.ndim == 3:
        image = image[..., 0]

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError(f"{pfm_path}: image contains no finite values")

    local_vmin = (
        float(vmin)
        if vmin is not None
        else float(np.percentile(finite, percentile_low))
    )
    local_vmax = (
        float(vmax)
        if vmax is not None
        else float(np.percentile(finite, percentile_high))
    )

    if local_vmax <= local_vmin:
        local_vmax = local_vmin + 1.0e-8

    png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(
        png_path,
        image,
        cmap="gray",
        vmin=local_vmin,
        vmax=local_vmax,
    )

    print(
        f"Saved {png_path} "
        f"with display range [{local_vmin:.6g}, {local_vmax:.6g}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--percentile-low", type=float, default=0.0)
    parser.add_argument("--percentile-high", type=float, default=100.0)
    args = parser.parse_args()

    if args.input:
        if not args.output:
            raise ValueError("Provide an output PNG path.")
        save_png(
            Path(args.input),
            Path(args.output),
            vmin=args.vmin,
            vmax=args.vmax,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )
        return

    if not args.input_dir or not args.output_dir:
        raise ValueError(
            "Use either INPUT OUTPUT or --input-dir with --output-dir."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pfm_files = sorted(input_dir.glob("*.pfm"))
    if not pfm_files:
        raise FileNotFoundError(f"No PFM files found in {input_dir}")

    for pfm_path in pfm_files:
        save_png(
            pfm_path,
            output_dir / f"{pfm_path.stem}.png",
            vmin=args.vmin,
            vmax=args.vmax,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )


if __name__ == "__main__":
    main()
