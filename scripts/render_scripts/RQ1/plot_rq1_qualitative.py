#!/usr/bin/env python3
"""Create the RQ1 qualitative reconstruction grid from timed PFM outputs.

Example
-------
python plot_rq1_qualitative.py \
  --input-dir results/rq1_multiblock \
  --output-dir results/rq1_multiblock/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATTERN = re.compile(
    r"render_(?P<block>b\d{3}_gaussians_\d+)_ratio_"
    r"(?P<ratio>\d{3})_repeat_(?P<repeat>\d+)\.pfm$"
)

GROUND_TRUTH_BY_GAUSSIAN = {
    "016": "block_z0_y1_x6.h5",
    "017": "block_z0_y1_x7.h5",
    "026": "block_z0_y2_x6.h5",
    "027": "block_z0_y2_x7.h5",
    "028": "block_z0_y2_x8.h5",
}


def read_pfm(path: Path) -> np.ndarray:
    """Read a grayscale or RGB PFM and restore its top-to-bottom orientation."""
    with path.open("rb") as file:
        header = file.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Unsupported PFM header in {path}: {header!r}")
        channels = 1 if header == "Pf" else 3

        dimensions = file.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = file.readline().decode("ascii").strip()
        width, height = map(int, dimensions.split())

        scale = float(file.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        values = np.fromfile(file, dtype=dtype)

    expected = width * height * channels
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")

    shape = (height, width) if channels == 1 else (height, width, channels)
    image = np.flipud(values.reshape(shape)).astype(np.float32, copy=False)
    if image.ndim != 2:
        raise ValueError(f"Expected a scalar reconstruction in {path}, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite reconstruction values found in {path}")
    return image


def discover_outputs(
    input_dir: Path, ratios: list[float], repeat_id: int
) -> tuple[list[str], dict[tuple[str, int], Path]]:
    """Find exactly one timed output for every block/ratio selection."""
    ratio_codes = {int(round(ratio * 100)) for ratio in ratios}
    selected: dict[tuple[str, int], Path] = {}

    for path in sorted(input_dir.glob("render_*_repeat_*.pfm")):
        match = OUTPUT_PATTERN.fullmatch(path.name)
        if match is None or int(match.group("repeat")) != repeat_id:
            continue
        ratio_code = int(match.group("ratio"))
        if ratio_code not in ratio_codes:
            continue
        key = (match.group("block"), ratio_code)
        if key in selected:
            raise ValueError(f"Duplicate output for {key}: {selected[key]} and {path}")
        selected[key] = path

    blocks = sorted({block for block, _ in selected})
    if not blocks:
        raise ValueError(f"No matching PFM outputs found in {input_dir}")

    missing = [
        (block, ratio_code)
        for block in blocks
        for ratio_code in sorted(ratio_codes)
        if (block, ratio_code) not in selected
    ]
    if missing:
        raise ValueError(f"Missing block/ratio outputs: {missing}")
    return blocks, selected


def block_label(block: str) -> str:
    """Convert b000_gaussians_016 to the concise report label gaussians 016."""
    return block.split("_", maxsplit=1)[1].replace("_", " ")


def render_gt_mip(
    path: Path,
    height: int,
    width: int,
    depth_samples: int,
    fov_y_degrees: float,
) -> np.ndarray:
    """Render normalized GT with the benchmark's center-camera ray geometry."""
    import h5py
    from scipy.ndimage import map_coordinates

    if depth_samples < 2:
        raise ValueError("depth_samples must be at least 2")
    with h5py.File(path, "r") as file:
        volume = file["raw"][:].astype(np.float32)
    low, high = float(volume.min()), float(volume.max())
    if high <= low:
        raise ValueError(f"Constant GT volume: {path}")
    volume = (volume - low) / (high - low)

    ndc_x = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
    ndc_y = 1.0 - 2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height
    grid_x, grid_y = np.meshgrid(ndc_x, ndc_y)
    tan_half_fov_y = np.tan(np.deg2rad(fov_y_degrees) / 2.0)
    aspect = width / height
    directions = np.stack(
        [
            grid_x * aspect * tan_half_fov_y,
            grid_y * tan_half_fov_y,
            np.ones_like(grid_x),
        ],
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    t_exit = np.min(
        1.0 / np.maximum(np.abs(directions), 1e-12), axis=-1
    )
    fractions = np.linspace(0.0, 1.0, depth_samples, dtype=np.float64)
    distances = (
        1e-4
        + (t_exit[None, :, :] - 1e-4) * fractions[:, None, None]
    )
    points = distances[:, :, :, None] * directions[None, :, :, :]

    depth, vol_height, vol_width = volume.shape
    coordinates = np.stack(
        [
            (points[:, :, :, 2] + 1.0) * 0.5 * (depth - 1),
            (points[:, :, :, 1] + 1.0) * 0.5 * (vol_height - 1),
            (points[:, :, :, 0] + 1.0) * 0.5 * (vol_width - 1),
        ]
    )
    samples = map_coordinates(
        volume,
        coordinates.reshape(3, -1),
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(depth_samples, height, width)
    image = samples.max(axis=0).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite GT render from {path}")
    return image

def plot_grid(
    blocks: list[str],
    ratios: list[float],
    paths: dict[tuple[str, int], Path],
    output_dir: Path,
    full_gaussians: int,
    gt_dir: Path,
    depth_samples: int,
    fov_y_degrees: float,
) -> None:
    ratio_codes = [int(round(ratio * 100)) for ratio in ratios]
    images = {
        (block, code): read_pfm(paths[(block, code)])
        for block in blocks
        for code in ratio_codes
    }
    height, width = next(iter(images.values())).shape

    ground_truth = {}
    for block in blocks:
        gaussian_suffix = block.rsplit("_", maxsplit=1)[-1]
        try:
            gt_name = GROUND_TRUTH_BY_GAUSSIAN[gaussian_suffix]
        except KeyError as error:
            raise ValueError(f"No GT mapping configured for {block}") from error
        ground_truth[block] = render_gt_mip(
            gt_dir / gt_name,
            height=height,
            width=width,
            depth_samples=depth_samples,
            fov_y_degrees=fov_y_degrees,
        )

    differences = {
        (block, code): np.abs(images[(block, code)] - ground_truth[block])
        for block in blocks
        for code in ratio_codes
    }

    # GT and reconstructions share one response scale; all absolute-difference
    # panels share a separate zero-based error scale.
    response_vmax = max(
        max(float(image.max()) for image in images.values()),
        max(float(image.max()) for image in ground_truth.values()),
    )
    difference_vmax = max(float(image.max()) for image in differences.values())
    if response_vmax <= 0 or difference_vmax <= 0:
        raise ValueError("Selected GT/reconstruction panels have invalid ranges")

    figure = plt.figure(figsize=(12.8, 8.2), layout="constrained")
    grid = figure.add_gridspec(
        len(blocks),
        2 * len(ratios) + 3,
        width_ratios=[1.0] * (2 * len(ratios) + 1) + [0.08, 0.08],
    )
    axes = [
        [figure.add_subplot(grid[row, col]) for col in range(2 * len(ratios) + 1)]
        for row in range(len(blocks))
    ]
    response_cax = figure.add_subplot(grid[:, -2])
    difference_cax = figure.add_subplot(grid[:, -1])

    response_artist = None
    difference_artist = None
    for row, block in enumerate(blocks):
        gt_axis = axes[row][0]
        response_artist = gt_axis.imshow(
            ground_truth[block], cmap="gray", vmin=0.0, vmax=response_vmax
        )
        gt_axis.set_ylabel(block_label(block), fontsize=10, labelpad=7)
        if row == 0:
            gt_axis.set_title("GT", fontsize=10, pad=5)

        for ratio_index, (ratio, code) in enumerate(zip(ratios, ratio_codes)):
            render_col = 1 + 2 * ratio_index
            difference_col = render_col + 1
            render_axis = axes[row][render_col]
            difference_axis = axes[row][difference_col]

            render_axis.imshow(
                images[(block, code)], cmap="gray", vmin=0.0, vmax=response_vmax
            )
            difference_artist = difference_axis.imshow(
                differences[(block, code)],
                cmap="gray",
                vmin=0.0,
                vmax=difference_vmax,
            )

            if row == 0:
                active = int(round(full_gaussians * ratio))
                render_axis.set_title(
                    f"{ratio:.0%} render\n({active:,})", fontsize=9, pad=5
                )
                difference_axis.set_title(
                    f"{ratio:.0%} |diff|", fontsize=9, pad=5
                )

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        "Ground truth, retained-payload reconstructions, and absolute differences",
        fontsize=13,
    )
    assert response_artist is not None and difference_artist is not None
    response_bar = figure.colorbar(response_artist, cax=response_cax)
    response_bar.set_label("GT / rendered response")
    difference_bar = figure.colorbar(difference_artist, cax=difference_cax)
    difference_bar.set_label("Absolute difference")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "rq1_qualitative_reconstructions"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot representative timed reconstructions for the RQ1 payload sweep."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.10, 0.50, 1.00]
    )
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--full-gaussians", type=int, default=50_000)
    parser.add_argument(
        "--gt-dir", type=Path, default=Path("data/smoke_data/blocks")
    )
    parser.add_argument("--depth-samples", type=int, default=64)
    parser.add_argument("--fov-y", type=float, default=90.0)
    args = parser.parse_args()

    if not args.ratios or any(ratio <= 0 or ratio > 1 for ratio in args.ratios):
        parser.error("--ratios must contain values in the interval (0, 1]")
    ratios = sorted(set(args.ratios))
    blocks, outputs = discover_outputs(args.input_dir, ratios, args.repeat_id)
    plot_grid(
        blocks,
        ratios,
        outputs,
        args.output_dir,
        args.full_gaussians,
        args.gt_dir,
        args.depth_samples,
        args.fov_y,
    )

    print(f"Blocks: {len(blocks)}")
    print(f"Retention ratios: {', '.join(f'{ratio:.0%}' for ratio in ratios)}")
    print(f"Output: {args.output_dir / 'rq1_qualitative_reconstructions.pdf'}")


if __name__ == "__main__":
    main()
