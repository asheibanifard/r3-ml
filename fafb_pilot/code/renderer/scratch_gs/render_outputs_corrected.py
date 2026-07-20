#!/usr/bin/env python3
"""Evaluate and visualise frames produced by gaussian_splat_scratch_corrected.cu.

The CUDA program writes three paired image sequences:

    gt_XXXX.bin      Ground-truth voxel grid rendered by DVR
    baked_XXXX.bin   Gaussian mixture baked to a grid, then rendered by DVR
    rec_XXXX.bin     Live Gaussian rasterizer output

Each binary frame contains:
    int32 width
    int32 height
    width * height float32 pixels, row-major, expected in [0, 1]

This script:
  1. validates and pairs frames by numeric frame index;
  2. computes PSNR, SSIM and LPIPS against GT for both reconstructions;
  3. reads stable GPU-only FPS values from fps_summary.txt;
  4. writes videos, comparison figures, per-frame curves and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

FRAME_PATTERN = re.compile(r"^(gt|baked|rec)_(\d+)\.bin$")


@dataclass(frozen=True)
class MetricSeries:
    """Per-frame quality measurements for one reconstruction method."""

    psnr_db: np.ndarray
    ssim: np.ndarray
    lpips: np.ndarray

    @property
    def mean_psnr(self) -> float:
        finite = self.psnr_db[np.isfinite(self.psnr_db)]
        return float(finite.mean()) if finite.size else float("inf")

    @property
    def mean_ssim(self) -> float:
        return float(self.ssim.mean())

    @property
    def mean_lpips(self) -> float:
        return float(self.lpips.mean())


# -----------------------------------------------------------------------------
# Binary frame I/O and validation
# -----------------------------------------------------------------------------
def read_frame(path: Path) -> np.ndarray:
    """Read one binary frame and fail clearly on malformed data."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: file is too small to contain width and height")

    width, height = struct.unpack_from("<ii", raw, offset=0)
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: invalid dimensions {width} x {height}")

    expected_size = 8 + width * height * 4
    if len(raw) != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} bytes for a {width} x {height} "
            f"float32 frame, found {len(raw)}"
        )

    frame = np.frombuffer(raw, dtype="<f4", offset=8).reshape(height, width).copy()
    if not np.isfinite(frame).all():
        raise ValueError(f"{path}: frame contains NaN or infinity")

    minimum = float(frame.min())
    maximum = float(frame.max())
    if minimum < -1e-5 or maximum > 1.0 + 1e-5:
        raise ValueError(
            f"{path}: expected values in [0,1], observed [{minimum:.6g}, {maximum:.6g}]"
        )

    # Tiny floating-point excursions are harmless for display and perceptual metrics.
    return np.clip(frame, 0.0, 1.0)


def discover_frame_paths(frames_dir: Path) -> dict[str, dict[int, Path]]:
    """Return paths grouped by sequence name and numeric frame index."""
    groups: dict[str, dict[int, Path]] = {"gt": {}, "baked": {}, "rec": {}}

    for path in frames_dir.glob("*.bin"):
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        sequence, index_text = match.groups()
        index = int(index_text)
        if index in groups[sequence]:
            raise ValueError(f"Duplicate {sequence} frame index {index} in {frames_dir}")
        groups[sequence][index] = path

    if not groups["gt"]:
        raise FileNotFoundError(f"No gt_XXXX.bin frames found in {frames_dir}")

    gt_indices = set(groups["gt"])
    for sequence in ("baked", "rec"):
        indices = set(groups[sequence])
        if indices != gt_indices:
            missing = sorted(gt_indices - indices)
            extra = sorted(indices - gt_indices)
            raise ValueError(
                f"{sequence} frame indices do not match GT. "
                f"Missing={missing}; extra={extra}"
            )

    return groups


def load_paired_frames(
    paths: dict[str, dict[int, Path]],
) -> tuple[list[int], dict[str, list[np.ndarray]]]:
    """Load all sequences in the same verified numeric order."""
    indices = sorted(paths["gt"])
    frames = {
        name: [read_frame(paths[name][index]) for index in indices]
        for name in ("gt", "baked", "rec")
    }

    reference_shape = frames["gt"][0].shape
    for name, sequence in frames.items():
        for index, frame in zip(indices, sequence):
            if frame.shape != reference_shape:
                raise ValueError(
                    f"{name}_{index:04d} has shape {frame.shape}; "
                    f"expected {reference_shape}"
                )

    return indices, frames


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def calculate_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """PSNR for images whose physical data range is exactly one.

    MSE = mean((reference - test)^2)
    PSNR = 10 log10(MAX^2 / MSE), with MAX=1 here.
    """
    difference = reference.astype(np.float64) - test.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def calculate_metrics(
    references: list[np.ndarray],
    tests: list[np.ndarray],
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> MetricSeries:
    """Compute PSNR, local-window SSIM and LPIPS for every paired frame."""
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for reference, test in zip(references, tests, strict=True):
        psnr_values.append(calculate_psnr(reference, test))
        ssim_values.append(
            float(structural_similarity(reference, test, data_range=1.0))
        )

        # LPIPS was trained on three-channel images in [-1,1]. Repeating a
        # grayscale channel does not invent colour; it only satisfies the network
        # interface while preserving the same luminance image in every channel.
        reference_tensor = (
            torch.from_numpy(reference)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .mul(2.0)
            .sub(1.0)
        )
        test_tensor = (
            torch.from_numpy(test)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .mul(2.0)
            .sub(1.0)
        )

        with torch.inference_mode():
            value = lpips_model(reference_tensor, test_tensor)
        lpips_values.append(float(value.item()))

    return MetricSeries(
        psnr_db=np.asarray(psnr_values, dtype=np.float64),
        ssim=np.asarray(ssim_values, dtype=np.float64),
        lpips=np.asarray(lpips_values, dtype=np.float64),
    )


# -----------------------------------------------------------------------------
# FPS metadata
# -----------------------------------------------------------------------------
def read_summary_file(path: Path) -> dict[str, str]:
    """Read key/value metadata while accepting numeric and textual values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corrected CUDA program before this script."
        )

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: malformed line {raw_line!r}")
        key, value = parts
        values[key] = value
    return values


def required_float(values: dict[str, str], key: str) -> float:
    if key not in values:
        raise KeyError(f"fps_summary.txt does not contain required key {key!r}")
    try:
        return float(values[key])
    except ValueError as error:
        raise ValueError(f"Summary value {key}={values[key]!r} is not numeric") from error


# -----------------------------------------------------------------------------
# Video and plots
# -----------------------------------------------------------------------------
def write_grayscale_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> None:
    """Encode float images in [0,1] as an H.264 grayscale MP4."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Cannot encode an empty frame sequence")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    height, width = frame_list[0].shape
    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "gray8",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None

    try:
        for frame_number, frame in enumerate(frame_list):
            if frame.shape != (height, width):
                raise ValueError(
                    f"Video frame {frame_number} has shape {frame.shape}; "
                    f"expected {(height, width)}"
                )
            image_u8 = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            process.stdin.write(image_u8.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        process.kill()
        raise

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {path}:\n{stderr}")
    print(f"Wrote {path}")


def scaled_difference_frames(
    references: list[np.ndarray], tests: list[np.ndarray], percentile: float = 99.5
) -> tuple[list[np.ndarray], float]:
    """Scale absolute error for visibility using one global robust scale."""
    differences = [np.abs(a - b) for a, b in zip(references, tests, strict=True)]
    all_values = np.concatenate([difference.ravel() for difference in differences])
    scale = max(float(np.percentile(all_values, percentile)), 1e-8)
    return [np.clip(difference / scale, 0.0, 1.0) for difference in differences], scale


def save_comparison_plots(
    out_dir: Path,
    indices: list[int],
    frames: dict[str, list[np.ndarray]],
    baked_metrics: MetricSeries,
    rec_metrics: MetricSeries,
) -> None:
    plot_dir = out_dir / "frame_comparisons"
    plot_dir.mkdir(parents=True, exist_ok=True)

    count = len(indices)
    positions = sorted({0, count // 4, count // 2, 3 * count // 4, count - 1})

    for position in positions:
        gt = frames["gt"][position]
        baked = frames["baked"][position]
        rec = frames["rec"][position]
        baked_diff = np.abs(gt - baked)
        rec_diff = np.abs(gt - rec)
        difference_limit = max(float(baked_diff.max()), float(rec_diff.max()), 1e-8)

        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        panels = [
            (axes[0, 0], gt, "GT DVR", "gray", 0.0, 1.0),
            (axes[0, 1], baked, "Baked + DVR", "gray", 0.0, 1.0),
            (axes[0, 2], baked_diff,
             f"|GT-Baked|\nPSNR {baked_metrics.psnr_db[position]:.2f} dB, "
             f"SSIM {baked_metrics.ssim[position]:.4f}",
             "hot", 0.0, difference_limit),
            (axes[1, 0], gt, "GT DVR", "gray", 0.0, 1.0),
            (axes[1, 1], rec, "Live Gaussian rasterizer", "gray", 0.0, 1.0),
            (axes[1, 2], rec_diff,
             f"|GT-Raster|\nPSNR {rec_metrics.psnr_db[position]:.2f} dB, "
             f"SSIM {rec_metrics.ssim[position]:.4f}",
             "hot", 0.0, difference_limit),
        ]

        for axis, image, title, cmap, vmin, vmax in panels:
            handle = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel("pixel x")
            axis.set_ylabel("pixel y")
            figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

        figure.suptitle(f"Frame {indices[position]}")
        figure.tight_layout()
        path = plot_dir / f"frame_{indices[position]:04d}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"Wrote {path}")


def save_metric_curves(
    out_dir: Path,
    indices: list[int],
    baked: MetricSeries,
    rec: MetricSeries,
) -> None:
    metric_specs: list[tuple[str, str, Callable[[MetricSeries], np.ndarray]]] = [
        ("PSNR per frame", "PSNR (dB)", lambda value: value.psnr_db),
        ("SSIM per frame", "SSIM", lambda value: value.ssim),
        ("LPIPS per frame", "LPIPS (lower is better)", lambda value: value.lpips),
    ]

    for filename, (title, ylabel, extractor) in zip(
        ("psnr_over_frames.png", "ssim_over_frames.png", "lpips_over_frames.png"),
        metric_specs,
        strict=True,
    ):
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(indices, extractor(baked), label="Baked + DVR")
        axis.plot(indices, extractor(rec), label="Gaussian rasterizer")
        axis.set_title(title)
        axis.set_xlabel("frame index")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = out_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"Wrote {path}")


def save_summary_csv(
    path: Path,
    fps: dict[str, float],
    baked_metrics: MetricSeries,
    rec_metrics: MetricSeries,
) -> None:
    rows = [
        {
            "representation": "GT DVR",
            "gpu_fps": fps["gt"],
            "psnr_db": "",
            "ssim": "",
            "lpips": "",
        },
        {
            "representation": "Baked + DVR",
            "gpu_fps": fps["baked"],
            "psnr_db": baked_metrics.mean_psnr,
            "ssim": baked_metrics.mean_ssim,
            "lpips": baked_metrics.mean_lpips,
        },
        {
            "representation": "Live Gaussian rasterizer",
            "gpu_fps": fps["rec"],
            "psnr_db": rec_metrics.mean_psnr,
            "ssim": rec_metrics.mean_ssim,
            "lpips": rec_metrics.mean_lpips,
        },
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["representation", "gpu_fps", "psnr_db", "ssim", "lpips"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def save_per_frame_csv(
    path: Path,
    indices: list[int],
    baked: MetricSeries,
    rec: MetricSeries,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "baked_psnr_db", "baked_ssim", "baked_lpips",
                "raster_psnr_db", "raster_ssim", "raster_lpips",
            ]
        )
        for position, index in enumerate(indices):
            writer.writerow(
                [
                    index,
                    baked.psnr_db[position], baked.ssim[position], baked.lpips[position],
                    rec.psnr_db[position], rec.ssim[position], rec.lpips[position],
                ]
            )
    print(f"Wrote {path}")


def save_summary_table_png(
    path: Path,
    fps: dict[str, float],
    baked: MetricSeries,
    rec: MetricSeries,
) -> None:
    rows = [
        ["GT DVR", f"{fps['gt']:.2f}", "-", "-", "-"],
        ["Baked + DVR", f"{fps['baked']:.2f}", f"{baked.mean_psnr:.2f}",
         f"{baked.mean_ssim:.4f}", f"{baked.mean_lpips:.4f}"],
        ["Live Gaussian rasterizer", f"{fps['rec']:.2f}", f"{rec.mean_psnr:.2f}",
         f"{rec.mean_ssim:.4f}", f"{rec.mean_lpips:.4f}"],
    ]

    figure, axis = plt.subplots(figsize=(11, 2.6))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Representation", "GPU FPS", "PSNR (dB)", "SSIM", "LPIPS"],
        loc="center",
        cellLoc="center",
        colWidths=[0.42, 0.145, 0.145, 0.145, 0.145],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)
    for row in range(1, len(rows) + 1):
        table[(row, 0)].set_text_props(ha="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


# -----------------------------------------------------------------------------
# Program entry point
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", type=Path, default=script_dir / "frames")
    parser.add_argument("--out_dir", type=Path, default=script_dir / "results")
    parser.add_argument("--video_fps", type=int, default=24)
    parser.add_argument(
        "--lpips_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used only for LPIPS evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    frames_dir = args.frames_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video_fps <= 0:
        raise ValueError("--video_fps must be positive")

    print("Step 1/6: discovering and validating paired frame files")
    paths = discover_frame_paths(frames_dir)
    indices, frames = load_paired_frames(paths)
    print(f"  Loaded {len(indices)} paired frames with shape {frames['gt'][0].shape}")

    print("Step 2/6: reading stable GPU benchmark metadata")
    summary = read_summary_file(frames_dir / "fps_summary.txt")
    timing_kind = summary.get("timing_kind", "unknown")
    if timing_kind != "gpu_events_repeated":
        print(f"  WARNING: timing_kind={timing_kind!r}; expected 'gpu_events_repeated'")
    fps = {
        "gt": required_float(summary, "dvr_gpu_fps"),
        "baked": required_float(summary, "baked_gpu_fps"),
        "rec": required_float(summary, "rasterizer_gpu_fps"),
    }

    if args.lpips_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--lpips_device cuda was requested, but CUDA is unavailable")
        device = torch.device("cuda")
    elif args.lpips_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Step 3/6: computing metrics (LPIPS device: {device})")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    baked_metrics = calculate_metrics(frames["gt"], frames["baked"], lpips_model, device)
    rec_metrics = calculate_metrics(frames["gt"], frames["rec"], lpips_model, device)

    print("Step 4/6: encoding videos")
    write_grayscale_video(frames["gt"], out_dir / "gt.mp4", args.video_fps)
    write_grayscale_video(frames["baked"], out_dir / "baked.mp4", args.video_fps)
    write_grayscale_video(frames["rec"], out_dir / "reconstruction.mp4", args.video_fps)

    baked_diff, baked_scale = scaled_difference_frames(frames["gt"], frames["baked"])
    rec_diff, rec_scale = scaled_difference_frames(frames["gt"], frames["rec"])
    write_grayscale_video(baked_diff, out_dir / "baked_difference_scaled.mp4", args.video_fps)
    write_grayscale_video(rec_diff, out_dir / "raster_difference_scaled.mp4", args.video_fps)

    print("Step 5/6: writing plots and CSV files")
    save_comparison_plots(out_dir, indices, frames, baked_metrics, rec_metrics)
    save_metric_curves(out_dir, indices, baked_metrics, rec_metrics)
    save_summary_csv(out_dir / "metrics_summary.csv", fps, baked_metrics, rec_metrics)
    save_per_frame_csv(out_dir / "metrics_per_frame.csv", indices, baked_metrics, rec_metrics)
    save_summary_table_png(out_dir / "metrics_summary.png", fps, baked_metrics, rec_metrics)

    print("Step 6/6: summary")
    print(f"  Difference-video scale, Baked+DVR: {baked_scale:.6g} (99.5th percentile)")
    print(f"  Difference-video scale, Raster:    {rec_scale:.6g} (99.5th percentile)")
    print(f"  GT DVR GPU FPS:                     {fps['gt']:.2f}")
    print(
        f"  Baked + DVR:                        {fps['baked']:.2f} FPS, "
        f"PSNR={baked_metrics.mean_psnr:.2f} dB, "
        f"SSIM={baked_metrics.mean_ssim:.4f}, "
        f"LPIPS={baked_metrics.mean_lpips:.4f}"
    )
    print(
        f"  Live Gaussian rasterizer:           {fps['rec']:.2f} FPS, "
        f"PSNR={rec_metrics.mean_psnr:.2f} dB, "
        f"SSIM={rec_metrics.mean_ssim:.4f}, "
        f"LPIPS={rec_metrics.mean_lpips:.4f}"
    )


if __name__ == "__main__":
    main()
