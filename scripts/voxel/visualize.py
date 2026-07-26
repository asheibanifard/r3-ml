#!/usr/bin/env python3
"""Visualize static outputs and optionally convert orbit frames to MP4."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--show", action="store_true", help="Open the static figure")
    parser.add_argument(
        "--make-video",
        action="store_true",
        help="Create MP4 orbit videos with ffmpeg",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    return parser.parse_args()


def make_static_figure(output_dir: Path) -> Path:
    paths = [
        output_dir / "gt_projection.pgm",
        output_dir / "pred_projection.pgm",
        output_dir / "diff.pgm",
    ]
    titles = [
        "GT: direct volume rendering",
        "Prediction: voxel rasterization",
        "Absolute difference",
    ]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing renderer outputs: " + ", ".join(missing))

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, path, title in zip(axes, paths, titles):
        image = Image.open(path)
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        # axis.axis("off")

    figure.tight_layout()
    destination = output_dir / "comparison.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    return destination


def make_video(pattern: Path, destination: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    static_path = make_static_figure(args.output_dir)
    print(f"Saved {static_path}")

    if args.make_video:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed. Install it with: sudo apt install ffmpeg")

        video_jobs = [
            (
                args.output_dir / "orbit_gt" / "gt_%04d.pgm",
                args.output_dir / "gt_orbit.mp4",
            ),
            (
                args.output_dir / "orbit_pred" / "pred_%04d.pgm",
                args.output_dir / "pred_orbit.mp4",
            ),
            (
                args.output_dir / "orbit_diff" / "diff_%04d.pgm",
                args.output_dir / "diff_orbit.mp4",
            ),
            (
                args.output_dir / "orbit_comparison" / "comparison_%04d.ppm",
                args.output_dir / "comparison_orbit.mp4",
            ),
        ]

        for pattern, destination in video_jobs:
            make_video(pattern, destination, args.fps)
            print(f"Saved {destination}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
