#!/usr/bin/env python3
"""Compute MIP-image quality metrics (PSNR/SSIM/MAE/max-abs-error) between a
dense ground-truth MIP render and a Gaussian-reconstruction MIP render.

Both PFMs must already be in the same aligned intensity space (e.g. both
exported with --normalise minmax against the same per-volume min/max) —
this script does not independently renormalise either image.

Usage
-----
python compute_mip_metrics.py \
    --gt dense/view_yaw000_pitch00.pfm \
    --rec gauss_50k/view_yaw000_pitch00.pfm \
    --data-range 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


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
        data = np.fromfile(file, dtype=endian + "f4")
        image = data.reshape(height, width)
        return np.flipud(image)


def compute_metrics(gt: np.ndarray, rec: np.ndarray, data_range: float) -> dict:
    diff = rec.astype(np.float64) - gt.astype(np.float64)
    mae = float(np.abs(diff).mean())
    max_abs_error = float(np.abs(diff).max())
    mse = float((diff ** 2).mean())
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10((data_range ** 2) / mse)
    ssim = float(structural_similarity(gt, rec, data_range=data_range))
    return {
        "mae": mae,
        "max_abs_error": max_abs_error,
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "gt_range": [float(gt.min()), float(gt.max())],
        "rec_range": [float(rec.min()), float(rec.max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--rec", required=True)
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    gt = read_pfm(Path(args.gt))
    rec = read_pfm(Path(args.rec))

    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape} rec={rec.shape}")

    metrics = compute_metrics(gt, rec, args.data_range)
    metrics["gt_path"] = args.gt
    metrics["rec_path"] = args.rec

    print(json.dumps(metrics, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
