#!/usr/bin/env python3
"""Run the CUDA voxel experiment and print FPS, PSNR, SSIM, and LPIPS.

Examples
--------
Run the CUDA executable, capture its FPS output, and evaluate images:
    python3 evaluate.py --run ./siren_voxel 2000 0.001 120 1000

Evaluate images that already exist:
    python3 evaluate.py --output-dir output

Evaluate explicit image paths:
    python3 evaluate.py --gt output/gt_projection.pgm \
                        --pred output/pred_projection.pgm

Evaluate the 3D volumes directly (more exact than the rendered projection
above — compares every voxel, not one rendered view):
    python3 evaluate.py --gt output/ground_truth_volume.tif \
                        --pred output/reconstructed_volume.tif
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print renderer FPS and GT/prediction image metrics."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing gt_projection.pgm and pred_projection.pgm",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        help="Explicit ground-truth path: a rendered image, or a 3D volume (.tif/.raw)",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        help="Explicit predicted path: a rendered image, or a 3D volume (.tif/.raw)",
    )
    parser.add_argument(
        "--run",
        nargs=argparse.REMAINDER,
        help=(
            "Command used to run the CUDA program. This option must be last. "
            "Example: --run ./siren_voxel 2000 0.001 120 1000"
        ),
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="Pretrained LPIPS backbone",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used for LPIPS",
    )
    parser.add_argument(
        "--volume",
        type=Path,
        help=(
            "Ground-truth voxel volume, for the compression-rate metric. "
            "Defaults to <output-dir>/ground_truth_volume.tif."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "SIREN parameter checkpoint (init.pth/best.pth/last.pth/<iter>.pth), "
            "for the compression-rate metric. Defaults to <output-dir>/best.pth."
        ),
    )
    return parser.parse_args()


def run_renderer(command: Sequence[str]) -> str:
    if not command:
        raise ValueError("--run was supplied without a command")

    print("Running CUDA experiment:")
    print("  " + " ".join(command))
    print()

    process = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(process.stdout, end="")

    if process.returncode != 0:
        raise RuntimeError(
            f"CUDA executable failed with exit status {process.returncode}"
        )
    return process.stdout


def extract_renderer_fps(stdout: str) -> dict[str, float]:
    """Extract FPS lines printed by the rewritten CUDA executable."""
    results: dict[str, float] = {}

    patterns = {
        "GT DVR FPS": r"GT direct volume renderer.*?([0-9]+(?:\.[0-9]+)?)\s+FPS",
        "Voxel rasterizer FPS": r"Pred voxel rasterizer.*?([0-9]+(?:\.[0-9]+)?)\s+FPS",
    }

    for label, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            results[label] = float(match.group(1))

    return results


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_raw_volume(path: Path) -> np.ndarray:
    """
    Load a legacy flat float32 voxel dump (main.cu's original save_raw_volume
    format, e.g. ground_truth_volume_f32.raw) and reshape it back into a
    [D,H,W] cube. The grid size is inferred from the file's element count
    rather than hardcoded, so it stays correct regardless of grid size.

    New runs write .tif volumes instead (see load_tiff_volume) — this
    remains only to read old .raw dumps.
    """

    if not path.exists():
        raise FileNotFoundError(f"Raw volume not found: {path}")

    flat = np.fromfile(path, dtype=np.float32)
    grid_size = round(flat.size ** (1.0 / 3.0))

    if grid_size**3 != flat.size:
        raise ValueError(
            f"{path} contains {flat.size} float32 values, which is not a "
            "perfect cube; expected a GRID_SIZE^3 raw volume dump."
        )

    return flat.reshape(grid_size, grid_size, grid_size)


def load_tiff_volume(path: Path) -> np.ndarray:
    """
    Load a [D,H,W] float32 volume written by voxel.io.save_volume
    (e.g. ground_truth_volume.tif / reconstructed_volume.tif).
    """
    import tifffile

    if not path.exists():
        raise FileNotFoundError(f"TIFF volume not found: {path}")

    array = tifffile.imread(path)
    if array.ndim != 3:
        raise ValueError(f"{path} must be a 3D [D,H,W] volume, got shape {array.shape}.")

    return array.astype(np.float32)


def load_array(path: Path) -> tuple[np.ndarray, bool]:
    """
    Load either a rendered image (any PIL-readable format), a .tif 3D
    volume, or a legacy raw float32 volume (.raw), dispatching on file
    extension. Returns (array, is_volume).
    """

    if path.suffix.lower() == ".raw":
        return load_raw_volume(path), True

    if path.suffix.lower() in (".tif", ".tiff"):
        return load_tiff_volume(path), True

    return load_image(path), False


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((gt - pred) ** 2, dtype=np.float64))
    if mse == 0.0:
        return mse, float("inf")
    psnr = 10.0 * np.log10(1.0 / mse)
    return mse, float(psnr)


def compute_ssim(gt: np.ndarray, pred: np.ndarray, is_volume: bool) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "SSIM requires scikit-image. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    # Raw volumes are single-channel [D,H,W]; rendered images are [H,W,3].
    return float(
        structural_similarity(
            gt,
            pred,
            data_range=1.0,
            channel_axis=None if is_volume else 2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def compute_lpips(
    gt: np.ndarray,
    pred: np.ndarray,
    network: str,
    requested_device: str,
    is_volume: bool,
) -> tuple[float, str, float]:
    """
    LPIPS is inherently a 2D perceptual metric (it runs a pretrained image
    network). For a raw 3D volume, this averages LPIPS over every axial
    (Z) slice, converting each single-channel slice to 3-channel RGB by
    replication; for a rendered image (already RGB), it scores the image
    directly, as before.
    """

    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS requires torch and lpips. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable in PyTorch")

    def to_tensor(image: np.ndarray) -> "torch.Tensor":
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return (tensor * 2.0 - 1.0).to(device)

    def grayscale_slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
        return np.repeat(slice_2d[:, :, None], 3, axis=-1)

    model = lpips.LPIPS(net=network).to(device)

    if is_volume:
        gt_slices = [grayscale_slice_to_rgb(gt[z]) for z in range(gt.shape[0])]
        pred_slices = [grayscale_slice_to_rgb(pred[z]) for z in range(pred.shape[0])]
    else:
        gt_slices = [gt]
        pred_slices = [pred]

    gt_tensor = to_tensor(gt_slices[0])
    pred_tensor = to_tensor(pred_slices[0])

    # Warm up CUDA before timing the LPIPS forward pass(es).
    if device == "cuda":
        with torch.no_grad():
            _ = model(gt_tensor, pred_tensor)
        torch.cuda.synchronize()

    start = time.perf_counter()
    values: list[float] = []
    with torch.no_grad():
        for gt_slice, pred_slice in zip(gt_slices, pred_slices):
            values.append(
                model(to_tensor(gt_slice), to_tensor(pred_slice)).item()
            )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return float(np.mean(values)), device, elapsed_ms


def compute_compression_rate(
    volume_path: Path,
    checkpoint_path: Path,
) -> tuple[int, int, float] | None:
    """
    Compare the raw ground-truth voxel volume's file size against a SIREN
    parameter checkpoint's file size. Returns (volume_bytes, model_bytes,
    ratio), or None if either file is missing (e.g. main.cu hasn't been
    rebuilt/rerun with checkpoint saving yet). ratio > 1x means the model is
    smaller than the raw volume.
    """

    if not volume_path.exists() or not checkpoint_path.exists():
        return None

    volume_bytes = volume_path.stat().st_size
    model_bytes = checkpoint_path.stat().st_size

    if model_bytes == 0:
        return None

    return volume_bytes, model_bytes, volume_bytes / model_bytes


def main() -> int:
    args = parse_args()

    renderer_stdout = ""
    if args.run is not None:
        renderer_stdout = run_renderer(args.run)

    gt_path = args.gt or (args.output_dir / "gt_projection.pgm")
    pred_path = args.pred or (args.output_dir / "pred_projection.pgm")
    volume_path = args.volume or (args.output_dir / "ground_truth_volume.tif")
    checkpoint_path = args.checkpoint or (args.output_dir / "best.pth")

    gt, gt_is_volume = load_array(gt_path)
    pred, pred_is_volume = load_array(pred_path)

    if gt_is_volume != pred_is_volume:
        raise ValueError(
            "GT and prediction must both be images or both be raw volumes: "
            f"GT is_volume={gt_is_volume} ({gt_path}), "
            f"prediction is_volume={pred_is_volume} ({pred_path})"
        )

    is_volume = gt_is_volume

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shapes differ: GT {gt.shape}, prediction {pred.shape}"
        )

    metric_start = time.perf_counter()
    mse, psnr = compute_psnr(gt, pred)
    ssim = compute_ssim(gt, pred, is_volume)
    lpips_value, lpips_device, lpips_ms = compute_lpips(
        gt, pred, args.lpips_net, args.device, is_volume
    )
    total_metric_ms = (time.perf_counter() - metric_start) * 1000.0

    compression = compute_compression_rate(volume_path, checkpoint_path)

    renderer_fps = extract_renderer_fps(renderer_stdout)

    print("\n" + "=" * 62)
    print("VOXEL RENDERING EVALUATION")
    print("=" * 62)
    print(f"GT input                 : {gt_path}")
    print(f"Predicted input          : {pred_path}")
    if is_volume:
        depth, height, width = gt.shape
        print(f"Volume size               : {depth} x {height} x {width}")
    else:
        print(f"Image size                : {gt.shape[1]} x {gt.shape[0]}")
    print("-" * 62)

    if renderer_fps:
        for label, value in renderer_fps.items():
            print(f"{label:<26}: {value:12.2f}")
    elif args.run is None:
        print("Renderer FPS              : not available (use --run)")
    else:
        print("Renderer FPS              : not found in executable output")

    print("-" * 62)
    print(f"MSE                       : {mse:12.8f}")
    print(f"PSNR                      : {psnr:12.4f} dB")
    print(f"SSIM                      : {ssim:12.6f}")
    lpips_label = (
        f"LPIPS ({args.lpips_net}, mean over Z slices)"
        if is_volume
        else f"LPIPS ({args.lpips_net})"
    )
    print(f"{lpips_label:<26}: {lpips_value:12.6f}")
    print(f"LPIPS device              : {lpips_device}")
    print(f"LPIPS forward time        : {lpips_ms:12.3f} ms")
    print(f"All metric computation    : {total_metric_ms:12.3f} ms")
    print("-" * 62)

    if compression is not None:
        volume_bytes, model_bytes, ratio = compression
        print(f"Ground-truth volume       : {volume_path} ({volume_bytes:,} bytes)")
        print(f"SIREN checkpoint          : {checkpoint_path} ({model_bytes:,} bytes)")
        print(f"Compression rate          : {ratio:12.2f}x")
    else:
        print(
            "Compression rate          : not available "
            f"(missing {volume_path} or {checkpoint_path})"
        )

    print("=" * 62)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
