#!/usr/bin/env python3
"""
Evaluate CUDA output against a reference array.

Metrics:
- MSE
- PSNR
- SSIM
- Max absolute error
- Output minimum
- Output maximum

Supported formats:
- .npy
- .npz
- .pt / .pth
- raw binary files, when --shape and --dtype are supplied

Examples
--------
python evaluate_cuda_output.py \
    --reference ground_truth.npy \
    --output cuda_output.npy

python evaluate_cuda_output.py \
    --reference gt.raw \
    --output rendered.raw \
    --shape 512 512 512 \
    --dtype float32 \
    --data-range 1.0

python evaluate_cuda_output.py \
    --reference gt.npy \
    --output output.npy \
    --ssim-mode volume \
    --save-json metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CUDA output with a reference array."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Path to the ground-truth/reference array.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the CUDA output array.",
    )
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=None,
        help="Array shape for raw binary files, e.g. --shape 512 512 512.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Data type for raw binary files. Default: float32.",
    )
    parser.add_argument(
        "--reference-key",
        default=None,
        help="Array key for a reference .npz file or dictionary-like .pt file.",
    )
    parser.add_argument(
        "--output-key",
        default=None,
        help="Array key for an output .npz file or dictionary-like .pt file.",
    )
    parser.add_argument(
        "--data-range",
        type=float,
        default=None,
        help=(
            "Signal range used for PSNR and SSIM. "
            "Default: max(reference) - min(reference)."
        ),
    )
    parser.add_argument(
        "--ssim-mode",
        choices=("auto", "volume", "slice"),
        default="auto",
        help=(
            "'volume' computes N-D SSIM directly; "
            "'slice' averages 2-D SSIM over the first axis; "
            "'auto' uses slice mode for 3-D arrays. Default: auto."
        ),
    )
    parser.add_argument(
        "--channel-axis",
        type=int,
        default=None,
        help="Channel axis for colour/multi-channel arrays, e.g. -1.",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Ignore this many pixels/voxels from every border.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path for saving the metrics as JSON.",
    )
    return parser.parse_args()


def _select_object(obj: Any, key: str | None, path: Path) -> Any:
    """Select an array/tensor from a loaded container."""
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.files)
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {keys}")
            return obj[key]
        if len(keys) != 1:
            raise ValueError(
                f"{path} contains multiple arrays: {keys}. "
                "Specify the required key."
            )
        return obj[keys[0]]

    if isinstance(obj, dict):
        keys = list(obj.keys())
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {keys}")
            return obj[key]
        if len(keys) != 1:
            raise ValueError(
                f"{path} contains multiple entries: {keys}. "
                "Specify the required key."
            )
        return obj[keys[0]]

    return obj


def load_array(
    path: Path,
    shape: Sequence[int] | None,
    dtype: str,
    key: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)

    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
            array = np.asarray(_select_object(loaded, key, path))

    elif suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to load .pt/.pth files."
            ) from exc

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        selected = _select_object(loaded, key, path)
        if isinstance(selected, torch.Tensor):
            array = selected.detach().cpu().numpy()
        else:
            array = np.asarray(selected)

    else:
        if shape is None:
            raise ValueError(
                f"Raw file detected: {path}. Supply --shape and --dtype."
            )
        array = np.fromfile(path, dtype=np.dtype(dtype))
        expected = int(np.prod(shape))
        if array.size != expected:
            raise ValueError(
                f"{path} contains {array.size} values, but shape {tuple(shape)} "
                f"requires {expected}."
            )
        array = array.reshape(tuple(shape))

    return np.asarray(array)


def crop_border(array: np.ndarray, border: int, channel_axis: int | None) -> np.ndarray:
    if border < 0:
        raise ValueError("--crop-border must be non-negative.")
    if border == 0:
        return array

    normalized_channel_axis = None
    if channel_axis is not None:
        normalized_channel_axis = channel_axis % array.ndim

    slices = []
    for axis, size in enumerate(array.shape):
        if axis == normalized_channel_axis:
            slices.append(slice(None))
            continue
        if size <= 2 * border:
            raise ValueError(
                f"Cannot crop border {border} from axis {axis} with size {size}."
            )
        slices.append(slice(border, -border))

    return array[tuple(slices)]


def finite_check(name: str, array: np.ndarray) -> None:
    non_finite = np.size(array) - int(np.isfinite(array).sum())
    if non_finite:
        raise ValueError(f"{name} contains {non_finite} NaN or Inf values.")


def calculate_ssim(
    reference: np.ndarray,
    output: np.ndarray,
    data_range: float,
    mode: str,
    channel_axis: int | None,
) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for SSIM. Install it with:\n"
            "  pip install scikit-image"
        ) from exc

    if mode == "auto":
        mode = "slice" if reference.ndim == 3 and channel_axis is None else "volume"

    if mode == "slice":
        if reference.ndim < 3:
            mode = "volume"
        else:
            values = []
            for index in range(reference.shape[0]):
                ref_slice = reference[index]
                out_slice = output[index]

                slice_channel_axis = None
                if channel_axis is not None:
                    normalized = channel_axis % reference.ndim
                    if normalized == 0:
                        raise ValueError(
                            "Slice mode cannot slice over the channel axis."
                        )
                    slice_channel_axis = normalized - 1

                values.append(
                    structural_similarity(
                        ref_slice,
                        out_slice,
                        data_range=data_range,
                        channel_axis=slice_channel_axis,
                    )
                )
            return float(np.mean(values))

    return float(
        structural_similarity(
            reference,
            output,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )


def calculate_metrics(
    reference: np.ndarray,
    output: np.ndarray,
    data_range: float | None,
    ssim_mode: str,
    channel_axis: int | None,
) -> dict[str, float | list[int] | str]:
    if reference.shape != output.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape}, output {output.shape}"
        )

    reference = reference.astype(np.float64, copy=False)
    output = output.astype(np.float64, copy=False)

    finite_check("Reference", reference)
    finite_check("Output", output)

    difference = output - reference
    squared_difference = difference * difference

    mse = float(np.mean(squared_difference))
    max_error = float(np.max(np.abs(difference)))
    output_min = float(np.min(output))
    output_max = float(np.max(output))
    reference_min = float(np.min(reference))
    reference_max = float(np.max(reference))

    if data_range is None:
        data_range = reference_max - reference_min

    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError(
            f"Invalid data range: {data_range}. "
            "Supply a positive value with --data-range."
        )

    psnr = math.inf if mse == 0.0 else float(
        10.0 * math.log10((data_range * data_range) / mse)
    )

    ssim = calculate_ssim(
        reference,
        output,
        float(data_range),
        ssim_mode,
        channel_axis,
    )

    return {
        "shape": list(reference.shape),
        "data_range": float(data_range),
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "max_absolute_error": max_error,
        "output_min": output_min,
        "output_max": output_max,
        "reference_min": reference_min,
        "reference_max": reference_max,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\nEvaluation results")
    print("=" * 48)
    print(f"Shape             : {tuple(metrics['shape'])}")
    print(f"Data range        : {metrics['data_range']:.10g}")
    print(f"MSE               : {metrics['mse']:.10g}")

    psnr = metrics["psnr_db"]
    if math.isinf(psnr):
        print("PSNR              : inf dB")
    else:
        print(f"PSNR              : {psnr:.6f} dB")

    print(f"SSIM              : {metrics['ssim']:.10f}")
    print(f"Max error         : {metrics['max_absolute_error']:.10g}")
    print(f"Output min        : {metrics['output_min']:.10g}")
    print(f"Output max        : {metrics['output_max']:.10g}")
    print(f"Reference min     : {metrics['reference_min']:.10g}")
    print(f"Reference max     : {metrics['reference_max']:.10g}")
    print("=" * 48)


def main() -> None:
    args = parse_args()

    reference = load_array(
        args.reference,
        args.shape,
        args.dtype,
        args.reference_key,
    )
    output = load_array(
        args.output,
        args.shape,
        args.dtype,
        args.output_key,
    )

    reference = crop_border(reference, args.crop_border, args.channel_axis)
    output = crop_border(output, args.crop_border, args.channel_axis)

    metrics = calculate_metrics(
        reference=reference,
        output=output,
        data_range=args.data_range,
        ssim_mode=args.ssim_mode,
        channel_axis=args.channel_axis,
    )

    print_metrics(metrics)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, allow_nan=True)
        print(f"Saved JSON metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
