#!/usr/bin/env python3
"""
Calculate volumetric reconstruction metrics:

    MSE
    PSNR
    SSIM
    Max Error
    Output Min
    Output Max

Supported inputs:
    - HDF5: .h5 / .hdf5
    - NumPy: .npy / .npz
    - Raw float32: .raw, using <file>.raw.json metadata
    - PFM: .pfm

Typical usage:

    python volume_metrics.py \
        --reference ground_truth.h5 \
        --prediction reconstructed_volume.raw \
        --reference-dataset raw

For two HDF5 files:

    python volume_metrics.py \
        --reference ground_truth.h5 \
        --prediction reconstructed.h5 \
        --reference-dataset raw \
        --prediction-dataset raw

Save results:

    python volume_metrics.py \
        --reference ground_truth.h5 \
        --prediction reconstructed_volume.raw \
        --reference-dataset raw \
        --output-json metrics.json \
        --output-csv metrics.csv

Dependencies:
    pip install numpy h5py scikit-image
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = file.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"Invalid PFM header in {path}: {header!r}")

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
        raise ValueError(
            f"{path}: expected {expected} values, found {values.size}"
        )

    if channels == 1:
        image = values.reshape(height, width)
    else:
        image = values.reshape(height, width, channels)

    return np.flipud(image).astype(np.float32, copy=False)


def load_raw(path: Path) -> np.ndarray:
    metadata_path = Path(str(path) + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"RAW metadata not found: {metadata_path}\n"
            "Expected a JSON file containing a 'shape' field."
        )

    metadata = json.loads(metadata_path.read_text())
    if "shape" not in metadata:
        raise KeyError(f"{metadata_path} does not contain 'shape'.")

    shape = tuple(int(v) for v in metadata["shape"])
    dtype = np.dtype(metadata.get("dtype", "float32"))

    endianness = metadata.get("endianness", "little")
    if endianness == "little":
        dtype = dtype.newbyteorder("<")
    elif endianness == "big":
        dtype = dtype.newbyteorder(">")

    values = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))

    if values.size != expected:
        raise ValueError(
            f"{path}: metadata shape {shape} requires {expected} values, "
            f"but the file contains {values.size}."
        )

    return values.reshape(shape)


def choose_h5_dataset(file: Any, requested: str | None) -> str:
    import h5py

    if requested is not None:
        if requested not in file:
            available: list[str] = []
            file.visititems(
                lambda name, obj: (
                    available.append(name)
                    if isinstance(obj, h5py.Dataset)
                    else None
                )
            )
            raise KeyError(
                f"Dataset {requested!r} was not found. "
                f"Available datasets: {available}"
            )
        return requested

    preferred = ("raw", "volume", "data", "prediction", "output")
    for name in preferred:
        if name in file and isinstance(file[name], h5py.Dataset):
            return name

    datasets: list[str] = []
    file.visititems(
        lambda name, obj: (
            datasets.append(name)
            if isinstance(obj, h5py.Dataset)
            else None
        )
    )

    if len(datasets) == 1:
        return datasets[0]

    raise ValueError(
        "The HDF5 file contains multiple datasets. "
        f"Choose one with --reference-dataset or --prediction-dataset. "
        f"Available datasets: {datasets}"
    )


def load_array(path: Path, dataset: str | None) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as file:
            dataset_name = choose_h5_dataset(file, dataset)
            array = file[dataset_name][...]
            print(f"Loaded HDF5 dataset {dataset_name!r} from {path}")

    elif suffix == ".npy":
        array = np.load(path)

    elif suffix == ".npz":
        archive = np.load(path)
        if dataset is not None:
            if dataset not in archive:
                raise KeyError(
                    f"Array {dataset!r} not found in {path}. "
                    f"Available arrays: {list(archive.files)}"
                )
            array = archive[dataset]
        elif len(archive.files) == 1:
            array = archive[archive.files[0]]
        elif "raw" in archive:
            array = archive["raw"]
        else:
            raise ValueError(
                f"{path} contains multiple arrays: {list(archive.files)}. "
                "Select one with the dataset argument."
            )

    elif suffix == ".raw":
        array = load_raw(path)

    elif suffix == ".pfm":
        array = read_pfm(path)

    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r} for {path}."
        )

    return np.asarray(array)


def squeeze_to_matching_shape(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.squeeze(reference)
    prediction = np.squeeze(prediction)

    if reference.shape == prediction.shape:
        return reference, prediction

    # Common axis-order mismatch: prediction is Z,Y,X while reference is X,Y,Z.
    if reference.ndim == prediction.ndim == 3:
        permutations = (
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
        for permutation in permutations:
            candidate = np.transpose(prediction, permutation)
            if candidate.shape == reference.shape:
                raise ValueError(
                    "Reference and prediction shapes differ but can be matched "
                    f"by transposing prediction with axes {permutation}. "
                    "Apply the correct axis order explicitly before computing "
                    "metrics; automatic transposition is intentionally avoided."
                )

    raise ValueError(
        f"Shape mismatch: reference {reference.shape}, "
        f"prediction {prediction.shape}."
    )


def calculate_ssim(
    reference: np.ndarray,
    prediction: np.ndarray,
    data_range: float,
) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise ImportError(
            "SSIM requires scikit-image. Install it with:\n"
            "    pip install scikit-image"
        ) from error

    smallest_dimension = min(reference.shape)

    # skimage's default window is 7. Use the largest valid odd window for
    # unusually small inputs.
    win_size = min(7, smallest_dimension)
    if win_size % 2 == 0:
        win_size -= 1

    if win_size < 3:
        raise ValueError(
            f"SSIM requires each spatial dimension to be at least 3; "
            f"received shape {reference.shape}."
        )

    return float(
        structural_similarity(
            reference,
            prediction,
            data_range=data_range,
            win_size=win_size,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)

    parser.add_argument("--reference-dataset", type=str, default=None)
    parser.add_argument("--prediction-dataset", type=str, default=None)

    parser.add_argument(
        "--data-range",
        type=float,
        default=None,
        help=(
            "Dynamic range used for PSNR and SSIM. By default, "
            "reference.max() - reference.min()."
        ),
    )

    parser.add_argument(
        "--normalize",
        choices=("none", "reference", "independent"),
        default="none",
        help=(
            "Optional normalization. 'reference' applies the reference min/max "
            "to both arrays. 'independent' normalizes each array independently."
        ),
    )

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)

    args = parser.parse_args()

    reference = load_array(
        args.reference,
        args.reference_dataset,
    ).astype(np.float64, copy=False)

    prediction = load_array(
        args.prediction,
        args.prediction_dataset,
    ).astype(np.float64, copy=False)

    reference, prediction = squeeze_to_matching_shape(
        reference,
        prediction,
    )

    if not np.isfinite(reference).all():
        raise ValueError("Reference contains NaN or infinite values.")

    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values.")

    original_output_min = float(prediction.min())
    original_output_max = float(prediction.max())

    if args.normalize == "reference":
        minimum = float(reference.min())
        maximum = float(reference.max())
        scale = maximum - minimum
        if scale <= 0:
            raise ValueError("Reference has zero dynamic range.")

        reference = (reference - minimum) / scale
        prediction = (prediction - minimum) / scale

    elif args.normalize == "independent":
        reference_range = float(reference.max() - reference.min())
        prediction_range = float(prediction.max() - prediction.min())

        if reference_range <= 0 or prediction_range <= 0:
            raise ValueError(
                "Independent normalization requires non-zero ranges."
            )

        reference = (
            reference - float(reference.min())
        ) / reference_range

        prediction = (
            prediction - float(prediction.min())
        ) / prediction_range

    difference = prediction - reference
    squared_difference = difference * difference

    mse = float(np.mean(squared_difference))
    max_error = float(np.max(np.abs(difference)))

    if args.data_range is not None:
        data_range = float(args.data_range)
    elif args.normalize != "none":
        data_range = 1.0
    else:
        data_range = float(reference.max() - reference.min())

    if data_range <= 0:
        raise ValueError(
            "PSNR/SSIM data range is zero. Supply --data-range explicitly."
        )

    if mse == 0.0:
        psnr = math.inf
    else:
        psnr = float(
            10.0 * math.log10((data_range * data_range) / mse)
        )

    ssim = calculate_ssim(reference, prediction, data_range)

    metrics = {
        "shape": list(reference.shape),
        "voxel_count": int(reference.size),
        "normalization": args.normalize,
        "data_range": data_range,
        "MSE": mse,
        "PSNR_dB": psnr,
        "SSIM": ssim,
        "Max_Error": max_error,
        "Output_Min": original_output_min,
        "Output_Max": original_output_max,
        "Reference_Min": float(reference.min()),
        "Reference_Max": float(reference.max()),
    }

    print()
    print(f"Shape       : {tuple(metrics['shape'])}")
    print(f"Voxel count : {metrics['voxel_count']}")
    print(f"MSE         : {mse:.10g}")
    print(f"PSNR        : {psnr:.6f} dB")
    print(f"SSIM        : {ssim:.8f}")
    print(f"Max Error   : {max_error:.10g}")
    print(f"Output Min  : {original_output_min:.10g}")
    print(f"Output Max  : {original_output_max:.10g}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metrics, indent=2, allow_nan=True) + "\n"
        )
        print(f"Saved JSON  : {args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Saved CSV   : {args.output_csv}")


if __name__ == "__main__":
    main()
