# voxel/io.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_volume(path: str | Path, *, device: torch.device) -> Tensor:
    """
    Load a real [G,G,G] EM block (e.g. processed_data/*.tif) as a
    normalized-to-[0,1] float32 CUDA tensor, for use as the SIREN's
    training target in place of the synthetic sphere/torus/blob volume.

    Normalizes by the loaded volume's own min/max (mirrors
    gaussian_volume.io.load_volume's default normalization), so it works
    for both uint8 EM blocks ([0,255]) and any other integer/float range.
    """
    import tifffile

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Volume file not found: {file_path}")

    array = tifffile.imread(file_path)
    if array.ndim != 3:
        raise ValueError(f"Loaded volume must be 3D, got shape {array.shape}.")

    if array.shape[0] != array.shape[1] or array.shape[1] != array.shape[2]:
        raise ValueError(f"Loaded volume must be a cube [G,G,G], got shape {array.shape}.")

    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)

    minimum = tensor.min()
    maximum = tensor.max()
    span = torch.clamp(maximum - minimum, min=1e-8)
    return ((tensor - minimum) / span).contiguous()


def save_volume(path: str | Path, volume: Tensor) -> None:
    """
    Save a [G,G,G] float32 tensor (SIREN's native [0,1] output range) as a
    .tif volume.
    """
    import tifffile

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(file_path, volume.detach().cpu().numpy().astype(np.float32))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def save_pgm(path: str | Path, image: np.ndarray) -> None:
    """
    Save a single-channel [H,W] float image in [0,1] as a binary (P5) PGM.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(image), mode="L").save(file_path)


def save_comparison_ppm(
    path: str | Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    difference: np.ndarray,
) -> None:
    """
    Save GT | prediction | amplified difference as one RGB (P6) PPM image,
    three [H,W] panels side by side. GT and prediction are grayscale; the
    difference panel uses a red-yellow heat map (matching main.cu's
    save_comparison_ppm), amplified 4x and clamped to [0,1].
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = ground_truth.shape
    panels = np.empty((height, width * 3, 3), dtype=np.uint8)

    gt_gray = _to_uint8(np.clip(ground_truth, 0.0, 1.0))
    pred_gray = _to_uint8(np.clip(prediction, 0.0, 1.0))
    panels[:, 0:width, :] = gt_gray[..., None]
    panels[:, width:2 * width, :] = pred_gray[..., None]

    amplified = np.clip(difference * 4.0, 0.0, 1.0)
    red = _to_uint8(np.clip(2.0 * amplified, 0.0, 1.0))
    green = _to_uint8(np.clip(2.0 * amplified - 0.5, 0.0, 1.0))
    blue = np.zeros_like(red)
    panels[:, 2 * width:3 * width, 0] = red
    panels[:, 2 * width:3 * width, 1] = green
    panels[:, 2 * width:3 * width, 2] = blue

    Image.fromarray(panels, mode="RGB").save(file_path)
