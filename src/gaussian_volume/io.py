# gaussian_volume/io.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def normalize_volume(
    volume: Tensor,
    *,
    minimum: float,
    maximum: float,
    eps: float,
) -> Tensor:
    if maximum <= minimum:
        raise ValueError(f"maximum ({maximum}) must exceed minimum ({minimum}).")

    span = maximum - minimum
    return (volume - minimum) / max(span, eps)


def denormalize_volume(
    volume: Tensor,
    *,
    minimum: float,
    maximum: float,
) -> Tensor:
    if maximum <= minimum:
        raise ValueError(f"maximum ({maximum}) must exceed minimum ({minimum}).")

    span = maximum - minimum
    return volume * span + minimum


def load_volume(
    path: str,
    *,
    dataset: Optional[str],
    device: torch.device,
    normalize: bool,
    normalization_minimum: Optional[float],
    normalization_maximum: Optional[float],
) -> tuple[Tensor, dict]:
    """
    Load a target volume and return (volume, metadata).

    normalization_minimum/normalization_maximum may be omitted (None), in
    which case they are taken from the loaded volume's own min/max — either
    way, the actual bounds used are reported back in
    metadata["normalization"] so a downstream checkpoint can record them and
    a later reconstruction can invert normalize_volume exactly via
    denormalize_volume.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Volume file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".npz":
        array = _load_npz(file_path, dataset)
    elif suffix in (".h5", ".hdf5"):
        array = _load_hdf5(file_path, dataset)
    elif suffix in (".tif", ".tiff"):
        array = _load_tiff(file_path)
    elif suffix in (".pt", ".pth"):
        array = _load_torch_file(file_path, dataset)
    else:
        raise ValueError(f"Unsupported volume file extension: {suffix}")

    raw_volume = _to_volume_tensor(array, device=device)

    minimum = (
        normalization_minimum
        if normalization_minimum is not None
        else float(raw_volume.min().item())
    )
    maximum = (
        normalization_maximum
        if normalization_maximum is not None
        else float(raw_volume.max().item())
    )

    if normalize:
        volume = normalize_volume(raw_volume, minimum=minimum, maximum=maximum, eps=1e-8)
    else:
        volume = raw_volume

    metadata = {
        "source_path": str(file_path),
        "dataset": dataset,
        "shape": tuple(volume.shape),
        "normalization": {"minimum": minimum, "maximum": maximum},
    }

    return volume, metadata


def save_volume(
    volume: Tensor,
    path: str,
    *,
    dataset: Optional[str],
    metadata: Optional[dict],
) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()
    array = volume.detach().cpu().numpy()

    if suffix in (".h5", ".hdf5"):
        _save_hdf5(file_path, array, dataset=dataset or "volume", metadata=metadata)
    elif suffix in (".tif", ".tiff"):
        _save_tiff(file_path, array)
    elif suffix == ".npz":
        np.savez_compressed(file_path, volume=array)
    else:
        raise ValueError(f"Unsupported volume file extension: {suffix}")

    return file_path.resolve()


def _load_npz(path: Path, dataset: Optional[str]) -> np.ndarray:
    with np.load(path) as data:
        key = dataset if dataset is not None else data.files[0]
        return data[key]


def _load_hdf5(path: Path, dataset: Optional[str]) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        key = dataset if dataset is not None else _list_hdf5_datasets(handle)[0]
        return handle[key][()]


def _load_tiff(path: Path) -> np.ndarray:
    import tifffile

    return tifffile.imread(path)


def _load_torch_file(path: Path, dataset: Optional[str]) -> np.ndarray:
    data = torch.load(path, map_location="cpu")

    if isinstance(data, torch.Tensor):
        return data.numpy()

    if isinstance(data, dict):
        key = dataset if dataset is not None else next(iter(data))
        value = data[key]
        return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    raise ValueError(f"Unrecognized contents in torch file: {path}")


def _to_volume_tensor(array: np.ndarray, *, device: torch.device) -> Tensor:
    array = _remove_singleton_dimensions(array)

    if array.ndim != 3:
        raise ValueError(f"Loaded volume must be 3D, got shape {array.shape}.")

    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)
    return tensor


def _remove_singleton_dimensions(array: np.ndarray) -> np.ndarray:
    while array.ndim > 3 and 1 in array.shape:
        singleton_axis = array.shape.index(1)
        array = np.squeeze(array, axis=singleton_axis)
    return array


def _list_hdf5_datasets(handle) -> list[str]:
    names: list[str] = []
    handle.visit(lambda name: names.append(name))
    return names


def _save_hdf5(path: Path, array: np.ndarray, *, dataset: str, metadata: Optional[dict]) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=array)
        if metadata:
            for key, value in metadata.items():
                handle.attrs[key] = value


def _save_tiff(path: Path, array: np.ndarray) -> None:
    import tifffile

    tifffile.imwrite(path, array)
