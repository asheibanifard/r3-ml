# voxel/checkpoint.py

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def save_siren_checkpoint(path: str | Path, parameters: Tensor) -> None:
    """
    Save the flat SIREN parameter buffer as a torch checkpoint (init.pth/
    best.pth/last.pth/<iteration>.pth), matching gaussian_volume's
    checkpoint convention.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters.detach().cpu(), file_path)


def load_siren_checkpoint(path: str | Path, *, device: torch.device) -> Tensor:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    parameters = torch.load(file_path, map_location=device, weights_only=True)
    return parameters.to(device=device).contiguous()
