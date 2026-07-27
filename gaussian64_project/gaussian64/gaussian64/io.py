from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import torch
from torch import Tensor


def save_tiff(path: Path, volume: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, volume.detach().cpu().numpy().astype(np.float32))


def save_png(path: Path, image: Tensor) -> None:
    array = image.detach().cpu().clamp(0, 1).numpy()
    Image.fromarray(np.round(array * 255).astype(np.uint8), mode="L").save(path)


def save_mips(directory: Path, prefix: str, volume: Tensor) -> None:
    save_png(directory / f"{prefix}_mip_xy.png", volume.amax(dim=0))
    save_png(directory / f"{prefix}_mip_xz.png", volume.amax(dim=1))
    save_png(directory / f"{prefix}_mip_yz.png", volume.amax(dim=2))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
