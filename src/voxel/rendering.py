# voxel/rendering.py

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .extension.loader import load_voxel_extension


def make_camera(
    azimuth_radians: float,
    elevation_radians: float,
    radius: float,
    vertical_fov_degrees: float,
    *,
    device: torch.device,
) -> Tensor:
    """
    Build the 13-float packed camera basis (origin, forward, right, up,
    tan_half_fov) consumed by render_direct_volume/render_voxel_rasterizer,
    orbiting a camera around the volume's centre.
    """
    origin = np.array(
        [
            radius * math.cos(elevation_radians) * math.cos(azimuth_radians),
            radius * math.sin(elevation_radians),
            radius * math.cos(elevation_radians) * math.sin(azimuth_radians),
        ],
        dtype=np.float32,
    )

    target = np.zeros(3, dtype=np.float32)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = _normalize(target - origin)
    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    tan_half_fov = math.tan(vertical_fov_degrees * math.pi / 360.0)

    packed = np.concatenate([origin, forward, right, up, [tan_half_fov]]).astype(np.float32)
    return torch.from_numpy(packed).to(device=device)


def _normalize(v: np.ndarray) -> np.ndarray:
    length = math.sqrt(max(float(np.dot(v, v)), 1.0e-20))
    return v / length


def render_direct_volume(
    volume: Tensor,
    camera: Tensor,
    *,
    image_width: int,
    image_height: int,
    dvr_steps: int,
    density_scale: float,
    hidden_size: int,
    hidden_layers: int,
) -> Tensor:
    extension = load_voxel_extension(hidden_size, hidden_layers)
    return extension.render_direct_volume(
        volume, camera, image_width, image_height, dvr_steps, density_scale
    )


def render_voxel_rasterizer(
    volume: Tensor,
    camera: Tensor,
    *,
    image_width: int,
    image_height: int,
    density_scale: float,
    hidden_size: int,
    hidden_layers: int,
) -> Tensor:
    extension = load_voxel_extension(hidden_size, hidden_layers)
    return extension.render_voxel_rasterizer(
        volume, camera, image_width, image_height, density_scale
    )


def evaluate_images(
    image_a: Tensor, image_b: Tensor, *, hidden_size: int, hidden_layers: int
) -> tuple[float, float, float]:
    """Returns (mse, psnr, ssim)."""
    extension = load_voxel_extension(hidden_size, hidden_layers)
    return extension.evaluate_images(image_a, image_b)


def compute_volume_psnr(
    volume_a: Tensor, volume_b: Tensor, *, hidden_size: int, hidden_layers: int
) -> float:
    extension = load_voxel_extension(hidden_size, hidden_layers)
    return extension.compute_volume_psnr(volume_a, volume_b)
