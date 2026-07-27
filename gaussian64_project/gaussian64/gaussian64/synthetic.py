from __future__ import annotations

import torch
from torch import Tensor


def make_synthetic_volume(shape: tuple[int, int, int], device: torch.device) -> Tensor:
    """Create a structured volume with spheres, a torus, and thin filaments."""
    d, h, w = shape
    z, y, x = torch.meshgrid(
        torch.arange(d, device=device, dtype=torch.float32) + 0.5,
        torch.arange(h, device=device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )

    def soft_sphere(cx: float, cy: float, cz: float, radius: float, edge: float = 0.8) -> Tensor:
        distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        return torch.sigmoid((radius - distance) / edge)

    sphere_a = 0.85 * soft_sphere(20.0, 21.0, 22.0, 10.0)
    sphere_b = 0.65 * soft_sphere(45.0, 42.0, 39.0, 8.0)

    radial = torch.sqrt((x - 39.0) ** 2 + (y - 21.0) ** 2)
    torus_distance = torch.sqrt((radial - 10.0) ** 2 + (z - 43.0) ** 2)
    torus = 0.75 * torch.sigmoid((2.7 - torus_distance) / 0.55)

    curve_y = 33.0 + 7.0 * torch.sin((x - 8.0) / 8.0)
    curve_z = 15.0 + 0.55 * (x - 8.0)
    filament_distance = torch.sqrt((y - curve_y) ** 2 + (z - curve_z) ** 2)
    filament_mask = ((x >= 8.0) & (x <= 55.0)).float()
    filament = 0.9 * torch.exp(-0.5 * (filament_distance / 1.1) ** 2) * filament_mask

    volume = torch.maximum(torch.maximum(sphere_a, sphere_b), torch.maximum(torus, filament))
    return volume.clamp(0.0, 1.0).contiguous()
