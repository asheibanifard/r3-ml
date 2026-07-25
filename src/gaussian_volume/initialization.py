# gaussian_volume/initialization.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

InitializationStrategy = Literal[
    "intensity_weighted",
    "uniform",
    "top_intensity",
    "grid",
]


@dataclass
class GaussianInitialization:
    means: Tensor
    scales: Tensor
    quaternions: Tensor
    confidences: Tensor
    features: Tensor

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quaternions": self.quaternions,
            "confidences": self.confidences,
            "features": self.features,
        }


def initialize_gaussians(
    volume: Tensor,
    num_gaussians: int,
    *,
    strategy: InitializationStrategy,
    initial_scale: float,
    initial_confidence: float,
    uniform_sample_fraction: float,
    intensity_power: float,
    minimum_sampling_weight: float,
    jitter: float,
    feature_sampling: Literal["nearest", "trilinear"],
    generator: torch.Generator,
) -> GaussianInitialization:
    """
    Create the initial Gaussian mixture used before training begins.

    All keyword arguments are required, per this codebase's convention of
    never providing default values in function signatures.
    """
    _validate_initialization_arguments(
        volume=volume,
        num_gaussians=num_gaussians,
        uniform_sample_fraction=uniform_sample_fraction,
        intensity_power=intensity_power,
        minimum_sampling_weight=minimum_sampling_weight,
        jitter=jitter,
    )

    shape = volume.shape
    device = volume.device
    dtype = volume.dtype

    if strategy == "intensity_weighted":
        indices = _sample_intensity_weighted_means(
            volume=volume,
            num_gaussians=num_gaussians,
            uniform_sample_fraction=uniform_sample_fraction,
            intensity_power=intensity_power,
            minimum_sampling_weight=minimum_sampling_weight,
            generator=generator,
        )
    elif strategy == "uniform":
        indices = _sample_uniform_means(
            shape=shape,
            num_gaussians=num_gaussians,
            device=device,
            generator=generator,
        )
    elif strategy == "top_intensity":
        indices = _sample_top_intensity_means(
            volume=volume,
            num_gaussians=num_gaussians,
        )
    elif strategy == "grid":
        indices = _sample_grid_means(
            shape=shape,
            num_gaussians=num_gaussians,
            device=device,
        )
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy!r}.")

    means = linear_indices_to_means(indices, shape, dtype)

    if jitter > 0:
        means = _add_position_jitter(means, jitter=jitter, generator=generator)

    n = means.shape[0]

    scales = torch.full((n, 3), initial_scale, device=device, dtype=dtype)
    quaternions = identity_quaternions(n, device, dtype)
    confidences = torch.full((n,), initial_confidence, device=device, dtype=dtype)

    sampled_intensity = sample_volume_at_means(volume, means, mode=feature_sampling)
    features = sampled_intensity.to(dtype)

    return GaussianInitialization(
        means=means,
        scales=scales,
        quaternions=quaternions,
        confidences=confidences,
        features=features,
    )


def _sample_intensity_weighted_means(
    volume: Tensor,
    num_gaussians: int,
    uniform_sample_fraction: float,
    intensity_power: float,
    minimum_sampling_weight: float,
    generator: torch.Generator,
) -> Tensor:
    flat = volume.reshape(-1).clamp_min(0.0)
    weights = flat.pow(intensity_power) + minimum_sampling_weight

    num_uniform = int(round(num_gaussians * uniform_sample_fraction))
    num_weighted = num_gaussians - num_uniform

    weighted_indices = torch.multinomial(
        weights,
        num_weighted,
        replacement=True,
        generator=generator,
    )

    if num_uniform > 0:
        uniform_indices = torch.randint(
            0,
            flat.shape[0],
            (num_uniform,),
            device=flat.device,
            generator=generator,
        )
        return torch.cat([weighted_indices, uniform_indices], dim=0)

    return weighted_indices


def _sample_uniform_means(
    shape: tuple[int, ...],
    num_gaussians: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    total_voxels = 1
    for dimension in shape:
        total_voxels *= dimension

    return torch.randint(
        0,
        total_voxels,
        (num_gaussians,),
        device=device,
        generator=generator,
    )


def _sample_top_intensity_means(volume: Tensor, num_gaussians: int) -> Tensor:
    flat = volume.reshape(-1)
    _, indices = torch.topk(flat, k=num_gaussians)
    return indices


def _sample_grid_means(
    shape: tuple[int, ...],
    num_gaussians: int,
    device: torch.device,
) -> Tensor:
    depth, height, width = shape
    per_axis = max(1, round(num_gaussians ** (1.0 / 3.0)))

    z_indices = torch.linspace(0, depth - 1, per_axis, device=device).round().long()
    y_indices = torch.linspace(0, height - 1, per_axis, device=device).round().long()
    x_indices = torch.linspace(0, width - 1, per_axis, device=device).round().long()

    grid_z, grid_y, grid_x = torch.meshgrid(z_indices, y_indices, x_indices, indexing="ij")

    linear = (grid_z * height + grid_y) * width + grid_x
    linear = linear.reshape(-1)

    if linear.shape[0] > num_gaussians:
        linear = linear[:num_gaussians]

    return linear


def linear_indices_to_means(
    indices: Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    depth, height, width = shape

    z = indices // (height * width)
    remainder = indices % (height * width)
    y = remainder // width
    x = remainder % width

    means = torch.stack(
        [x.to(dtype) + 0.5, y.to(dtype) + 0.5, z.to(dtype) + 0.5],
        dim=-1,
    )

    return means


def sample_volume_at_means(
    volume: Tensor,
    means: Tensor,
    mode: Literal["nearest", "trilinear"],
) -> Tensor:
    if mode == "nearest":
        return _sample_volume_nearest(volume, means)
    elif mode == "trilinear":
        return _sample_volume_trilinear(volume, means)
    else:
        raise ValueError(f"Unknown feature_sampling mode: {mode!r}.")


def _sample_volume_nearest(volume: Tensor, means: Tensor) -> Tensor:
    depth, height, width = volume.shape

    x = means[:, 0].round().long().clamp(0, width - 1)
    y = means[:, 1].round().long().clamp(0, height - 1)
    z = means[:, 2].round().long().clamp(0, depth - 1)

    return volume[z, y, x]


def _sample_volume_trilinear(volume: Tensor, means: Tensor) -> Tensor:
    depth, height, width = volume.shape

    grid = torch.stack(
        [
            (means[:, 0] / (width - 1)) * 2 - 1,
            (means[:, 1] / (height - 1)) * 2 - 1,
            (means[:, 2] / (depth - 1)) * 2 - 1,
        ],
        dim=-1,
    ).view(1, -1, 1, 1, 3)

    volume_5d = volume.unsqueeze(0).unsqueeze(0)

    sampled = torch.nn.functional.grid_sample(
        volume_5d,
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )

    return sampled.view(-1)


def identity_quaternions(
    num_gaussians: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    quaternions = torch.zeros((num_gaussians, 4), device=device, dtype=dtype)
    quaternions[:, 0] = 1.0
    return quaternions


def _add_position_jitter(
    means: Tensor,
    jitter: float,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(
        means.shape,
        device=means.device,
        dtype=means.dtype,
        generator=generator,
    )
    return means + noise * jitter


def _validate_initialization_arguments(
    volume: Tensor,
    num_gaussians: int,
    uniform_sample_fraction: float,
    intensity_power: float,
    minimum_sampling_weight: float,
    jitter: float,
) -> None:
    if volume.ndim != 3:
        raise ValueError(f"volume must have shape [D,H,W], got {tuple(volume.shape)}.")

    if num_gaussians <= 0:
        raise ValueError(f"num_gaussians must be positive, got {num_gaussians}.")

    if not (0.0 <= uniform_sample_fraction <= 1.0):
        raise ValueError(
            f"uniform_sample_fraction must be in [0,1], got {uniform_sample_fraction}."
        )

    if intensity_power <= 0:
        raise ValueError(f"intensity_power must be positive, got {intensity_power}.")

    if minimum_sampling_weight < 0:
        raise ValueError(
            f"minimum_sampling_weight must be non-negative, got {minimum_sampling_weight}."
        )

    if jitter < 0:
        raise ValueError(f"jitter must be non-negative, got {jitter}.")
