# gaussian_volume/pruning.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from .model import GaussianVolumeModel


@dataclass
class PruningConfig:
    enabled: bool
    from_step: int
    until_step: int
    interval: int
    minimum_confidence: float
    minimum_feature: float
    minimum_scale: float
    maximum_scale: Optional[float]
    remove_outside_volume: bool
    outside_margin: float
    minimum_gaussians: int
    confidence_score_weight: float
    feature_score_weight: float


@dataclass
class PruningResult:
    means: Tensor
    scales: Tensor
    quaternions: Tensor
    confidences: Tensor
    features: Tensor
    removed_count: int
    final_count: int


def should_prune(step: int, config: PruningConfig) -> bool:
    _validate_config(config)

    if not config.enabled:
        return False

    if step < config.from_step or step > config.until_step:
        return False

    return (step - config.from_step) % config.interval == 0


def prune_gaussians(
    model: GaussianVolumeModel,
    *,
    config: PruningConfig,
    volume_shape: Sequence[int],
) -> PruningResult:
    _validate_config(config)

    current_count = model.num_gaussians

    if current_count <= config.minimum_gaussians:
        return PruningResult(
            means=model.means.detach().clone(),
            scales=model.scales.detach().clone(),
            quaternions=model.normalized_quaternions.detach().clone(),
            confidences=model.confidences.detach().clone(),
            features=model.features.detach().clone(),
            removed_count=0,
            final_count=current_count,
        )

    confidences = model.confidences.detach()
    features = model.features.detach()
    scales = model.scales.detach()
    means = model.means.detach()

    score = (
        config.confidence_score_weight * confidences
        + config.feature_score_weight * features
    )

    dead_mask = (confidences < config.minimum_confidence) | (
        features.abs() < config.minimum_feature
    )

    max_scale = scales.max(dim=-1).values
    min_scale = scales.min(dim=-1).values

    scale_mask = min_scale < config.minimum_scale
    if config.maximum_scale is not None:
        scale_mask = scale_mask | (max_scale > config.maximum_scale)

    remove_mask = dead_mask | scale_mask

    if config.remove_outside_volume:
        remove_mask = remove_mask | outside_volume_mask(
            means, volume_shape, margin=config.outside_margin
        )

    keep_mask = ~remove_mask

    num_would_remain = int(keep_mask.sum().item())
    if num_would_remain < config.minimum_gaussians:
        num_to_remove = current_count - config.minimum_gaussians
        if num_to_remove <= 0:
            keep_mask = torch.ones_like(keep_mask)
        else:
            removable_indices = remove_mask.nonzero(as_tuple=True)[0]
            removable_scores = score[removable_indices]
            drop_count = min(num_to_remove, removable_indices.numel())
            worst_indices = torch.topk(
                removable_scores, k=drop_count, largest=False
            ).indices
            final_remove_indices = removable_indices[worst_indices]

            keep_mask = torch.ones_like(remove_mask)
            keep_mask[final_remove_indices] = False

    removed_count = current_count - int(keep_mask.sum().item())

    quaternions = model.normalized_quaternions.detach()

    return PruningResult(
        means=means[keep_mask],
        scales=scales[keep_mask],
        quaternions=quaternions[keep_mask],
        confidences=confidences[keep_mask],
        features=features[keep_mask],
        removed_count=removed_count,
        final_count=int(keep_mask.sum().item()),
    )


def outside_volume_mask(
    means: Tensor,
    shape: Sequence[int],
    *,
    margin: float,
) -> Tensor:
    depth, height, width = shape

    outside_x = (means[:, 0] < -margin) | (means[:, 0] > width + margin)
    outside_y = (means[:, 1] < -margin) | (means[:, 1] > height + margin)
    outside_z = (means[:, 2] < -margin) | (means[:, 2] > depth + margin)

    return outside_x | outside_y | outside_z


def _validate_config(config: PruningConfig) -> None:
    if config.interval <= 0:
        raise ValueError(f"interval must be positive, got {config.interval}.")

    if config.from_step < 0:
        raise ValueError(f"from_step must be non-negative, got {config.from_step}.")

    if config.until_step < config.from_step:
        raise ValueError("until_step must be >= from_step.")

    if config.minimum_gaussians <= 0:
        raise ValueError(
            f"minimum_gaussians must be positive, got {config.minimum_gaussians}."
        )

    if config.maximum_scale is not None and config.minimum_scale >= config.maximum_scale:
        raise ValueError("minimum_scale must be less than maximum_scale.")
