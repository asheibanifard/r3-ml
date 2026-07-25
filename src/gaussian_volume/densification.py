# gaussian_volume/densification.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .math3d import quaternion_to_rotation_matrix
from .model import GaussianVolumeModel


@dataclass
class DensificationConfig:
    enabled: bool
    from_step: int
    until_step: int
    interval: int
    gradient_threshold: float
    population_exponent: float
    clone_scale_threshold: float
    split_scale_threshold: float
    max_new_gaussians: int
    max_gaussians: int
    clone_jitter_fraction: float
    clone_confidence_fraction: float
    split_children: int
    split_offset_fraction: float
    split_scale_factor: float
    minimum_confidence: float


@dataclass
class DensificationResult:
    means: Tensor
    scales: Tensor
    quaternions: Tensor
    confidences: Tensor
    features: Tensor
    selected_count: int
    cloned_count: int
    split_parent_count: int
    new_count: int
    final_count: int


def should_densify(step: int, config: DensificationConfig) -> bool:
    _validate_config(config)

    if not config.enabled:
        return False

    if step < config.from_step or step > config.until_step:
        return False

    return (step - config.from_step) % config.interval == 0


def effective_gradient_threshold(
    *,
    current_count: int,
    initial_count: int,
    config: DensificationConfig,
) -> float:
    _validate_config(config)

    if initial_count <= 0:
        raise ValueError(f"initial_count must be positive, got {initial_count}.")

    ratio = current_count / initial_count
    scale = ratio ** config.population_exponent

    return config.gradient_threshold * scale


def densify_gaussians(
    model: GaussianVolumeModel,
    *,
    initial_count: int,
    config: DensificationConfig,
    volume_shape: Sequence[int],
) -> DensificationResult:
    _validate_config(config)

    if model.means.grad is None:
        return _unchanged_result(model)

    current_count = model.num_gaussians
    threshold = effective_gradient_threshold(
        current_count=current_count,
        initial_count=initial_count,
        config=config,
    )

    mean_gradient_norm = model.means.grad.norm(dim=-1)
    candidate_mask = mean_gradient_norm >= threshold

    if config.max_new_gaussians > 0:
        room_left = config.max_gaussians - current_count
        max_candidates = min(config.max_new_gaussians, max(room_left, 0))

        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        if candidate_indices.numel() > max_candidates:
            candidate_scores = mean_gradient_norm[candidate_indices]
            top_indices = torch.topk(candidate_scores, k=max_candidates).indices
            selected_indices = candidate_indices[top_indices]

            candidate_mask = torch.zeros_like(candidate_mask)
            candidate_mask[selected_indices] = True

    if not candidate_mask.any():
        return _unchanged_result(model)

    scales = model.scales.detach()
    max_scale = scales.max(dim=-1).values

    clone_mask = candidate_mask & (max_scale <= config.clone_scale_threshold)
    split_mask = candidate_mask & (max_scale > config.split_scale_threshold)

    # All of means/scales/quaternions/confidences/features assembled below
    # are in PHYSICAL space (matching model.scales/model.confidences/
    # model.features properties, not the raw pre-transform parameters) —
    # DensificationResult feeds directly into a fresh GaussianVolumeModel
    # constructor call (see training.py's rebuild_model), whose scales/
    # confidences/features arguments are physical-space values that get
    # inverse-transformed internally.
    means = model.means.detach()
    quaternions = model.normalized_quaternions.detach()
    confidences = model.confidences.detach()
    features = model.features.detach()

    keep_mask = ~split_mask

    new_means_parts = [means[keep_mask]]
    new_scales_parts = [scales[keep_mask]]
    new_quaternions_parts = [quaternions[keep_mask]]
    new_confidences_parts = [confidences[keep_mask]]
    new_features_parts = [features[keep_mask]]

    cloned_count = int(clone_mask.sum().item())
    if cloned_count > 0:
        clone_means = means[clone_mask]
        clone_scales = scales[clone_mask]
        clone_quaternions = quaternions[clone_mask]
        clone_confidences = confidences[clone_mask]
        clone_features = features[clone_mask]

        clone_scale_magnitude = clone_scales.max(dim=-1, keepdim=True).values
        jitter = torch.randn_like(clone_means) * clone_scale_magnitude * config.clone_jitter_fraction

        new_means_parts.append(clone_means + jitter)
        new_scales_parts.append(clone_scales)
        new_quaternions_parts.append(clone_quaternions)
        new_confidences_parts.append(clone_confidences * config.clone_confidence_fraction)
        new_features_parts.append(clone_features)

    split_parent_count = int(split_mask.sum().item())
    if split_parent_count > 0:
        parent_means = means[split_mask]
        parent_scales = scales[split_mask]
        parent_quaternions = quaternions[split_mask]
        parent_confidences = confidences[split_mask]
        parent_features = features[split_mask]

        rotations = quaternion_to_rotation_matrix(parent_quaternions, eps=1e-8)
        principal_axis = rotations[:, :, 0]
        offset_magnitude = parent_scales[:, 0:1] * config.split_offset_fraction

        for _ in range(config.split_children):
            sign = torch.randint(
                0, 2, (parent_means.shape[0], 1), device=parent_means.device
            ).float() * 2 - 1

            child_means = parent_means + principal_axis * offset_magnitude * sign
            child_scales = parent_scales * config.split_scale_factor

            new_means_parts.append(child_means)
            new_scales_parts.append(child_scales)
            new_quaternions_parts.append(parent_quaternions)
            new_confidences_parts.append(parent_confidences)
            new_features_parts.append(parent_features)

    final_means = torch.cat(new_means_parts, dim=0)
    final_scales = torch.cat(new_scales_parts, dim=0)
    final_quaternions = torch.cat(new_quaternions_parts, dim=0)
    final_confidences = torch.cat(new_confidences_parts, dim=0)
    final_features = torch.cat(new_features_parts, dim=0)

    final_means = clamp_means(final_means, volume_shape)

    new_count = final_means.shape[0] - current_count

    return DensificationResult(
        means=final_means,
        scales=final_scales,
        quaternions=final_quaternions,
        confidences=final_confidences,
        features=final_features,
        selected_count=int(candidate_mask.sum().item()),
        cloned_count=cloned_count,
        split_parent_count=split_parent_count,
        new_count=new_count,
        final_count=final_means.shape[0],
    )


def _clone_gaussians(*args, **kwargs):
    raise NotImplementedError(
        "Cloning is performed inline within densify_gaussians."
    )


def _split_gaussians(*args, **kwargs):
    raise NotImplementedError(
        "Splitting is performed inline within densify_gaussians."
    )


def clamp_means(means: Tensor, shape: Sequence[int]) -> Tensor:
    depth, height, width = shape

    clamped = means.clone()
    clamped[:, 0].clamp_(0.0, width)
    clamped[:, 1].clamp_(0.0, height)
    clamped[:, 2].clamp_(0.0, depth)

    return clamped


def _unchanged_result(model: GaussianVolumeModel) -> DensificationResult:
    n = model.num_gaussians

    return DensificationResult(
        means=model.means.detach().clone(),
        scales=model.scales.detach().clone(),
        quaternions=model.normalized_quaternions.detach().clone(),
        confidences=model.confidences.detach().clone(),
        features=model.features.detach().clone(),
        selected_count=0,
        cloned_count=0,
        split_parent_count=0,
        new_count=0,
        final_count=n,
    )


def _validate_config(config: DensificationConfig) -> None:
    if config.interval <= 0:
        raise ValueError(f"interval must be positive, got {config.interval}.")

    if config.from_step < 0:
        raise ValueError(f"from_step must be non-negative, got {config.from_step}.")

    if config.until_step < config.from_step:
        raise ValueError("until_step must be >= from_step.")

    if config.max_gaussians <= 0:
        raise ValueError(f"max_gaussians must be positive, got {config.max_gaussians}.")

    if config.split_children <= 0:
        raise ValueError(
            f"split_children must be positive, got {config.split_children}."
        )
