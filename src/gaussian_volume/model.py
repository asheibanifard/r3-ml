# gaussian_volume/model.py

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import math3d
from . import reconstruction


def inverse_softplus(value: Tensor, minimum: float) -> Tensor:
    """
    Inverse of softplus(x) + minimum, solved elementwise for x.

    Used by the constructor to convert a physical-space value (e.g. a
    desired scale or confidence) into the raw parameter that reproduces it
    exactly under softplus(raw) + minimum — so both fresh initialization
    (physical `initial_scale`/`initial_confidence`) and checkpoint reload
    (physical values from `export_gaussians()`) can pass the same
    physical-space tensors into the constructor.
    """
    if torch.any(value <= minimum):
        raise ValueError(
            f"All values must be greater than minimum ({minimum})."
        )

    shifted = value - minimum
    return shifted + torch.log(-torch.expm1(-shifted))


def _inverse_sigmoid_affine(value: Tensor, minimum: float, maximum: float) -> Tensor:
    """
    Inverse of sigmoid(x) * (maximum - minimum) + minimum, solved for x.
    """
    span = maximum - minimum
    normalized = ((value - minimum) / span).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(normalized)


class GaussianVolumeModel(nn.Module):
    """
    A mixture of anisotropic 3D Gaussians reconstructing a dense voxel grid.

    V(x) = sum_i w_i(x) f_i / (sum_i w_i(x) + epsilon)
    w_i(x) = confidence_i * exp(-0.5 * (x - mean_i)^T Q_i (x - mean_i))

    All constructor arguments are required — this codebase's convention is
    that no numerical defaults appear in function/constructor signatures;
    every value must be supplied explicitly by the caller (config or CLI).
    """

    def __init__(
        self,
        means: Tensor,
        scales: Tensor,
        features: Tensor,
        confidences: Tensor,
        quaternions: Tensor,
        *,
        minimum_scale: float,
        minimum_confidence: float,
        constrain_features: bool,
        feature_min: float,
        feature_max: float,
    ) -> None:
        super().__init__()

        self._validate_constructor_inputs(
            means, scales, features, confidences, quaternions
        )

        if minimum_scale <= 0:
            raise ValueError(f"minimum_scale must be positive, got {minimum_scale}.")

        if minimum_confidence < 0:
            raise ValueError(
                f"minimum_confidence must be non-negative, got {minimum_confidence}."
            )

        if constrain_features and feature_min >= feature_max:
            raise ValueError(
                "feature_min must be less than feature_max, got "
                f"{feature_min} and {feature_max}."
            )

        # scales/confidences/features arrive in physical space (the same
        # space `initialize_gaussians` and `export_gaussians` both use), so
        # every construction path — fresh init or checkpoint reload — is
        # inverse-transformed here into raw storage once, uniformly.
        raw_scales = inverse_softplus(scales, minimum_scale)
        raw_confidences = inverse_softplus(confidences, minimum_confidence)

        if constrain_features:
            raw_features = _inverse_sigmoid_affine(features, feature_min, feature_max)
        else:
            raw_features = features.clone()

        self.means = nn.Parameter(means.clone())
        self.raw_scales = nn.Parameter(raw_scales)
        self.raw_features = nn.Parameter(raw_features)
        self.raw_confidences = nn.Parameter(raw_confidences)
        self.quats = nn.Parameter(quaternions.clone())

        self.minimum_scale = minimum_scale
        self.minimum_confidence = minimum_confidence
        self.constrain_features = constrain_features
        self.feature_min = feature_min
        self.feature_max = feature_max

    @staticmethod
    def _validate_constructor_inputs(
        means: Tensor,
        scales: Tensor,
        features: Tensor,
        confidences: Tensor,
        quaternions: Tensor,
    ) -> None:
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError(f"means must have shape [N,3], got {tuple(means.shape)}.")

        n = means.shape[0]

        if scales.shape != (n, 3):
            raise ValueError(f"scales must have shape [{n},3], got {tuple(scales.shape)}.")

        if features.shape != (n,):
            raise ValueError(f"features must have shape [{n}], got {tuple(features.shape)}.")

        if confidences.shape != (n,):
            raise ValueError(
                f"confidences must have shape [{n}], got {tuple(confidences.shape)}."
            )

        if quaternions.shape != (n, 4):
            raise ValueError(
                f"quaternions must have shape [{n},4], got {tuple(quaternions.shape)}."
            )

        tensors = (means, scales, features, confidences, quaternions)
        devices = {t.device for t in tensors}
        if len(devices) != 1:
            raise ValueError("All constructor tensors must be on the same device.")

        dtypes = {t.dtype for t in tensors}
        if len(dtypes) != 1:
            raise ValueError("All constructor tensors must share the same dtype.")

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    @property
    def scales(self) -> Tensor:
        return F.softplus(self.raw_scales) + self.minimum_scale

    @property
    def normalized_quaternions(self) -> Tensor:
        return math3d.normalize_quaternion(self.quats, eps=1e-8)

    @property
    def confidences(self) -> Tensor:
        return F.softplus(self.raw_confidences) + self.minimum_confidence

    @property
    def features(self) -> Tensor:
        if self.constrain_features:
            span = self.feature_max - self.feature_min
            return torch.sigmoid(self.raw_features) * span + self.feature_min
        return self.raw_features

    @property
    def precision6(self) -> Tensor:
        return math3d.precision_from_scale_rotation(
            self.scales,
            self.normalized_quaternions,
            minimum_scale=1e-8,
        )

    def forward(
        self,
        shape: Sequence[int],
        cutoff_sigma: float,
        epsilon: float,
    ) -> Tensor:
        self._validate_output_shape(shape)

        return reconstruction.reconstruct_volume(
            self.means,
            self.precision6,
            self.confidences,
            self.features,
            shape,
            cutoff_sigma,
            epsilon,
        )

    @staticmethod
    def _validate_output_shape(shape: Sequence[int]) -> None:
        if len(shape) != 3:
            raise ValueError(f"shape must have 3 elements [D,H,W], got {shape}.")

        if any(dimension <= 0 for dimension in shape):
            raise ValueError(f"All entries of shape must be positive, got {shape}.")

    def clamp_means_to_volume(self, shape: Sequence[int], margin: float) -> None:
        depth, height, width = shape

        with torch.no_grad():
            self.means[:, 0].clamp_(margin, width - margin)
            self.means[:, 1].clamp_(margin, height - margin)
            self.means[:, 2].clamp_(margin, depth - margin)

    def parameter_groups(
        self,
        *,
        mean_lr: float,
        scale_lr: float,
        rotation_lr: float,
        confidence_lr: float,
        feature_lr: float,
    ) -> list[dict]:
        return [
            {"params": [self.means], "lr": mean_lr, "name": "means"},
            {"params": [self.raw_scales], "lr": scale_lr, "name": "scales"},
            {"params": [self.quats], "lr": rotation_lr, "name": "quats"},
            {"params": [self.raw_confidences], "lr": confidence_lr, "name": "confidences"},
            {"params": [self.raw_features], "lr": feature_lr, "name": "features"},
        ]

    def export_gaussians(self) -> dict[str, Tensor]:
        return {
            "means": self.means.detach().clone(),
            "scales": self.scales.detach().clone(),
            "quaternions": self.normalized_quaternions.detach().clone(),
            "confidences": self.confidences.detach().clone(),
            "features": self.features.detach().clone(),
        }

    def extra_repr(self) -> str:
        return (
            f"num_gaussians={self.num_gaussians}, "
            f"minimum_scale={self.minimum_scale}, "
            f"minimum_confidence={self.minimum_confidence}, "
            f"constrain_features={self.constrain_features}"
        )
