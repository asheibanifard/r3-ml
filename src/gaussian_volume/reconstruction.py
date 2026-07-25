# gaussian_volume/reconstruction.py

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.autograd import Function

from .extension.loader import load_gaussian_volume_extension


def _validate_tensor(
    tensor: Tensor,
    *,
    name: str,
    shape_suffix: tuple[int, ...],
    ndim: int,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got {tensor.ndim}."
        )

    if shape_suffix and tensor.shape[-len(shape_suffix):] != shape_suffix:
        raise ValueError(
            f"{name} must have trailing shape {shape_suffix}, "
            f"got {tuple(tensor.shape)}."
        )

    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")

    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}.")


def _validate_inputs(
    means: Tensor,
    precision6: Tensor,
    confidences: Tensor,
    features: Tensor,
) -> None:
    _validate_tensor(means, name="means", shape_suffix=(3,), ndim=2)
    _validate_tensor(precision6, name="precision6", shape_suffix=(6,), ndim=2)
    _validate_tensor(confidences, name="confidences", shape_suffix=(), ndim=1)
    _validate_tensor(features, name="features", shape_suffix=(), ndim=1)

    n = means.shape[0]
    if precision6.shape[0] != n:
        raise ValueError("precision6 must have the same N as means.")
    if confidences.shape[0] != n:
        raise ValueError("confidences must have the same N as means.")
    if features.shape[0] != n:
        raise ValueError("features must have the same N as means.")


def _validate_shape(shape: Sequence[int]) -> None:
    if len(shape) != 3:
        raise ValueError(f"shape must have 3 elements [D,H,W], got {shape}.")

    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"All entries of shape must be positive, got {shape}.")


class _GaussianVolumeFunction(Function):
    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        precision6: Tensor,
        confidences: Tensor,
        features: Tensor,
        depth: int,
        height: int,
        width: int,
        cutoff_squared: float,
        epsilon: float,
    ) -> Tensor:
        extension = load_gaussian_volume_extension()

        means_c = means.contiguous()
        precision6_c = precision6.contiguous()
        confidences_c = confidences.contiguous()
        features_c = features.contiguous()

        output = extension.forward(
            means_c,
            precision6_c,
            confidences_c,
            features_c,
            depth,
            height,
            width,
            cutoff_squared,
            epsilon,
        )

        ctx.save_for_backward(means_c, precision6_c, confidences_c, features_c)
        ctx.cutoff_squared = cutoff_squared
        ctx.epsilon = epsilon

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        means, precision6, confidences, features = ctx.saved_tensors
        extension = load_gaussian_volume_extension()

        grad_means, grad_precision6, grad_confidences, grad_features = extension.backward(
            grad_output.contiguous(),
            means,
            precision6,
            confidences,
            features,
            ctx.cutoff_squared,
            ctx.epsilon,
        )

        return (
            grad_means,
            grad_precision6,
            grad_confidences,
            grad_features,
            None,
            None,
            None,
            None,
            None,
        )


def reconstruct_volume(
    means: Tensor,
    precision6: Tensor,
    confidences: Tensor,
    features: Tensor,
    shape: Sequence[int],
    cutoff_sigma: float,
    epsilon: float,
) -> Tensor:
    """
    Reconstruct a dense [D,H,W] voxel grid from a Gaussian mixture using the
    compiled CUDA extension.

    V(x) = sum_i w_i(x) f_i / (sum_i w_i(x) + epsilon)
    w_i(x) = confidence_i * exp(-0.5 * mahalanobis_squared_i(x))

    Gaussians whose mahalanobis distance from a voxel exceeds cutoff_sigma
    standard deviations are ignored for that voxel (both value and gradient).
    """
    _validate_inputs(means, precision6, confidences, features)
    _validate_shape(shape)

    if cutoff_sigma <= 0:
        raise ValueError(f"cutoff_sigma must be positive, got {cutoff_sigma}.")

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    depth, height, width = shape
    cutoff_squared = float(cutoff_sigma) ** 2

    return _GaussianVolumeFunction.apply(
        means,
        precision6,
        confidences,
        features,
        depth,
        height,
        width,
        cutoff_squared,
        epsilon,
    )


def reconstruct_volume_reference(
    means: Tensor,
    precision6: Tensor,
    confidences: Tensor,
    features: Tensor,
    shape: Sequence[int],
    cutoff_sigma: float,
    epsilon: float,
    voxel_chunk_size: int,
) -> Tensor:
    """
    Pure-PyTorch reference implementation of reconstruct_volume.

    Used for CPU execution, unit testing, and cross-checking the CUDA
    extension. Processes voxels in chunks of voxel_chunk_size to bound peak
    memory usage for large volumes.
    """
    _validate_inputs(means, precision6, confidences, features)
    _validate_shape(shape)

    if cutoff_sigma <= 0:
        raise ValueError(f"cutoff_sigma must be positive, got {cutoff_sigma}.")

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    if voxel_chunk_size <= 0:
        raise ValueError(
            f"voxel_chunk_size must be positive, got {voxel_chunk_size}."
        )

    depth, height, width = shape
    device = means.device
    dtype = means.dtype
    cutoff_squared = float(cutoff_sigma) ** 2

    from .math3d import compact6_to_symmetric_matrix

    precision_full = compact6_to_symmetric_matrix(precision6)

    z_coords = torch.arange(depth, device=device, dtype=dtype) + 0.5
    y_coords = torch.arange(height, device=device, dtype=dtype) + 0.5
    x_coords = torch.arange(width, device=device, dtype=dtype) + 0.5

    grid_z, grid_y, grid_x = torch.meshgrid(
        z_coords, y_coords, x_coords, indexing="ij"
    )

    points = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)],
        dim=-1,
    )

    total_voxels = points.shape[0]
    output = torch.empty(total_voxels, device=device, dtype=dtype)

    for start in range(0, total_voxels, voxel_chunk_size):
        end = min(start + voxel_chunk_size, total_voxels)
        chunk_points = points[start:end]

        delta = chunk_points[:, None, :] - means[None, :, :]
        mahalanobis = torch.einsum(
            "mni,nij,mnj->mn", delta, precision_full, delta
        )

        mask = mahalanobis < cutoff_squared
        weights = confidences[None, :] * torch.exp(-0.5 * mahalanobis) * mask

        numerator = (weights * features[None, :]).sum(dim=-1)
        denominator = weights.sum(dim=-1) + epsilon

        output[start:end] = numerator / denominator

    return output.reshape(depth, height, width)
