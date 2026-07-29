"""
Volume reconstruction from anisotropic 3D Gaussian mixtures.

This module provides both CUDA-accelerated and pure-PyTorch implementations of
Gaussian volume reconstruction:

    V(x) = Σᵢ wᵢ(x) · fᵢ / (Σᵢ wᵢ(x) + ε)
    wᵢ(x) = confidenceᵢ · exp(-0.5 · (x - μᵢ)ᵀ Qᵢ (x - μᵢ))

where:
- wᵢ(x): Gaussian weight at point x (exponential decay from mean)
- fᵢ: Feature value (intensity/color) of Gaussian i
- Qᵢ: Precision matrix (inverse covariance)
- ε: Numerical stability constant

Key components:
- `reconstruct_volume()`: Fast CUDA-accelerated reconstruction (default).
- `reconstruct_volume_reference()`: Pure-PyTorch reference (CPU-compatible, slower).
- `_GaussianVolumeFunction`: Autograd wrapper for CUDA kernel with gradient support.
- Input validation and shape handling.

Both implementations support:
- Cutoff distance: Ignore Gaussians beyond cutoff_sigma standard deviations.
- Differentiability: Full gradient support for training.
- Arbitrary output shapes: Reconstruct at any resolution.
"""

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
    require_cuda: bool = True,
) -> None:
    """
    Validate tensor for reconstruction operations.

    Checks:
    - Is a torch.Tensor
    - Has exactly ndim dimensions
    - Ends with expected shape (e.g., last dim is 3 for 3D vectors)
    - Optionally requires GPU (CUDA) for kernel-based reconstruction
    - Is float32 precision

    Parameters
    ----------
    tensor : Tensor
        Tensor to validate.
    name : str
        Tensor name for error messages.
    shape_suffix : tuple[int, ...]
        Expected trailing shape dimensions (e.g., (3,) for 3D vectors).
    ndim : int
        Expected total number of dimensions.
    require_cuda : bool
        If True, tensor must be on GPU (for CUDA kernel path).
        If False, tensor can be on CPU or GPU (for pure-PyTorch path).

    Raises
    ------
    TypeError
        If tensor is not a torch.Tensor.
    ValueError
        If shape or device/dtype requirements are not met.
    """
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

    if require_cuda and not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")

    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}.")


def _validate_inputs(
    means: Tensor,
    precision6: Tensor,
    confidences: Tensor,
    features: Tensor,
    require_cuda: bool = True,
) -> None:
    """
    Validate all Gaussian parameters for reconstruction.

    Checks:
    - All tensors are float32
    - All tensors on same device (CPU or GPU)
    - All tensors have consistent Gaussian count N
    - means: [N, 3]
    - precision6: [N, 6] (compact symmetric matrices)
    - confidences: [N]
    - features: [N]

    Parameters
    ----------
    means : Tensor
        Gaussian centers, shape [N, 3].
    precision6 : Tensor
        Compact precision matrices, shape [N, 6].
    confidences : Tensor
        Gaussian weights, shape [N].
    features : Tensor
        Feature values, shape [N].
    require_cuda : bool
        If True, all tensors must be CUDA (for CUDA kernel path).
        If False, tensors can be CPU or CUDA (for pure-PyTorch reference).

    Raises
    ------
    TypeError
        If any tensor is not torch.Tensor.
    ValueError
        If shapes or device/dtype don't match.
    """
    _validate_tensor(means, name="means", shape_suffix=(3,), ndim=2, require_cuda=require_cuda)
    _validate_tensor(precision6, name="precision6", shape_suffix=(6,), ndim=2, require_cuda=require_cuda)
    _validate_tensor(confidences, name="confidences", shape_suffix=(), ndim=1, require_cuda=require_cuda)
    _validate_tensor(features, name="features", shape_suffix=(), ndim=1, require_cuda=require_cuda)

    n = means.shape[0]
    if precision6.shape[0] != n:
        raise ValueError("precision6 must have the same N as means.")
    if confidences.shape[0] != n:
        raise ValueError("confidences must have the same N as means.")
    if features.shape[0] != n:
        raise ValueError("features must have the same N as means.")

    # Verify all tensors are on the same device
    devices = {t.device for t in (means, precision6, confidences, features)}
    if len(devices) > 1:
        raise ValueError("All tensors must be on the same device.")


def _validate_shape(shape: Sequence[int]) -> None:
    """
    Validate output volume shape.

    Parameters
    ----------
    shape : Sequence[int]
        Volume shape [D, H, W]. All entries must be positive.

    Raises
    ------
    ValueError
        If shape has != 3 elements or contains non-positive dimensions.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must have 3 elements [D,H,W], got {shape}.")

    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"All entries of shape must be positive, got {shape}.")


class _GaussianVolumeFunction(Function):
    """
    Autograd Function wrapping CUDA kernel for Gaussian volume reconstruction.

    Implements forward and backward passes for the Gaussian mixture volume formula:

        V(x) = Σᵢ wᵢ(x) · fᵢ / (Σᵢ wᵢ(x) + ε)
        wᵢ(x) = confidenceᵢ · exp(-0.5 · (x - μᵢ)ᵀ Qᵢ (x - μᵢ))

    The forward pass delegates to the JIT-compiled CUDA extension; backward
    computes gradients w.r.t. means, precision, confidences, and features.
    """

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
        """
        Reconstruct volume via CUDA kernel.

        Parameters
        ----------
        ctx : Context
            Autograd context for gradient bookkeeping.
        means : Tensor
            [N, 3] Gaussian centers (CUDA float32).
        precision6 : Tensor
            [N, 6] Compact precision matrices (CUDA float32).
        confidences : Tensor
            [N] Gaussian weights (CUDA float32).
        features : Tensor
            [N] Feature values (CUDA float32).
        depth, height, width : int
            Output volume shape.
        cutoff_squared : float
            Squared cutoff distance (Mahalanobis) in standard deviations.
        epsilon : float
            Numerical stability constant.

        Returns
        -------
        Tensor
            Reconstructed volume [D, H, W], dtype float32, on GPU.
        """
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
    _validate_inputs(means, precision6, confidences, features, require_cuda=True)
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
    Pure-PyTorch reference implementation of Gaussian volume reconstruction.

    Reconstructs the volume using NumPy-like operations (einsum, masking, etc.)
    instead of a compiled CUDA kernel. Fully differentiable but slower than
    the CUDA version.

    Used for:
    - CPU execution (when GPU unavailable)
    - GPU execution (alternative to CUDA kernel, always works)
    - Unit testing and validation
    - Cross-checking CUDA kernel correctness
    - Debugging gradient computation

    Processes voxels in chunks to bound peak memory usage for large volumes.

    Parameters
    ----------
    means : Tensor
        Gaussian centers, shape [N, 3]. Can be on CPU or GPU.
    precision6 : Tensor
        Compact precision matrices, shape [N, 6]. Must be on same device as means.
    confidences : Tensor
        Gaussian weights, shape [N]. Must be on same device as means.
    features : Tensor
        Feature values, shape [N]. Must be on same device as means.
    shape : Sequence[int]
        Output volume shape [D, H, W].
    cutoff_sigma : float
        Cutoff distance in standard deviations. Gaussians beyond this distance
        contribute zero weight. Typical: 3.0.
    epsilon : float
        Numerical stability constant (added to denominator).
    voxel_chunk_size : int
        Process voxels in chunks of this size to limit memory usage.
        Larger chunks are faster but use more memory. Typical: 65536 (256³/4).

    Returns
    -------
    Tensor
        Reconstructed volume [D, H, W], same dtype/device as inputs.

    Notes
    -----
    Implementation:
    1. Expand Gaussian centers to full 3×3 precision matrices (compact6 → full).
    2. Create voxel coordinate grid.
    3. For each chunk of voxels:
       a. Compute Mahalanobis squared distance from each voxel to each Gaussian.
       b. Mask out Gaussians beyond cutoff_sigma.
       c. Compute weights = confidence * exp(-0.5 * distance).
       d. Blend features: output = Σ(weight * feature) / (Σweight + ε).
    """
    _validate_inputs(means, precision6, confidences, features, require_cuda=False)
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
