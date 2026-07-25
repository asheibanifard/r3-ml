# gaussian_volume/math3d.py

from __future__ import annotations

import torch
from torch import Tensor


def normalize_quaternion(
    quaternions: Tensor,
    eps: float,
) -> Tensor:
    """
    Normalize quaternions to unit length.

    Parameters
    ----------
    quaternions:
        Tensor with shape [N, 4].

        Quaternion order is:

            [w, x, y, z]

    eps:
        Numerical stability value.

    Returns
    -------
    Tensor
        Unit quaternions with shape [N, 4].
    """
    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError(
            "quaternions must have shape [N,4], "
            f"got {tuple(quaternions.shape)}."
        )

    norms = torch.linalg.vector_norm(
        quaternions,
        dim=-1,
        keepdim=True,
    )

    if torch.any(norms <= eps):
        raise ValueError(
            "Every quaternion must have a nonzero norm."
        )

    return quaternions / norms.clamp_min(eps)


def quaternion_to_rotation_matrix(
    quaternions: Tensor,
    eps: float,
) -> Tensor:
    """
    Convert quaternions into 3x3 rotation matrices.

    Quaternion convention:

        q = [w, x, y, z]

    Parameters
    ----------
    quaternions:
        Tensor with shape [N, 4].

    eps:
        Numerical stability value used during normalization.

    Returns
    -------
    Tensor
        Rotation matrices with shape [N, 3, 3].
    """
    q = normalize_quaternion(
        quaternions,
        eps=eps,
    )

    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z
    yz = y * z

    r00 = ww + xx - yy - zz
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = ww - xx - yy + zz

    row0 = torch.stack(
        [r00, r01, r02],
        dim=-1,
    )

    row1 = torch.stack(
        [r10, r11, r12],
        dim=-1,
    )

    row2 = torch.stack(
        [r20, r21, r22],
        dim=-1,
    )

    return torch.stack(
        [row0, row1, row2],
        dim=-2,
    )


def covariance_from_scale_rotation(
    scales: Tensor,
    quaternions: Tensor,
    minimum_scale: float,
) -> Tensor:
    """
    Construct full 3D covariance matrices.

    For each Gaussian:

        Sigma = R diag(sx^2, sy^2, sz^2) R^T

    Parameters
    ----------
    scales:
        Positive anisotropic Gaussian standard deviations.

        Shape:
            [N, 3]

    quaternions:
        Quaternion rotations in [w,x,y,z] order.

        Shape:
            [N, 4]

    minimum_scale:
        Numerical lower bound for scales.

    Returns
    -------
    Tensor
        Covariance matrices with shape [N, 3, 3].
    """
    _validate_scale_rotation_inputs(
        scales,
        quaternions,
    )

    if minimum_scale <= 0:
        raise ValueError(
            "minimum_scale must be positive, "
            f"got {minimum_scale}."
        )

    safe_scales = scales.clamp_min(
        minimum_scale
    )

    rotations = quaternion_to_rotation_matrix(
        quaternions,
        eps=1e-8,
    )

    variances = safe_scales.square()

    diagonal_covariance = torch.diag_embed(
        variances
    )

    return (
        rotations
        @ diagonal_covariance
        @ rotations.transpose(-1, -2)
    )


def precision_from_scale_rotation_full(
    scales: Tensor,
    quaternions: Tensor,
    minimum_scale: float,
) -> Tensor:
    """
    Construct full 3D precision matrices directly.

    Since:

        Sigma = R diag(s^2) R^T

    and R is orthonormal:

        Sigma^{-1}
        =
        R diag(1 / s^2) R^T

    This direct construction avoids explicitly inverting covariance
    matrices.

    Parameters
    ----------
    scales:
        Positive anisotropic standard deviations.

        Shape:
            [N, 3]

    quaternions:
        Rotations in [w,x,y,z] order.

        Shape:
            [N, 4]

    minimum_scale:
        Numerical lower bound.

    Returns
    -------
    Tensor
        Precision matrices with shape [N, 3, 3].
    """
    _validate_scale_rotation_inputs(
        scales,
        quaternions,
    )

    if minimum_scale <= 0:
        raise ValueError(
            "minimum_scale must be positive, "
            f"got {minimum_scale}."
        )

    safe_scales = scales.clamp_min(
        minimum_scale
    )

    rotations = quaternion_to_rotation_matrix(
        quaternions,
        eps=1e-8,
    )

    inverse_variances = safe_scales.reciprocal().square()

    diagonal_precision = torch.diag_embed(
        inverse_variances
    )

    precision = (
        rotations
        @ diagonal_precision
        @ rotations.transpose(-1, -2)
    )

    # Remove very small asymmetries caused by floating-point arithmetic.
    return 0.5 * (
        precision
        + precision.transpose(-1, -2)
    )


def symmetric_matrix_to_compact6(
    matrices: Tensor,
) -> Tensor:
    """
    Convert symmetric 3x3 matrices into six-value storage.

    Input matrix:

        [ qxx  qxy  qxz ]
        [ qxy  qyy  qyz ]
        [ qxz  qyz  qzz ]

    Output ordering:

        [qxx, qyy, qzz, qxy, qxz, qyz]

    Parameters
    ----------
    matrices:
        Tensor with shape [N,3,3].

    Returns
    -------
    Tensor
        Compact tensor with shape [N,6].
    """
    if matrices.ndim != 3:
        raise ValueError(
            "matrices must have shape [N,3,3], "
            f"got {tuple(matrices.shape)}."
        )

    if matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "matrices must have shape [N,3,3], "
            f"got {tuple(matrices.shape)}."
        )

    qxx = matrices[:, 0, 0]
    qyy = matrices[:, 1, 1]
    qzz = matrices[:, 2, 2]

    qxy = 0.5 * (
        matrices[:, 0, 1]
        + matrices[:, 1, 0]
    )

    qxz = 0.5 * (
        matrices[:, 0, 2]
        + matrices[:, 2, 0]
    )

    qyz = 0.5 * (
        matrices[:, 1, 2]
        + matrices[:, 2, 1]
    )

    return torch.stack(
        [qxx, qyy, qzz, qxy, qxz, qyz],
        dim=-1,
    )


def compact6_to_symmetric_matrix(
    compact: Tensor,
) -> Tensor:
    """
    Convert compact six-value matrices into full symmetric 3x3 form.

    Input ordering:

        [qxx, qyy, qzz, qxy, qxz, qyz]

    Parameters
    ----------
    compact:
        Tensor with shape [N,6].

    Returns
    -------
    Tensor
        Symmetric matrices with shape [N,3,3].
    """
    if compact.ndim != 2 or compact.shape[-1] != 6:
        raise ValueError(
            "compact must have shape [N,6], "
            f"got {tuple(compact.shape)}."
        )

    qxx, qyy, qzz, qxy, qxz, qyz = compact.unbind(
        dim=-1
    )

    row0 = torch.stack(
        [qxx, qxy, qxz],
        dim=-1,
    )

    row1 = torch.stack(
        [qxy, qyy, qyz],
        dim=-1,
    )

    row2 = torch.stack(
        [qxz, qyz, qzz],
        dim=-1,
    )

    return torch.stack(
        [row0, row1, row2],
        dim=-2,
    )


def precision_from_scale_rotation(
    scales: Tensor,
    quaternions: Tensor,
    minimum_scale: float,
) -> Tensor:
    """
    Construct compact six-value precision matrices.

    This is the main function used by model.py.

    For each Gaussian:

        Q = R diag(
            1/sx^2,
            1/sy^2,
            1/sz^2
        ) R^T

    Parameters
    ----------
    scales:
        Shape [N,3].

    quaternions:
        Shape [N,4], in [w,x,y,z] order.

    minimum_scale:
        Lower numerical scale bound.

    Returns
    -------
    Tensor
        Compact precision matrices with shape [N,6].

        Ordering:

            [qxx, qyy, qzz, qxy, qxz, qyz]
    """
    full_precision = precision_from_scale_rotation_full(
        scales=scales,
        quaternions=quaternions,
        minimum_scale=minimum_scale,
    )

    return symmetric_matrix_to_compact6(
        full_precision
    )


def mahalanobis_squared(
    points: Tensor,
    means: Tensor,
    precision6: Tensor,
) -> Tensor:
    """
    PyTorch reference implementation of squared Mahalanobis distance.

    Parameters
    ----------
    points:
        Query points with shape [M,3].

    means:
        Gaussian centres with shape [N,3].

    precision6:
        Compact precision matrices with shape [N,6].

    Returns
    -------
    Tensor
        Pairwise squared distances with shape [M,N].
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(
            "points must have shape [M,3], "
            f"got {tuple(points.shape)}."
        )

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(
            "means must have shape [N,3], "
            f"got {tuple(means.shape)}."
        )

    if precision6.ndim != 2 or precision6.shape[-1] != 6:
        raise ValueError(
            "precision6 must have shape [N,6], "
            f"got {tuple(precision6.shape)}."
        )

    if means.shape[0] != precision6.shape[0]:
        raise ValueError(
            "means and precision6 must contain the same "
            "number of Gaussians."
        )

    delta = (
        points[:, None, :]
        - means[None, :, :]
    )

    precision = compact6_to_symmetric_matrix(
        precision6
    )

    return torch.einsum(
        "mni,nij,mnj->mn",
        delta,
        precision,
        delta,
    )


def _validate_scale_rotation_inputs(
    scales: Tensor,
    quaternions: Tensor,
) -> None:
    if not isinstance(scales, Tensor):
        raise TypeError(
            "scales must be a torch.Tensor."
        )

    if not isinstance(quaternions, Tensor):
        raise TypeError(
            "quaternions must be a torch.Tensor."
        )

    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError(
            "scales must have shape [N,3], "
            f"got {tuple(scales.shape)}."
        )

    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError(
            "quaternions must have shape [N,4], "
            f"got {tuple(quaternions.shape)}."
        )

    if scales.shape[0] != quaternions.shape[0]:
        raise ValueError(
            "scales and quaternions must contain the same "
            "number of Gaussians, got "
            f"{scales.shape[0]} and {quaternions.shape[0]}."
        )

    if scales.device != quaternions.device:
        raise ValueError(
            "scales and quaternions must be on the same device."
        )

    if scales.dtype != quaternions.dtype:
        raise ValueError(
            "scales and quaternions must have the same dtype."
        )

    if not torch.is_floating_point(scales):
        raise TypeError(
            "scales must use a floating-point dtype."
        )

    if not torch.is_floating_point(quaternions):
        raise TypeError(
            "quaternions must use a floating-point dtype."
        )

    if not torch.isfinite(scales).all():
        raise ValueError(
            "scales contains NaN or infinite values."
        )

    if not torch.isfinite(quaternions).all():
        raise ValueError(
            "quaternions contains NaN or infinite values."
        )

    if torch.any(scales <= 0):
        raise ValueError(
            "All Gaussian scales must be positive."
        )
