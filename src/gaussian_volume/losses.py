# Enables postponed evaluation of type hints (so annotations like torch.Tensor
# don't need torch imported at module load time for the hints themselves).
from __future__ import annotations

# Core tensor library — used for all math and autograd below.
import torch
# Functional API (conv3d, l1_loss, etc.) — stateless ops, no learnable params.
import torch.nn.functional as F


def gaussian_window_3d(
    window_size: int,   # side length (in voxels) of the cubic 3D window
    sigma: float,        # standard deviation of the Gaussian, in voxels
    channels: int,       # number of channels to repeat the window for (grouped conv)
    device: torch.device,  # where to allocate the window (CPU/GPU)
    dtype: torch.dtype,     # floating point precision to use
) -> torch.Tensor:
    """
    Build a separable 3D Gaussian blur kernel for use as a conv3d weight.

    Returns a tensor of shape [channels, 1, window_size, window_size, window_size],
    ready to pass as the `weight` argument to F.conv3d with groups=channels
    (depthwise convolution — each channel is blurred independently).
    """
    # Build a 1D coordinate axis centered on zero, e.g. window_size=11 -> [-5, ..., 5].
    coordinates = torch.arange(
        window_size,
        device=device,
        dtype=dtype,
    ) - window_size // 2

    # 1D Gaussian curve evaluated at each centered coordinate.
    gaussian_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    # Normalize so the 1D kernel sums to 1 (keeps convolution output on the same scale).
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    # Outer product of the 1D kernel with itself along all 3 axes gives a
    # separable 3D Gaussian: window_3d[i,j,k] = g(i) * g(j) * g(k).
    window_3d = (
        gaussian_1d[:, None, None]
        * gaussian_1d[None, :, None]
        * gaussian_1d[None, None, :]
    )

    # Reshape to conv3d's expected weight layout [out_channels, in_channels/groups, D, H, W]
    # and repeat it once per channel so each channel is convolved independently
    # (paired with groups=channels in F.conv3d — a depthwise 3D convolution).
    return window_3d.view(
        1,
        1,
        window_size,
        window_size,
        window_size,
    ).repeat(channels, 1, 1, 1, 1)


def ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,   # 11x11x11 local window, as in classic 2D SSIM
    sigma: float,        # Gaussian width, as in classic 2D SSIM
    data_range: float,  # max possible pixel value (1.0 since inputs are normalized [0,1])
    eps: float,          # small constant to avoid divide-by-zero in the final ratio
) -> torch.Tensor:
    """
    Compute differentiable 3D SSIM.

    Accepted input shapes:
        [D, H, W]
        [N, D, H, W]
        [N, C, D, H, W]
    """

    # Step 1: shapes must match element-for-element to compare pred vs target.
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, "
            f"got {pred.shape} and {target.shape}"
        )

    # Step 2: normalize input rank to 5D [N, C, D, H, W], since conv3d requires it.
    if pred.ndim == 3:
        # Bare volume [D, H, W] -> add batch and channel dims of size 1.
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)

    elif pred.ndim == 4:
        # Batched volume [N, D, H, W] -> insert a single channel dim.
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    elif pred.ndim != 5:
        # Anything else (not 3D, 4D, or already 5D) is unsupported.
        raise ValueError(
            "Expected [D,H,W], [N,D,H,W], or [N,C,D,H,W], "
            f"got shape {pred.shape}"
        )

    # Step 3: read off channel count for the grouped/depthwise convolution below.
    channels = pred.shape[1]

    # Step 4: shrink the window if the volume is smaller than the requested window_size,
    # so conv3d padding never has to exceed the input's own spatial extent.
    minimum_dimension = min(pred.shape[-3:])
    actual_window_size = min(window_size, minimum_dimension)

    # Keep the Gaussian window odd.
    if actual_window_size % 2 == 0:
        actual_window_size -= 1

    # Step 5: a window smaller than 1 voxel is meaningless — guard against empty volumes.
    if actual_window_size < 1:
        raise ValueError("The spatial dimensions must be at least 1.")

    # Step 6: build the separable 3D Gaussian blur kernel used to compute local
    # (windowed) means and variances below, one copy per channel.
    window = gaussian_window_3d(
        window_size=actual_window_size,
        sigma=sigma,
        channels=channels,
        device=pred.device,
        dtype=pred.dtype,
    )

    # "Same" padding: keeps conv3d output the same spatial size as the input.
    padding = actual_window_size // 2

    # Step 7: local mean of pred and target, via Gaussian-weighted blur
    # (groups=channels makes this a per-channel/depthwise convolution).
    mu_pred = F.conv3d(
        pred,
        window,
        padding=padding,
        groups=channels,
    )

    mu_target = F.conv3d(
        target,
        window,
        padding=padding,
        groups=channels,
    )

    # Step 8: precompute squared/cross means, needed for the variance identity
    # Var(X) = E[X^2] - E[X]^2 used below.
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_pred_target = mu_pred * mu_target

    # Step 9: local variance of pred = E[pred^2] (blurred) minus E[pred]^2.
    sigma_pred_sq = (
        F.conv3d(
            pred * pred,
            window,
            padding=padding,
            groups=channels,
        )
        - mu_pred_sq
    )

    # Step 10: local variance of target, same identity as above.
    sigma_target_sq = (
        F.conv3d(
            target * target,
            window,
            padding=padding,
            groups=channels,
        )
        - mu_target_sq
    )

    # Step 11: local covariance of pred and target = E[pred*target] (blurred)
    # minus E[pred]*E[target].
    sigma_pred_target = (
        F.conv3d(
            pred * target,
            window,
            padding=padding,
            groups=channels,
        )
        - mu_pred_target
    )

    # Prevent tiny negative variances caused by floating-point error.
    sigma_pred_sq = sigma_pred_sq.clamp_min(0.0)
    sigma_target_sq = sigma_target_sq.clamp_min(0.0)

    # Step 12: SSIM stability constants (standard formula: C1 = (K1*L)^2, C2 = (K2*L)^2
    # with K1=0.01, K2=0.03, L=data_range) — avoid division blowing up when
    # local means/variances are near zero.
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Step 13: SSIM numerator — combines the luminance-similarity term
    # (2*mu_pred*mu_target + c1) with the contrast/structure-similarity term
    # (2*covariance + c2).
    numerator = (
        (2.0 * mu_pred_target + c1)
        * (2.0 * sigma_pred_target + c2)
    )

    # Step 14: SSIM denominator — normalizes by the sum of squared means and
    # the sum of variances, so the whole ratio lands in roughly [-1, 1].
    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (sigma_pred_sq + sigma_target_sq + c2)
    )

    # Step 15: per-voxel SSIM map; clamp the denominator away from zero for
    # numerical safety before dividing.
    ssim_map = numerator / denominator.clamp_min(eps)

    # Step 16: reduce the per-voxel map to a single scalar SSIM score.
    return ssim_map.mean()


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
):
    """
    Combined L1 + SSIM reconstruction loss for volumetric predictions.

    Weighted 0.7 * L1 + 0.3 * (1 - SSIM): L1 drives voxel-wise accuracy,
    while (1 - SSIM) penalizes loss of local structure/contrast that a
    pure per-voxel loss would miss.

    Returns:
        total: scalar tensor, the weighted loss to backprop through.
        metrics: dict of detached (no-grad) scalars for logging —
            raw L1, raw SSIM, and the SSIM-derived loss term.
    """
    # Step 1: mean absolute per-voxel error between prediction and ground truth.
    l1 = F.l1_loss(pred, target)

    # Step 2: structural similarity score in [-1, 1] (1 = identical structure).
    ssim = ssim_3d(
        pred,
        target,
        window_size=11,
        sigma=1.5,
        data_range=data_range,
        eps=1e-8,
    )

    # Step 3: combine into a single scalar loss; (1 - ssim) turns "higher is
    # better" similarity into a "lower is better" loss term.
    total = 0.7 * l1 + 0.3 * (1.0 - ssim)

    # Step 4: return the differentiable total plus detached scalars for logging
    # (detach so these don't keep the autograd graph alive after this call).
    return total, {
        "l1": l1.detach(),
        "ssim": ssim.detach(),
        "ssim_loss": (1.0 - ssim).detach(),
    }
