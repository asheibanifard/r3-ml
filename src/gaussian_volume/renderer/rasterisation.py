# gaussian_volume/renderer/rasterisation.py
"""
Camera/projection-style rasterisation of a GaussianVolumeModel: dense
per-voxel evaluation and orthogonal true Maximum Intensity Projection
(MIP) via the CUDA kernels in extension/ (reconstruct_volume.cu,
splat_mip.cu, loaded by extension/loader.py) — evolved from the same
kernels _3dgs.py's render_splatted_mips used (its own copies now live
under renderer/extension/ too — see _3dgs.py's _load_eval_kernel), now
exposed as a standalone module for the newer, modular GaussianVolumeModel
instead of the legacy GaussianCloud.

This is a separate reconstruction path from reconstruction.py's
reconstruct_volume(): that one implements the model's own normalized
formula (V(x) = sum w_i f_i / (sum w_i + epsilon)) via the training
extension, and is what training/checkpointing actually optimizes against.
render_volume here instead accumulates an unnormalized sum per voxel
(matching the original GaussianCloud convention: softplus(inten) *
exp(-0.5*mahalanobis)) — a rendering-time visualization, not a training
target. The two will not produce identical images.

confidence_i * feature_i (this model's density weight times its value) is
inverse-softplus'd into the "inten" the kernels expect, so the kernel's
internal softplus recovers that product exactly. Values are clamped to a
small positive floor first, since inverse_softplus requires strictly
positive input — relevant if constrain_features=False allows negative
features.

render_mip/splat_mip computes a true max-intensity projection: each 3D
Gaussian's own peak value along the view axis has a closed form (see
splat_mip.cu's header comment), so per-pixel max over every Gaussian
covering that pixel is an exact MIP through the mixture — raw peak
density, no opacity mapping needed. dvr_ground_truth is the ground-truth
counterpart: also a true MIP (ray-marched/interpolated instead of
analytic — see its docstring), so the two ARE now an equation-matched
pair for direct comparison.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..representation.model import GaussianVolumeModel, inverse_softplus
from .extension.loader import load_rasterisation_extension


def _model_to_kernel_tensors(model: GaussianVolumeModel) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Convert GaussianVolumeModel's physical-space parameters into the flat
    (means, log_s, quats, inten) tensors the eval kernels expect.

    confidence has no upper bound during training: the model optimizes a
    *normalized* reconstruction (V(x) = sum w_i f_i / (sum w_i + eps)), so
    the denominator absorbs any individual w_i (confidence) regardless of
    its magnitude -- confidence values well above 1 (even ~10+) are a
    normal, correct outcome of training, not an error. But the
    rasterisation kernels use an unnormalized formula (no division by the
    total weight), so opacity is clamped to <= 1.0 here -- its intended
    meaning as a per-Gaussian opacity, not an unbounded training weight --
    before it reaches them. Without this ceiling, a handful of
    high-confidence Gaussians saturate a large fraction of the true-MIP
    output (see render_mip's docstring), since their raw peak values
    (softplus(confidence*feature), unclamped) can reach into the tens.
    """
    means = model.means.detach().contiguous()
    log_s = torch.log(model.scales.detach()).contiguous()
    quats = model.normalized_quaternions.detach().contiguous()

    opacity = (model.confidences.detach() * model.features.detach()).clamp(1e-6, 1.0)
    inten = inverse_softplus(opacity, minimum=0.0).contiguous()

    return means, log_s, quats, inten


def _default_bounds(shape: Sequence[int]) -> tuple[float, float, float, float, float, float]:
    """
    Default axis-aligned bounds spanning the full voxel grid, matching
    model.py's convention that means live in voxel coordinates (voxel
    index i along an axis corresponds to coordinate i + 0.5, over
    [0, size]).
    """
    depth, height, width = shape
    return 0.0, float(width), 0.0, float(height), 0.0, float(depth)


def render_volume(
    model: GaussianVolumeModel,
    shape: Sequence[int],
    *,
    cutoff_sigma: float,
    bounds: tuple[float, float, float, float, float, float] | None = None,
) -> Tensor:
    """
    Dense per-voxel rasterisation via the fused CUDA kernel
    (reconstruct_volume): an unnormalized, clamped-to-[0,1] accumulation —
    a visualization aid, not the model's training-time reconstruction (see
    module docstring).
    """
    if len(shape) != 3:
        raise ValueError(f"shape must have 3 elements [D,H,W], got {shape}.")

    if cutoff_sigma <= 0:
        raise ValueError(f"cutoff_sigma must be positive, got {cutoff_sigma}.")

    depth, height, width = shape
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = bounds or _default_bounds(shape)

    extension = load_rasterisation_extension()
    means, log_s, quats, inten = _model_to_kernel_tensors(model)

    flat = extension.reconstruct_volume(
        means, log_s, quats, inten,
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        depth, height, width,
        model.minimum_scale, cutoff_sigma ** 2,
    )
    return flat.reshape(depth, height, width)


def render_mip(
    model: GaussianVolumeModel,
    view_axis: int,
    out_shape: Sequence[int],
    *,
    cutoff_sigma: float,
    bounds: tuple[float, float, float, float, float, float],
    depth_samples: int = 32,
    max_gauss_per_tile: int = 0,
) -> Tensor:
    """
    Tiled true Maximum Intensity Projection of the Gaussian mixture along
    one of three orthogonal view axes, via the fused CUDA kernel
    (splat_mip):

        view_axis=0: project onto the (x,y) plane (looking down z) -> 'xy'
        view_axis=1: project onto the (x,z) plane (looking down y) -> 'xz'
        view_axis=2: project onto the (y,z) plane (looking down x) -> 'yz'

    Each 3D Gaussian's own peak value along the view axis has a closed
    form (marginalizing/maximizing a Gaussian along one axis analytically
    yields an effective 2D Gaussian in screen space — see splat_mip.cu's
    header comment), so per-pixel max over every Gaussian covering that
    pixel is an exact MIP through the mixture: no ray marching, no
    opacity mapping, no auto-exposure — the output is each pixel's raw
    peak density value, clamped to [0,1].

    Each pixel evaluates a Gaussian's contribution using the nearest
    point within its own footprint (half a pixel-width around its
    centre), not just its exact centre — so even a Gaussian sharp enough
    to fall between two sample points is still captured exactly by
    whichever pixel's footprint contains its true peak (see
    splat_mip.cu's header comment for why this is exact, not an
    approximation).

    dvr_ground_truth is the equation-matched ground-truth counterpart
    (also a true MIP, ray-marched/interpolated instead of analytic — see
    its docstring), so the two are now directly comparable for PSNR/SSIM,
    not just qualitatively.

    depth_samples is accepted for API parity with the legacy CPU fallback
    this replaced; the CUDA kernel handles the full MIP projection in one
    launch regardless of its value.
    """
    if view_axis not in (0, 1, 2):
        raise ValueError(f"view_axis must be 0, 1, or 2, got {view_axis}.")

    if len(out_shape) != 2:
        raise ValueError(f"out_shape must have 2 elements [H,W], got {out_shape}.")

    if cutoff_sigma <= 0:
        raise ValueError(f"cutoff_sigma must be positive, got {cutoff_sigma}.")

    out_h, out_w = out_shape
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = bounds

    extension = load_rasterisation_extension()
    means, log_s, quats, inten = _model_to_kernel_tensors(model)

    flat = extension.splat_mip(
        means, log_s, quats, inten,
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        out_h, out_w, depth_samples, view_axis,
        model.minimum_scale, cutoff_sigma ** 2,
        max_gauss_per_tile,
        False, True,
    )
    return flat.reshape(out_h, out_w)


def render_orthogonal_mips(
    model: GaussianVolumeModel,
    volume_shape: Sequence[int],
    *,
    cutoff_sigma: float,
    screen_size: tuple[int, int] | None = None,
    depth_samples: int = 32,
    max_gauss_per_tile: int = 0,
) -> dict[str, Tensor]:
    """
    True Maximum Intensity Projection of all three orthogonal views at
    once: 'xy' (looking down z), 'xz' (looking down y), 'yz' (looking
    down x).

    screen_size=None (the default) sizes each view to match the volume
    exactly, no downsampling: 'xy' (H,W), 'xz' (D,W), 'yz' (D,H). Passing an
    explicit (screen_height, screen_width) instead renders all three views
    at that same independent projection resolution — decoupled from the
    volume's own voxel-grid dimensions, the way a camera's screen
    resolution is independent of scene scale.
    """
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must have 3 elements [D,H,W], got {volume_shape}.")

    if screen_size is not None and len(screen_size) != 2:
        raise ValueError(f"screen_size must have 2 elements [H,W], got {screen_size}.")

    depth, height, width = volume_shape
    bounds = _default_bounds(volume_shape)

    xy_shape = screen_size or (height, width)
    xz_shape = screen_size or (depth, width)
    yz_shape = screen_size or (depth, height)

    common = dict(
        cutoff_sigma=cutoff_sigma,
        bounds=bounds,
        depth_samples=depth_samples,
        max_gauss_per_tile=max_gauss_per_tile,
    )

    return {
        "xy": render_mip(model, 0, xy_shape, **common),
        "xz": render_mip(model, 1, xz_shape, **common),
        "yz": render_mip(model, 2, yz_shape, **common),
    }


def _normalized_axis_coords(size: int, num: int, *, device: torch.device) -> Tensor:
    """
    num sample positions spanning voxel indices [0, size-1] (align_corners=True
    convention: index 0 maps to -1, index size-1 maps to +1), for use as one
    axis of a grid_sample sampling grid.
    """
    if size <= 1:
        return torch.zeros(num, device=device)
    idx = torch.linspace(0, size - 1, steps=num, device=device)
    return idx * (2.0 / (size - 1)) - 1.0


def dvr_ground_truth(
    volume: Tensor,
    view_axis: int,
    *,
    step_size: float = 1.0,
    screen_size: tuple[int, int] | None = None,
) -> Tensor:
    """
    Ray-marched Maximum Intensity Projection of a dense ground-truth voxel
    grid: one ray per output pixel, marched straight through the volume
    along view_axis, sampling at each step via trilinear interpolation
    (torch.nn.functional.grid_sample) and taking the max value
    encountered — the ground-truth counterpart to render_mip/splat_mip's
    own true MIP (an analytic per-Gaussian max instead of ray marching;
    the name "DVR" is kept only for continuity with the existing
    CLI/config surface — this does NOT accumulate opacity/transmittance
    the way the voxel package's camera-based DVR does).

    Marching via interpolation (rather than an exact per-voxel amax)
    means this supports an independent output resolution (screen_size)
    on the two in-plane axes, decoupled from the volume's own voxel-grid
    dimensions — the same way render_mip's screen_size does for the
    Gaussian side. screen_size=None (the default) samples at the
    volume's own native in-plane resolution, in which case this reduces
    to a plain volume.amax(dim=view_axis) up to interpolation error (not
    bit-exact, since every sample still goes through grid_sample rather
    than a raw index lookup).

    step_size sets the ray-marching sample spacing in voxel units along
    view_axis: 1.0 samples once per voxel; smaller values sample more
    densely between voxel centres (a plain max has no compositing to
    "smooth over" an under-sampled ray, so a coarse step can miss a thin
    bright feature a finer step would catch); larger values are coarser
    and cheaper.

    Upsampling here (screen_size larger than the volume's native shape,
    or step_size < 1) is a visual/comparison convenience only — the
    volume has no real detail between voxel centres, so interpolation
    invents smooth-looking values there rather than recovering anything.

    view_axis=0/1/2: view_axis=0 projects along depth (z) -> (H,W) 'xy';
    view_axis=1 projects along height (y) -> (D,W) 'xz'; view_axis=2
    projects along width (x) -> (D,H) 'yz' — matching
    render_mip/render_orthogonal_mips's own convention exactly.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be a 3D [D,H,W] tensor, got shape {tuple(volume.shape)}.")

    if view_axis not in (0, 1, 2):
        raise ValueError(f"view_axis must be 0, 1, or 2, got {view_axis}.")

    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}.")

    if screen_size is not None and len(screen_size) != 2:
        raise ValueError(f"screen_size must have 2 elements [H,W], got {screen_size}.")

    device = volume.device
    native_sizes = tuple(volume.shape)  # (D, H, W)
    other_axes = [axis for axis in (0, 1, 2) if axis != view_axis]

    if screen_size is not None:
        out_sizes = {other_axes[0]: screen_size[0], other_axes[1]: screen_size[1]}
    else:
        out_sizes = {axis: native_sizes[axis] for axis in other_axes}

    march_size = native_sizes[view_axis]
    num_samples = int(round((march_size - 1) / step_size)) + 1 if march_size > 1 else 1

    axis_coords: list[Tensor | None] = [None, None, None]
    axis_coords[view_axis] = _normalized_axis_coords(march_size, num_samples, device=device)
    for axis in other_axes:
        axis_coords[axis] = _normalized_axis_coords(native_sizes[axis], out_sizes[axis], device=device)

    grid_d, grid_h, grid_w = torch.meshgrid(*axis_coords, indexing="ij")
    # grid_sample's last dim orders coordinates (x, y, z) = (W, H, D).
    grid = torch.stack((grid_w, grid_h, grid_d), dim=-1).unsqueeze(0)

    volume_5d = volume.unsqueeze(0).unsqueeze(0).to(torch.float32)
    sampled = torch.nn.functional.grid_sample(
        volume_5d, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    sampled = sampled.squeeze(0).squeeze(0)  # (D', H', W'), with size num_samples along view_axis

    return sampled.amax(dim=view_axis)


def render_orthogonal_dvr_ground_truth(
    volume: Tensor,
    *,
    step_size: float = 1.0,
    screen_size: tuple[int, int] | None = None,
) -> dict[str, Tensor]:
    """
    Ray-marched MIP of all three orthogonal views of a dense ground-truth
    voxel grid at once: 'xy', 'xz', 'yz' — matching
    render_orthogonal_mips's view conventions exactly. See
    dvr_ground_truth for the step_size/screen_size semantics.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be a 3D [D,H,W] tensor, got shape {tuple(volume.shape)}.")

    common = dict(step_size=step_size, screen_size=screen_size)

    return {
        "xy": dvr_ground_truth(volume, 0, **common),
        "xz": dvr_ground_truth(volume, 1, **common),
        "yz": dvr_ground_truth(volume, 2, **common),
    }
