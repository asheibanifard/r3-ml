# gaussian_volume/rasterisation.py
"""
Camera/projection-style rasterisation of a GaussianVolumeModel: dense
per-voxel evaluation and orthogonal MIP splatting via the CUDA kernels in
csrc/eval/ (reconstruct_volume.cu, splat_mip.cu) — the same kernels
_3dgs.py's render_splatted_mips used, now exposed as a standalone module
for the newer, modular GaussianVolumeModel instead of the legacy
GaussianCloud.

This is a separate reconstruction path from reconstruction.py's
reconstruct_volume(): that one implements the model's own normalized
formula (V(x) = sum w_i f_i / (sum w_i + epsilon)) via the training
extension, and is what training/checkpointing actually optimizes against.
The kernels here instead accumulate an unnormalized, Beer-Lambert-style
opacity per Gaussian (matching the original GaussianCloud convention:
softplus(inten) * exp(-0.5*mahalanobis)) — a rendering-time visualization,
not a training target. The two will not produce identical images.

confidence_i * feature_i (this model's density weight times its value) is
inverse-softplus'd into the "inten" the kernels expect, so the kernel's
internal softplus recovers that product exactly. Values are clamped to a
small positive floor first, since inverse_softplus requires strictly
positive input — relevant if constrain_features=False allows negative
features.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

from .model import GaussianVolumeModel, inverse_softplus


@lru_cache(maxsize=1)
def load_rasterisation_extension():
    """
    JIT-compile and load the Gaussian rasterisation CUDA extension,
    exposing:

        reconstruct_volume(means, log_s, quats, inten, lo_x, hi_x, lo_y,
                            hi_y, lo_z, hi_z, D, H, W, scale_min,
                            mahal_clamp) -> flat float32 CUDA tensor

        splat_mip(means, log_s, quats, inten, lo_x, hi_x, lo_y, hi_y,
                  lo_z, hi_z, out_h, out_w, depth_samples, view_axis,
                  scale_min, mahal_clamp, density_scale, max_gauss_per_tile,
                  print_stats, clamp_output) -> flat float32 CUDA tensor
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The Gaussian rasterisation extension requires an NVIDIA GPU."
        )

    source_dir = Path(__file__).resolve().parent / "csrc" / "eval"
    sources = [
        source_dir / "reconstruct_volume.cu",
        source_dir / "splat_mip.cu",
        source_dir / "bindings.cpp",
    ]

    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing rasterisation extension source files:\n" + "\n".join(missing)
        )

    verbose = os.environ.get("GAUSSIAN_VOLUME_VERBOSE_BUILD", "0") == "1"

    extension = load(
        name="gaussian_volume_rasterisation_cuda",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=verbose,
    )

    for function_name in ("reconstruct_volume", "splat_mip"):
        if not hasattr(extension, function_name):
            raise RuntimeError(
                "Compiled rasterisation extension is missing required "
                f"function {function_name!r}."
            )

    return extension


def _model_to_kernel_tensors(model: GaussianVolumeModel) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Convert GaussianVolumeModel's physical-space parameters into the flat
    (means, log_s, quats, inten) tensors the eval kernels expect.
    """
    means = model.means.detach().contiguous()
    log_s = torch.log(model.scales.detach()).contiguous()
    quats = model.normalized_quaternions.detach().contiguous()

    opacity = (model.confidences.detach() * model.features.detach()).clamp_min(1e-6)
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


def _auto_scale_opacity(raw: Tensor, *, reference_density_scale: float, target_opacity: float) -> Tensor:
    """
    Rescale a Beer-Lambert opacity image (splat_mip's output, or
    dvr_ground_truth's) computed at reference_density_scale so its
    brightest pixel reaches target_opacity, without recomputing it.

    Every Beer-Lambert mapping used in this module has the form `mapped =
    1 - exp(-density_scale * acc)` for some non-negative per-pixel
    accumulated density `acc` that doesn't itself depend on density_scale.
    Since `1 - mapped = exp(-density_scale * acc)`, raising (1 - raw) to the
    power `ratio = new_density_scale / reference_density_scale` gives
    exactly `exp(-new_density_scale * acc)` — i.e. the image that density_scale
    would have produced, algebraically, with no extra rendering work:

        new_mapped = 1 - (1 - raw) ** ratio

    ratio is solved so the raw image's own max reaches target_opacity:
    `1 - (1 - raw_max) ** ratio = target_opacity`.
    """
    raw_max = raw.max().clamp_min(1e-12)

    if raw_max < 1e-8:
        # Nothing visible at all at the (very small) reference scale — no
        # amount of rescaling recovers detail that was never accumulated
        # (e.g. cutoff_sigma too tight, or the model has ~zero density here).
        return raw

    log_survival = torch.log1p(-raw_max)  # log(1 - raw_max); raw_max < 1 always (see module docstring)
    ratio = torch.log(torch.tensor(1.0 - target_opacity, device=raw.device)) / log_survival

    return 1.0 - (1.0 - raw).clamp_min(0.0).pow(ratio)


def render_mip(
    model: GaussianVolumeModel,
    view_axis: int,
    out_shape: Sequence[int],
    *,
    cutoff_sigma: float,
    bounds: tuple[float, float, float, float, float, float],
    depth_samples: int = 32,
    density_scale: float | None = None,
    target_opacity: float = 0.9,
    reference_density_scale: float = 1.0e-4,
    max_gauss_per_tile: int = 0,
) -> Tensor:
    """
    Tiled MIP splatting of the Gaussian mixture along one of three
    orthogonal view axes, via the fused CUDA kernel (splat_mip):

        view_axis=0: project onto the (x,y) plane (looking down z) -> 'xy'
        view_axis=1: project onto the (x,z) plane (looking down y) -> 'xz'
        view_axis=2: project onto the (y,z) plane (looking down x) -> 'yz'

    density_scale=None (the default) auto-exposes: one kernel launch at
    reference_density_scale, then an analytic rescale (see
    _auto_scale_opacity) so the brightest pixel reaches target_opacity —
    no manual tuning needed, and no second kernel launch. Pass an explicit
    density_scale to bypass auto-exposure and use splat_mip's raw mapping
    directly.

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

    if not (0.0 < target_opacity < 1.0):
        raise ValueError(f"target_opacity must be in (0, 1), got {target_opacity}.")

    out_h, out_w = out_shape
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = bounds

    extension = load_rasterisation_extension()
    means, log_s, quats, inten = _model_to_kernel_tensors(model)

    effective_density_scale = density_scale if density_scale is not None else reference_density_scale

    flat = extension.splat_mip(
        means, log_s, quats, inten,
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        out_h, out_w, depth_samples, view_axis,
        model.minimum_scale, cutoff_sigma ** 2,
        effective_density_scale, max_gauss_per_tile,
        False, True,
    )
    image = flat.reshape(out_h, out_w)

    if density_scale is None:
        image = _auto_scale_opacity(
            image, reference_density_scale=reference_density_scale, target_opacity=target_opacity
        )

    return image


def render_orthogonal_mips(
    model: GaussianVolumeModel,
    volume_shape: Sequence[int],
    *,
    cutoff_sigma: float,
    screen_size: tuple[int, int] | None = None,
    depth_samples: int = 32,
    density_scale: float | None = None,
    target_opacity: float = 0.9,
    reference_density_scale: float = 1.0e-4,
    max_gauss_per_tile: int = 0,
) -> dict[str, Tensor]:
    """
    MIP-splat all three orthogonal views at once: 'xy' (looking down z),
    'xz' (looking down y), 'yz' (looking down x).

    screen_size=None (the default) sizes each view to match the volume
    exactly, no downsampling: 'xy' (H,W), 'xz' (D,W), 'yz' (D,H). Passing an
    explicit (screen_height, screen_width) instead renders all three views
    at that same independent projection resolution — decoupled from the
    volume's own voxel-grid dimensions, the way a camera's screen
    resolution is independent of scene scale.

    Each view auto-exposes independently by default (density_scale=None —
    see render_mip), since the three projections can accumulate very
    different densities.
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
        density_scale=density_scale,
        target_opacity=target_opacity,
        reference_density_scale=reference_density_scale,
        max_gauss_per_tile=max_gauss_per_tile,
    )

    return {
        "xy": render_mip(model, 0, xy_shape, **common),
        "xz": render_mip(model, 1, xz_shape, **common),
        "yz": render_mip(model, 2, yz_shape, **common),
    }


def mip_ground_truth(volume: Tensor, view_axis: int) -> Tensor:
    """
    Plain Maximum Intensity Projection directly on a dense ground-truth
    voxel grid (no Gaussian mixture involved) — the "what does the raw
    data actually look like" reference to compare render_mip's
    Gaussian-rasterised MIP against. No CUDA kernel needed: MIP is just a
    max-reduction along one axis.

        view_axis=0: project along depth (z) -> (H,W), 'xy'
        view_axis=1: project along height (y) -> (D,W), 'xz'
        view_axis=2: project along width (x) -> (D,H), 'yz'

    This lines up exactly with render_mip/render_orthogonal_mips's own
    view_axis convention: volume (shaped [D,H,W]) reduced along dimension
    view_axis gives precisely the same output shape those functions do.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be a 3D [D,H,W] tensor, got shape {tuple(volume.shape)}.")

    if view_axis not in (0, 1, 2):
        raise ValueError(f"view_axis must be 0, 1, or 2, got {view_axis}.")

    return volume.amax(dim=view_axis)


def render_orthogonal_mips_ground_truth(volume: Tensor) -> dict[str, Tensor]:
    """
    MIP all three orthogonal views of a dense ground-truth voxel grid at
    once: 'xy', 'xz', 'yz' — matching render_orthogonal_mips's view
    conventions exactly, so the two are directly comparable side by side.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be a 3D [D,H,W] tensor, got shape {tuple(volume.shape)}.")

    return {
        "xy": mip_ground_truth(volume, 0),
        "xz": mip_ground_truth(volume, 1),
        "yz": mip_ground_truth(volume, 2),
    }


def dvr_ground_truth(
    volume: Tensor,
    view_axis: int,
    *,
    density_scale: float | None = None,
    step_size: float = 1.0,
    target_opacity: float = 0.9,
    reference_density_scale: float = 1.0e-4,
) -> Tensor:
    """
    Front-to-back Beer-Lambert direct volume rendering (DVR) of a dense
    ground-truth voxel grid, marching parallel (orthographic) rays along
    one grid axis — the same accumulation model as the voxel package's
    camera-based DVR (extension/render.cu's render_direct_volume):

        alpha_k    = 1 - exp(-density_scale * volume_k * step_size)
        T_k        = prod_{j<k} (1 - alpha_j)         (transmittance before sample k)
        C          = sum_k T_k * alpha_k

    Marching parallel rays along a grid axis (rather than the voxel package's
    perspective camera) needs no ray-AABB intersection or per-ray geometry,
    so this is a pure-PyTorch cumulative product/sum along view_axis — no
    CUDA kernel involved, and numerically exact (log(1-alpha_k) equals
    -density_scale*volume_k*step_size exactly, so the running transmittance
    is one exclusive cumsum + exp, not an iterative loop).

    density_scale=None (the default) auto-exposes exactly like render_mip:
    accumulate once at reference_density_scale, then analytically rescale
    (_auto_scale_opacity) so the brightest pixel reaches target_opacity —
    the same trick applies here since C is Beer-Lambert in exactly the
    same way. Pass an explicit density_scale to bypass auto-exposure.

    view_axis=0/1/2 matches mip_ground_truth's convention exactly, so the
    two are directly comparable side by side.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be a 3D [D,H,W] tensor, got shape {tuple(volume.shape)}.")

    if view_axis not in (0, 1, 2):
        raise ValueError(f"view_axis must be 0, 1, or 2, got {view_axis}.")

    if density_scale is not None and density_scale <= 0:
        raise ValueError(f"density_scale must be positive, got {density_scale}.")

    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}.")

    if not (0.0 < target_opacity < 1.0):
        raise ValueError(f"target_opacity must be in (0, 1), got {target_opacity}.")

    effective_density_scale = density_scale if density_scale is not None else reference_density_scale

    log_survival = -effective_density_scale * volume.clamp_min(0.0) * step_size
    exclusive_log_transmittance = torch.cumsum(log_survival, dim=view_axis) - log_survival
    transmittance = torch.exp(exclusive_log_transmittance)
    alpha = 1.0 - torch.exp(log_survival)

    image = (transmittance * alpha).sum(dim=view_axis).clamp(0.0, 1.0)

    if density_scale is None:
        image = _auto_scale_opacity(
            image, reference_density_scale=reference_density_scale, target_opacity=target_opacity
        )

    return image


def render_orthogonal_dvr_ground_truth(
    volume: Tensor,
    *,
    density_scale: float | None = None,
    step_size: float = 1.0,
    target_opacity: float = 0.9,
    reference_density_scale: float = 1.0e-4,
) -> dict[str, Tensor]:
    """
    DVR all three orthogonal views of a dense ground-truth voxel grid at
    once: 'xy', 'xz', 'yz' — matching render_orthogonal_mips_ground_truth's
    view conventions exactly. Each view auto-exposes independently by
    default (density_scale=None — see dvr_ground_truth).
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be a 3D [D,H,W] tensor, got shape {tuple(volume.shape)}.")

    common = dict(
        density_scale=density_scale,
        step_size=step_size,
        target_opacity=target_opacity,
        reference_density_scale=reference_density_scale,
    )

    return {
        "xy": dvr_ground_truth(volume, 0, **common),
        "xz": dvr_ground_truth(volume, 1, **common),
        "yz": dvr_ground_truth(volume, 2, **common),
    }
