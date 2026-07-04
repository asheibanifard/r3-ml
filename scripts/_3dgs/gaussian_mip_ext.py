"""
CUDA extension for fast 3D Gaussian MIP splatting.

Provides a pure-CUDA renderer that evaluates full anisotropic 3D Gaussians
and returns maximum-intensity projections along a chosen view axis.
"""

import os
import torch
from torch.utils.cpp_extension import load as _cpp_load
from pathlib import Path

_EXT_DIR = Path(__file__).parent
_ext = None


def _find_cuda_include():
    """Locate CUDA headers."""
    import shutil
    candidates = []

    # 1. Explicit CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        candidates.append(Path(cuda_home) / 'include')

    # 2. Conda environments
    try:
        import sys as _sys
        conda_root = Path(_sys.executable).parent.parent
        candidates.insert(0, Path(_sys.prefix) / 'include')
        for env_path in (conda_root / 'envs').glob('*/include'):
            candidates.append(env_path)
        # NVIDIA packages in site-packages
        site_pkgs = Path(_sys.prefix) / 'lib' / f'python{_sys.version_info.major}.{_sys.version_info.minor}' / 'site-packages'
        cuda_runtime_pkg = site_pkgs / 'nvidia' / 'cuda_runtime' / 'include'
        if cuda_runtime_pkg.exists():
            candidates.insert(0, cuda_runtime_pkg)
    except Exception:
        pass

    # 3. System CUDA
    candidates.extend([
        Path('/usr/local/cuda/include'),
        Path('/usr/include/cuda'),
    ])

    # Return first valid candidate with cuda_runtime.h
    for p in candidates:
        if (p / 'cuda_runtime.h').exists():
            return [str(p)]

    for p in candidates:
        if p.exists():
            return [str(p)]

    return []


def _get_ext():
    """Lazily compile and cache the CUDA extension."""
    global _ext
    if _ext is None:
        # Detect GPU arch
        major, minor = (0, 0)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'

        # Save/restore CC/CXX to avoid cross-compiler issues
        _saved_cc = os.environ.pop('CC', None)
        _saved_cxx = os.environ.pop('CXX', None)

        try:
            _ext = _cpp_load(
                name='gaussian_mip_cuda',
                sources=[str(_EXT_DIR / 'gaussian_mip_cuda.cu')],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                extra_include_paths=_find_cuda_include(),
                verbose=False,
            )
        finally:
            if _saved_cc is not None:
                os.environ['CC'] = _saved_cc
            if _saved_cxx is not None:
                os.environ['CXX'] = _saved_cxx

    return _ext


def render_mip(
    means,
    log_scales,
    quats,
    intensities,
    lo,
    hi,
    out_h,
    out_w,
    depth_samples=32,
    view_axis=0,
    scale_min=1e-3,
    mahal_clamp=40.0,
):
    """Render a full 3D Gaussian MIP projection.

    Args:
        means: (N, 3) float32, Gaussian centers
        log_scales: (N, 3) float32, log of per-axis scales
        quats: (N, 4) float32, quaternions [w, x, y, z]
        intensities: (N,) float32, raw intensities
        lo, hi: 3-element bounds for the volume AABB
        out_h, out_w: output dimensions
        depth_samples: samples along each ray
        view_axis: 0=XY, 1=XZ, 2=YZ
        scale_min: minimum per-axis scale
        mahal_clamp: Mahalanobis cutoff for culling far Gaussians

    Returns:
        (out_h, out_w) float32 tensor with MIP values
    """
    ext = _get_ext()
    lo_x, lo_y, lo_z = float(lo[0]), float(lo[1]), float(lo[2])
    hi_x, hi_y, hi_z = float(hi[0]), float(hi[1]), float(hi[2])
    return ext.render_mip(
        means,
        log_scales,
        quats,
        intensities,
        lo_x,
        hi_x,
        lo_y,
        hi_y,
        lo_z,
        hi_z,
        out_h,
        out_w,
        depth_samples,
        view_axis,
        scale_min,
        mahal_clamp,
    )


def render_mip_xy(
    means,
    log_scales,
    quats,
    intensities,
    lo,
    hi,
    height,
    width,
    **kwargs,
):
    return render_mip(
        means,
        log_scales,
        quats,
        intensities,
        lo,
        hi,
        height,
        width,
        view_axis=0,
        **kwargs,
    )


def render_mip_xz(
    means,
    log_scales,
    quats,
    intensities,
    lo,
    hi,
    depth,
    width,
    **kwargs,
):
    return render_mip(
        means,
        log_scales,
        quats,
        intensities,
        lo,
        hi,
        depth,
        width,
        view_axis=1,
        **kwargs,
    )


def render_mip_yz(
    means,
    log_scales,
    quats,
    intensities,
    lo,
    hi,
    depth,
    height,
    **kwargs,
):
    return render_mip(
        means,
        log_scales,
        quats,
        intensities,
        lo,
        hi,
        depth,
        height,
        view_axis=2,
        **kwargs,
    )
