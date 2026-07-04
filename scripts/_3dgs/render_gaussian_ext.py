"""
PyTorch extension for CUDA-accelerated Gaussian field rendering.
Compiles and caches the CUDA kernel on first use.
"""

from pathlib import Path
from torch.utils.cpp_extension import load
import torch

# Cache compiled module
_render_module = None


def _find_cuda_include():
    """Find CUDA include paths."""
    import os
    import sys as _sys
    candidates = []

    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        candidates.append(Path(cuda_home) / 'include')

    try:
        import torch
        prefix = Path(_sys.prefix)

        # PyTorch headers (needed for torch/extension.h)
        torch_inc = Path(torch.utils.cpp_extension.include_paths()[0])
        if torch_inc.exists():
            candidates.insert(0, torch_inc)

        # Conda CUDA targets path (contains nv/target headers)
        targets_inc = prefix / 'targets' / 'x86_64-linux' / 'include'
        if targets_inc.exists():
            candidates.insert(0, targets_inc)

        site_pkgs = prefix / 'lib' / f'python{_sys.version_info.major}.{_sys.version_info.minor}' / 'site-packages'
        cuda_runtime_pkg = site_pkgs / 'nvidia' / 'cuda_runtime' / 'include'
        if cuda_runtime_pkg.exists():
            candidates.insert(0, cuda_runtime_pkg)

        # Also add cu13 headers
        cu13_pkg = site_pkgs / 'nvidia' / 'cu13' / 'include'
        if cu13_pkg.exists():
            candidates.insert(0, cu13_pkg)
    except Exception as e:
        print(f"Warning: Could not add some include paths: {e}")
        pass

    candidates.extend([
        Path('/usr/local/cuda/include'),
        Path('/usr/include/cuda'),
    ])

    result = []
    for p in candidates:
        if p.exists():
            result.append(str(p))

    return result if result else []


def _load_render_module():
    """Lazily compile and cache the rendering kernel."""
    global _render_module
    if _render_module is None:
        src = Path(__file__).parent / "render_gaussian_cuda.cu"
        extra_inc = _find_cuda_include()
        extra_flags = ["-O3", "--use_fast_math"] + [f"-I{p}" for p in extra_inc]

        print(f"Compiling CUDA rendering kernel from {src}...")
        print(f"Include paths: {extra_inc}")
        _render_module = load(
            name="render_gaussian_cuda",
            sources=[str(src)],
            extra_cuda_cflags=extra_flags,
            extra_include_paths=extra_inc,
            verbose=True,
        )
        print("✓ CUDA rendering kernel compiled successfully")
    return _render_module


def render_gaussian_cuda(pts, means, log_scales, quats, intensities):
    """
    Render Gaussian mixture on query points using optimized CUDA kernel.

    Parameters
    ----------
    pts : torch.Tensor
        Query points, shape (N, 3), on CUDA device
    means : torch.Tensor
        Gaussian centers, shape (M, 3), on CUDA device
    log_scales : torch.Tensor
        Log scales, shape (M, 3), on CUDA device
    quats : torch.Tensor
        Quaternions, shape (M, 4), on CUDA device, normalized
    intensities : torch.Tensor
        Intensities (sigmoid(alpha)), shape (M,), on CUDA device

    Returns
    -------
    torch.Tensor
        Predicted intensities, shape (N,), values in [0, 1]
    """
    module = _load_render_module()

    # Ensure all inputs are on CUDA and contiguous
    pts = pts.contiguous()
    means = means.contiguous()
    log_scales = log_scales.contiguous()
    quats = quats.contiguous()
    intensities = intensities.contiguous()

    return module.render_gaussian_cuda(pts, means, log_scales, quats, intensities)
