"""
Pure CUDA kernel wrapper using ctypes.
No PyTorch extension compilation - just raw CUDA .so file.
"""

import ctypes
import numpy as np
import torch
from pathlib import Path

# Load the compiled kernel library
_lib = None

def _load_kernel():
    """Load the compiled CUDA kernel library."""
    global _lib
    if _lib is None:
        kernel_path = Path(__file__).parent / "librender_gaussian.so"
        if not kernel_path.exists():
            raise FileNotFoundError(
                f"CUDA kernel not compiled. Run: bash compile_cuda_kernel.sh\n"
                f"Expected: {kernel_path}"
            )
        _lib = ctypes.CDLL(str(kernel_path))

        # Define function signature
        # render_gaussian_cuda(
        #   const float* d_pts,        // GPU pointer
        #   const float* d_means,      // GPU pointer
        #   const float* d_log_scales, // GPU pointer
        #   const float* d_quats,      // GPU pointer
        #   const float* d_intensities,// GPU pointer
        #   float* d_output,           // GPU pointer
        #   int N_pts,
        #   int M
        # )
        _lib.render_gaussian_cuda.argtypes = [
            ctypes.c_void_p,  # d_pts
            ctypes.c_void_p,  # d_means
            ctypes.c_void_p,  # d_log_scales
            ctypes.c_void_p,  # d_quats
            ctypes.c_void_p,  # d_intensities
            ctypes.c_void_p,  # d_output
            ctypes.c_int,     # N_pts
            ctypes.c_int,     # M
        ]
        _lib.render_gaussian_cuda.restype = ctypes.c_int  # cudaError_t

    return _lib


def render_gaussian_cuda(pts, means, log_scales, quats, intensities):
    """
    Render Gaussian mixture using pure CUDA kernel.

    Parameters
    ----------
    pts : torch.Tensor
        Query points, shape (N, 3), on CUDA device
    means : torch.Tensor
        Gaussian centers, shape (M, 3), on CUDA device
    log_scales : torch.Tensor
        Log scales, shape (M, 3), on CUDA device
    quats : torch.Tensor
        Quaternions, shape (M, 4), on CUDA device (normalized)
    intensities : torch.Tensor
        Intensities (sigmoid(alpha)), shape (M,), on CUDA device

    Returns
    -------
    torch.Tensor
        Predicted intensities, shape (N,), values in [0, 1]
    """
    lib = _load_kernel()

    # Ensure all inputs are on GPU and contiguous
    pts = pts.contiguous().float()
    means = means.contiguous().float()
    log_scales = log_scales.contiguous().float()
    quats = quats.contiguous().float()
    intensities = intensities.contiguous().float()

    # Output tensor
    N_pts = pts.shape[0]
    M = means.shape[0]
    output = torch.zeros(N_pts, dtype=torch.float32, device=pts.device)

    # Get device pointers (these are just integers in Python)
    pts_ptr = pts.data_ptr()
    means_ptr = means.data_ptr()
    log_scales_ptr = log_scales.data_ptr()
    quats_ptr = quats.data_ptr()
    intensities_ptr = intensities.data_ptr()
    output_ptr = output.data_ptr()

    # Call the CUDA kernel
    err = lib.render_gaussian_cuda(
        ctypes.c_void_p(pts_ptr),
        ctypes.c_void_p(means_ptr),
        ctypes.c_void_p(log_scales_ptr),
        ctypes.c_void_p(quats_ptr),
        ctypes.c_void_p(intensities_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int(N_pts),
        ctypes.c_int(M),
    )

    if err != 0:
        error_msgs = {
            1: "cudaErrorInvalidValue",
            11: "cudaErrorInvalidDevicePointer",
            35: "cudaErrorInvalidPitchedAddress",
            46: "cudaErrorInvalidAddressSpace",
        }
        error_name = error_msgs.get(err, f"CUDA error {err}")
        raise RuntimeError(
            f"CUDA kernel failed: {error_name}\n"
            f"  N_pts={N_pts}, M={M}\n"
            f"  pts dtype={pts.dtype}, device={pts.device}\n"
            f"  means dtype={means.dtype}, device={means.device}"
        )

    return output
