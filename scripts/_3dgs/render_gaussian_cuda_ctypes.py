"""
CUDA kernel wrapper using ctypes with proper CUDA context management.
"""

import ctypes
import torch
from pathlib import Path

def _load_cudart():
    """Load the CUDA runtime that matches the active PyTorch installation."""
    torch_site = Path(torch.__file__).resolve().parent.parent
    candidates = [
        torch_site / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12",
        torch_site / "nvidia" / "cu12" / "lib" / "libcudart.so.12",
        Path("/home/armin/miniconda3/envs/gaussian-3d/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"),
        Path("/home/armin/miniconda3/envs/gaussian-3d/lib/libcudart.so.12"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    raise RuntimeError("Could not find a compatible CUDA runtime library (libcudart.so.12)")


cudart = _load_cudart()

# Define CUDA functions
cudart.cudaSetDevice.argtypes = [ctypes.c_int]
cudart.cudaSetDevice.restype = ctypes.c_int

cudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
cudart.cudaGetDevice.restype = ctypes.c_int

cudart.cudaDeviceSynchronize.argtypes = []
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

# Load the compiled kernel library
_lib = None

def _load_kernel():
    """Load the compiled CUDA kernel library with proper initialization."""
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

        if hasattr(_lib, "reconstruct_gaussian_volume_cuda"):
            _lib.reconstruct_gaussian_volume_cuda.argtypes = [
                ctypes.c_void_p,  # d_means
                ctypes.c_void_p,  # d_log_scales
                ctypes.c_void_p,  # d_quats
                ctypes.c_void_p,  # d_intensities
                ctypes.c_void_p,  # d_output
                ctypes.c_int,     # Dz
                ctypes.c_int,     # Dy
                ctypes.c_int,     # Dx
                ctypes.c_int,     # M
            ]
            _lib.reconstruct_gaussian_volume_cuda.restype = ctypes.c_int  # cudaError_t
        else:
            _lib.reconstruct_gaussian_volume_cuda = None

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

    # Get device number
    device_idx = pts.get_device()

    # Set the device before kernel launch
    err = cudart.cudaSetDevice(ctypes.c_int(device_idx))
    if err != 0:
        raise RuntimeError(f"cudaSetDevice failed with error {err}")

    # Output tensor
    N_pts = pts.shape[0]
    M = means.shape[0]
    output = torch.zeros(N_pts, dtype=torch.float32, device=pts.device)

    # Get device pointers
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
            3: "cudaErrorInitializationError",
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


def render_gaussian_volume(means, log_scales, quats, intensities, Dz, Dy, Dx):
    """Reconstruct a dense voxel grid directly from fitted Gaussians."""
    lib = _load_kernel()

    means = means.contiguous().float()
    log_scales = log_scales.contiguous().float()
    quats = quats.contiguous().float()
    intensities = intensities.contiguous().float()

    device_idx = means.get_device()

    err = cudart.cudaSetDevice(ctypes.c_int(device_idx))
    if err != 0:
        raise RuntimeError(f"cudaSetDevice failed with error {err}")

    output = torch.zeros((Dz, Dy, Dx), dtype=torch.float32, device=means.device)

    if getattr(lib, "reconstruct_gaussian_volume_cuda", None) is None:
        zg = torch.linspace(-1, 1, Dz, device=means.device, dtype=torch.float32)
        yg = torch.linspace(-1, 1, Dy, device=means.device, dtype=torch.float32)
        xg = torch.linspace(-1, 1, Dx, device=means.device, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(zg, yg, xg, indexing="ij")
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1).contiguous()

        flat = render_gaussian_cuda(pts, means, log_scales, quats, intensities)
        return flat.reshape(Dz, Dy, Dx)

    err = lib.reconstruct_gaussian_volume_cuda(
        ctypes.c_void_p(means.data_ptr()),
        ctypes.c_void_p(log_scales.data_ptr()),
        ctypes.c_void_p(quats.data_ptr()),
        ctypes.c_void_p(intensities.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(Dz),
        ctypes.c_int(Dy),
        ctypes.c_int(Dx),
        ctypes.c_int(means.shape[0]),
    )

    if err != 0:
        error_msgs = {
            1: "cudaErrorInvalidValue",
            3: "cudaErrorInitializationError",
            11: "cudaErrorInvalidDevicePointer",
            35: "cudaErrorInvalidPitchedAddress",
            46: "cudaErrorInvalidAddressSpace",
        }
        error_name = error_msgs.get(err, f"CUDA error {err}")
        raise RuntimeError(
            f"CUDA volume kernel failed: {error_name}\n"
            f"  volume_shape=({Dz}, {Dy}, {Dx}), M={means.shape[0]}\n"
            f"  means dtype={means.dtype}, device={means.device}"
        )

    return output
