# gaussian_volume/renderer/extension/loader.py

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


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
                  scale_min, mahal_clamp, max_gauss_per_tile,
                  print_stats, clamp_output) -> flat float32 CUDA tensor
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The Gaussian rasterisation extension requires an NVIDIA GPU."
        )

    # reconstruct_volume.cu, splat_mip.cu, bindings.cpp, common.cuh, and
    # ext.h are siblings of this file, inside the same
    # renderer/extension/ package directory.
    source_dir = Path(__file__).resolve().parent
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
