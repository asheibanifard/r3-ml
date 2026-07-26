# voxel/extension/loader.py

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=None)
def load_voxel_extension(hidden_size: int, hidden_layers: int):
    """
    JIT-compile and load the SIREN voxel-field CUDA extension.

    hidden_size/hidden_layers are baked into the compiled kernel as
    preprocessor defines (siren.cu's per-thread activation buffer needs a
    compile-time size), so each distinct (hidden_size, hidden_layers) pair
    triggers its own compile; the result is cached per pair for the
    lifetime of the process. Every other shape (grid_size, image
    dimensions, train_batch, dvr_steps, ...) is a plain runtime kernel
    argument and never triggers a recompile.

    Returns
    -------
    Python module
        Compiled PyTorch extension exposing:

            siren_param_count() -> int
            initialize_siren(seed, device) -> Tensor
            siren_training_step(target_volume, parameters, grid_size,
                                 train_batch, iteration) -> (gradients, loss)
            adam_update(parameters, gradients, first_moment, second_moment,
                        iteration, learning_rate)
            reconstruct_voxel_grid(parameters, grid_size) -> Tensor
            create_ground_truth_volume(grid_size, device) -> Tensor
            render_direct_volume(volume, camera, image_width, image_height,
                                  dvr_steps, density_scale) -> Tensor
            render_voxel_rasterizer(volume, camera, image_width,
                                     image_height, density_scale) -> Tensor
            evaluate_images(image_a, image_b) -> (mse, psnr, ssim)
            compute_volume_psnr(volume_a, volume_b) -> float
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The voxel volume extension requires an NVIDIA GPU."
        )

    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}.")

    if hidden_layers <= 0:
        raise ValueError(f"hidden_layers must be positive, got {hidden_layers}.")

    source_dir = Path(__file__).resolve().parent

    sources = [
        source_dir / "bindings.cpp",
        source_dir / "siren.cu",
        source_dir / "render.cu",
        source_dir / "metrics.cu",
    ]

    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing extension source files:\n" + "\n".join(missing)
        )

    verbose = os.environ.get("VOXEL_VERBOSE_BUILD", "0") == "1"

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        f"-DHIDDEN_SIZE={hidden_size}",
        f"-DHIDDEN_LAYERS={hidden_layers}",
    ]

    extra_cflags = ["-O3"]

    extension = load(
        name=f"voxel_cuda_h{hidden_size}_l{hidden_layers}",
        sources=[str(path) for path in sources],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=verbose,
    )

    required_functions = (
        "siren_param_count",
        "initialize_siren",
        "siren_training_step",
        "adam_update",
        "reconstruct_voxel_grid",
        "create_ground_truth_volume",
        "render_direct_volume",
        "render_voxel_rasterizer",
        "evaluate_images",
        "compute_volume_psnr",
    )

    for function_name in required_functions:
        if not hasattr(extension, function_name):
            raise RuntimeError(
                f"Compiled extension is missing required function {function_name!r}."
            )

    return extension
