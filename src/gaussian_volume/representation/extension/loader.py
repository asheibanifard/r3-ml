# gaussian_volume/representation/extension/loader.py

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_gaussian_volume_extension():
    """
    JIT-compile and load the Gaussian volume CUDA extension.

    Returns
    -------
    Python module
        Compiled PyTorch extension exposing:

            forward(...)
            backward(...)

    Notes
    -----
    Compilation happens once per Python environment/build configuration.
    PyTorch caches the compiled extension under its extension cache.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The Gaussian volume extension requires an NVIDIA GPU."
        )

    # bindings.cpp and gaussian_volume.cu are siblings of this file, inside
    # the same extension/ package directory.
    source_dir = Path(__file__).resolve().parent

    cpp_source = source_dir / "bindings.cpp"
    cuda_source = source_dir / "gaussian_volume.cu"

    missing = [
        str(path)
        for path in (cpp_source, cuda_source)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing extension source files:\n"
            + "\n".join(missing)
        )

    verbose = os.environ.get(
        "GAUSSIAN_VOLUME_VERBOSE_BUILD",
        "0",
    ) == "1"

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]

    extra_cflags = [
        "-O3",
    ]

    extension = load(
        name="gaussian_volume_cuda",
        sources=[
            str(cpp_source),
            str(cuda_source),
        ],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=verbose,
    )

    required_functions = (
        "forward",
        "backward",
    )

    for function_name in required_functions:
        if not hasattr(extension, function_name):
            raise RuntimeError(
                "Compiled extension is missing required function "
                f"{function_name!r}."
            )

    return extension
