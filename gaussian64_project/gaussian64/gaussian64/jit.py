from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

from .model import GaussianVolume


@lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    root = Path(__file__).resolve().parents[1]
    return load(
        name="gaussian64_normalized_cuda",
        sources=[
            str(root / "csrc" / "bindings.cpp"),
            str(root / "csrc" / "normalized_rasterize.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=verbose,
    )


@torch.no_grad()
def reconstruct_jit(
    model: GaussianVolume,
    shape: tuple[int, int, int],
    verbose_build: bool = False,
) -> Tensor:
    if model.means.device.type != "cuda":
        raise RuntimeError("The JIT rasteriser requires a CUDA model.")
    extension = load_extension(verbose_build)
    return extension.normalized_rasterize(
        model.means.detach().float().contiguous(),
        model.precision6().detach().float().contiguous(),
        model.confidences.detach().float().contiguous(),
        model.features.detach().float().contiguous(),
        *shape,
        model.epsilon,
        model.cutoff_sigma,
    )


@torch.no_grad()
def reconstruct_torch(
    model: GaussianVolume,
    shape: tuple[int, int, int],
    voxel_chunk: int,
    gaussian_chunk: int,
) -> Tensor:
    d, h, w = shape
    z, y, x = torch.meshgrid(
        torch.arange(d, device=model.means.device, dtype=torch.float32) + 0.5,
        torch.arange(h, device=model.means.device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=model.means.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    coordinates = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
    pieces = []
    for start in range(0, coordinates.shape[0], voxel_chunk):
        pieces.append(
            model.evaluate(
                coordinates[start : start + voxel_chunk],
                gaussian_chunk=gaussian_chunk,
            )
        )
    return torch.cat(pieces).reshape(d, h, w)
