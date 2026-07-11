#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# Modified for forward-only scalar depth-layered Gaussian MIP splatting.
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_INCLUDE_DIR = os.path.join(ROOT_DIR, "third_party", "glm")


setup(
    name="mip_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-I" + GLM_INCLUDE_DIR,
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)