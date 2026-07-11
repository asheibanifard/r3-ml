/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * Modified for forward-only scalar depth-layered Gaussian MIP splatting.
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// Tile/block size used by the rasterization kernels.
#define BLOCK_X 16
#define BLOCK_Y 16

// Number of threads per tile block:
// BLOCK_SIZE = BLOCK_X * BLOCK_Y
// Defined in auxiliary.h.

// Maximum number of depth slabs used inside the per-pixel MIP kernel.
// This must be compile-time fixed because slab_sum[MAX_DEPTH_SLABS]
// is a local CUDA array.
#define MAX_DEPTH_SLABS 256

#endif