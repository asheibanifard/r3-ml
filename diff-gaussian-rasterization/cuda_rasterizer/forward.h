/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * Modified for depth-layered Gaussian MIP splatting.
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	void preprocess(
		int P,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* intensities,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const int W,
		int H,
		const float focal_x,
		float focal_y,
		const float tan_fovx,
		float tan_fovy,
		const float low_pass,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float4* conic_intensity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	void render_mip(
		const dim3 grid,
		dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W,
		int H,
		const float2* means2D,
		const float* depths,
		const float4* conic_intensity,
		const int num_depth_slabs,
		const float depth_min,
		const float depth_max,
		float* out_mip,
		float* out_depth);
}

#endif