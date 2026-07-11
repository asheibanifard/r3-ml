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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"

#include <stdio.h>
#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS  (BLOCK_SIZE / 32)

// -----------------------------------------------------------------------------
// Convert normalized device coordinate [-1, 1] to pixel coordinate.
// Same convention as GraphDECO.
// -----------------------------------------------------------------------------
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

// -----------------------------------------------------------------------------
// Compute tile rectangle touched by a Gaussian with screen-space radius.
// rect_min inclusive, rect_max exclusive.
// -----------------------------------------------------------------------------
__forceinline__ __device__ void getRect(
	const float2 p,
	int max_radius,
	uint2& rect_min,
	uint2& rect_max,
	dim3 grid)
{
	rect_min = {
		min(grid.x, max(0, static_cast<int>((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max(0, static_cast<int>((p.y - max_radius) / BLOCK_Y)))
	};

	rect_max = {
		min(grid.x, max(0, static_cast<int>((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max(0, static_cast<int>((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

// -----------------------------------------------------------------------------
// Transform point by 4x3 matrix.
// Assumes GraphDECO/GLM-style column-major layout.
// -----------------------------------------------------------------------------
__forceinline__ __device__ float3 transformPoint4x3(
	const float3& p,
	const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14]
	};

	return transformed;
}

// -----------------------------------------------------------------------------
// Transform point by 4x4 matrix.
// Assumes GraphDECO/GLM-style column-major layout.
// -----------------------------------------------------------------------------
__forceinline__ __device__ float4 transformPoint4x4(
	const float3& p,
	const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};

	return transformed;
}

// -----------------------------------------------------------------------------
// Transform vector by 4x3 matrix, ignoring translation.
// Useful if later you need view-direction transforms.
// -----------------------------------------------------------------------------
__forceinline__ __device__ float3 transformVec4x3(
	const float3& p,
	const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z
	};

	return transformed;
}

// -----------------------------------------------------------------------------
// Transform vector by transpose of 4x3 matrix, ignoring translation.
// Kept because covariance/projection code may need this convention.
// -----------------------------------------------------------------------------
__forceinline__ __device__ float3 transformVec4x3Transpose(
	const float3& p,
	const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2]  * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6]  * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z
	};

	return transformed;
}

// -----------------------------------------------------------------------------
// Frustum test.
//
// Original GraphDECO assumes positive camera-space z in front of the camera.
// If your camera convention is different, change the z test here.
//
// For your orthographic/identity smoke tests, you may want to disable the
// z-forward condition or replace it with explicit volume bounds.
// -----------------------------------------------------------------------------
__forceinline__ __device__ bool in_frustum(
	int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	const float3 p_orig = {
		orig_points[3 * idx + 0],
		orig_points[3 * idx + 1],
		orig_points[3 * idx + 2]
	};

	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float inv_w = 1.0f / (p_hom.w + 1e-7f);

	const float3 p_proj = {
		p_hom.x * inv_w,
		p_hom.y * inv_w,
		p_hom.z * inv_w
	};

	p_view = transformPoint4x3(p_orig, viewmatrix);

	// Original GraphDECO near-plane test.
	// For standard perspective 3DGS, this is fine.
	// For volume/MIP rendering with arbitrary camera conventions, verify this.
	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This should not happen.\n");
			__trap();
		}

		return false;
	}

	// Optional stricter screen-space culling.
	// Usually disabled because Gaussians slightly outside screen can still
	// contribute through their footprint.
	//
	// if (p_proj.x < -1.3f || p_proj.x > 1.3f ||
	//     p_proj.y < -1.3f || p_proj.y > 1.3f)
	// {
	//     return false;
	// }

	return true;
}

// -----------------------------------------------------------------------------
// CUDA error checker.
// Same macro style as GraphDECO.
// -----------------------------------------------------------------------------
#define CHECK_CUDA(A, debug)                                      \
	A;                                                            \
	if (debug)                                                    \
	{                                                            \
		auto ret = cudaDeviceSynchronize();                       \
		if (ret != cudaSuccess)                                  \
		{                                                        \
			std::cerr << "\n[CUDA ERROR] in " << __FILE__         \
			          << "\nLine " << __LINE__ << ": "           \
			          << cudaGetErrorString(ret) << std::endl;   \
			throw std::runtime_error(cudaGetErrorString(ret));   \
		}                                                        \
	}

#endif