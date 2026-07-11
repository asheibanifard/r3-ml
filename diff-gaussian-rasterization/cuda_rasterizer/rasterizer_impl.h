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

#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(
		char*& chunk,
		T*& ptr,
		std::size_t count,
		std::size_t alignment)
	{
		std::size_t offset =
			(reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1)
			& ~(alignment - 1);

		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// -------------------------------------------------------------------------
	// GeometryState
	//
	// Original GraphDECO state stored:
	//   clamped, rgb, conic_opacity
	//
	// MIP-splat version stores:
	//   depths            : camera-space Gaussian depth
	//   means2D           : projected 2D Gaussian centre
	//   cov3D             : optional internally computed 3D covariance
	//   conic_intensity   : (A, B, C, intensity)
	//   tiles_touched     : number of tiles touched per Gaussian
	//   point_offsets     : prefix sum over tiles_touched
	// -------------------------------------------------------------------------
	struct GeometryState
	{
		size_t scan_size;

		float* depths;
		int* internal_radii;
		float2* means2D;
		float* cov3D;

		// x, y, z = inverse 2D conic parameters
		// w       = scalar Gaussian intensity
		float4* conic_intensity;

		uint32_t* tiles_touched;

		char* scanning_space;
		uint32_t* point_offsets;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	// -------------------------------------------------------------------------
	// ImageState
	//
	// Original GraphDECO state stored:
	//   accum_alpha, n_contrib, ranges
	//
	// MIP-splat version only needs:
	//   ranges : per-tile [start, end) in sorted Gaussian instance list
	// -------------------------------------------------------------------------
	struct ImageState
	{
		uint2* ranges;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	// -------------------------------------------------------------------------
	// BinningState
	//
	// Same as GraphDECO.
	// Stores duplicated Gaussian/tile instances and radix-sort buffers.
	// -------------------------------------------------------------------------
	struct BinningState
	{
		size_t sorting_size;

		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;

		uint32_t* point_list_unsorted;
		uint32_t* point_list;

		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template <typename T>
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return reinterpret_cast<size_t>(size) + 128;
	}
}