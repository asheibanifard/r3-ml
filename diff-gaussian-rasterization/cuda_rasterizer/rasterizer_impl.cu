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

#include "rasterizer_impl.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"


// -----------------------------------------------------------------------------
// Helper function to find the next-highest bit of the MSB on the CPU.
// Same as GraphDECO.
// -----------------------------------------------------------------------------
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;

	while (step > 1)
	{
		step /= 2;

		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}

	if (n >> msb)
		msb++;

	return msb;
}

// -----------------------------------------------------------------------------
// Coarse frustum test.
// Same as GraphDECO.
// -----------------------------------------------------------------------------
__global__ void checkFrustum(
	int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	const auto idx = cg::this_grid().thread_rank();

	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(
		idx,
		orig_points,
		viewmatrix,
		projmatrix,
		false,
		p_view);
}

// -----------------------------------------------------------------------------
// Generate one key/value pair for each Gaussian/tile overlap.
//
// Same GraphDECO structure:
//   key   = [tile_id | depth_bits]
//   value = gaussian_id
//
// For MIP splatting, depth sorting is not strictly required, because we do
// slab-wise max aggregation rather than alpha front-to-back compositing.
// However, keeping the same key layout is useful because:
//   1. it preserves GraphDECO's tile-range infrastructure,
//   2. it gives deterministic ordering,
//   3. it avoids larger changes to the binning pipeline.
// -----------------------------------------------------------------------------
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	const auto idx = cg::this_grid().thread_rank();

	if (idx >= P)
		return;

	if (radii[idx] <= 0)
		return;

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	for (int y = rect_min.y; y < rect_max.y; ++y)
	{
		for (int x = rect_min.x; x < rect_max.x; ++x)
		{
			uint64_t key = y * grid.x + x;
			key <<= 32;

			// Keep depth in low bits for compatibility with original sorting.
			key |= *((uint32_t*)&depths[idx]);

			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			off++;
		}
	}
}

// -----------------------------------------------------------------------------
// Identify start/end range for each tile in the sorted Gaussian list.
// Same as GraphDECO.
// -----------------------------------------------------------------------------
__global__ void identifyTileRanges(
	int L,
	uint64_t* point_list_keys,
	uint2* ranges)
{
	const auto idx = cg::this_grid().thread_rank();

	if (idx >= L)
		return;

	const uint64_t key = point_list_keys[idx];
	const uint32_t currtile = key >> 32;

	if (idx == 0)
	{
		ranges[currtile].x = 0;
	}
	else
	{
		const uint32_t prevtile = point_list_keys[idx - 1] >> 32;

		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}

	if (idx == L - 1)
	{
		ranges[currtile].y = L;
	}
}

// -----------------------------------------------------------------------------
// Public visibility marker.
// Same as GraphDECO.
// -----------------------------------------------------------------------------
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		viewmatrix,
		projmatrix,
		present);
}

// -----------------------------------------------------------------------------
// Geometry state.
//
// Modified from GraphDECO:
//   - conic_opacity becomes conic_intensity
//   - rgb and clamped are removed from the MIP-only forward path
//
// You must also update the GeometryState struct in rasterizer_impl.h
// to contain:
//   float* depths;
//   int* internal_radii;
//   float2* means2D;
//   float* cov3D;
//   float4* conic_intensity;
//   uint32_t* tiles_touched;
//   void* scanning_space;
//   size_t scan_size;
//   uint32_t* point_offsets;
// -----------------------------------------------------------------------------
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(
	char*& chunk,
	size_t P)
{
	GeometryState geom;

	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_intensity, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);

	cub::DeviceScan::InclusiveSum(
		nullptr,
		geom.scan_size,
		geom.tiles_touched,
		geom.tiles_touched,
		P);

	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);

	return geom;
}

// -----------------------------------------------------------------------------
// Image state.
//
// Modified from GraphDECO:
//   - no accum_alpha
//   - no n_contrib
//   - only tile ranges are needed
//
// You must update ImageState in rasterizer_impl.h to contain:
//   uint2* ranges;
// -----------------------------------------------------------------------------
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(
	char*& chunk,
	size_t N)
{
	ImageState img;

	obtain(chunk, img.ranges, N, 128);

	return img;
}

// -----------------------------------------------------------------------------
// Binning state.
// Same as GraphDECO.
// P here means the number of duplicated Gaussian/tile instances.
// -----------------------------------------------------------------------------
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(
	char*& chunk,
	size_t P)
{
	BinningState binning;

	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);

	cub::DeviceRadixSort::SortPairs(
		nullptr,
		binning.sorting_size,
		binning.point_list_keys_unsorted,
		binning.point_list_keys,
		binning.point_list_unsorted,
		binning.point_list,
		P);

	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);

	return binning;
}

// -----------------------------------------------------------------------------
// Forward MIP rendering procedure.
//
// This replaces GraphDECO's RGB alpha-composited forward renderer.
//
// Original GraphDECO forward:
//   means3D + opacity + SH/color
//   -> projected Gaussian splats
//   -> front-to-back alpha compositing
//   -> RGB image
//
// Modified MIP forward:
//   means3D + intensity
//   -> projected Gaussian splats
//   -> depth-slab accumulation
//   -> max over slabs
//   -> scalar MIP image
//
// Required caller-side API change:
//   - background removed
//   - shs/colors_precomp removed
//   - opacities renamed/reinterpreted as intensities
//   - out_color replaced by out_mip
//   - out_depth optional
//   - num_depth_slabs, depth_min, depth_max added
// -----------------------------------------------------------------------------
int CudaRasterizer::Rasterizer::forward_mip(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,

	const int P,

	const int width,
	const int height,

	const float* means3D,
	const float* intensities,

	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,

	const float* viewmatrix,
	const float* projmatrix,

	const float tan_fovx,
	const float tan_fovy,
	const float low_pass,

	const bool prefiltered,

	const int num_depth_slabs,
	const float depth_min,
	const float depth_max,

	float* out_mip,
	float* out_depth,

	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// Geometry buffers.
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);

	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Image buffers.
	// Only ranges are required for this forward-only MIP renderer.
	size_t img_chunk_size = required<ImageState>(tile_grid.x * tile_grid.y);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, tile_grid.x * tile_grid.y);

	// -------------------------------------------------------------------------
	// Preprocess:
	//   - frustum culling
	//   - covariance construction/projection
	//   - 2D conic computation
	//   - radius/tile coverage
	//   - pack conic + intensity
	// -------------------------------------------------------------------------
	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		intensities,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		width,
		height,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		low_pass,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.conic_intensity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// -------------------------------------------------------------------------
	// Prefix sum over tile counts.
	// Example:
	//   tiles_touched = [2, 3, 0, 2, 1]
	//   point_offsets = [2, 5, 5, 7, 8]
	// -------------------------------------------------------------------------
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space,
		geomState.scan_size,
		geomState.tiles_touched,
		geomState.point_offsets,
		P), debug)

	// Total duplicated Gaussian/tile instances.
	int num_rendered = 0;

	if (P > 0)
	{
		CHECK_CUDA(cudaMemcpy(
			&num_rendered,
			geomState.point_offsets + P - 1,
			sizeof(int),
			cudaMemcpyDeviceToHost), debug)
	}

	// If no Gaussian touches any tile, return empty output.
	if (num_rendered == 0)
	{
		CHECK_CUDA(cudaMemset(out_mip, 0, width * height * sizeof(float)), debug)

		if (out_depth != nullptr)
		{
			// cudaMemset cannot set +inf, so leave this to caller or add a small kernel
			// if you require +inf empty depth.
			CHECK_CUDA(cudaMemset(out_depth, 0, width * height * sizeof(float)), debug)
		}

		return 0;
	}

	// -------------------------------------------------------------------------
	// Binning buffers sized by duplicated Gaussian/tile instances.
	// -------------------------------------------------------------------------
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// -------------------------------------------------------------------------
	// Duplicate each visible Gaussian into all touched tiles.
	// -------------------------------------------------------------------------
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid);
	CHECK_CUDA(, debug)

	// -------------------------------------------------------------------------
	// Sort duplicated Gaussian instances by tile ID and depth.
	// Only tile grouping is required for MIP, but depth is kept in the key
	// for compatibility/determinism.
	// -------------------------------------------------------------------------
	const int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted,
		binningState.point_list_keys,
		binningState.point_list_unsorted,
		binningState.point_list,
		num_rendered,
		0,
		32 + bit), debug)

	// -------------------------------------------------------------------------
	// Identify per-tile Gaussian ranges.
	// -------------------------------------------------------------------------
	CHECK_CUDA(cudaMemset(
		imgState.ranges,
		0,
		tile_grid.x * tile_grid.y * sizeof(uint2)), debug)

	identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
		num_rendered,
		binningState.point_list_keys,
		imgState.ranges);
	CHECK_CUDA(, debug)

	// -------------------------------------------------------------------------
	// Render scalar MIP image.
	// -------------------------------------------------------------------------
	CHECK_CUDA(FORWARD::render_mip(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width,
		height,
		geomState.means2D,
		geomState.depths,
		geomState.conic_intensity,
		num_depth_slabs,
		depth_min,
		depth_max,
		out_mip,
		out_depth), debug)

	return num_rendered;
}



