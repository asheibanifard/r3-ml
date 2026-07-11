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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		// ---------------------------------------------------------------------
		// Forward-only scalar MIP splatting.
		//
		// This replaces the original GraphDECO RGB alpha-composited forward().
		//
		// Original:
		//   means3D + SH/color + opacity
		//   -> alpha-composited RGB image
		//
		// Modified:
		//   means3D + covariance/scale/rotation + scalar intensity
		//   -> depth-layered splat accumulation
		//   -> max over depth slabs
		//   -> scalar MIP image
		//
		// out_mip:
		//   [height, width] scalar MIP image
		//
		// out_depth:
		//   [height, width] optional winning-depth map.
		//   Pass nullptr if not needed.
		// ---------------------------------------------------------------------
		static int forward_mip(
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

			int* radii = nullptr,
			bool debug = false);
	};
};

#endif