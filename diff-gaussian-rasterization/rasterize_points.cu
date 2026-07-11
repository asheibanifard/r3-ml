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

#include <math.h>
#include <torch/extension.h>

#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({ static_cast<long long>(N) });
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};

	return lambda;
}


// -----------------------------------------------------------------------------
// Forward-only scalar MIP rasterization.
//
// Inputs:
//   means3D:         [P, 3]
//   intensities:     [P] or [P, 1]
//   scales:          [P, 3], optional if cov3D_precomp is provided
//   rotations:       [P, 4], optional if cov3D_precomp is provided
//   cov3D_precomp:   [P, 6], optional
//   viewmatrix:      [4, 4]
//   projmatrix:      [4, 4]
//   tan_fovx/y:      camera parameters
//   image_height/w:  output resolution
//   num_depth_slabs: number of MIP depth slabs
//   depth_min/max:   camera-space depth range for slab assignment
//
// Outputs:
//   rendered:        number of duplicated Gaussian/tile instances
//   out_mip:         [H, W]
//   out_depth:       [H, W]
//   radii:           [P]
//   geomBuffer, binningBuffer, imgBuffer
// -----------------------------------------------------------------------------
std::tuple<
	int,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor,
	torch::Tensor>
RasterizeGaussiansMIPCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& intensities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float low_pass,
	const int image_height,
	const int image_width,
	const int num_depth_slabs,
	const float depth_min,
	const float depth_max,
	const bool prefiltered,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions [num_points, 3]");
	}

	if (intensities.ndimension() != 1 &&
		!(intensities.ndimension() == 2 && intensities.size(1) == 1))
	{
		AT_ERROR("intensities must have dimensions [num_points] or [num_points, 1]");
	}

	if (intensities.size(0) != means3D.size(0))
	{
		AT_ERROR("intensities and means3D must have the same number of Gaussians");
	}

	if (viewmatrix.numel() != 16)
	{
		AT_ERROR("viewmatrix must contain 16 values");
	}

	if (projmatrix.numel() != 16)
	{
		AT_ERROR("projmatrix must contain 16 values");
	}

	if (num_depth_slabs <= 0)
	{
		AT_ERROR("num_depth_slabs must be positive");
	}

	if (num_depth_slabs > MAX_DEPTH_SLABS)
	{
		AT_ERROR("num_depth_slabs exceeds MAX_DEPTH_SLABS");
	}

	if (depth_max <= depth_min)
	{
		AT_ERROR("depth_max must be greater than depth_min");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means3D.options().dtype(torch::kFloat32);
	auto int_opts = means3D.options().dtype(torch::kInt32);

	torch::Tensor out_mip = torch::full({ H, W }, 0.0f, float_opts);
	torch::Tensor out_depth = torch::full({ H, W }, 0.0f, float_opts);
	torch::Tensor radii = torch::full({ P }, 0, int_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);

	torch::Tensor geomBuffer = torch::empty({ 0 }, byte_options);
	torch::Tensor binningBuffer = torch::empty({ 0 }, byte_options);
	torch::Tensor imgBuffer = torch::empty({ 0 }, byte_options);

	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;

	if (P != 0)
	{
		rendered = CudaRasterizer::Rasterizer::forward_mip(
			geomFunc,
			binningFunc,
			imgFunc,

			P,

			W,
			H,

			means3D.contiguous().data_ptr<float>(),
			intensities.contiguous().data_ptr<float>(),

			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),

			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),

			tan_fovx,
			tan_fovy,
			low_pass,

			prefiltered,

			num_depth_slabs,
			depth_min,
			depth_max,

			out_mip.contiguous().data_ptr<float>(),
			out_depth.contiguous().data_ptr<float>(),

			radii.contiguous().data_ptr<int>(),
			debug);
	}

	return std::make_tuple(
		rendered,
		out_mip,
		out_depth,
		radii,
		geomBuffer,
		binningBuffer,
		imgBuffer);
}


// -----------------------------------------------------------------------------
// Visibility check.
// Kept from GraphDECO.
// -----------------------------------------------------------------------------
torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full(
		{ P },
		false,
		means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(
			P,
			means3D.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			present.contiguous().data_ptr<bool>());
	}

	return present;
}