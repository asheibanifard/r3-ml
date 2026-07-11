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

#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

// -----------------------------------------------------------------------------
// Forward-only scalar depth-layered Gaussian MIP splatting.
//
// Inputs:
//   means3D           [P, 3]
//   intensities       [P] or [P, 1]
//   scales            [P, 3], used if cov3D_precomp is empty/null
//   rotations         [P, 4], used if cov3D_precomp is empty/null
//   scale_modifier    scalar multiplier for Gaussian scales
//   cov3D_precomp     [P, 6], optional precomputed upper-triangular covariance
//   viewmatrix        [4, 4]
//   projmatrix        [4, 4]
//   tan_fovx/y        camera parameters
//   image_height/width
//   num_depth_slabs   number of depth slabs used for MIP aggregation
//   depth_min/max     camera-space slab range
//
// Outputs tuple:
//   rendered          number of duplicated Gaussian/tile instances
//   out_mip           [H, W]
//   out_depth         [H, W]
//   radii             [P]
//   geomBuffer
//   binningBuffer
//   imgBuffer
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
	const bool debug);

// -----------------------------------------------------------------------------
// Visibility check.
// Kept from GraphDECO.
// -----------------------------------------------------------------------------
torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix);