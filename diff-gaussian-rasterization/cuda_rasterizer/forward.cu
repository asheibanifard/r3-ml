/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * Modified for scalar depth-layered Gaussian MIP splatting.
 */

#include "forward.h"
#include "auxiliary.h"
#include <math.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#ifndef MAX_DEPTH_SLABS
#define MAX_DEPTH_SLABS 64
#endif

// -----------------------------------------------------------------------------
// Forward version of 2D covariance matrix computation.
// Same role as GraphDECO:
//     3D covariance + camera projection -> 2D screen-space covariance.
// -----------------------------------------------------------------------------
__device__ float3 computeCov2D(
	const float3& mean,
	float focal_x,
	float focal_y,
	float tan_fovx,
	float tan_fovy,
	const float* cov3D,
	const float* viewmatrix,
	float low_pass)
{
	// Based on EWA splatting.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;

	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;

	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0f, 0.0f, 0.0f);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Low-pass filter inherited from GraphDECO.
	// Prevents sub-pixel singular splats. Tunable: the original constant
	// (0.3 px^2) assumes multi-pixel splats; voxel-fit Gaussians can be
	// sub-pixel by design, so a caller may pass a much smaller value.
	cov[0][0] += low_pass;
	cov[1][1] += low_pass;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// -----------------------------------------------------------------------------
// Convert scale + quaternion to 3D covariance.
// Same as GraphDECO, but quaternion is explicitly normalized.
// Assumes rotation layout glm::vec4(r, x, y, z), i.e. wxyz.
// -----------------------------------------------------------------------------
__device__ void computeCov3D(
	const glm::vec3 scale,
	float mod,
	const glm::vec4 rot,
	float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot / glm::length(rot);

	const float r = q.x;
	const float x = q.y;
	const float y = q.z;
	const float z = q.w;

	glm::mat3 R = glm::mat3(
		1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - r * z),       2.0f * (x * z + r * y),
		2.0f * (x * y + r * z),       1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - r * x),
		2.0f * (x * z - r * y),       2.0f * (y * z + r * x),       1.0f - 2.0f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Store upper-triangular covariance.
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// -----------------------------------------------------------------------------
// Preprocess each Gaussian prior to MIP splatting.
//
// Difference from GraphDECO:
//   - opacities -> intensities
//   - conic_opacity -> conic_intensity
//   - no SH/color computation required
//
// conic_intensity layout:
//   x = inverse 2D conic A
//   y = inverse 2D conic B
//   z = inverse 2D conic C
//   w = scalar Gaussian intensity
// -----------------------------------------------------------------------------
__global__ void preprocessCUDA(
	int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* intensities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W,
	const int H,
	const float tan_fovx,
	const float tan_fovy,
	const float focal_x,
	const float focal_y,
	const float low_pass,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* conic_intensity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	const auto idx = cg::this_grid().thread_rank();

	if (idx >= P)
		return;

	// If unchanged, this Gaussian will not be rasterized.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Frustum culling.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	const float3 p_orig = {
		orig_points[3 * idx + 0],
		orig_points[3 * idx + 1],
		orig_points[3 * idx + 2]
	};

	// Project mean to NDC.
	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float p_w = 1.0f / (p_hom.w + 1e-7f);
	const float3 p_proj = {
		p_hom.x * p_w,
		p_hom.y * p_w,
		p_hom.z * p_w
	};

	// Use precomputed covariance if provided, otherwise build from scale/rotation.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Project covariance to 2D screen space.
	const float3 cov = computeCov2D(
		p_orig,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		cov3D,
		viewmatrix,
		low_pass);

	// Invert 2D covariance.
	const float det = cov.x * cov.z - cov.y * cov.y;
	if (det <= 0.0f || !isfinite(det))
		return;

	const float det_inv = 1.0f / det;
	const float3 conic = {
		cov.z * det_inv,
		-cov.y * det_inv,
		cov.x * det_inv
	};

	if (!isfinite(conic.x) || !isfinite(conic.y) || !isfinite(conic.z))
		return;

	// Compute projected radius from largest eigenvalue.
	const float mid = 0.5f * (cov.x + cov.z);
	const float discr = max(0.1f, mid * mid - det);
	const float lambda1 = mid + sqrtf(discr);
	const float lambda2 = mid - sqrtf(discr);
	const float my_radius = ceilf(3.0f * sqrtf(max(lambda1, lambda2)));

	const float2 point_image = {
		ndc2Pix(p_proj.x, W),
		ndc2Pix(p_proj.y, H)
	};

	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	const float intensity = intensities[idx];

	if (!isfinite(intensity) || intensity <= 0.0f)
		return;

	// Store helper data for binning and rendering.
	depths[idx] = p_view.z;
	radii[idx] = static_cast<int>(my_radius);
	points_xy_image[idx] = point_image;

	// Pack conic and scalar intensity.
	conic_intensity[idx] = { conic.x, conic.y, conic.z, intensity };

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// -----------------------------------------------------------------------------
// Convert camera-space depth to slab index.
// -----------------------------------------------------------------------------
__device__ __forceinline__ int depthToSlab(
	float depth,
	float depth_min,
	float depth_max,
	int num_slabs)
{
	const float span = fmaxf(depth_max - depth_min, 1e-6f);
	int slab = static_cast<int>((depth - depth_min) / span * static_cast<float>(num_slabs));
	return min(max(slab, 0), num_slabs - 1);
}

// -----------------------------------------------------------------------------
// MIP splatting render kernel.
//
// Difference from GraphDECO:
//   - no T/transmittance
//   - no alpha-composited RGB
//   - no early termination by opacity
//   - each splat contributes scalar intensity to a depth slab
//   - output is max over slab sums
//
// Formula:
//   M(p) = max_k sum_{i in slab k} intensity_i exp(-0.5 * mahal_i(p))
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderMIPCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W,
	int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_intensity,
	const int num_depth_slabs,
	const float depth_min,
	const float depth_max,
	float* __restrict__ out_mip,
	float* __restrict__ out_depth)
{
	const auto block = cg::this_thread_block();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	const uint2 pix_min = {
		block.group_index().x * BLOCK_X,
		block.group_index().y * BLOCK_Y
	};

	const uint2 pix_max = {
		min(pix_min.x + BLOCK_X, W),
		min(pix_min.y + BLOCK_Y, H)
	};

	const uint2 pix = {
		pix_min.x + block.thread_index().x,
		pix_min.y + block.thread_index().y
	};

	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { static_cast<float>(pix.x), static_cast<float>(pix.y) };

	const bool inside = pix.x < W && pix.y < H;
	const bool done = !inside;

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float4 collected_conic_intensity[BLOCK_SIZE];

	float slab_sum[MAX_DEPTH_SLABS];

	const int slabs = min(max(num_depth_slabs, 1), MAX_DEPTH_SLABS);
	for (int s = 0; s < MAX_DEPTH_SLABS; ++s)
		slab_sum[s] = 0.0f;

	for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
	{
		// All threads participate in fetching, even invalid-pixel threads.
		const int progress = i * BLOCK_SIZE + block.thread_rank();

		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];

			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_depth[block.thread_rank()] = depths[coll_id];
			collected_conic_intensity[block.thread_rank()] = conic_intensity[coll_id];
		}

		block.sync();

		if (!done)
		{
			const int batch_count = min(BLOCK_SIZE, toDo);

			for (int j = 0; j < batch_count; ++j)
			{
				const float2 xy = collected_xy[j];
				const float2 d = {
					xy.x - pixf.x,
					xy.y - pixf.y
				};

				const float4 con_i = collected_conic_intensity[j];

				// GraphDECO conic form:
				// power = -0.5 * (A dx^2 + C dy^2) - B dx dy
				//       = -0.5 * (A dx^2 + 2B dxdy + C dy^2)
				const float power =
					-0.5f * (con_i.x * d.x * d.x + con_i.z * d.y * d.y)
					- con_i.y * d.x * d.y;

				// Outside Gaussian support or invalid numerical value.
				if (power > 0.0f || !isfinite(power))
					continue;

				const float weight = expf(power);
				const float contrib = con_i.w * weight;

				if (!isfinite(contrib) || contrib <= 0.0f)
					continue;

				const int slab = depthToSlab(
					collected_depth[j],
					depth_min,
					depth_max,
					slabs);

				slab_sum[slab] += contrib;
			}
		}

		block.sync();
	}

	if (inside)
	{
		float best = 0.0f;
		int best_slab = -1;

		for (int s = 0; s < slabs; ++s)
		{
			const float v = slab_sum[s];

			if (v > best)
			{
				best = v;
				best_slab = s;
			}
		}

		out_mip[pix_id] = best;

		if (out_depth != nullptr)
		{
			if (best_slab >= 0)
			{
				const float slab_width = fmaxf(depth_max - depth_min, 1e-6f) / static_cast<float>(slabs);
				out_depth[pix_id] = depth_min + (static_cast<float>(best_slab) + 0.5f) * slab_width;
			}
			else
			{
				out_depth[pix_id] = INFINITY;
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Host wrapper: MIP render.
// This replaces FORWARD::render from the original alpha-compositing version.
// -----------------------------------------------------------------------------
void FORWARD::render_mip(
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
	float* out_depth)
{
	renderMIPCUDA<<<grid, block>>>(
		ranges,
		point_list,
		W,
		H,
		means2D,
		depths,
		conic_intensity,
		num_depth_slabs,
		depth_min,
		depth_max,
		out_mip,
		out_depth);
}

// -----------------------------------------------------------------------------
// Host wrapper: preprocess.
// This replaces opacities with intensities and removes color/SH dependency.
// -----------------------------------------------------------------------------
void FORWARD::preprocess(
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
	bool prefiltered)
{
	preprocessCUDA<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		intensities,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		W,
		H,
		tan_fovx,
		tan_fovy,
		focal_x,
		focal_y,
		low_pass,
		radii,
		means2D,
		depths,
		cov3Ds,
		conic_intensity,
		grid,
		tiles_touched,
		prefiltered);
}