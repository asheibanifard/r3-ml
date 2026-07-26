/*
 * splat_mip.cu — tiled true Maximum Intensity Projection of a Gaussian
 * mixture, with conservative per-tile culling.
 *
 * Each 3D Gaussian's peak value along the view axis, at any given
 * screen-space (u,v), has a closed form: marginalizing/maximizing a
 * Gaussian along one axis analytically yields an effective 2D Gaussian
 * in the remaining two axes (see "Effective 2D precision after
 * maximizing along the view axis" below), whose value at (u,v) is
 * exactly that Gaussian's own maximum over the view axis at that pixel.
 * So per-pixel `max_i( intensity_i * exp(-0.5 * effective_2D_mahalanobis_i) )`
 * over every Gaussian covering that pixel IS a true max-intensity
 * projection through the mixture -- no ray marching/depth sampling
 * needed, and no Beer-Lambert opacity mapping: the output is the raw
 * peak density value itself (same convention as
 * gaussian_volume.renderer.rasterisation.dvr_ground_truth's ray-marched
 * ground-truth MIP).
 *
 * Each pixel's Mahalanobis distance to a Gaussian is computed after
 * clamping the offset to that pixel's own footprint (half a pixel-width
 * on each side of its centre), not at its exact centre point -- so every
 * pixel reports a Gaussian's true maximum anywhere within its footprint,
 * and a Gaussian sharp enough to fall between two sample points is still
 * captured exactly by whichever pixel's footprint contains its peak
 * (footprints tile the plane with no gaps). The tile-culling radius
 * below is padded by that same half pixel to stay a conservative bound.
 *
 * Exports:
 *   splat_mip(means, log_s, quats, inten, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
 *             out_h, out_w, depth_samples, view_axis,
 *             scale_min, mahal_clamp,
 *             max_gauss_per_tile=0,
 *             print_stats=false, clamp_output=true)
 */

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "common.cuh"

namespace py = pybind11;

__global__ void splat_mip_tiled_kernel(
        const float* __restrict__ proj_u,
        const float* __restrict__ proj_v,
        const float* __restrict__ si_uu,
        const float* __restrict__ si_uv,
        const float* __restrict__ si_vv,
        const float* __restrict__ inten,
        const int* __restrict__ tile_offsets,
        const int* __restrict__ tile_indices,
        int tiles_x,
        int out_h, int out_w,
        float u_lo, float u_hi,
        float v_lo, float v_hi,
        float mahal_clamp,
        int clamp_output,
        float* __restrict__ out) {
    __shared__ float s_u[K_TILE_GAUSS];
    __shared__ float s_v[K_TILE_GAUSS];
    __shared__ float s_i[K_TILE_GAUSS];
    __shared__ float s_a[K_TILE_GAUSS];
    __shared__ float s_b[K_TILE_GAUSS];
    __shared__ float s_d[K_TILE_GAUSS];

    const int tile_id = blockIdx.x;
    const int tile_y = tile_id / tiles_x;
    const int tile_x = tile_id % tiles_x;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int ir = tile_y * kTilePixH + ty;
    const int ic = tile_x * kTilePixW + tx;
    const bool active = (ir < out_h && ic < out_w);

    const int lane = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;
    const int start = tile_offsets[tile_id];
    const int end = tile_offsets[tile_id + 1];

    // Pixel-centre convention, matching model.py's "voxel index i has
    // centre i+0.5, over [0,size]" (see rasterisation.py's _default_bounds):
    // pixel ic spans [u_lo + ic*pixel_size_u, u_lo + (ic+1)*pixel_size_u),
    // centred at u_lo + (ic+0.5)*pixel_size_u. This also removes the old
    // out_w==1 special case for free (it reduces to the u_lo/u_hi midpoint
    // automatically).
    const float pixel_size_u = (u_hi - u_lo) / static_cast<float>(out_w);
    const float pixel_size_v = (v_hi - v_lo) / static_cast<float>(out_h);
    const float px = u_lo + (static_cast<float>(ic) + 0.5f) * pixel_size_u;
    const float py = v_lo + (static_cast<float>(ir) + 0.5f) * pixel_size_v;

    // Half this pixel's footprint in physical (u,v) units, on each side of
    // its centre. Every point in the (u,v) plane belongs to exactly one
    // pixel's footprint, so clamping each Gaussian's offset into this
    // range before computing its Mahalanobis distance means whichever
    // pixel's footprint contains a Gaussian's true peak always reports
    // that peak exactly (offset clamps to (0,0), mah=0) -- a sharp
    // Gaussian can no longer fall entirely between two sample points.
    // Every other pixel's contribution is now this Gaussian's true
    // maximum anywhere within ITS footprint too, not just its value at
    // one infinitesimal centre point.
    const float half_pixel_u = 0.5f * pixel_size_u;
    const float half_pixel_v = 0.5f * pixel_size_v;

    // Shared-memory loading and both barriers must run for every thread in
    // the block regardless of `active`, or a partial (edge) tile leaves
    // some threads never reaching a __syncthreads() that others do --
    // undefined behaviour per the CUDA barrier semantics. Only the
    // per-pixel evaluation and the final write are safe to gate on
    // `active`.
    float peak = 0.f;
    for (int t0 = start; t0 < end; t0 += K_TILE_GAUSS) {
        const int tn = std::min(K_TILE_GAUSS, end - t0);

        for (int i = lane; i < tn; i += nthreads) {
            const int g = tile_indices[t0 + i];
            s_u[i] = proj_u[g];
            s_v[i] = proj_v[g];
            s_a[i] = si_uu[g];
            s_b[i] = si_uv[g];
            s_d[i] = si_vv[g];
            s_i[i] = inten[g];
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tn; ++i) {
                const float raw_du = px - s_u[i];
                const float raw_dv = py - s_v[i];
                // Residual distance from the pixel footprint's *nearest edge*
                // to the Gaussian, not the clamped value itself: 0 when the
                // Gaussian's peak is within this pixel's footprint, else the
                // true distance shrunk by exactly half a pixel-width.
                const float du = raw_du - fminf(fmaxf(raw_du, -half_pixel_u), half_pixel_u);
                const float dv = raw_dv - fminf(fmaxf(raw_dv, -half_pixel_v), half_pixel_v);
                const float mah = du * (s_a[i] * du + s_b[i] * dv)
                                + dv * (s_b[i] * du + s_d[i] * dv);
                if (mah < mahal_clamp) {
                    const float contribution = s_i[i] * __expf(-0.5f * mah);
                    peak = fmaxf(peak, contribution);
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        out[ir * out_w + ic] = clamp_output ? fminf(fmaxf(peak, 0.f), 1.f) : peak;
    }
}

torch::Tensor splat_mip(
        torch::Tensor means, torch::Tensor log_s, torch::Tensor quats, torch::Tensor inten,
        float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z,
        int out_h, int out_w, int depth_samples, int view_axis,
        float scale_min, float mahal_clamp,
        int max_gauss_per_tile,
        bool print_stats,
        bool clamp_output) {
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(log_s.is_contiguous(), "log_s must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(inten.is_contiguous(), "inten must be contiguous");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(view_axis >= 0 && view_axis <= 2, "view_axis must be 0, 1, or 2");
    TORCH_CHECK(max_gauss_per_tile >= 0, "max_gauss_per_tile must be >= 0; use 0 for unlimited");

    const int N = static_cast<int>(means.size(0));
    auto out = torch::zeros({out_h * out_w}, means.options());

    int u_axis = 0, v_axis = 1;
    float u_lo = 0.f, u_hi = 1.f, v_lo = 0.f, v_hi = 1.f;
    projected_axes_for_view_axis(view_axis, u_axis, v_axis, u_lo, u_hi, v_lo, v_hi,
                                 lo_x, hi_x, lo_y, hi_y, lo_z, hi_z);
    // Pixel-index scale matching the kernel's pixel-centre convention
    // (pixel ic centred at u_lo + (ic+0.5)/u_scale): 1 pixel-index unit ==
    // one pixel-width, out_w pixels spanning [u_lo,u_hi] -- not out_w-1.
    const float u_scale = static_cast<float>(out_w) / (u_hi - u_lo);
    const float v_scale = static_cast<float>(out_h) / (v_hi - v_lo);
    const int tiles_x = (out_w + kTilePixW - 1) / kTilePixW;
    const int tiles_y = (out_h + kTilePixH - 1) / kTilePixH;
    const int num_tiles = tiles_x * tiles_y;

    auto means_cpu = means.contiguous().to(torch::kCPU);
    auto log_s_cpu = log_s.contiguous().to(torch::kCPU);
    auto quats_cpu = quats.contiguous().to(torch::kCPU);
    auto inten_cpu = inten.contiguous().to(torch::kCPU);

    const float* m_ptr = means_cpu.data_ptr<float>();
    const float* ls_ptr = log_s_cpu.data_ptr<float>();
    const float* q_ptr = quats_cpu.data_ptr<float>();
    const float* i_ptr = inten_cpu.data_ptr<float>();

    std::vector<float> proj_u(N), proj_v(N);
    std::vector<float> si_uu(N), si_uv(N), si_vv(N);
    std::vector<float> proj_iv(N);
    std::vector<std::vector<int>> tile_lists(num_tiles);
    const float radius_scale = std::sqrt(std::max(mahal_clamp, 0.f));

    for (int g = 0; g < N; ++g) {
        float R00, R01, R02, R10, R11, R12, R20, R21, R22;
        quat_to_rotmat_host(q_ptr[g * 4 + 0], q_ptr[g * 4 + 1], q_ptr[g * 4 + 2], q_ptr[g * 4 + 3],
                            R00, R01, R02, R10, R11, R12, R20, R21, R22);

        const float sx = std::max(std::exp(ls_ptr[g * 3 + 0]), scale_min);
        const float sy = std::max(std::exp(ls_ptr[g * 3 + 1]), scale_min);
        const float sz = std::max(std::exp(ls_ptr[g * 3 + 2]), scale_min);

        const float sx2 = sx * sx, sy2 = sy * sy, sz2 = sz * sz;
        const float C00 = R00 * R00 * sx2 + R01 * R01 * sy2 + R02 * R02 * sz2;
        const float C01 = R00 * R10 * sx2 + R01 * R11 * sy2 + R02 * R12 * sz2;
        const float C02 = R00 * R20 * sx2 + R01 * R21 * sy2 + R02 * R22 * sz2;
        const float C11 = R10 * R10 * sx2 + R11 * R11 * sy2 + R12 * R12 * sz2;
        const float C12 = R10 * R20 * sx2 + R11 * R21 * sy2 + R12 * R22 * sz2;
        const float C22 = R20 * R20 * sx2 + R21 * R21 * sy2 + R22 * R22 * sz2;

        const float isx = 1.f / sx;
        const float isy = 1.f / sy;
        const float isz = 1.f / sz;
        const float D00 = R00 * isx, D01 = R01 * isy, D02 = R02 * isz;
        const float D10 = R10 * isx, D11 = R11 * isy, D12 = R12 * isz;
        const float D20 = R20 * isx, D21 = R21 * isy, D22 = R22 * isz;
        const float Ixx = D00 * D00 + D01 * D01 + D02 * D02;
        const float Ixy = D00 * D10 + D01 * D11 + D02 * D12;
        const float Ixz = D00 * D20 + D01 * D21 + D02 * D22;
        const float Iyy = D10 * D10 + D11 * D11 + D12 * D12;
        const float Iyz = D10 * D20 + D11 * D21 + D12 * D22;
        const float Izz = D20 * D20 + D21 * D21 + D22 * D22;

        const float mx = m_ptr[g * 3 + 0];
        const float my = m_ptr[g * 3 + 1];
        const float mz = m_ptr[g * 3 + 2];

        float mu_u_w = 0.f, mu_v_w = 0.f;
        float cuu = 0.f, cuv = 0.f, cvv = 0.f;
        if (view_axis == 0) {
            mu_u_w = mx; mu_v_w = my;
            cuu = C00; cuv = C01; cvv = C11;
        } else if (view_axis == 1) {
            mu_u_w = mx; mu_v_w = mz;
            cuu = C00; cuv = C02; cvv = C22;
        } else {
            mu_u_w = my; mu_v_w = mz;
            cuu = C11; cuv = C12; cvv = C22;
        }

        // Effective 2D precision after maximizing along the view axis.
        float q00 = 0.f, q01 = 0.f, q11 = 0.f, q02 = 0.f, q12 = 0.f, q22 = 1.f;
        if (view_axis == 0) {
            q00 = Ixx; q01 = Ixy; q11 = Iyy; q02 = Ixz; q12 = Iyz; q22 = Izz;
        } else if (view_axis == 1) {
            q00 = Ixx; q01 = Ixz; q11 = Izz; q02 = Ixy; q12 = Iyz; q22 = Iyy;
        } else {
            q00 = Iyy; q01 = Iyz; q11 = Izz; q02 = Ixy; q12 = Ixz; q22 = Ixx;
        }
        const float inv_q22 = 1.f / std::max(q22, 1e-8f);
        const float eff00 = q00 - q02 * q02 * inv_q22;
        const float eff01 = q01 - q02 * q12 * inv_q22;
        const float eff11 = q11 - q12 * q12 * inv_q22;

        const float puu = cuu * u_scale * u_scale;
        const float puv = cuv * u_scale * v_scale;
        const float pvv = cvv * v_scale * v_scale;
        const float tr = puu + pvv;
        const float disc = std::sqrt(std::max(0.f, (puu - pvv) * (puu - pvv) + 4.f * puv * puv));
        const float lam = 0.5f * (tr + disc);
        // + 0.5 pixel: the kernel now clamps each Gaussian's query offset
        // to the covering pixel's footprint before evaluating it (see
        // splat_mip_tiled_kernel), which lets a Gaussian influence a pixel
        // up to half a pixel-width beyond the raw cutoff-ellipse distance
        // used below. Padding the culling radius by that same half pixel
        // keeps tile assignment a conservative superset, so a Gaussian is
        // never excluded from a tile it can now actually affect.
        const float radius = radius_scale * std::sqrt(std::max(lam, 1e-12f)) + 0.5f;

        // -0.5: converts physical position to pixel-INDEX space, so a
        // Gaussian exactly at pixel ic's centre maps to mu_u_px == ic (not
        // ic+0.5) -- matching the kernel's own px/py pixel-centre formula.
        const float mu_u_px = (mu_u_w - u_lo) * u_scale - 0.5f;
        const float mu_v_px = (mu_v_w - v_lo) * v_scale - 0.5f;

        int u0 = static_cast<int>(std::floor(mu_u_px - radius));
        int u1 = static_cast<int>(std::ceil(mu_u_px + radius));
        int v0 = static_cast<int>(std::floor(mu_v_px - radius));
        int v1 = static_cast<int>(std::ceil(mu_v_px + radius));
        if (u1 < 0 || v1 < 0 || u0 >= out_w || v0 >= out_h) continue;
        u0 = std::max(0, u0);
        u1 = std::min(out_w - 1, u1);
        v0 = std::max(0, v0);
        v1 = std::min(out_h - 1, v1);

        const int tx0 = u0 / kTilePixW;
        const int tx1 = u1 / kTilePixW;
        const int ty0 = v0 / kTilePixH;
        const int ty1 = v1 / kTilePixH;
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                tile_lists[ty * tiles_x + tx].push_back(g);
            }
        }

        proj_u[g] = mu_u_w;
        proj_v[g] = mu_v_w;
        // eff00/eff01/eff11 (the Schur complement computed above), not the
        // raw Iuu/Iuv/Ivv slice: for a Gaussian whose principal axes aren't
        // aligned with the coordinate axes, the raw 2x2 slice of the 3x3
        // precision matrix is NOT the correct effective 2D precision for
        // its analytic maximum along the view axis -- only the Schur
        // complement is (maximizing a joint Gaussian quadratic form over
        // one variable at fixed others yields exactly this same reduced
        // form). Iuu/Iuv/Ivv only coincide with eff00/eff01/eff11 for
        // Gaussians whose rotation makes the view axis one of their own
        // principal axes.
        si_uu[g] = eff00;
        si_uv[g] = eff01;
        si_vv[g] = eff11;
        proj_iv[g] = softplus_stable(i_ptr[g]);
    }

    if (max_gauss_per_tile > 0) {
        for (int t = 0; t < num_tiles; ++t) {
            auto& lst = tile_lists[t];
            if (static_cast<int>(lst.size()) > max_gauss_per_tile) {
                std::partial_sort(
                    lst.begin(),
                    lst.begin() + max_gauss_per_tile,
                    lst.end(),
                    [&](int a, int b) { return proj_iv[a] > proj_iv[b]; }
                );
                lst.resize(max_gauss_per_tile);
            }
        }
    }

    if (print_stats) {
        int max_count = 0;
        int over128 = 0, over256 = 0, over512 = 0, over1024 = 0;
        double avg_count = 0.0;
        long long total_candidates = 0;
        for (int t = 0; t < num_tiles; ++t) {
            const int c = static_cast<int>(tile_lists[t].size());
            total_candidates += c;
            avg_count += c;
            max_count = std::max(max_count, c);
            if (c > 128) ++over128;
            if (c > 256) ++over256;
            if (c > 512) ++over512;
            if (c > 1024) ++over1024;
        }
        avg_count /= std::max(1, num_tiles);
        std::cout << "[splat_mip] tiles=" << num_tiles
                  << " total candidates=" << total_candidates
                  << " avg/tile=" << avg_count
                  << " max/tile=" << max_count
                  << " >128=" << over128
                  << " >256=" << over256
                  << " >512=" << over512
                  << " >1024=" << over1024
                  << " max_gauss_per_tile=" << max_gauss_per_tile
                  << std::endl;
    }

    std::vector<int> tile_offsets(num_tiles + 1, 0);
    int total = 0;
    for (int t = 0; t < num_tiles; ++t) {
        tile_offsets[t] = total;
        total += static_cast<int>(tile_lists[t].size());
    }
    tile_offsets[num_tiles] = total;

    std::vector<int> tile_indices(total);
    for (int t = 0; t < num_tiles; ++t) {
        const int base = tile_offsets[t];
        const auto& lst = tile_lists[t];
        for (int i = 0; i < static_cast<int>(lst.size()); ++i) {
            tile_indices[base + i] = lst[i];
        }
    }

    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto mkf = [&](std::vector<float>& v) {
        auto t = torch::empty({N}, fo);
        std::memcpy(t.data_ptr<float>(), v.data(), sizeof(float) * N);
        return t;
    };

    auto t_u = mkf(proj_u);
    auto t_v = mkf(proj_v);
    auto t_uu = mkf(si_uu);
    auto t_uv = mkf(si_uv);
    auto t_vv = mkf(si_vv);
    auto t_iv = mkf(proj_iv);

    auto t_off = torch::empty({num_tiles + 1}, io);
    auto t_idx = torch::empty({total}, io);
    std::memcpy(t_off.data_ptr<int>(), tile_offsets.data(), sizeof(int) * (num_tiles + 1));
    if (total > 0) {
        std::memcpy(t_idx.data_ptr<int>(), tile_indices.data(), sizeof(int) * total);
    }

    const auto dev = means.device();
    auto d_u = t_u.to(dev);
    auto d_v = t_v.to(dev);
    auto d_uu = t_uu.to(dev);
    auto d_uv = t_uv.to(dev);
    auto d_vv = t_vv.to(dev);
    auto d_iv = t_iv.to(dev);
    auto d_off = t_off.to(dev);
    auto d_idx = t_idx.to(dev);

    const dim3 block(kTilePixW, kTilePixH);
    const dim3 grid(num_tiles);
    splat_mip_tiled_kernel<<<grid, block>>>(
        d_u.data_ptr<float>(), d_v.data_ptr<float>(),
        d_uu.data_ptr<float>(), d_uv.data_ptr<float>(), d_vv.data_ptr<float>(),
        d_iv.data_ptr<float>(),
        d_off.data_ptr<int>(), d_idx.data_ptr<int>(),
        tiles_x, out_h, out_w,
        u_lo, u_hi, v_lo, v_hi,
        mahal_clamp, clamp_output ? 1 : 0,
        out.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}
