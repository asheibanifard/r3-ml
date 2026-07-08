/*
 * Clean CUDA/C++ implementation for 3D Gaussian evaluation.
 *
 * Exports:
 *   reconstruct_volume(means, log_s, quats, inten, lo_x, hi_x, lo_y, hi_y,
 *                      lo_z, hi_z, D, H, W, scale_min, mahal_clamp)
 *   splat_mip(means, log_s, quats, inten, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
 *             out_h, out_w, depth_samples, view_axis,
 *             scale_min, mahal_clamp,
 *             density_scale=1.0e-4, max_gauss_per_tile=0,
 *             print_stats=false, clamp_output=true)
 */

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace py = pybind11;

static constexpr int kBlockSize = 256;
static constexpr int kTilePixW = 16;
static constexpr int kTilePixH = 16;
#define K_TILE_GAUSS 128

__host__ __device__ static inline float softplus_stable(float x) {
    return (x > 20.f) ? x : std::log1pf(std::exp(x));
}

static inline void quat_to_rotmat_host(
        float qw, float qx, float qy, float qz,
        float& R00, float& R01, float& R02,
        float& R10, float& R11, float& R12,
        float& R20, float& R21, float& R22) {
    const float n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    const float inv_n = (n > 0.f) ? (1.f / n) : 1.f;
    qw *= inv_n;
    qx *= inv_n;
    qy *= inv_n;
    qz *= inv_n;

    R00 = 1.f - 2.f * (qy * qy + qz * qz);
    R01 = 2.f * (qx * qy - qw * qz);
    R02 = 2.f * (qx * qz + qw * qy);
    R10 = 2.f * (qx * qy + qw * qz);
    R11 = 1.f - 2.f * (qx * qx + qz * qz);
    R12 = 2.f * (qy * qz - qw * qx);
    R20 = 2.f * (qx * qz - qw * qy);
    R21 = 2.f * (qy * qz + qw * qx);
    R22 = 1.f - 2.f * (qx * qx + qy * qy);
}

static inline void projected_axes_for_view_axis(
        int view_axis,
        int& u_axis, int& v_axis,
        float& u_lo, float& u_hi,
        float& v_lo, float& v_hi,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z) {
    if (view_axis == 0) {
        u_axis = 0; v_axis = 1;
        u_lo = lo_x; u_hi = hi_x;
        v_lo = lo_y; v_hi = hi_y;
    } else if (view_axis == 1) {
        u_axis = 0; v_axis = 2;
        u_lo = lo_x; u_hi = hi_x;
        v_lo = lo_z; v_hi = hi_z;
    } else {
        u_axis = 1; v_axis = 2;
        u_lo = lo_y; u_hi = hi_y;
        v_lo = lo_z; v_hi = hi_z;
    }
}

static inline void depth_bounds_for_view_axis(
        int view_axis,
        float& depth_lo, float& depth_hi,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z) {
    if (view_axis == 0) {
        depth_lo = lo_z; depth_hi = hi_z;
    } else if (view_axis == 1) {
        depth_lo = lo_y; depth_hi = hi_y;
    } else {
        depth_lo = lo_x; depth_hi = hi_x;
    }
}

__global__ void eval_volume_kernel(
        const float* __restrict__ means,
        const float* __restrict__ log_s,
        const float* __restrict__ quats,
        const float* __restrict__ inten,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z,
        int D, int H, int W, int N,
        float scale_min, float mahal_clamp,
        float* __restrict__ out) {
    __shared__ float s_mu[K_TILE_GAUSS][3];
    __shared__ float s_ls[K_TILE_GAUSS][3];
    __shared__ float s_qu[K_TILE_GAUSS][4];
    __shared__ float s_iv[K_TILE_GAUSS];

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int DHW = D * H * W;

    float cx = 0.f, cy = 0.f, cz = 0.f;
    const bool active = (n < DHW);
    if (active) {
        const int iz = n / (H * W);
        const int ih = (n % (H * W)) / W;
        const int iw = n % W;
        cx = lo_x + ((W > 1) ? (static_cast<float>(iw) / static_cast<float>(W - 1)) : 0.5f) * (hi_x - lo_x);
        cy = lo_y + ((H > 1) ? (static_cast<float>(ih) / static_cast<float>(H - 1)) : 0.5f) * (hi_y - lo_y);
        cz = lo_z + ((D > 1) ? (static_cast<float>(iz) / static_cast<float>(D - 1)) : 0.5f) * (hi_z - lo_z);
    }

    float acc = 0.f;
    for (int t0 = 0; t0 < N; t0 += K_TILE_GAUSS) {
        const int tn = std::min(K_TILE_GAUSS, N - t0);

        for (int i = threadIdx.x; i < tn; i += blockDim.x) {
            const int g = t0 + i;
            s_mu[i][0] = means[g * 3 + 0];
            s_mu[i][1] = means[g * 3 + 1];
            s_mu[i][2] = means[g * 3 + 2];
            s_ls[i][0] = log_s[g * 3 + 0];
            s_ls[i][1] = log_s[g * 3 + 1];
            s_ls[i][2] = log_s[g * 3 + 2];
            s_qu[i][0] = quats[g * 4 + 0];
            s_qu[i][1] = quats[g * 4 + 1];
            s_qu[i][2] = quats[g * 4 + 2];
            s_qu[i][3] = quats[g * 4 + 3];
            s_iv[i] = softplus_stable(inten[g]);
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tn; ++i) {
                float qw = s_qu[i][0], qx = s_qu[i][1], qy = s_qu[i][2], qz = s_qu[i][3];
                const float inv_n = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
                qw *= inv_n; qx *= inv_n; qy *= inv_n; qz *= inv_n;

                const float R00 = 1.f - 2.f * (qy * qy + qz * qz);
                const float R01 = 2.f * (qx * qy - qw * qz);
                const float R02 = 2.f * (qx * qz + qw * qy);
                const float R10 = 2.f * (qx * qy + qw * qz);
                const float R11 = 1.f - 2.f * (qx * qx + qz * qz);
                const float R12 = 2.f * (qy * qz - qw * qx);
                const float R20 = 2.f * (qx * qz - qw * qy);
                const float R21 = 2.f * (qy * qz + qw * qx);
                const float R22 = 1.f - 2.f * (qx * qx + qy * qy);

                const float sx = std::max(std::exp(s_ls[i][0]), scale_min);
                const float sy = std::max(std::exp(s_ls[i][1]), scale_min);
                const float sz = std::max(std::exp(s_ls[i][2]), scale_min);
                const float isx = 1.f / sx;
                const float isy = 1.f / sy;
                const float isz = 1.f / sz;

                const float D00 = R00 * isx, D01 = R01 * isy, D02 = R02 * isz;
                const float D10 = R10 * isx, D11 = R11 * isy, D12 = R12 * isz;
                const float D20 = R20 * isx, D21 = R21 * isy, D22 = R22 * isz;

                const float S00 = D00 * D00 + D01 * D01 + D02 * D02;
                const float S01 = D00 * D10 + D01 * D11 + D02 * D12;
                const float S02 = D00 * D20 + D01 * D21 + D02 * D22;
                const float S11 = D10 * D10 + D11 * D11 + D12 * D12;
                const float S12 = D10 * D20 + D11 * D21 + D12 * D22;
                const float S22 = D20 * D20 + D21 * D21 + D22 * D22;

                const float d0 = cx - s_mu[i][0];
                const float d1 = cy - s_mu[i][1];
                const float d2 = cz - s_mu[i][2];
                const float mah = d0 * (S00 * d0 + S01 * d1 + S02 * d2)
                                + d1 * (S01 * d0 + S11 * d1 + S12 * d2)
                                + d2 * (S02 * d0 + S12 * d1 + S22 * d2);
                if (mah >= mahal_clamp) continue;

                acc += s_iv[i] * __expf(-0.5f * mah);
            }
        }
        __syncthreads();
    }

    if (active) {
        out[n] = std::min(std::max(acc, 0.f), 1.f);
    }
}

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
        float density_scale,
        int clamp_output,
        float* __restrict__ out) {
    __shared__ float s_u[K_TILE_GAUSS];
    __shared__ float s_v[K_TILE_GAUSS];
    __shared__ float s_w[K_TILE_GAUSS];
    __shared__ float s_i[K_TILE_GAUSS];
    __shared__ float s_a[K_TILE_GAUSS];
    __shared__ float s_b[K_TILE_GAUSS];
    __shared__ float s_c[K_TILE_GAUSS];
    __shared__ float s_d[K_TILE_GAUSS];
    __shared__ float s_e[K_TILE_GAUSS];
    __shared__ float s_f[K_TILE_GAUSS];

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

    const float px = (out_w > 1)
        ? (u_lo + (static_cast<float>(ic) / static_cast<float>(out_w - 1)) * (u_hi - u_lo))
        : 0.5f * (u_lo + u_hi);
    const float py = (out_h > 1)
        ? (v_lo + (static_cast<float>(ir) / static_cast<float>(out_h - 1)) * (v_hi - v_lo))
        : 0.5f * (v_lo + v_hi);

    if (active) {
        float acc = 0.f;
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

            for (int i = 0; i < tn; ++i) {
                const float du = px - s_u[i];
                const float dv = py - s_v[i];
                const float mah = du * (s_a[i] * du + s_b[i] * dv)
                                + dv * (s_b[i] * du + s_d[i] * dv);
                if (mah >= mahal_clamp) continue;
                acc += s_i[i] * __expf(-0.5f * mah);
            }
            __syncthreads();
        }

        const float positive = fmaxf(acc, 0.f);
        const float mapped = 1.f - __expf(-density_scale * positive);
        out[ir * out_w + ic] = clamp_output ? fminf(fmaxf(mapped, 0.f), 1.f) : mapped;
    }
}

torch::Tensor reconstruct_volume(
        torch::Tensor means, torch::Tensor log_s, torch::Tensor quats, torch::Tensor inten,
        float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z,
        int D, int H, int W, float scale_min, float mahal_clamp) {
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(log_s.is_contiguous(), "log_s must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(inten.is_contiguous(), "inten must be contiguous");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");

    const int N = static_cast<int>(means.size(0));
    const int DHW = D * H * W;
    auto out = torch::empty({DHW}, means.options());

    const int blocks = (DHW + kBlockSize - 1) / kBlockSize;
    eval_volume_kernel<<<blocks, kBlockSize>>>(
        means.data_ptr<float>(), log_s.data_ptr<float>(),
        quats.data_ptr<float>(), inten.data_ptr<float>(),
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        D, H, W, N, scale_min, mahal_clamp,
        out.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor splat_mip(
        torch::Tensor means, torch::Tensor log_s, torch::Tensor quats, torch::Tensor inten,
        float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z,
        int out_h, int out_w, int depth_samples, int view_axis,
        float scale_min, float mahal_clamp,
        float density_scale,
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
    TORCH_CHECK(density_scale >= 0.f, "density_scale must be non-negative");
    TORCH_CHECK(max_gauss_per_tile >= 0, "max_gauss_per_tile must be >= 0; use 0 for unlimited");

    const int N = static_cast<int>(means.size(0));
    auto out = torch::zeros({out_h * out_w}, means.options());

    int u_axis = 0, v_axis = 1;
    float u_lo = 0.f, u_hi = 1.f, v_lo = 0.f, v_hi = 1.f;
    projected_axes_for_view_axis(view_axis, u_axis, v_axis, u_lo, u_hi, v_lo, v_hi,
                                 lo_x, hi_x, lo_y, hi_y, lo_z, hi_z);
    const float u_scale = (out_w > 1)
        ? (static_cast<float>(out_w - 1) / (u_hi - u_lo))
        : 1.f;
    const float v_scale = (out_h > 1)
        ? (static_cast<float>(out_h - 1) / (v_hi - v_lo))
        : 1.f;
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

    std::vector<float> proj_u(N), proj_v(N), proj_w(N);
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
        float Iuu = 0.f, Iuv = 0.f, Ivv = 0.f;
        float cuu = 0.f, cuv = 0.f, cvv = 0.f;
        if (view_axis == 0) {
            mu_u_w = mx; mu_v_w = my;
            Iuu = Ixx; Iuv = Ixy; Ivv = Iyy;
            cuu = C00; cuv = C01; cvv = C11;
        } else if (view_axis == 1) {
            mu_u_w = mx; mu_v_w = mz;
            Iuu = Ixx; Iuv = Ixz; Ivv = Izz;
            cuu = C00; cuv = C02; cvv = C22;
        } else {
            mu_u_w = my; mu_v_w = mz;
            Iuu = Iyy; Iuv = Iyz; Ivv = Izz;
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
        const float radius = radius_scale * std::sqrt(std::max(lam, 1e-12f));

        const float mu_u_px = (mu_u_w - u_lo) * u_scale;
        const float mu_v_px = (mu_v_w - v_lo) * v_scale;

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
        si_uu[g] = Iuu;
        si_uv[g] = Iuv;
        si_vv[g] = Ivv;
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
                  << " density_scale=" << density_scale
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
        mahal_clamp, density_scale, clamp_output ? 1 : 0,
        out.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reconstruct_volume", &reconstruct_volume,
          py::arg("means"), py::arg("log_s"), py::arg("quats"), py::arg("inten"),
          py::arg("lo_x"), py::arg("hi_x"), py::arg("lo_y"), py::arg("hi_y"),
          py::arg("lo_z"), py::arg("hi_z"),
          py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("scale_min"), py::arg("mahal_clamp"),
          "Evaluate Gaussian cloud at every voxel centre. Returns a flat float32 CUDA tensor.");

    m.def("splat_mip", &splat_mip,
          py::arg("means"), py::arg("log_s"), py::arg("quats"), py::arg("inten"),
          py::arg("lo_x"), py::arg("hi_x"), py::arg("lo_y"), py::arg("hi_y"),
          py::arg("lo_z"), py::arg("hi_z"),
          py::arg("out_h"), py::arg("out_w"), py::arg("depth_samples"), py::arg("view_axis"),
          py::arg("scale_min"), py::arg("mahal_clamp"),
          py::arg("density_scale") = 1.0e-4f,
          py::arg("max_gauss_per_tile") = 0,
          py::arg("print_stats") = false,
          py::arg("clamp_output") = true,
          "Tiled MIP splatting of a Gaussian mixture with conservative culling.");
}