/*
 * reconstruct_volume.cu — dense per-voxel evaluation of a Gaussian mixture.
 *
 * Exports:
 *   reconstruct_volume(means, log_s, quats, inten, lo_x, hi_x, lo_y, hi_y,
 *                       lo_z, hi_z, D, H, W, scale_min, mahal_clamp)
 */

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>

#include "common.cuh"

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
