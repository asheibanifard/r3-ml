#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda_runtime.h>
#include <cmath>

/*
 * Pure CUDA Gaussian MIP splatting for full 3D Gaussians.
 *
 * Each output pixel traces a ray through the volume, evaluates the anisotropic
 * 3D Gaussian field at a fixed number of depth samples, and keeps the maximum
 * response along that ray.
 */

#define BLOCK_MIP 256
#define TILE_MIP  128

__global__ void render_mip_kernel(
    const float* __restrict__ means,
    const float* __restrict__ log_s,
    const float* __restrict__ quats,
    const float* __restrict__ inten,
    float lo_x, float hi_x,
    float lo_y, float hi_y,
    float lo_z, float hi_z,
    int out_h, int out_w, int n_gauss, int depth_samples,
    int view_axis,
    float scale_min, float mahal_clamp,
    float* __restrict__ out)
{
    __shared__ float s_mu[TILE_MIP][3];
    __shared__ float s_ls[TILE_MIP][3];
    __shared__ float s_qu[TILE_MIP][4];
    __shared__ float s_iv[TILE_MIP];

    const int idx = blockIdx.x * BLOCK_MIP + threadIdx.x;
    const bool active = (idx < out_h * out_w);

    const int ir = active ? (idx / out_w) : 0;
    const int ic = active ? (idx % out_w) : 0;

    float pa = 0.f, pb = 0.f, depth_lo = 0.f, depth_hi = 0.f;
    if (view_axis == 0) {
        pa = lo_x + (out_w > 1 ? (float)ic / (float)(out_w - 1) : 0.5f) * (hi_x - lo_x);
        pb = lo_y + (out_h > 1 ? (float)ir / (float)(out_h - 1) : 0.5f) * (hi_y - lo_y);
        depth_lo = lo_z;
        depth_hi = hi_z;
    } else if (view_axis == 1) {
        pa = lo_x + (out_w > 1 ? (float)ic / (float)(out_w - 1) : 0.5f) * (hi_x - lo_x);
        pb = lo_z + (out_h > 1 ? (float)ir / (float)(out_h - 1) : 0.5f) * (hi_z - lo_z);
        depth_lo = lo_y;
        depth_hi = hi_y;
    } else {
        pa = lo_y + (out_w > 1 ? (float)ic / (float)(out_w - 1) : 0.5f) * (hi_y - lo_y);
        pb = lo_z + (out_h > 1 ? (float)ir / (float)(out_h - 1) : 0.5f) * (hi_z - lo_z);
        depth_lo = lo_x;
        depth_hi = hi_x;
    }

    float max_val = 0.f;

    for (int d = 0; d < depth_samples; ++d) {
        const float t = (depth_samples > 1) ? ((float)d / (float)(depth_samples - 1)) : 0.5f;
        const float dv = depth_lo + t * (depth_hi - depth_lo);

        float cx, cy, cz;
        if (view_axis == 0) {
            cx = pa; cy = pb; cz = dv;
        } else if (view_axis == 1) {
            cx = pa; cy = dv; cz = pb;
        } else {
            cx = dv; cy = pa; cz = pb;
        }

        float acc = 0.f;

        for (int t0 = 0; t0 < n_gauss; t0 += TILE_MIP) {
            const int tn = min(TILE_MIP, n_gauss - t0);

            for (int i = threadIdx.x; i < tn; i += BLOCK_MIP) {
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
                const float r = inten[g];
                s_iv[i] = (r > 20.f) ? r : log1pf(__expf(r));
            }
            __syncthreads();

            if (active) {
                for (int i = 0; i < tn; ++i) {
                    float qw = s_qu[i][0], qx = s_qu[i][1];
                    float qy = s_qu[i][2], qz = s_qu[i][3];
                    float ni = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
                    qw *= ni; qx *= ni; qy *= ni; qz *= ni;

                    float R00 = 1.f - 2.f * (qy * qy + qz * qz);
                    float R01 = 2.f * (qx * qy - qw * qz);
                    float R02 = 2.f * (qx * qz + qw * qy);
                    float R10 = 2.f * (qx * qy + qw * qz);
                    float R11 = 1.f - 2.f * (qx * qx + qz * qz);
                    float R12 = 2.f * (qy * qz - qw * qx);
                    float R20 = 2.f * (qx * qz - qw * qy);
                    float R21 = 2.f * (qy * qz + qw * qx);
                    float R22 = 1.f - 2.f * (qx * qx + qy * qy);

                    float sx = fmaxf(__expf(s_ls[i][0]), scale_min);
                    float sy = fmaxf(__expf(s_ls[i][1]), scale_min);
                    float sz = fmaxf(__expf(s_ls[i][2]), scale_min);
                    float isx = 1.f / sx;
                    float isy = 1.f / sy;
                    float isz = 1.f / sz;

                    float D00 = R00 * isx, D01 = R01 * isy, D02 = R02 * isz;
                    float D10 = R10 * isx, D11 = R11 * isy, D12 = R12 * isz;
                    float D20 = R20 * isx, D21 = R21 * isy, D22 = R22 * isz;

                    float S00 = D00 * D00 + D01 * D01 + D02 * D02;
                    float S01 = D00 * D10 + D01 * D11 + D02 * D12;
                    float S02 = D00 * D20 + D01 * D21 + D02 * D22;
                    float S11 = D10 * D10 + D11 * D11 + D12 * D12;
                    float S12 = D10 * D20 + D11 * D21 + D12 * D22;
                    float S22 = D20 * D20 + D21 * D21 + D22 * D22;

                    float d0 = cx - s_mu[i][0];
                    float d1 = cy - s_mu[i][1];
                    float d2 = cz - s_mu[i][2];
                    float mah = d0 * (S00 * d0 + S01 * d1 + S02 * d2)
                              + d1 * (S01 * d0 + S11 * d1 + S12 * d2)
                              + d2 * (S02 * d0 + S12 * d1 + S22 * d2);
                    if (mah >= mahal_clamp) continue;

                    acc += s_iv[i] * __expf(-0.5f * mah);
                }
            }

            __syncthreads();
        }

        max_val = fmaxf(max_val, acc);
    }

    if (active) {
        out[idx] = fminf(fmaxf(max_val, 0.f), 1.f);
    }
}

torch::Tensor render_mip(
    torch::Tensor means,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor intensities,
    float lo_x, float hi_x,
    float lo_y, float hi_y,
    float lo_z, float hi_z,
    int out_h, int out_w,
    int depth_samples,
    int view_axis,
    float scale_min,
    float mahal_clamp)
{
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(log_scales.is_cuda(), "log_scales must be a CUDA tensor");
    TORCH_CHECK(quats.is_cuda(), "quats must be a CUDA tensor");
    TORCH_CHECK(intensities.is_cuda(), "intensities must be a CUDA tensor");

    auto means_gpu = means.contiguous().to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto scales_gpu = log_scales.contiguous().to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto quats_gpu = quats.contiguous().to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto intens_gpu = intensities.contiguous().to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const int n_gauss = static_cast<int>(means_gpu.size(0));
    auto output = torch::zeros({out_h * out_w}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const int pixels = out_h * out_w;
    const int blocks = (pixels + BLOCK_MIP - 1) / BLOCK_MIP;
    render_mip_kernel<<<blocks, BLOCK_MIP>>>(
        means_gpu.data_ptr<float>(),
        scales_gpu.data_ptr<float>(),
        quats_gpu.data_ptr<float>(),
        intens_gpu.data_ptr<float>(),
        lo_x, hi_x,
        lo_y, hi_y,
        lo_z, hi_z,
        out_h, out_w, n_gauss, depth_samples,
        view_axis,
        scale_min, mahal_clamp,
        output.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor render_mip_xy(
    torch::Tensor means,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor intensities,
    float lo_x, float hi_x,
    float lo_y, float hi_y,
    float lo_z, float hi_z,
    int height, int width,
    int depth_samples,
    float scale_min,
    float mahal_clamp)
{
    return render_mip(
        means, log_scales, quats, intensities,
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        height, width, depth_samples, 0,
        scale_min, mahal_clamp);
}

torch::Tensor render_mip_xz(
    torch::Tensor means,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor intensities,
    float lo_x, float hi_x,
    float lo_y, float hi_y,
    float lo_z, float hi_z,
    int depth, int width,
    int depth_samples,
    float scale_min,
    float mahal_clamp)
{
    return render_mip(
        means, log_scales, quats, intensities,
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        depth, width, depth_samples, 1,
        scale_min, mahal_clamp);
}

torch::Tensor render_mip_yz(
    torch::Tensor means,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor intensities,
    float lo_x, float hi_x,
    float lo_y, float hi_y,
    float lo_z, float hi_z,
    int depth, int height,
    int depth_samples,
    float scale_min,
    float mahal_clamp)
{
    return render_mip(
        means, log_scales, quats, intensities,
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        depth, height, depth_samples, 2,
        scale_min, mahal_clamp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_mip", &render_mip, "Render 3D Gaussian MIP by pure CUDA splatting");
    m.def("render_mip_xy", &render_mip_xy, "Render XY MIP by splatting anisotropic 3D Gaussians");
    m.def("render_mip_xz", &render_mip_xz, "Render XZ MIP by splatting anisotropic 3D Gaussians");
    m.def("render_mip_yz", &render_mip_yz, "Render YZ MIP by splatting anisotropic 3D Gaussians");
}
