/*
 * forward.cu — fused CUDA forward pass for 3-D Gaussian field evaluation.
 *
 * f(x) = Σ_k  gain_k · inten_k · exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))
 *
 * where inten is passed as the post-softplus value (softplus applied in
 * Python before calling kernel.forward), so the kernel treats it as a plain
 * positive float. Covariance: Σ_k = R_k diag(s_k²) R_kᵀ, parameterised by
 * [w,x,y,z] + log_s.
 *
 * One thread per sample point. Gaussians are loaded cooperatively into
 * shared memory, tiled TILE_FWD at a time.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.cuh"

// ─── Forward kernel (shared-memory tiled) ────────────────────────────────────
/*
 * Each block of BLOCK_FWD threads handles BLOCK_FWD sample points.
 * Gaussians are loaded cooperatively into shared memory TILE_FWD at a time,
 * reducing global-memory reads by a factor of BLOCK_FWD vs the naïve kernel.
 *
 * Shared-mem per block = TILE_FWD × (3+3+4+1+1) × 4 = TILE_FWD × 48 B.
 * With TILE_FWD=256: 12 288 B — well within the 48 KB limit.
 */
#define BLOCK_FWD 256
#define TILE_FWD  256

__global__ void gaussian_forward_kernel(
        const float* __restrict__ pts,
        const float* __restrict__ means,
        const float* __restrict__ log_s,
        const float* __restrict__ quats,
        const float* __restrict__ gain,
        const float* __restrict__ inten,
        float scale_min, float mahal_clamp,
        int M, int N,
        float* __restrict__ out)
{
    __shared__ float s_mu[TILE_FWD][3];
    __shared__ float s_ls[TILE_FWD][3];
    __shared__ float s_qu[TILE_FWD][4];
    __shared__ float s_ga[TILE_FWD];
    __shared__ float s_iv[TILE_FWD];

    const int m      = blockIdx.x * BLOCK_FWD + threadIdx.x;
    const bool active = (m < M);

    float px=0.f, py=0.f, pz=0.f;
    if (active) { px = pts[m*3+0]; py = pts[m*3+1]; pz = pts[m*3+2]; }
    float acc = 0.f;

    for (int t0 = 0; t0 < N; t0 += TILE_FWD) {
        const int tn = min(TILE_FWD, N - t0);

        /* cooperative load: each thread loads one Gaussian (if in range) */
        for (int i = threadIdx.x; i < tn; i += BLOCK_FWD) {
            const int k = t0 + i;
            s_mu[i][0] = means[k*3+0]; s_mu[i][1] = means[k*3+1]; s_mu[i][2] = means[k*3+2];
            s_ls[i][0] = log_s[k*3+0]; s_ls[i][1] = log_s[k*3+1]; s_ls[i][2] = log_s[k*3+2];
            s_qu[i][0] = quats[k*4+0]; s_qu[i][1] = quats[k*4+1];
            s_qu[i][2] = quats[k*4+2]; s_qu[i][3] = quats[k*4+3];
            s_ga[i] = gain[k];
            s_iv[i] = inten[k];
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tn; i++) {
                float qn[4], R[9], inv_norm;
                normalize_quat(s_qu[i], qn, inv_norm);
                quat_to_rotmat(qn, R);

                const float s0 = fmaxf(expf(s_ls[i][0]), scale_min);
                const float s1 = fmaxf(expf(s_ls[i][1]), scale_min);
                const float s2 = fmaxf(expf(s_ls[i][2]), scale_min);

                const float diff[3] = {px-s_mu[i][0], py-s_mu[i][1], pz-s_mu[i][2]};
                float u[3];
                mat_t_vec(R, diff, u);

                const float mahal = u[0]*u[0]/(s0*s0) + u[1]*u[1]/(s1*s1) + u[2]*u[2]/(s2*s2);
                if (mahal >= mahal_clamp) continue;

                acc += s_ga[i] * s_iv[i] * expf(-0.5f * mahal);
            }
        }
        __syncthreads();
    }

    if (active) out[m] = acc;
}


// ─── Python-visible entry point ───────────────────────────────────────────────

torch::Tensor gaussian_forward(
        torch::Tensor pts,
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor gain,
        torch::Tensor inten,
        float scale_min,
        float mahal_clamp)
{
    TORCH_CHECK(pts.is_cuda()   && pts.is_contiguous(),   "pts must be contiguous CUDA float32");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be contiguous CUDA float32");
    TORCH_CHECK(log_s.is_cuda() && log_s.is_contiguous(), "log_s must be contiguous CUDA float32");
    TORCH_CHECK(quats.is_cuda() && quats.is_contiguous(), "quats must be contiguous CUDA float32");
    TORCH_CHECK(gain.is_cuda()  && gain.is_contiguous(),  "gain must be contiguous CUDA float32");
    TORCH_CHECK(inten.is_cuda() && inten.is_contiguous(), "inten must be contiguous CUDA float32");
    TORCH_CHECK(pts.scalar_type()   == torch::kFloat32, "pts must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(log_s.scalar_type() == torch::kFloat32, "log_s must be float32");
    TORCH_CHECK(quats.scalar_type() == torch::kFloat32, "quats must be float32");
    TORCH_CHECK(gain.scalar_type()  == torch::kFloat32, "gain must be float32");
    TORCH_CHECK(inten.scalar_type() == torch::kFloat32, "inten must be float32");
    TORCH_CHECK(pts.dim() == 2   && pts.size(1) == 3,   "pts must be (M, 3)");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be (N, 3)");
    TORCH_CHECK(log_s.dim() == 2 && log_s.size(1) == 3, "log_s must be (N, 3)");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be (N, 4)");
    TORCH_CHECK(gain.dim()  == 1, "gain must be (N,)");
    TORCH_CHECK(inten.dim() == 1, "inten must be (N,)");

    const int M = static_cast<int>(pts.size(0));
    const int N = static_cast<int>(means.size(0));
    auto out = torch::zeros({M}, pts.options());

    const int blocks = (M + BLOCK_FWD - 1) / BLOCK_FWD;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_forward_kernel<<<blocks, BLOCK_FWD, 0, stream>>>(
        pts.data_ptr<float>(),
        means.data_ptr<float>(),
        log_s.data_ptr<float>(),
        quats.data_ptr<float>(),
        gain.data_ptr<float>(),
        inten.data_ptr<float>(),
        scale_min, mahal_clamp, M, N,
        out.data_ptr<float>()
    );
    return out;
}
