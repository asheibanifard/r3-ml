/*
 * backward.cu — fused CUDA backward pass for 3-D Gaussian field evaluation.
 *
 * Gradients w.r.t. Gaussian parameters (means, log_s, quats, gain, inten).
 * One CUDA block per Gaussian, threads share the M sample-point loop, and
 * warp-shuffle + shared-memory reduction avoids all atomicAdd contention
 * (unlike a naive one-thread-per-sample-point design).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

#include "common.cuh"

namespace py = pybind11;

// ─── Backward kernel v2 (N blocks × BLOCK_BWD threads, warp-shuffle reduce) ──
/*
 * One CUDA block per Gaussian — BLOCK_BWD threads share the M sample-point loop.
 *
 * v1 problem: ceil(N/256) = 20 blocks for N=5000 → only 20 SMs active out of 60+.
 * v2 fix:     N           = 5000 blocks → all SMs saturated; each thread handles
 *             only M/BLOCK_BWD = 8 points instead of M=2048.
 *
 * Algorithm per block (Gaussian k = blockIdx.x):
 *   1. Thread 0 loads Gaussian params into shared memory (broadcast to block).
 *   2. Each thread t handles points t, t+BLOCK_BWD, … accumulating in registers.
 *   3. Warp-level shuffle reduces 32 threads → 1 warp leader.
 *   4. Warp leaders write to a 17×WARPS_BWD shared buffer; thread 0 sums it.
 *   5. Thread 0 writes directly to grad_* — zero atomicAdd contention.
 *
 * Shared memory per block ≈ (17 × WARPS_BWD + ~50 param floats) × 4 B ≈ 750 B.
 */
#define BLOCK_BWD  256
#define WARPS_BWD  (BLOCK_BWD / 32)   /* = 8 */

__global__ void gaussian_backward_kernel(
        const float* __restrict__ grad_out,
        const float* __restrict__ pts,
        const float* __restrict__ means,
        const float* __restrict__ log_s,
        const float* __restrict__ quats,
        const float* __restrict__ gain,
        const float* __restrict__ inten,
        float scale_min, float mahal_clamp,
        int M, int N,
        float* __restrict__ grad_means,
        float* __restrict__ grad_log_s,
        float* __restrict__ grad_quats,
        float* __restrict__ grad_gain,
        float* __restrict__ grad_inten)
{
    const int k   = blockIdx.x;          /* Gaussian index — one block per Gaussian */
    if (k >= N) return;
    const int t   = threadIdx.x;
    const int wid = t >> 5;              /* warp id   (t / 32) */
    const int lid = t & 31;             /* lane id   (t % 32) */

    /* ── Broadcast Gaussian k params into shared memory ──────────────────── */
    __shared__ float sh_mu[3], sh_ls[3], sh_qu[4], sh_G, sh_V;
    __shared__ float sh_qn[4], sh_R[9], sh_s[3], sh_is2[3], sh_invn;
    __shared__ bool  sh_cl[3];

    if (t < 3) { sh_mu[t] = means[k*3+t]; sh_ls[t] = log_s[k*3+t]; }
    if (t < 4)   sh_qu[t] = quats[k*4+t];
    if (t == 0) { sh_G = gain[k]; sh_V = inten[k]; }
    __syncthreads();

    if (t == 0) {
        normalize_quat(sh_qu, sh_qn, sh_invn);
        quat_to_rotmat(sh_qn, sh_R);
        const float rs0 = expf(sh_ls[0]), rs1 = expf(sh_ls[1]), rs2 = expf(sh_ls[2]);
        sh_cl[0] = (rs0 <= scale_min);
        sh_cl[1] = (rs1 <= scale_min);
        sh_cl[2] = (rs2 <= scale_min);
        sh_s[0]   = fmaxf(rs0, scale_min);
        sh_s[1]   = fmaxf(rs1, scale_min);
        sh_s[2]   = fmaxf(rs2, scale_min);
        sh_is2[0] = 1.f / (sh_s[0]*sh_s[0]);
        sh_is2[1] = 1.f / (sh_s[1]*sh_s[1]);
        sh_is2[2] = 1.f / (sh_s[2]*sh_s[2]);
    }
    __syncthreads();

    /* ── Strided loop: thread t covers points t, t+BLOCK_BWD, … ─────────── */
    float gm0=0.f, gm1=0.f, gm2=0.f;
    float gls0=0.f, gls1=0.f, gls2=0.f;
    float gR[9] = {0.f,0.f,0.f, 0.f,0.f,0.f, 0.f,0.f,0.f};
    float g_G=0.f, g_V=0.f;

    for (int m = t; m < M; m += BLOCK_BWD) {
        const float g_out = grad_out[m];
        if (g_out == 0.f) continue;

        const float d0 = pts[m*3+0] - sh_mu[0];
        const float d1 = pts[m*3+1] - sh_mu[1];
        const float d2 = pts[m*3+2] - sh_mu[2];

        const float u0 = sh_R[0]*d0 + sh_R[3]*d1 + sh_R[6]*d2;
        const float u1 = sh_R[1]*d0 + sh_R[4]*d1 + sh_R[7]*d2;
        const float u2 = sh_R[2]*d0 + sh_R[5]*d1 + sh_R[8]*d2;

        const float mahal = u0*u0*sh_is2[0] + u1*u1*sh_is2[1] + u2*u2*sh_is2[2];
        if (mahal >= mahal_clamp) continue;

        const float w  = expf(-0.5f * mahal);
        const float gf = g_out * sh_G * sh_V * w;
        g_G += g_out * sh_V * w;
        g_V += g_out * sh_G * w;

        const float t0 = u0 * sh_is2[0];
        const float t1 = u1 * sh_is2[1];
        const float t2 = u2 * sh_is2[2];

        gm0 += gf * (sh_R[0]*t0 + sh_R[1]*t1 + sh_R[2]*t2);
        gm1 += gf * (sh_R[3]*t0 + sh_R[4]*t1 + sh_R[5]*t2);
        gm2 += gf * (sh_R[6]*t0 + sh_R[7]*t1 + sh_R[8]*t2);

        if (!sh_cl[0]) gls0 += gf * u0*u0*sh_is2[0];
        if (!sh_cl[1]) gls1 += gf * u1*u1*sh_is2[1];
        if (!sh_cl[2]) gls2 += gf * u2*u2*sh_is2[2];

        gR[0]-=gf*d0*t0; gR[1]-=gf*d0*t1; gR[2]-=gf*d0*t2;
        gR[3]-=gf*d1*t0; gR[4]-=gf*d1*t1; gR[5]-=gf*d1*t2;
        gR[6]-=gf*d2*t0; gR[7]-=gf*d2*t1; gR[8]-=gf*d2*t2;
    }

    /* ── Warp-level reduction (shuffle down) ─────────────────────────────── */
    #define WRED(x) \
        x += __shfl_down_sync(0xffffffffu, x, 16); \
        x += __shfl_down_sync(0xffffffffu, x,  8); \
        x += __shfl_down_sync(0xffffffffu, x,  4); \
        x += __shfl_down_sync(0xffffffffu, x,  2); \
        x += __shfl_down_sync(0xffffffffu, x,  1);
    WRED(gm0)  WRED(gm1)  WRED(gm2)
    WRED(gls0) WRED(gls1) WRED(gls2)
    WRED(gR[0]) WRED(gR[1]) WRED(gR[2])
    WRED(gR[3]) WRED(gR[4]) WRED(gR[5])
    WRED(gR[6]) WRED(gR[7]) WRED(gR[8])
    WRED(g_G)  WRED(g_V)
    #undef WRED

    /* ── Inter-warp reduction via shared memory (17 values × 8 warps) ────── */
    __shared__ float sm[17][WARPS_BWD];
    if (lid == 0) {
        sm[ 0][wid]=gm0;   sm[ 1][wid]=gm1;   sm[ 2][wid]=gm2;
        sm[ 3][wid]=gls0;  sm[ 4][wid]=gls1;  sm[ 5][wid]=gls2;
        sm[ 6][wid]=gR[0]; sm[ 7][wid]=gR[1]; sm[ 8][wid]=gR[2];
        sm[ 9][wid]=gR[3]; sm[10][wid]=gR[4]; sm[11][wid]=gR[5];
        sm[12][wid]=gR[6]; sm[13][wid]=gR[7]; sm[14][wid]=gR[8];
        sm[15][wid]=g_G;   sm[16][wid]=g_V;
    }
    __syncthreads();

    /* ── Thread 0: sum warp results, write to global (no atomics needed) ─── */
    if (t == 0) {
        float s0=0.f,s1=0.f,s2=0.f, sl0=0.f,sl1=0.f,sl2=0.f;
        float sR[9]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float sG=0.f, sV=0.f;
        for (int w = 0; w < WARPS_BWD; ++w) {
            s0  += sm[ 0][w]; s1  += sm[ 1][w]; s2  += sm[ 2][w];
            sl0 += sm[ 3][w]; sl1 += sm[ 4][w]; sl2 += sm[ 5][w];
            for (int i = 0; i < 9; ++i) sR[i] += sm[6+i][w];
            sG  += sm[15][w]; sV  += sm[16][w];
        }
        grad_means[k*3+0] = s0;
        grad_means[k*3+1] = s1;
        grad_means[k*3+2] = s2;
        grad_log_s[k*3+0] = sl0;
        grad_log_s[k*3+1] = sl1;
        grad_log_s[k*3+2] = sl2;
        grad_gain[k]  = sG;
        grad_inten[k] = sV;

        float gq[4];
        quat_grad_from_rot_grad(sR, sh_qn, sh_invn, gq);
        grad_quats[k*4+0] = gq[0];
        grad_quats[k*4+1] = gq[1];
        grad_quats[k*4+2] = gq[2];
        grad_quats[k*4+3] = gq[3];
    }
}


// ─── Python-visible entry point ───────────────────────────────────────────────

py::tuple gaussian_backward(
        torch::Tensor grad_out,
        torch::Tensor pts,
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor gain,
        torch::Tensor inten,
        float scale_min,
        float mahal_clamp)
{
    TORCH_CHECK(grad_out.is_cuda() && grad_out.is_contiguous(), "grad_out must be contiguous CUDA");
    TORCH_CHECK(pts.is_cuda()   && pts.is_contiguous(),   "pts must be contiguous CUDA");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be contiguous CUDA");
    TORCH_CHECK(log_s.is_cuda() && log_s.is_contiguous(), "log_s must be contiguous CUDA");
    TORCH_CHECK(quats.is_cuda() && quats.is_contiguous(), "quats must be contiguous CUDA");
    TORCH_CHECK(gain.is_cuda()  && gain.is_contiguous(),  "gain must be contiguous CUDA");
    TORCH_CHECK(inten.is_cuda() && inten.is_contiguous(), "inten must be contiguous CUDA");

    const int M = static_cast<int>(pts.size(0));
    const int N = static_cast<int>(means.size(0));

    auto grad_means = torch::zeros_like(means);
    auto grad_log_s = torch::zeros_like(log_s);
    auto grad_quats = torch::zeros_like(quats);
    auto grad_gain  = torch::zeros_like(gain);
    auto grad_inten = torch::zeros_like(inten);

    /* v2: one block per Gaussian (N blocks), BLOCK_BWD threads share M-loop. */
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_backward_kernel<<<N, BLOCK_BWD, 0, stream>>>(
        grad_out.data_ptr<float>(),
        pts.data_ptr<float>(),
        means.data_ptr<float>(),
        log_s.data_ptr<float>(),
        quats.data_ptr<float>(),
        gain.data_ptr<float>(),
        inten.data_ptr<float>(),
        scale_min, mahal_clamp, M, N,
        grad_means.data_ptr<float>(),
        grad_log_s.data_ptr<float>(),
        grad_quats.data_ptr<float>(),
        grad_gain.data_ptr<float>(),
        grad_inten.data_ptr<float>()
    );

    return py::make_tuple(grad_means, grad_log_s, grad_quats, grad_gain, grad_inten);
}
