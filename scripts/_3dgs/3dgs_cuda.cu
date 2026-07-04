/*
 * 3dgs_cuda.cu — Fused CUDA kernel for 3-D Gaussian field evaluation
 *
 * Implements forward and backward passes for:
 *   f(x) = Σ_k  gain_k · inten_k · exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))
 *
 * where inten is passed as the post-softplus value (softplus applied in Python
 * before calling kernel.forward), so the kernel treats it as a plain positive float.
 * The softplus chain-rule is applied in Python after kernel.backward returns.
 *
 * Covariance:  Σ_k = R_k diag(s_k²) R_kᵀ,   parameterised by [w,x,y,z] + log_s.
 *
 * Both kernels are one-thread-per-sample-point. Gradients w.r.t. Gaussian
 * parameters are accumulated via atomicAdd (safe for concurrent thread access
 * since multiple sample points may hit the same Gaussian).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ─── Device-side helper functions ────────────────────────────────────────────

/* Normalize a raw quaternion q → qn and store inv_norm = 1/||q||.
 * inv_norm is saved so that the gradient through normalisation can be
 * computed without a second rsqrt in the backward pass. */
__device__ __forceinline__ void normalize_quat(
        const float* __restrict__ q,
        float* qn,
        float& inv_norm)
{
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    inv_norm  = rsqrtf(fmaxf(n2, 1e-12f));
    qn[0] = q[0] * inv_norm;
    qn[1] = q[1] * inv_norm;
    qn[2] = q[2] * inv_norm;
    qn[3] = q[3] * inv_norm;
}

/* Closed-form Rodrigues formula: unit quaternion [w,x,y,z] → 3×3 rotation.
 * Row-major flat layout: R[i*3+j] = R_{ij}. */
__device__ __forceinline__ void quat_to_rotmat(const float* qn, float* R)
{
    const float w=qn[0], x=qn[1], y=qn[2], z=qn[3];
    R[0] = 1.f - 2.f*(y*y + z*z);
    R[1] = 2.f*(x*y - w*z);
    R[2] = 2.f*(x*z + w*y);
    R[3] = 2.f*(x*y + w*z);
    R[4] = 1.f - 2.f*(x*x + z*z);
    R[5] = 2.f*(y*z - w*x);
    R[6] = 2.f*(x*z - w*y);
    R[7] = 2.f*(y*z + w*x);
    R[8] = 1.f - 2.f*(x*x + y*y);
}

/* u = Rᵀ v   (transposed matrix–vector product) */
__device__ __forceinline__ void mat_t_vec(
        const float* __restrict__ R,
        const float* __restrict__ v,
        float* u)
{
    u[0] = R[0]*v[0] + R[3]*v[1] + R[6]*v[2];
    u[1] = R[1]*v[0] + R[4]*v[1] + R[7]*v[2];
    u[2] = R[2]*v[0] + R[5]*v[1] + R[8]*v[2];
}

/* u = R v */
__device__ __forceinline__ void mat_vec(
        const float* __restrict__ R,
        const float* __restrict__ v,
        float* u)
{
    u[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    u[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    u[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

/* Backprop through quat_to_rotmat (Rodrigues) and through normalize_quat.
 *
 * Given:
 *   grad_R[9]  — ∂L/∂R in flat row-major layout
 *   qn[4]      — normalised quaternion [w,x,y,z]
 *   inv_norm   — 1/||q_raw|| from normalize_quat
 *
 * Computes:
 *   grad_qraw[4] — ∂L/∂q_raw
 *
 * Two steps:
 *   1. ∂L/∂qn  by summing ∂R[i,j]/∂qn · grad_R[i,j] over all (i,j).
 *   2. Backprop through normalisation: qn = q_raw·inv_norm
 *      ∂L/∂q_raw = inv_norm · (∂L/∂qn  −  qn · (qnᵀ · ∂L/∂qn))
 */
__device__ __forceinline__ void quat_grad_from_rot_grad(
        const float* __restrict__ grad_R,
        const float* __restrict__ qn,
        float inv_norm,
        float* grad_qraw)
{
    const float w=qn[0], x=qn[1], y=qn[2], z=qn[3];
    /* Index helpers: g(i,j) = grad_R[i*3+j] */
    #define G(i,j) grad_R[(i)*3+(j)]

    float gw =  2.f*( -z*G(0,1) + y*G(0,2) + z*G(1,0) - x*G(1,2) - y*G(2,0) + x*G(2,1) );
    float gx =  2.f*(  y*G(0,1) + z*G(0,2) + y*G(1,0) - 2.f*x*G(1,1) - w*G(1,2) + z*G(2,0) + w*G(2,1) - 2.f*x*G(2,2) );
    float gy =  2.f*( -2.f*y*G(0,0) + x*G(0,1) + w*G(0,2) + x*G(1,0) + z*G(1,2) - w*G(2,0) + z*G(2,1) - 2.f*y*G(2,2) );
    float gz =  2.f*( -2.f*z*G(0,0) - w*G(0,1) + x*G(0,2) + w*G(1,0) - 2.f*z*G(1,1) + y*G(1,2) + x*G(2,0) + y*G(2,1) );
    #undef G

    /* Backprop through normalisation: remove the component along qn. */
    float dot = gw*w + gx*x + gy*y + gz*z;
    grad_qraw[0] = inv_norm * (gw - w*dot);
    grad_qraw[1] = inv_norm * (gx - x*dot);
    grad_qraw[2] = inv_norm * (gy - y*dot);
    grad_qraw[3] = inv_norm * (gz - z*dot);
}


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


// ─── Python-visible entry points ──────────────────────────────────────────────

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

// ─── Fused regularisation kernel ─────────────────────────────────────────────
/*
 * One thread per Gaussian.  A single GPU pass computes ALL parameter-only
 * regularisation losses and their analytic gradients simultaneously:
 *
 *   scale_reg      : w_scale  * s_max²
 *   scale_ceiling  : w_ceil   * relu(s_max − cap)
 *   scale_outlier  : w_out    * relu(s_max − out_thresh)   [thresh pre-computed]
 *   anisotropy     : w_aniso  * s_min²
 *   count          : w_count  * sigmoid(inten)
 *   L1_intensity   : w_L1    * softplus(inten)
 *   coverage       : w_cov   * (−log(clamp(s_max/s_ref, ε, cap/s_ref)))
 *   sparsity       : w_spar  * softplus(inten) * (1 − trilinear(vol, means))
 *
 * All weights are baked into the kernel; zero-weight terms are branch-free no-ops.
 * Gradients are written directly (one thread per Gaussian → no atomicAdd on grads).
 * The per-Gaussian loss contribution is atomicAdd'd to a single scalar.
 *
 * Coordinate convention for sparsity trilinear:
 *   means are in [-1,1]³ with align_corners=True → px = (mx+1)/2*(W-1), etc.
 */
#define REG_BLOCK 256

__global__ void gaussian_reg_kernel(
        const float* __restrict__ means,   /* (N,3) in [-1,1]³                */
        const float* __restrict__ log_s,   /* (N,3)                            */
        const float* __restrict__ inten,   /* (N,)  raw pre-softplus intensity */
        const float* __restrict__ volume,  /* (D*H*W,) float in [0,1]          */
        int N, int D, int H, int W,
        /* per-term weights — zero means skip */
        float w_scale, float w_ceil, float cap,
        float w_out,   float out_thresh,
        float w_aniso,
        float w_count, float w_L1,
        float w_cov,   float s_ref, float cap_over_sref,
        float w_spar,
        float inv_N,   /* 1/N — applied here so Python just sums the output */
        float* __restrict__ total_loss,    /* scalar (atomicAdd) */
        float* __restrict__ grad_means,    /* (N,3) */
        float* __restrict__ grad_log_s,    /* (N,3) */
        float* __restrict__ grad_inten     /* (N,) */)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    /* ── Load params ────────────────────────────────────────────────────────── */
    const float ls0 = log_s[k*3+0], ls1 = log_s[k*3+1], ls2 = log_s[k*3+2];
    const float s0 = expf(ls0), s1 = expf(ls1), s2 = expf(ls2);
    const float vi = inten[k];
    const float m0 = means[k*3+0], m1 = means[k*3+1], m2 = means[k*3+2];

    /* ── s_max / s_min with axis tracking ──────────────────────────────────── */
    float s_max = s0; int ax_max = 0;
    if (s1 > s_max) { s_max = s1; ax_max = 1; }
    if (s2 > s_max) { s_max = s2; ax_max = 2; }
    float s_min = s0; int ax_min = 0;
    if (s1 < s_min) { s_min = s1; ax_min = 1; }
    if (s2 < s_min) { s_min = s2; ax_min = 2; }

    /* ── Intensity helpers ──────────────────────────────────────────────────── */
    const float sig = 1.f / (1.f + expf(-vi));   /* sigmoid(vi) */
    const float v_k = log1pf(expf(vi));           /* softplus(vi) */

    /* ── Accumulators ───────────────────────────────────────────────────────── */
    float loss_k = 0.f;
    float gls0 = 0.f, gls1 = 0.f, gls2 = 0.f;
    float gm0  = 0.f, gm1  = 0.f, gm2  = 0.f;
    float gvi  = 0.f;
    float* gls_ptr[3] = {&gls0, &gls1, &gls2};

    /* scale_reg: s_max²  — grad: 2*s_max² w.r.t. log_s[ax_max] */
    if (w_scale > 0.f) {
        loss_k += w_scale * s_max * s_max;
        *gls_ptr[ax_max] += w_scale * 2.f * s_max * s_max;
    }

    /* scale_ceiling: relu(s_max − cap)  — grad: s_max if active */
    if (w_ceil > 0.f && s_max > cap) {
        loss_k += w_ceil * (s_max - cap);
        *gls_ptr[ax_max] += w_ceil * s_max;
    }

    /* scale_outlier: relu(s_max − out_thresh)  — same pattern */
    if (w_out > 0.f && s_max > out_thresh) {
        loss_k += w_out * (s_max - out_thresh);
        *gls_ptr[ax_max] += w_out * s_max;
    }

    /* anisotropy: s_min²  — grad: 2*s_min² w.r.t. log_s[ax_min] */
    if (w_aniso > 0.f) {
        loss_k += w_aniso * s_min * s_min;
        *gls_ptr[ax_min] += w_aniso * 2.f * s_min * s_min;
    }

    /* count: sigmoid(vi)  — grad: sig*(1-sig) */
    if (w_count > 0.f) {
        loss_k += w_count * sig;
        gvi += w_count * sig * (1.f - sig);
    }

    /* L1_intensity: softplus(vi)  — grad: sigmoid(vi) */
    if (w_L1 > 0.f) {
        loss_k += w_L1 * v_k;
        gvi += w_L1 * sig;
    }

    /* coverage: -log(clamp(s_max/s_ref, ε, cap/s_ref))
     * grad w.r.t. log_s[ax_max]: -1  (since log(s_max)=log_s[ax_max])
     * only when not clamped at either end                                    */
    if (w_cov > 0.f) {
        float ratio = s_max / s_ref;
        float ratio_c = fmaxf(1e-4f, fminf(ratio, cap_over_sref));
        loss_k += w_cov * (-logf(ratio_c));
        if (ratio >= 1e-4f && ratio <= cap_over_sref)
            *gls_ptr[ax_max] -= w_cov;
    }

    /* sparsity: v_k * (1 - trilinear(vol, means))
     * means are in [-1,1]³; align_corners=True → px=(m+1)/2*(W-1)           */
    if (w_spar > 0.f && D > 1 && H > 1 && W > 1) {
        float px = (m0 + 1.f) * 0.5f * (W - 1);
        float py = (m1 + 1.f) * 0.5f * (H - 1);
        float pz = (m2 + 1.f) * 0.5f * (D - 1);

        if (px >= 0.f && px <= W-1.f && py >= 0.f && py <= H-1.f &&
            pz >= 0.f && pz <= D-1.f) {
            int ix = min((int)px, W-2), iy = min((int)py, H-2), iz = min((int)pz, D-2);
            float fx = px-ix, fy = py-iy, fz = pz-iz;

            float v000 = volume[iz*H*W + iy*W + ix];
            float v001 = volume[iz*H*W + iy*W + ix+1];
            float v010 = volume[iz*H*W + (iy+1)*W + ix];
            float v011 = volume[iz*H*W + (iy+1)*W + ix+1];
            float v100 = volume[(iz+1)*H*W + iy*W + ix];
            float v101 = volume[(iz+1)*H*W + iy*W + ix+1];
            float v110 = volume[(iz+1)*H*W + (iy+1)*W + ix];
            float v111 = volume[(iz+1)*H*W + (iy+1)*W + ix+1];

            float c00 = v000*(1-fx) + v001*fx;
            float c01 = v010*(1-fx) + v011*fx;
            float c10 = v100*(1-fx) + v101*fx;
            float c11 = v110*(1-fx) + v111*fx;
            float c0  = c00*(1-fy) + c01*fy;
            float c1  = c10*(1-fy) + c11*fy;
            float gt_k = c0*(1-fz) + c1*fz;

            float onemgt = 1.f - gt_k;
            loss_k += w_spar * v_k * onemgt;
            gvi    += w_spar * sig * onemgt;

            /* d(gt)/d(px)*d(px)/d(m0), etc. */
            float dgt_dx = (1-fz)*((1-fy)*(v001-v000) + fy*(v011-v010))
                         + fz*    ((1-fy)*(v101-v100) + fy*(v111-v110));
            float dgt_dy = (1-fz)*((1-fx)*(v010-v000) + fx*(v011-v001))
                         + fz*    ((1-fx)*(v110-v100) + fx*(v111-v101));
            float dgt_dz = (1-fy)*((1-fx)*(v100-v000) + fx*(v101-v001))
                         + fy*    ((1-fx)*(v110-v010) + fx*(v111-v011));

            float gf = -w_spar * v_k;
            gm0 += gf * dgt_dx * 0.5f * (W-1);
            gm1 += gf * dgt_dy * 0.5f * (H-1);
            gm2 += gf * dgt_dz * 0.5f * (D-1);
        }
    }

    /* ── Normalise by 1/N and write ─────────────────────────────────────────── */
    atomicAdd(total_loss, loss_k * inv_N);
    grad_means[k*3+0] = gm0  * inv_N;
    grad_means[k*3+1] = gm1  * inv_N;
    grad_means[k*3+2] = gm2  * inv_N;
    grad_log_s[k*3+0] = gls0 * inv_N;
    grad_log_s[k*3+1] = gls1 * inv_N;
    grad_log_s[k*3+2] = gls2 * inv_N;
    grad_inten[k]     = gvi  * inv_N;
}

py::tuple gaussian_reg(
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor inten,
        torch::Tensor volume,      /* (D*H*W,) flat float on CUDA */
        int64_t N, int64_t D, int64_t H, int64_t W,
        float w_scale, float w_ceil,  float cap,
        float w_out,   float out_thresh,
        float w_aniso,
        float w_count, float w_L1,
        float w_cov,   float s_ref,   float cap_over_sref,
        float w_spar,
        float inv_N)
{
    TORCH_CHECK(means.is_cuda()  && means.is_contiguous(),  "means must be contiguous CUDA");
    TORCH_CHECK(log_s.is_cuda()  && log_s.is_contiguous(),  "log_s must be contiguous CUDA");
    TORCH_CHECK(inten.is_cuda()  && inten.is_contiguous(),  "inten must be contiguous CUDA");
    TORCH_CHECK(volume.is_cuda() && volume.is_contiguous(), "volume must be contiguous CUDA");

    auto total_loss = torch::zeros({1}, means.options());
    auto grad_means = torch::zeros_like(means);
    auto grad_log_s = torch::zeros_like(log_s);
    auto grad_inten = torch::zeros_like(inten);

    const int blocks = ((int)N + REG_BLOCK - 1) / REG_BLOCK;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_reg_kernel<<<blocks, REG_BLOCK, 0, stream>>>(
        means.data_ptr<float>(), log_s.data_ptr<float>(),
        inten.data_ptr<float>(), volume.data_ptr<float>(),
        (int)N, (int)D, (int)H, (int)W,
        w_scale, w_ceil, cap,
        w_out, out_thresh,
        w_aniso,
        w_count, w_L1,
        w_cov, s_ref, cap_over_sref,
        w_spar,
        inv_N,
        total_loss.data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_log_s.data_ptr<float>(),
        grad_inten.data_ptr<float>()
    );

    return py::make_tuple(total_loss, grad_means, grad_log_s, grad_inten);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &gaussian_forward,  "Gaussian splatting forward (CUDA)");
    m.def("backward", &gaussian_backward, "Gaussian splatting backward (CUDA)");
    m.def("gaussian_reg", &gaussian_reg,  "Fused regularisation loss + grad (CUDA)");
}
