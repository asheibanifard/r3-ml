/*
 * regularization.cu — fused per-Gaussian regularisation losses and their
 * analytic gradients, in a single GPU pass.
 *
 * One thread per Gaussian. Self-contained: unlike forward.cu/backward.cu,
 * this kernel does no quaternion/rotation math, so it does not need
 * common.cuh.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ─── Fused regularisation kernel ─────────────────────────────────────────────
/*
 * A single GPU pass computes ALL parameter-only regularisation losses and
 * their analytic gradients simultaneously:
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


// ─── Python-visible entry point ───────────────────────────────────────────────

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
