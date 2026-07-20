// ============================================================================
// gaussian_splat_scratch.cu
//
// Fully standalone, from-scratch pure-CUDA pipeline (no PyTorch, no reuse of
// scripts/_3dgs/3dgs_cuda.cu or Mip_Render_Inside_Volume.cu):
//
//   1. Generate a synthetic 64^3 voxel grid (sum of anisotropic blobs, fixed,
//      deterministic).
//   2. Fit a fixed-count mixture of anisotropic 3D Gaussians to that grid:
//      hand-written forward density model, analytic backward gradients
//      (mean, log-scale, quaternion incl. normalization Jacobian, intensity),
//      hand-written Adam optimizer. No autodiff.
//   3. Finite-difference self-test of the analytic gradients before trusting
//      the trained model.
//   4. Baseline renderer: vanilla DVR (ray-march the target voxel grid,
//      trilinear sampling, MIP) from a camera at the volume centre.
//   5. Gaussian rasterizer: camera at the volume centre, only Gaussians in
//      front of the camera considered, closed-form ray/Gaussian interval +
//      depth-binned accumulation (max-of-bins == max-of-continuous-ray-sum).
//   6. Camera yaw sweep -> per-frame GT / reconstruction frames written to
//      disk (simple raw binary format), FPS measured for both renderers.
//
// A companion Python script (render_outputs.py) assembles the frames into
// GT/Recon/Diff mp4s, plots, and a PSNR/SSIM/LPIPS/FPS metrics table.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Global configuration
// ---------------------------------------------------------------------------
constexpr int GRID = 64;                       // synthetic voxel grid resolution
constexpr int VOXEL_COUNT = GRID * GRID * GRID; // 262144

constexpr int N_GAUSSIANS = 800;
constexpr int TRAIN_ITERS = 3000;
constexpr float MAHALANOBIS_CUTOFF = 20.0f;     // matches project convention
constexpr float MIN_SCALE = 1.0e-3f;
constexpr float MAX_SCALE = 1.2f;

constexpr int N_FRAMES = 60;
constexpr int RENDER_BINS = 48;                 // depth bins, rasterizer
constexpr int DVR_SAMPLES = 128;                // ray samples, DVR baseline
constexpr float FOV_DEG = 90.0f;
constexpr float BOX_MIN = -1.0f, BOX_MAX = 1.0f;
constexpr float NEAR_PLANE = 1.0e-3f;            // camera-front cull threshold
constexpr int TILE_SIZE = 16;                   // tile-based rasterizer: pixels per tile side

// ---------------------------------------------------------------------------
// Composite training loss, matching the formula documented in CLAUDE.md for
// scripts/_3dgs/_3dgs.py (base L1 + SSIM-on-crop + 7 per-Gaussian
// regularizers). CLAUDE.md specifies the formula, not numeric weights/refs --
// the values below are our own reasonable choices, kept small relative to the
// L1 base term so they act as genuine regularizers rather than dominating.
// GRID==64 exactly matches the documented "64x64 crop" size, so the random
// crop here is simply one full XY-slice at a random Z index.
// ---------------------------------------------------------------------------
constexpr float LAMBDA_SSIM     = 0.05f;
constexpr float LAMBDA_SCALE    = 1.0e-3f;
constexpr float LAMBDA_CEILING  = 1.0f;
constexpr float LAMBDA_OUTLIER  = 1.0f;
constexpr float LAMBDA_SPARSITY = 1.0e-2f;
constexpr float LAMBDA_ANISO    = 1.0e-3f;
constexpr float LAMBDA_COUNT    = 1.0e-4f;
constexpr float LAMBDA_L1REG    = 1.0e-4f;      // CLAUDE.md's "lambda_L1" term (on softplus(raw_inten)), distinct from the base L1(pred,gt) data term
constexpr float LAMBDA_COVERAGE = 1.0e-3f;
constexpr float SCALE_CAP       = 0.30f;         // ceiling term cap on s_max
constexpr float SCALE_REF       = 0.15f;         // coverage term reference scale
constexpr float SSIM_K1 = 0.01f, SSIM_K2 = 0.03f; // data range L=1 (values in [0,1]) -> C1=(K1*L)^2, C2=(K2*L)^2
constexpr float SSIM_C1 = SSIM_K1 * SSIM_K1, SSIM_C2 = SSIM_K2 * SSIM_K2;

// Adam
constexpr float ADAM_BETA1 = 0.9f, ADAM_BETA2 = 0.999f, ADAM_EPS = 1e-8f;
constexpr float LR_MEAN = 4.0e-3f, LR_LOGSCALE = 5.0e-3f, LR_QUAT = 1.5e-3f, LR_INTEN = 1.5e-2f;

// ---------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------
__host__ __device__ inline float3 f3make(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 f3sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 f3add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 f3scale(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ inline float f3dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 f3cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float3 f3norm(float3 a) {
    float l = sqrtf(fmaxf(f3dot(a, a), 1e-20f));
    return f3scale(a, 1.0f / l);
}
__host__ __device__ inline float softplus(float x) {
    // numerically stable
    return x > 20.0f ? x : log1pf(expf(x));
}
__host__ __device__ inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// Voxel <-> world coordinate mapping (cell-centred, matches AABB convention)
// ---------------------------------------------------------------------------
__host__ __device__ inline float3 voxel_center(int ix, int iy, int iz) {
    float sx = (BOX_MAX - BOX_MIN) / GRID;
    float x = BOX_MIN + (ix + 0.5f) * sx;
    float y = BOX_MIN + (iy + 0.5f) * sx;
    float z = BOX_MIN + (iz + 0.5f) * sx;
    return f3make(x, y, z);
}

// Trilinear sample of a dense GRID^3 voxel array at a continuous world
// position. Used both by the DVR renderer (ray-march sampling) and, further
// below, by the training loss's sparsity regularizer (GT lookup at each
// Gaussian's continuous mean position, CLAUDE.md's "GT(mu_k)" term).
__host__ __device__ inline float sample_voxel_trilinear(const float* grid, float3 pos) {
    float gx = (pos.x - BOX_MIN) / (BOX_MAX - BOX_MIN) * GRID - 0.5f;
    float gy = (pos.y - BOX_MIN) / (BOX_MAX - BOX_MIN) * GRID - 0.5f;
    float gz = (pos.z - BOX_MIN) / (BOX_MAX - BOX_MIN) * GRID - 0.5f;
    int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
    float fx = gx - x0, fy = gy - y0, fz = gz - z0;
    float acc = 0.0f;
    for (int dz = 0; dz <= 1; ++dz)
        for (int dy = 0; dy <= 1; ++dy)
            for (int dx = 0; dx <= 1; ++dx) {
                int xi = x0 + dx, yi = y0 + dy, zi = z0 + dz;
                if (xi < 0 || xi >= GRID || yi < 0 || yi >= GRID || zi < 0 || zi >= GRID) continue;
                float wx = dx ? fx : (1 - fx);
                float wy = dy ? fy : (1 - fy);
                float wz = dz ? fz : (1 - fz);
                acc += wx * wy * wz * grid[xi + yi * GRID + zi * GRID * GRID];
            }
    return acc;
}

// ---------------------------------------------------------------------------
// Synthetic target: fixed sum of anisotropic blobs, deterministic, no RNG.
// ---------------------------------------------------------------------------
struct Blob { float3 center; float3 inv_sigma2; float amplitude; };

__constant__ Blob d_blobs[8];
constexpr int N_BLOBS = 8;

void make_blobs(Blob* blobs) {
    // Hand-picked, deterministic layout spread through the unit cube so the
    // target has real 3D structure (overlaps, elongated + round features).
    struct Spec { float cx, cy, cz, sx, sy, sz, amp; };
    Spec specs[N_BLOBS] = {
        { -0.45f, -0.35f, -0.30f, 0.22f, 0.14f, 0.14f, 1.00f },
        {  0.40f,  0.30f,  0.25f, 0.16f, 0.16f, 0.30f, 0.90f },
        {  0.10f, -0.40f,  0.35f, 0.10f, 0.10f, 0.10f, 1.10f },
        { -0.30f,  0.35f, -0.20f, 0.12f, 0.28f, 0.10f, 0.85f },
        {  0.35f, -0.25f, -0.40f, 0.20f, 0.10f, 0.10f, 0.75f },
        {  0.00f,  0.05f,  0.00f, 0.35f, 0.08f, 0.08f, 0.60f },
        { -0.10f,  0.15f,  0.45f, 0.09f, 0.09f, 0.20f, 0.95f },
        {  0.25f,  0.45f, -0.05f, 0.13f, 0.13f, 0.13f, 0.70f },
    };
    for (int i = 0; i < N_BLOBS; ++i) {
        Blob b;
        b.center = f3make(specs[i].cx, specs[i].cy, specs[i].cz);
        b.inv_sigma2 = f3make(1.0f / (specs[i].sx * specs[i].sx),
                               1.0f / (specs[i].sy * specs[i].sy),
                               1.0f / (specs[i].sz * specs[i].sz));
        b.amplitude = specs[i].amp;
        blobs[i] = b;
    }
}

__global__ void generate_target_kernel(float* target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= VOXEL_COUNT) return;
    int ix = idx % GRID;
    int iy = (idx / GRID) % GRID;
    int iz = idx / (GRID * GRID);
    float3 x = voxel_center(ix, iy, iz);

    float v = 0.0f;
    #pragma unroll
    for (int b = 0; b < N_BLOBS; ++b) {
        float3 d = f3sub(x, d_blobs[b].center);
        float q = d.x * d.x * d_blobs[b].inv_sigma2.x +
                  d.y * d.y * d_blobs[b].inv_sigma2.y +
                  d.z * d.z * d_blobs[b].inv_sigma2.z;
        v += d_blobs[b].amplitude * expf(-0.5f * q);
    }
    target[idx] = fminf(fmaxf(v, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// Trainable Gaussian parameters (SoA), gradients, Adam state
// ---------------------------------------------------------------------------
struct Params {
    float3* mean;
    float3* log_scale;
    float4* quat;       // raw (unnormalized), (w,x,y,z)
    float*  raw_inten;
};
struct Grads {
    float3* d_mean;
    float3* d_log_scale;
    float4* d_quat;
    float*  d_raw_inten;
};
struct AdamState {
    float3* m_mean;  float3* v_mean;
    float3* m_logs;  float3* v_logs;
    float4* m_quat;  float4* v_quat;
    float*  m_inten; float* v_inten;
};

// Per-Gaussian precomputed runtime data (rebuilt every iteration before the
// forward/backward passes, since params change every step).
struct GaussianRuntime {
    float3 mean;
    float  R[9];        // row-major rotation matrix, local->world (columns = local axes in world frame)
    float  qnorm[4];     // normalized quaternion (w,x,y,z), needed for backward chain
    float  qraw_len;     // ||raw quaternion||, needed for normalization backward
    float3 inv_s2;
    float3 s2;
    float  amplitude;
    float  support_radius;
    float  s_max, s_min;   // post-clamp scale extremes, needed by the scale/ceiling/outlier/aniso/coverage loss terms
    int    max_axis, min_axis; // which of x/y/z achieves s_max/s_min, for routing gradients to the right log_scale axis
};

void alloc_params(Params& p, int n) {
    CUDA_CHECK(cudaMalloc(&p.mean, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&p.log_scale, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&p.quat, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&p.raw_inten, n * sizeof(float)));
}
void alloc_grads(Grads& g, int n) {
    CUDA_CHECK(cudaMalloc(&g.d_mean, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.d_log_scale, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.d_quat, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g.d_raw_inten, n * sizeof(float)));
}
void alloc_adam(AdamState& a, int n) {
    CUDA_CHECK(cudaMalloc(&a.m_mean, n * sizeof(float3))); CUDA_CHECK(cudaMalloc(&a.v_mean, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&a.m_logs, n * sizeof(float3))); CUDA_CHECK(cudaMalloc(&a.v_logs, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&a.m_quat, n * sizeof(float4))); CUDA_CHECK(cudaMalloc(&a.v_quat, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&a.m_inten, n * sizeof(float))); CUDA_CHECK(cudaMalloc(&a.v_inten, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(a.m_mean, 0, n * sizeof(float3))); CUDA_CHECK(cudaMemset(a.v_mean, 0, n * sizeof(float3)));
    CUDA_CHECK(cudaMemset(a.m_logs, 0, n * sizeof(float3))); CUDA_CHECK(cudaMemset(a.v_logs, 0, n * sizeof(float3)));
    CUDA_CHECK(cudaMemset(a.m_quat, 0, n * sizeof(float4))); CUDA_CHECK(cudaMemset(a.v_quat, 0, n * sizeof(float4)));
    CUDA_CHECK(cudaMemset(a.m_inten, 0, n * sizeof(float))); CUDA_CHECK(cudaMemset(a.v_inten, 0, n * sizeof(float)));
}

// Deterministic host-side init (simple LCG, fixed seed -> fully reproducible)
struct LCG {
    uint64_t s;
    LCG(uint64_t seed) : s(seed) {}
    float next01() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 33) & 0xFFFFFF) / float(0xFFFFFF); }
    float nextRange(float lo, float hi) { return lo + next01() * (hi - lo); }
};

void init_params_host(Params& p, int n) {
    float3* h_mean = new float3[n];
    float3* h_logs = new float3[n];
    float4* h_quat = new float4[n];
    float*  h_int  = new float[n];
    LCG rng(123456789ULL);
    for (int i = 0; i < n; ++i) {
        h_mean[i] = f3make(rng.nextRange(-0.9f, 0.9f), rng.nextRange(-0.9f, 0.9f), rng.nextRange(-0.9f, 0.9f));
        float s0 = logf(rng.nextRange(0.08f, 0.16f));
        h_logs[i] = f3make(s0, s0, s0);
        h_quat[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // identity rotation
        h_int[i] = -2.0f; // softplus(-2) ~= 0.127
    }
    CUDA_CHECK(cudaMemcpy(p.mean, h_mean, n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.log_scale, h_logs, n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.quat, h_quat, n * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.raw_inten, h_int, n * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_mean; delete[] h_logs; delete[] h_quat; delete[] h_int;
}

// ---------------------------------------------------------------------------
// Quaternion -> rotation matrix (row-major, local->world convention: columns
// of R are the local axes expressed in world coordinates, so Sigma = R S^2 R^T)
// ---------------------------------------------------------------------------
__host__ __device__ inline void quat_to_R(float w, float x, float y, float z, float* R) {
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w);     R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);     R[7] = 2 * (y * z + x * w);     R[8] = 1 - 2 * (x * x + y * y);
}

__global__ void precompute_runtime_kernel(const Params p, GaussianRuntime* rt, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    // float4 layout: (w,x,y,z) stored in (.x,.y,.z,.w)
    float4 q = p.quat[k];
    float qlen = sqrtf(fmaxf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w, 1e-20f));
    float w = q.x / qlen, x = q.y / qlen, y = q.z / qlen, z = q.w / qlen;

    GaussianRuntime r;
    r.mean = p.mean[k];
    quat_to_R(w, x, y, z, r.R);
    r.qnorm[0] = w; r.qnorm[1] = x; r.qnorm[2] = y; r.qnorm[3] = z;
    r.qraw_len = qlen;

    float3 ls = p.log_scale[k];
    float sx = fminf(fmaxf(expf(ls.x), MIN_SCALE), MAX_SCALE);
    float sy = fminf(fmaxf(expf(ls.y), MIN_SCALE), MAX_SCALE);
    float sz = fminf(fmaxf(expf(ls.z), MIN_SCALE), MAX_SCALE);
    r.s2 = f3make(sx * sx, sy * sy, sz * sz);
    r.inv_s2 = f3make(1.0f / r.s2.x, 1.0f / r.s2.y, 1.0f / r.s2.z);
    r.amplitude = softplus(p.raw_inten[k]);
    float max_s2 = fmaxf(r.s2.x, fmaxf(r.s2.y, r.s2.z));
    r.support_radius = sqrtf(MAHALANOBIS_CUTOFF * max_s2);

    float s_arr[3] = { sx, sy, sz };
    r.max_axis = 0; r.min_axis = 0;
    #pragma unroll
    for (int i = 1; i < 3; ++i) {
        if (s_arr[i] > s_arr[r.max_axis]) r.max_axis = i;
        if (s_arr[i] < s_arr[r.min_axis]) r.min_axis = i;
    }
    r.s_max = s_arr[r.max_axis];
    r.s_min = s_arr[r.min_axis];
    rt[k] = r;
}

// world offset d -> local coords u = R^T d
__host__ __device__ inline float3 world_to_local(const float* R, float3 d) {
    return f3make(R[0] * d.x + R[3] * d.y + R[6] * d.z,
                  R[1] * d.x + R[4] * d.y + R[7] * d.z,
                  R[2] * d.x + R[5] * d.y + R[8] * d.z);
}
// local vector w -> world coords: R * w
__host__ __device__ inline float3 local_to_world(const float* R, float3 w) {
    return f3make(R[0] * w.x + R[1] * w.y + R[2] * w.z,
                  R[3] * w.x + R[4] * w.y + R[5] * w.z,
                  R[6] * w.x + R[7] * w.y + R[8] * w.z);
}

// Takes raw fields (rather than a GaussianRuntime&) so both the full
// GaussianRuntime struct and the lightweight shared-memory copy used by the
// tile-based rasterizer (below) can share this one implementation.
__host__ __device__ inline float eval_gaussian_density(float3 mean, const float* R, float3 inv_s2, float amplitude,
                                                        float3 x, float* out_Q = nullptr) {
    float3 d = f3sub(x, mean);
    float3 u = world_to_local(R, d);
    float Q = u.x * u.x * inv_s2.x + u.y * u.y * inv_s2.y + u.z * u.z * inv_s2.z;
    if (out_Q) *out_Q = Q;
    if (Q > MAHALANOBIS_CUTOFF) return 0.0f;
    return amplitude * expf(-0.5f * Q);
}

// ---------------------------------------------------------------------------
// Forward + per-voxel loss-gradient kernel: one thread per VOXEL, loop over
// all Gaussians (mirrors project's own forward design: many samples, fused
// tile over Gaussians -- here left as a plain loop for a from-scratch,
// dependency-free implementation).
// ---------------------------------------------------------------------------
__global__ void forward_kernel(const GaussianRuntime* rt, int n,
                                const float* target, float* pred, float* grad_out,
                                double* loss_accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= VOXEL_COUNT) return;
    int ix = idx % GRID;
    int iy = (idx / GRID) % GRID;
    int iz = idx / (GRID * GRID);
    float3 x = voxel_center(ix, iy, iz);

    float p = 0.0f;
    for (int k = 0; k < n; ++k) {
        float3 d = f3sub(x, rt[k].mean);
        if (f3dot(d, d) > rt[k].support_radius * rt[k].support_radius) continue;
        p += eval_gaussian_density(rt[k].mean, rt[k].R, rt[k].inv_s2, rt[k].amplitude, x);
    }
    pred[idx] = p;
    float t = target[idx];
    float diff = p - t;
    // Base data term is L1(pred,gt) (CLAUDE.md's documented loss), not MSE:
    // subgradient of mean(|diff|) is sign(diff)/M (0 at the measure-zero diff==0 case).
    float sign_diff = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    grad_out[idx] = sign_diff / VOXEL_COUNT;
    // Loss scalar accumulated in double: with 262144 single-precision atomicAdds
    // the partial sum grows large enough that float32 rounding noise (~1e-4
    // scale) swamps the tiny loss deltas a finite-difference check needs to
    // measure. Only this monitoring/self-test scalar needs double; per-voxel
    // gradients above are computed independently so they aren't affected.
    atomicAdd(loss_accum, (double)fabsf(diff));
}

// ---------------------------------------------------------------------------
// Differentiable SSIM on a random Z-slice (CLAUDE.md's "SSIM(random 64x64
// Z-crop)" term). GRID==64 exactly matches that crop size, so the "crop" is
// simply one full XY-slice at a random Z index. Uses GLOBAL (whole-slice)
// SSIM statistics rather than a sliding window, giving a clean closed-form
// per-pixel gradient instead of needing a windowed-convolution backward pass.
//
// Stats accumulated in double (same reasoning as the loss scalar above:
// 4096-term atomicAdd sums are large enough that float32 rounding would
// swamp the gradient-relevant precision).
// ---------------------------------------------------------------------------
__global__ void ssim_slice_stats_kernel(const float* pred, const float* target, int z_slice,
                                         double* stats /* [sum_x, sum_y, sum_x2, sum_y2, sum_xy] */) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= GRID || iy >= GRID) return;
    int idx = ix + iy * GRID + z_slice * GRID * GRID;
    double x = (double)pred[idx];
    double y = (double)target[idx];
    atomicAdd(&stats[0], x);
    atomicAdd(&stats[1], y);
    atomicAdd(&stats[2], x * x);
    atomicAdd(&stats[3], y * y);
    atomicAdd(&stats[4], x * y);
}

// Host-side helper: turns the 5 raw moment sums into (mu_x,mu_y,var_x,var_y,
// cov_xy,SSIM). Var/cov via the standard E[XY]-E[X]E[Y] identity -- fine in
// double precision for data in [0,1].
inline double ssim_from_stats(const double* stats, double& mu_x, double& mu_y,
                               double& var_x, double& var_y, double& cov_xy) {
    double n = (double)(GRID * GRID);
    mu_x = stats[0] / n;
    mu_y = stats[1] / n;
    var_x = stats[2] / n - mu_x * mu_x;
    var_y = stats[3] / n - mu_y * mu_y;
    cov_xy = stats[4] / n - mu_x * mu_y;
    double A = 2.0 * mu_x * mu_y + SSIM_C1;
    double B = 2.0 * cov_xy + SSIM_C2;
    double C = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
    double D = var_x + var_y + SSIM_C2;
    return (A * B) / (C * D);
}

// Adds -LAMBDA_SSIM * d(SSIM)/dx_i into grad_out for every pixel in the slice
// (additive on top of the L1 gradient forward_kernel already wrote there).
// Derivation (global/whole-slice SSIM, n=GRID*GRID pixels):
//   SSIM = (A*B)/(C*D),  A=2*mu_x*mu_y+C1, B=2*cov_xy+C2, C=mu_x^2+mu_y^2+C1, D=var_x+var_y+C2
//   dmu_x/dx_i = 1/n ;  dvar_x/dx_i = 2(x_i-mu_x)/n ;  dcov_xy/dx_i = (y_i-mu_y)/n
//   => dSSIM/dx_i = (B/CD)*dA/dx_i + (A/CD)*dB/dx_i - SSIM*(dC/dx_i)/C - SSIM*(dD/dx_i)/D
__global__ void ssim_slice_grad_kernel(const float* pred, const float* target, int z_slice,
                                        float* grad_out,
                                        float mu_x, float mu_y, float var_x, float var_y, float cov_xy) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= GRID || iy >= GRID) return;
    int idx = ix + iy * GRID + z_slice * GRID * GRID;
    float xi = pred[idx];
    float yi = target[idx];

    float A = 2.0f * mu_x * mu_y + SSIM_C1;
    float B = 2.0f * cov_xy + SSIM_C2;
    float C = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
    float D = var_x + var_y + SSIM_C2;
    float SSIM = (A * B) / (C * D);

    const float inv_n = 1.0f / (float)(GRID * GRID);
    float dA = 2.0f * mu_y * inv_n;
    float dB = 2.0f * (yi - mu_y) * inv_n;
    float dC = 2.0f * mu_x * inv_n;
    float dD = 2.0f * (xi - mu_x) * inv_n;

    float dSSIM = (B / (C * D)) * dA + (A / (C * D)) * dB - SSIM * (dC / C) - SSIM * (dD / D);
    grad_out[idx] += -LAMBDA_SSIM * dSSIM;
}

// Extracts per-Gaussian quantities needed to compute the (non-SSIM) regularizer
// loss VALUES on the host: s_max, s_min (already in GaussianRuntime, just
// need a contiguous array), and GT(mean_k) (trilinear lookup of the target
// grid at the Gaussian's continuous mean position -- CLAUDE.md's sparsity
// term). The corresponding GRADIENTS are computed directly inside
// backward_kernel below, not from these host-side arrays.
__global__ void extract_regularizer_inputs_kernel(const GaussianRuntime* rt, const float* target, int n,
                                                    float* out_s_max, float* out_s_min, float* out_gt_at_mean) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    out_s_max[k] = rt[k].s_max;
    out_s_min[k] = rt[k].s_min;
    out_gt_at_mean[k] = sample_voxel_trilinear(target, rt[k].mean);
}

// Median + MAD (median absolute deviation) of s_max across all Gaussians, for
// the outlier-suppression loss term. N_GAUSSIANS is tiny (800), so a plain
// host-side sort per iteration is simple and fast enough -- not worth a
// device-side selection/sort kernel for this size.
inline void compute_median_mad(const float* h_s_max, int n, float& median, float& mad) {
    std::vector<float> sorted(h_s_max, h_s_max + n);
    std::sort(sorted.begin(), sorted.end());
    median = (n % 2 == 0) ? 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]) : sorted[n / 2];
    std::vector<float> dev(n);
    for (int i = 0; i < n; ++i) dev[i] = fabsf(h_s_max[i] - median);
    std::sort(dev.begin(), dev.end());
    mad = (n % 2 == 0) ? 0.5f * (dev[n / 2 - 1] + dev[n / 2]) : dev[n / 2];
}

// ---------------------------------------------------------------------------
// Backward kernel: one thread per GAUSSIAN, loops over all voxels. Since each
// Gaussian's gradient is only ever written by its own thread, no atomics are
// needed at all (matches the project's own "transposed layout" backward
// design philosophy, described in CLAUDE.md for the real training kernel).
// ---------------------------------------------------------------------------
__global__ void backward_kernel(const Params p, const GaussianRuntime* rt, int n,
                                 const float* grad_out, const float* target,
                                 float outlier_median, float outlier_mad, Grads g) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    GaussianRuntime r = rt[k];
    float radius2 = r.support_radius * r.support_radius;

    float3 d_mean = f3make(0, 0, 0);
    float3 d_logs = f3make(0, 0, 0);
    float GradR[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float d_inten = 0.0f;

    for (int ix = 0; ix < GRID; ++ix) {
        for (int iy = 0; iy < GRID; ++iy) {
            for (int iz = 0; iz < GRID; ++iz) {
                float3 x = voxel_center(ix, iy, iz);
                float3 dd = f3sub(x, r.mean);
                if (f3dot(dd, dd) > radius2) continue;
                float3 u = world_to_local(r.R, dd);
                float Q = u.x * u.x * r.inv_s2.x + u.y * u.y * r.inv_s2.y + u.z * u.z * r.inv_s2.z;
                if (Q > MAHALANOBIS_CUTOFF) continue;
                float e = expf(-0.5f * Q);
                float f = r.amplitude * e;
                int j = ix + iy * GRID + iz * GRID * GRID;
                float gj = grad_out[j];

                // intensity
                d_inten += gj * e;

                // local-frame scale gradient: dL/d(log_scale_i) = gj*f*(u_i^2 * inv_s2_i)
                d_logs.x += gj * f * (u.x * u.x * r.inv_s2.x);
                d_logs.y += gj * f * (u.y * u.y * r.inv_s2.y);
                d_logs.z += gj * f * (u.z * u.z * r.inv_s2.z);

                // mean gradient (world frame): gj*f*Sigma^-1*dd = gj*f*R*(inv_s2 .* u)
                float3 w_local = f3make(u.x * r.inv_s2.x, u.y * r.inv_s2.y, u.z * r.inv_s2.z);
                float3 Rw = local_to_world(r.R, w_local);
                d_mean.x += gj * f * Rw.x;
                d_mean.y += gj * f * Rw.y;
                d_mean.z += gj * f * Rw.z;

                // rotation-matrix gradient: dL/dR[a][i] += -gj*f*dd_a*w_local_i
                float coef = -gj * f;
                float da[3] = { dd.x, dd.y, dd.z };
                float wi[3] = { w_local.x, w_local.y, w_local.z };
                #pragma unroll
                for (int a = 0; a < 3; ++a)
                    #pragma unroll
                    for (int i = 0; i < 3; ++i)
                        GradR[a * 3 + i] += coef * da[a] * wi[i];
            }
        }
    }

    // Chain GradR through dR/dq_norm (closed form, verified against direct
    // differentiation of quat_to_R), then through the normalization Jacobian.
    float qw = r.qnorm[0], qx = r.qnorm[1], qy = r.qnorm[2], qz = r.qnorm[3];
    // dR_{a,i}/dq[m] tables, indexed [a*3+i][m] for m in {w,x,y,z}
    float dRdq[9][4] = {
        /*R00*/ {0,      0,      -4*qy,  -4*qz},
        /*R01*/ {-2*qz,  2*qy,   2*qx,   -2*qw},
        /*R02*/ {2*qy,   2*qz,   2*qw,   2*qx},
        /*R10*/ {2*qz,   2*qy,   2*qx,   2*qw},
        /*R11*/ {0,      -4*qx,  0,      -4*qz},
        /*R12*/ {-2*qx,  -2*qw,  2*qz,   2*qy},
        /*R20*/ {-2*qy,  2*qz,   -2*qw,  2*qx},
        /*R21*/ {2*qx,   2*qw,   2*qz,   2*qy},
        /*R22*/ {0,      -4*qx,  -4*qy,  0},
    };
    float dL_dqnorm[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int e = 0; e < 9; ++e)
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            dL_dqnorm[m] += GradR[e] * dRdq[e][m];

    // Normalization backward: q_norm = q_raw/||q_raw||
    // dL/dq_raw = (dL/dq_norm - q_norm*(q_norm . dL/dq_norm)) / ||q_raw||
    float dot = qw * dL_dqnorm[0] + qx * dL_dqnorm[1] + qy * dL_dqnorm[2] + qz * dL_dqnorm[3];
    float inv_len = 1.0f / r.qraw_len;
    float dL_dqraw_w = (dL_dqnorm[0] - qw * dot) * inv_len;
    float dL_dqraw_x = (dL_dqnorm[1] - qx * dot) * inv_len;
    float dL_dqraw_y = (dL_dqnorm[2] - qy * dot) * inv_len;
    float dL_dqraw_z = (dL_dqnorm[3] - qz * dot) * inv_len;

    // ---- Per-Gaussian regularizer terms (CLAUDE.md's composite loss) ----
    // These don't depend on any voxel at all, just this Gaussian's own
    // precomputed s_max/s_min/amplitude/raw_inten, so they're added directly
    // here rather than through the per-voxel grad_out loop above.
    float inv_n = 1.0f / (float)n;
    float* d_logs_arr = &d_logs.x; // float3 is layout-compatible with 3 contiguous floats

    // scale: lambda_scale * mean(s_max^2) -> d/d(log_scale[max_axis]) = lambda_scale*2*s_max^2/n
    d_logs_arr[r.max_axis] += LAMBDA_SCALE * 2.0f * r.s_max * r.s_max * inv_n;
    // ceiling: lambda_ceiling * mean(relu(s_max-cap)) -> d/d(log_scale[max_axis]) = lambda_ceiling*(s_max>cap)*s_max/n
    if (r.s_max > SCALE_CAP) d_logs_arr[r.max_axis] += LAMBDA_CEILING * r.s_max * inv_n;
    // outlier: lambda_outlier * mean(relu(s_max-median-3*mad)) -> same form, median/mad treated as
    // constants w.r.t. this Gaussian's own gradient (standard treatment of population statistics).
    if (r.s_max > outlier_median + 3.0f * outlier_mad) d_logs_arr[r.max_axis] += LAMBDA_OUTLIER * r.s_max * inv_n;
    // aniso: lambda_aniso * mean(s_min^2) -> d/d(log_scale[min_axis]) = lambda_aniso*2*s_min^2/n
    d_logs_arr[r.min_axis] += LAMBDA_ANISO * 2.0f * r.s_min * r.s_min * inv_n;
    // coverage: lambda_coverage * mean(-log(s_max/s_ref)) -> d(-log(s_max))/d(log_scale[max_axis]) = -1
    // (constant: log(s_max)==log_scale[max_axis] when not clamped by MIN/MAX_SCALE, which never
    // triggers in practice for our trained scale ranges)
    d_logs_arr[r.max_axis] += -LAMBDA_COVERAGE * inv_n;

    g.d_mean[k] = d_mean;
    g.d_log_scale[k] = d_logs;
    // float4 storage order matches Params.quat: (.x,.y,.z,.w) <-> (w,x,y,z)
    g.d_quat[k] = make_float4(dL_dqraw_w, dL_dqraw_x, dL_dqraw_y, dL_dqraw_z);

    float raw_inten = p.raw_inten[k];
    float sig = sigmoidf(raw_inten);
    float d_raw_inten = d_inten * sig; // chain rule from the per-voxel (L1+SSIM) driven dL/dv_k term
    // sparsity: lambda_sparsity * mean(v_k*(1-GT(mean_k))) -> d/d(raw_inten) = lambda_sparsity*(1-GT)*sigmoid/n
    // (detached w.r.t. mean_k: the point is to suppress opacity of Gaussians sitting in empty
    // target space, not to nudge their position, matching how such priors are typically applied)
    float gt_at_mean = sample_voxel_trilinear(target, r.mean);
    d_raw_inten += LAMBDA_SPARSITY * (1.0f - gt_at_mean) * sig * inv_n;
    // count: lambda_count * mean(sigmoid(raw_inten)) -> d/d(raw_inten) = lambda_count*sigmoid*(1-sigmoid)/n
    d_raw_inten += LAMBDA_COUNT * sig * (1.0f - sig) * inv_n;
    // L1-on-amplitude: lambda_L1 * mean(softplus(raw_inten)) -> d/d(raw_inten) = lambda_L1*sigmoid/n
    d_raw_inten += LAMBDA_L1REG * sig * inv_n;
    g.d_raw_inten[k] = d_raw_inten;
}

// ---------------------------------------------------------------------------
// Adam update kernel: one thread per Gaussian.
// ---------------------------------------------------------------------------
__device__ inline float adam_step(float param, float grad, float& m, float& v, float lr, float bc1, float bc2) {
    m = ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1 - ADAM_BETA2) * grad * grad;
    float mhat = m / bc1;
    float vhat = v / bc2;
    return param - lr * mhat / (sqrtf(vhat) + ADAM_EPS);
}

__global__ void adam_update_kernel(Params p, Grads g, AdamState a, int n, int t) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float bc1 = 1.0f - powf(ADAM_BETA1, (float)t);
    float bc2 = 1.0f - powf(ADAM_BETA2, (float)t);

    float3 mean = p.mean[k], gm = g.d_mean[k];
    float3 mm = a.m_mean[k], vm = a.v_mean[k];
    mean.x = adam_step(mean.x, gm.x, mm.x, vm.x, LR_MEAN, bc1, bc2);
    mean.y = adam_step(mean.y, gm.y, mm.y, vm.y, LR_MEAN, bc1, bc2);
    mean.z = adam_step(mean.z, gm.z, mm.z, vm.z, LR_MEAN, bc1, bc2);
    p.mean[k] = mean; a.m_mean[k] = mm; a.v_mean[k] = vm;

    float3 logs = p.log_scale[k], gl = g.d_log_scale[k];
    float3 ml = a.m_logs[k], vl = a.v_logs[k];
    logs.x = adam_step(logs.x, gl.x, ml.x, vl.x, LR_LOGSCALE, bc1, bc2);
    logs.y = adam_step(logs.y, gl.y, ml.y, vl.y, LR_LOGSCALE, bc1, bc2);
    logs.z = adam_step(logs.z, gl.z, ml.z, vl.z, LR_LOGSCALE, bc1, bc2);
    p.log_scale[k] = logs; a.m_logs[k] = ml; a.v_logs[k] = vl;

    float4 q = p.quat[k], gq = g.d_quat[k];
    float4 mq = a.m_quat[k], vq = a.v_quat[k];
    q.x = adam_step(q.x, gq.x, mq.x, vq.x, LR_QUAT, bc1, bc2);
    q.y = adam_step(q.y, gq.y, mq.y, vq.y, LR_QUAT, bc1, bc2);
    q.z = adam_step(q.z, gq.z, mq.z, vq.z, LR_QUAT, bc1, bc2);
    q.w = adam_step(q.w, gq.w, mq.w, vq.w, LR_QUAT, bc1, bc2);
    p.quat[k] = q; a.m_quat[k] = mq; a.v_quat[k] = vq;

    float inten = p.raw_inten[k], gi = g.d_raw_inten[k];
    float mi = a.m_inten[k], vi = a.v_inten[k];
    inten = adam_step(inten, gi, mi, vi, LR_INTEN, bc1, bc2);
    p.raw_inten[k] = inten; a.m_inten[k] = mi; a.v_inten[k] = vi;
}

// ---------------------------------------------------------------------------
// Finite-difference gradient self-test (host-orchestrated, device eval)
// ---------------------------------------------------------------------------
// Computes the FULL composite loss value (L1 base + SSIM-on-slice + all 7
// per-Gaussian regularizers), matching exactly what backward_kernel computes
// the analytic gradient of -- required for the finite-difference self-test to
// mean anything. z_slice is an explicit parameter (not re-randomized inside)
// so a +eps/-eps perturbation pair evaluates the SAME SSIM slice; the training
// loop picks a fresh random z_slice once per iteration and passes it in.
// out_median/out_mad are returned so the caller can feed the SAME values into
// backward_kernel instead of recomputing them.
double compute_full_loss(const Params& p, GaussianRuntime* rt_buf, int n, const float* target,
                          float* pred_buf, float* grad_buf, int z_slice,
                          float& out_median, float& out_mad) {
    int threads = 256;
    precompute_runtime_kernel<<<(n + threads - 1) / threads, threads>>>(p, rt_buf, n);
    CUDA_CHECK(cudaGetLastError());

    double* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
    forward_kernel<<<(VOXEL_COUNT + threads - 1) / threads, threads>>>(rt_buf, n, target, pred_buf, grad_buf, d_loss);
    CUDA_CHECK(cudaGetLastError());
    double h_l1_loss;
    CUDA_CHECK(cudaMemcpy(&h_l1_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
    h_l1_loss /= VOXEL_COUNT;
    CUDA_CHECK(cudaFree(d_loss));

    // SSIM-on-slice term
    double* d_stats;
    CUDA_CHECK(cudaMalloc(&d_stats, 5 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_stats, 0, 5 * sizeof(double)));
    dim3 sliceBlock(16, 16), sliceGrid((GRID + 15) / 16, (GRID + 15) / 16);
    ssim_slice_stats_kernel<<<sliceGrid, sliceBlock>>>(pred_buf, target, z_slice, d_stats);
    CUDA_CHECK(cudaGetLastError());
    double h_stats[5];
    CUDA_CHECK(cudaMemcpy(h_stats, d_stats, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_stats));
    double mu_x, mu_y, var_x, var_y, cov_xy;
    double ssim_val = ssim_from_stats(h_stats, mu_x, mu_y, var_x, var_y, cov_xy);
    ssim_slice_grad_kernel<<<sliceGrid, sliceBlock>>>(pred_buf, target, z_slice, grad_buf,
                                                       (float)mu_x, (float)mu_y, (float)var_x, (float)var_y, (float)cov_xy);
    CUDA_CHECK(cudaGetLastError());
    double ssim_loss = LAMBDA_SSIM * (1.0 - ssim_val);

    // Per-Gaussian regularizer terms (values computed host-side on tiny N=800 arrays)
    float *d_s_max, *d_s_min, *d_gt_at_mean;
    CUDA_CHECK(cudaMalloc(&d_s_max, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s_min, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gt_at_mean, n * sizeof(float)));
    extract_regularizer_inputs_kernel<<<(n + 255) / 256, 256>>>(rt_buf, target, n, d_s_max, d_s_min, d_gt_at_mean);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> h_s_max(n), h_s_min(n), h_gt_at_mean(n), h_raw_inten(n);
    CUDA_CHECK(cudaMemcpy(h_s_max.data(), d_s_max, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s_min.data(), d_s_min, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gt_at_mean.data(), d_gt_at_mean, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_raw_inten.data(), p.raw_inten, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_s_max)); CUDA_CHECK(cudaFree(d_s_min)); CUDA_CHECK(cudaFree(d_gt_at_mean));

    float median, mad;
    compute_median_mad(h_s_max.data(), n, median, mad);
    out_median = median; out_mad = mad;

    double scale_loss = 0, ceiling_loss = 0, outlier_loss = 0, sparsity_loss = 0;
    double aniso_loss = 0, count_loss = 0, l1reg_loss = 0, coverage_loss = 0;
    for (int i = 0; i < n; ++i) {
        float s_max = h_s_max[i], s_min = h_s_min[i];
        float v = softplus(h_raw_inten[i]);
        float sig = sigmoidf(h_raw_inten[i]);
        scale_loss   += s_max * s_max;
        ceiling_loss += fmaxf(0.0f, s_max - SCALE_CAP);
        outlier_loss += fmaxf(0.0f, s_max - median - 3.0f * mad);
        sparsity_loss += v * (1.0f - h_gt_at_mean[i]);
        aniso_loss   += s_min * s_min;
        count_loss   += sig;
        l1reg_loss   += v;
        coverage_loss += -logf(s_max / SCALE_REF);
    }
    double reg_total = LAMBDA_SCALE * (scale_loss / n) + LAMBDA_CEILING * (ceiling_loss / n) +
                        LAMBDA_OUTLIER * (outlier_loss / n) + LAMBDA_SPARSITY * (sparsity_loss / n) +
                        LAMBDA_ANISO * (aniso_loss / n) + LAMBDA_COUNT * (count_loss / n) +
                        LAMBDA_L1REG * (l1reg_loss / n) + LAMBDA_COVERAGE * (coverage_loss / n);

    return h_l1_loss + ssim_loss + reg_total;
}

// Generic scalar-field perturbation helper: writes `val` into the float at
// device address `field_ptr + tk*stride_floats + comp_offset`, recomputes the
// full loss, returns it. Kept generic so mean/log_scale/quat/raw_inten all
// share one central-difference path instead of four near-duplicate blocks.
double loss_with_perturbed_scalar(Params& p, GaussianRuntime* rt_buf, int n, const float* target,
                                   float* pred_buf, float* grad_buf, int z_slice,
                                   float* field_base, int tk, int stride_floats, int comp_offset,
                                   float value) {
    float* addr = field_base + tk * stride_floats + comp_offset;
    CUDA_CHECK(cudaMemcpy(addr, &value, sizeof(float), cudaMemcpyHostToDevice));
    float dummy_median, dummy_mad;
    return compute_full_loss(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, dummy_median, dummy_mad);
}

void check_one_param(Params& p, GaussianRuntime* rt_buf, int n, const float* target,
                      float* pred_buf, float* grad_buf, int z_slice,
                      float* field_base, int stride_floats, int comp_offset,
                      int tk, float analytic, const char* label, float eps) {
    float orig;
    CUDA_CHECK(cudaMemcpy(&orig, field_base + tk * stride_floats + comp_offset, sizeof(float), cudaMemcpyDeviceToHost));

    double lp = loss_with_perturbed_scalar(p, rt_buf, n, target, pred_buf, grad_buf, z_slice,
                                            field_base, tk, stride_floats, comp_offset, orig + eps);
    double lm = loss_with_perturbed_scalar(p, rt_buf, n, target, pred_buf, grad_buf, z_slice,
                                            field_base, tk, stride_floats, comp_offset, orig - eps);
    CUDA_CHECK(cudaMemcpy(field_base + tk * stride_floats + comp_offset, &orig, sizeof(float), cudaMemcpyHostToDevice));

    double numeric = (lp - lm) / (2.0 * eps);
    double rel_err = 100.0 * fabs((double)analytic - numeric) / (fabs(numeric) + 1e-9);
    printf("  gaussian %4d  %-12s analytic=% .6e  numeric=% .6e  rel_err=%.4f%%\n",
           tk, label, analytic, numeric, rel_err);
}

void finite_difference_check(Params& p, int n, const float* target, float* pred_buf, float* grad_buf) {
    printf("=== Finite-difference gradient self-test ===\n");
    GaussianRuntime* rt_buf;
    CUDA_CHECK(cudaMalloc(&rt_buf, n * sizeof(GaussianRuntime)));
    Grads g; alloc_grads(g, n);

    // Gaussian n/2 gets a non-identity quaternion AND anisotropic scale for
    // this test only, so the quaternion backward is actually exercised: at
    // the identity quaternion dR/dw is exactly zero (infinitesimal rotations
    // are generated by the vector part only), and rotating an ISOTROPIC
    // Gaussian never changes its density (all Gaussians start isotropic, see
    // init_params_host) -- either alone leaves the true gradient at zero.
    int qtest = n / 2;
    float4 test_quat = make_float4(0.8f, 0.3f, -0.4f, 0.2f); // arbitrary, non-identity, unnormalized on purpose
    float3 orig_logscale;
    CUDA_CHECK(cudaMemcpy(&orig_logscale, p.log_scale + qtest, sizeof(float3), cudaMemcpyDeviceToHost));
    float3 aniso_logscale = f3make(logf(0.10f), logf(0.18f), logf(0.28f));
    CUDA_CHECK(cudaMemcpy(p.quat + qtest, &test_quat, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.log_scale + qtest, &aniso_logscale, sizeof(float3), cudaMemcpyHostToDevice));

    // Fixed z-slice for the whole self-test (a +eps/-eps perturbation pair
    // must evaluate the SAME SSIM slice, or the "numeric" gradient would mix
    // in slice-choice noise unrelated to the perturbation).
    const int z_slice = GRID / 2;

    // One forward+backward pass to get analytic gradients for all Gaussians.
    float median, mad;
    compute_full_loss(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, median, mad);
    precompute_runtime_kernel<<<(n + 255) / 256, 256>>>(p, rt_buf, n);
    CUDA_CHECK(cudaGetLastError());
    backward_kernel<<<(n + 255) / 256, 256>>>(p, rt_buf, n, grad_buf, target, median, mad, g);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float EPS = 2e-3f;
    int test_gaussians[3] = {0, qtest, n - 1};

    for (int tk : test_gaussians) {
        float3 gm, gl; float4 gq; float gi;
        CUDA_CHECK(cudaMemcpy(&gm, g.d_mean + tk, sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gl, g.d_log_scale + tk, sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gq, g.d_quat + tk, sizeof(float4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gi, g.d_raw_inten + tk, sizeof(float), cudaMemcpyDeviceToHost));

        check_one_param(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, (float*)p.mean, 3, 0, tk, gm.x, "mean.x", EPS);
        check_one_param(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, (float*)p.log_scale, 3, 0, tk, gl.x, "log_scale.x", EPS);
        check_one_param(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, (float*)p.quat, 4, 0, tk, gq.x, "quat.w", EPS);
        check_one_param(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, (float*)p.quat, 4, 1, tk, gq.y, "quat.x", EPS);
        check_one_param(p, rt_buf, n, target, pred_buf, grad_buf, z_slice, p.raw_inten, 1, 0, tk, gi, "raw_inten", EPS);
    }

    // Restore identity quaternion + original isotropic scale at qtest so
    // training starts from the intended initialization.
    float4 identity_quat = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    CUDA_CHECK(cudaMemcpy(p.quat + qtest, &identity_quat, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.log_scale + qtest, &orig_logscale, sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFree(rt_buf));
    CUDA_CHECK(cudaFree(g.d_mean)); CUDA_CHECK(cudaFree(g.d_log_scale));
    CUDA_CHECK(cudaFree(g.d_quat)); CUDA_CHECK(cudaFree(g.d_raw_inten));
    printf("=== End self-test ===\n\n");
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
struct Camera {
    float3 pos, forward, right, up;
    float tan_half_fov;
};

Camera make_camera(float yaw_deg) {
    float yaw = yaw_deg * (float)M_PI / 180.0f;
    Camera cam;
    cam.pos = f3make(0, 0, 0); // volume centre
    cam.forward = f3norm(f3make(sinf(yaw), 0.0f, cosf(yaw)));
    float3 world_up = f3make(0, 1, 0);
    cam.right = f3norm(f3cross(cam.forward, world_up));
    cam.up = f3norm(f3cross(cam.right, cam.forward));
    cam.tan_half_fov = tanf(0.5f * FOV_DEG * (float)M_PI / 180.0f);
    return cam;
}

__host__ __device__ inline float3 camera_ray_dir(const Camera& cam, int px, int py, int W, int H) {
    float u = (2.0f * ((px + 0.5f) / W) - 1.0f) * cam.tan_half_fov;
    float v = (1.0f - 2.0f * ((py + 0.5f) / H)) * cam.tan_half_fov;
    float3 dir = f3add(cam.forward, f3add(f3scale(cam.right, u), f3scale(cam.up, v)));
    return f3norm(dir);
}

// Ray-box intersection against [BOX_MIN,BOX_MAX]^3, returns false if no hit.
__host__ __device__ inline bool ray_box(float3 origin, float3 dir, float& t0, float& t1) {
    t0 = 0.0f; t1 = 1e9f;
    for (int a = 0; a < 3; ++a) {
        float od = (a == 0) ? origin.x : (a == 1 ? origin.y : origin.z);
        float dd = (a == 0) ? dir.x : (a == 1 ? dir.y : dir.z);
        if (fabsf(dd) < 1e-9f) {
            if (od < BOX_MIN || od > BOX_MAX) return false;
        } else {
            float ta = (BOX_MIN - od) / dd;
            float tb = (BOX_MAX - od) / dd;
            if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
            t0 = fmaxf(t0, ta);
            t1 = fminf(t1, tb);
        }
    }
    return t0 <= t1;
}

// ---------------------------------------------------------------------------
// DVR baseline: ray-march the target VOXEL GRID (trilinear interpolation),
// MIP compositing. This is the "vanilla DVR on voxel grid" baseline.
// (sample_voxel_trilinear itself now lives up near voxel_center, since the
// training loss's sparsity regularizer needs it too, before this point in the file)
// ---------------------------------------------------------------------------
__global__ void dvr_render_kernel(const float* grid, Camera cam, float* image, int W, int H) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    float3 dir = camera_ray_dir(cam, px, py, W, H);
    float t0, t1;
    float mip = 0.0f;
    if (ray_box(cam.pos, dir, t0, t1)) {
        float dt = (t1 - t0) / DVR_SAMPLES;
        for (int s = 0; s < DVR_SAMPLES; ++s) {
            float t = t0 + (s + 0.5f) * dt;
            float3 pos = f3add(cam.pos, f3scale(dir, t));
            float v = sample_voxel_trilinear(grid, pos);
            mip = fmaxf(mip, v);
        }
    }
    image[py * W + px] = fminf(fmaxf(mip, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// Gaussian rasterizer: camera-front-only, closed-form ray-interval solve +
// depth-binned accumulation (validated design: max_t[sum_k f_k] via binning,
// not max_t per Gaussian, which would silently compute the wrong quantity
// under overlap).
// ---------------------------------------------------------------------------
__device__ inline bool solve_gaussian_ray_interval(float3 origin, float3 dir, float3 mean, const float* R, float3 inv_s2,
                                                    float& t_lo, float& t_hi) {
    float3 oc = f3sub(origin, mean);
    float3 u0 = world_to_local(R, oc);
    float3 ud = world_to_local(R, dir);
    // Q(t) = a t^2 + b t + c <= CUTOFF
    float a = ud.x * ud.x * inv_s2.x + ud.y * ud.y * inv_s2.y + ud.z * ud.z * inv_s2.z;
    float b = 2.0f * (u0.x * ud.x * inv_s2.x + u0.y * ud.y * inv_s2.y + u0.z * ud.z * inv_s2.z);
    float c = u0.x * u0.x * inv_s2.x + u0.y * u0.y * inv_s2.y + u0.z * u0.z * inv_s2.z - MAHALANOBIS_CUTOFF;
    if (a < 1e-12f) return false;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0f) return false;
    float sq = sqrtf(disc);
    t_lo = (-b - sq) / (2 * a);
    t_hi = (-b + sq) / (2 * a);
    return true;
}

// Screen-space bounding box for one Gaussian (shared by both the legacy and
// tile-based rasterizers). Returns false if the Gaussian is entirely behind
// the camera. full_screen is set when the camera sits inside/near the
// Gaussian's support sphere, where the small-angle bbox approximation below
// divides by a near-zero/negative depth and breaks down.
__device__ inline bool gaussian_screen_bbox(const GaussianRuntime& r, Camera cam, int W, int H,
                                             int& x_lo, int& x_hi, int& y_lo, int& y_hi, bool& full_screen) {
    float3 to_center = f3sub(r.mean, cam.pos);
    float depth = f3dot(to_center, cam.forward);
    if (depth + r.support_radius <= NEAR_PLANE) return false;

    if (depth <= r.support_radius) {
        full_screen = true;
        x_lo = 0; x_hi = W - 1; y_lo = 0; y_hi = H - 1;
        return true;
    }
    full_screen = false;
    float dist = depth;
    float ang_radius = r.support_radius / dist; // small-angle approx, matches project convention
    float u_center = f3dot(to_center, cam.right) / dist;
    float v_center = f3dot(to_center, cam.up) / dist;

    float u_lo_ndc = (u_center - ang_radius) / cam.tan_half_fov;
    float u_hi_ndc = (u_center + ang_radius) / cam.tan_half_fov;
    float v_lo_ndc = (v_center - ang_radius) / cam.tan_half_fov;
    float v_hi_ndc = (v_center + ang_radius) / cam.tan_half_fov;

    x_lo = max(0, (int)floorf((u_lo_ndc + 1.0f) * 0.5f * W));
    x_hi = min(W - 1, (int)ceilf((u_hi_ndc + 1.0f) * 0.5f * W));
    y_lo = max(0, (int)floorf((1.0f - v_hi_ndc) * 0.5f * H));
    y_hi = min(H - 1, (int)ceilf((1.0f - v_lo_ndc) * 0.5f * H));
    return x_lo <= x_hi && y_lo <= y_hi;
}

// ---------------------------------------------------------------------------
// LEGACY rasterizer: one thread per Gaussian, scatters via atomicAdd into a
// global per-pixel-per-bin buffer. Only ~N_GAUSSIANS threads total, so most
// of the GPU sits idle regardless of screen size -- kept only as a
// correctness oracle to validate the tile-based rasterizer below against
// (same math, reorganized for parallelism), not used for the timed render.
// ---------------------------------------------------------------------------
__global__ void rasterize_gaussians_kernel_legacy(const GaussianRuntime* rt, int n, Camera cam,
                                                   float* bin_accum, int W, int H, int bins,
                                                   float t_far) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    GaussianRuntime r = rt[k];

    int x_lo, x_hi, y_lo, y_hi; bool full_screen;
    if (!gaussian_screen_bbox(r, cam, W, H, x_lo, x_hi, y_lo, y_hi, full_screen)) return;

    for (int py = y_lo; py <= y_hi; ++py) {
        for (int px = x_lo; px <= x_hi; ++px) {
            float3 dir = camera_ray_dir(cam, px, py, W, H);
            float t_lo, t_hi;
            if (!solve_gaussian_ray_interval(cam.pos, dir, r.mean, r.R, r.inv_s2, t_lo, t_hi)) continue;
            t_lo = fmaxf(t_lo, 0.0f);
            t_hi = fminf(t_hi, t_far);
            if (t_lo >= t_hi) continue;

            int bin_lo = max(0, (int)floorf(t_lo / t_far * bins));
            int bin_hi = min(bins - 1, (int)floorf(t_hi / t_far * bins));
            for (int bin = bin_lo; bin <= bin_hi; ++bin) {
                float t_mid = t_far * (bin + 0.5f) / bins;
                float3 pos = f3add(cam.pos, f3scale(dir, t_mid));
                float dens = eval_gaussian_density(r.mean, r.R, r.inv_s2, r.amplitude, pos);
                if (dens > 0.0f) {
                    int flat = (py * W + px) * bins + bin;
                    atomicAdd(&bin_accum[flat], dens);
                }
            }
        }
    }
}

__global__ void reduce_bins_kernel(const float* bin_accum, float* image, int W, int H, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    float m = 0.0f;
    for (int b = 0; b < bins; ++b) m = fmaxf(m, bin_accum[idx * bins + b]);
    image[idx] = fminf(fmaxf(m, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// TILE-BASED rasterizer (default): restores per-PIXEL parallelism (like the
// DVR kernel) instead of per-Gaussian parallelism.
//
// Pass 1 (build_tile_lists_kernel): one thread per Gaussian, same screen-bbox
// logic as the legacy path, but instead of looping over pixels itself, it
// just appends its own index into every tile's Gaussian list it overlaps
// (cheap: iterates over tiles, not pixels).
//
// Pass 2 (render_tiles_kernel): one thread BLOCK per tile, one thread per
// PIXEL within the tile (so thread count scales with screen resolution, same
// as DVR). Each block cooperatively streams its tile's Gaussian list through
// shared memory in batches; every thread then evaluates its own pixel against
// the batch and accumulates into a PRIVATE per-thread bin array (registers)
// -- since each thread owns its pixel exclusively, no atomics are needed at
// all. Same sum-within-bin-then-max-across-bins math as the legacy path.
// ---------------------------------------------------------------------------
struct RenderGaussian {
    float3 mean;
    float R[9];
    float3 inv_s2;
    float amplitude;
};

__global__ void build_tile_lists_kernel(const GaussianRuntime* rt, int n, Camera cam,
                                         int* tile_counts, int* tile_gaussian_list,
                                         int W, int H, int tiles_x, int tiles_y) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    GaussianRuntime r = rt[k];

    int x_lo, x_hi, y_lo, y_hi; bool full_screen;
    if (!gaussian_screen_bbox(r, cam, W, H, x_lo, x_hi, y_lo, y_hi, full_screen)) return;

    int tx_lo, tx_hi, ty_lo, ty_hi;
    if (full_screen) {
        tx_lo = 0; tx_hi = tiles_x - 1; ty_lo = 0; ty_hi = tiles_y - 1;
    } else {
        tx_lo = x_lo / TILE_SIZE; tx_hi = x_hi / TILE_SIZE;
        ty_lo = y_lo / TILE_SIZE; ty_hi = y_hi / TILE_SIZE;
    }

    // Capacity per tile is N_GAUSSIANS (== n), so every slot write below is
    // guaranteed in-bounds: a Gaussian can appear in a given tile's list at
    // most once, and there are only n Gaussians total.
    for (int ty = ty_lo; ty <= ty_hi; ++ty) {
        for (int tx = tx_lo; tx <= tx_hi; ++tx) {
            int tile_id = ty * tiles_x + tx;
            int slot = atomicAdd(&tile_counts[tile_id], 1);
            tile_gaussian_list[tile_id * n + slot] = k;
        }
    }
}

__global__ void render_tiles_kernel(const GaussianRuntime* rt, Camera cam,
                                     const int* tile_counts, const int* tile_gaussian_list,
                                     float* image, int W, int H, int tiles_x, int n,
                                     float t_far, int bins) {
    __shared__ RenderGaussian batch[TILE_SIZE * TILE_SIZE];

    int tile_x = blockIdx.x, tile_y = blockIdx.y;
    int tile_id = tile_y * tiles_x + tile_x;
    int count = tile_counts[tile_id];

    int local_x = threadIdx.x, local_y = threadIdx.y;
    int px = tile_x * TILE_SIZE + local_x;
    int py = tile_y * TILE_SIZE + local_y;
    bool active = (px < W && py < H);

    float bin_vals[RENDER_BINS];
    #pragma unroll
    for (int b = 0; b < RENDER_BINS; ++b) bin_vals[b] = 0.0f;

    float3 dir = make_float3(0, 0, 0);
    if (active) dir = camera_ray_dir(cam, px, py, W, H);

    int tid = local_y * TILE_SIZE + local_x;
    const int batch_size = TILE_SIZE * TILE_SIZE;

    for (int base = 0; base < count; base += batch_size) {
        int idx_in_tile = base + tid;
        if (idx_in_tile < count) {
            int gid = tile_gaussian_list[tile_id * n + idx_in_tile];
            GaussianRuntime g = rt[gid];
            batch[tid].mean = g.mean;
            #pragma unroll
            for (int i = 0; i < 9; ++i) batch[tid].R[i] = g.R[i];
            batch[tid].inv_s2 = g.inv_s2;
            batch[tid].amplitude = g.amplitude;
        }
        __syncthreads();

        int this_batch_count = min(batch_size, count - base);
        if (active) {
            for (int j = 0; j < this_batch_count; ++j) {
                float t_lo, t_hi;
                if (!solve_gaussian_ray_interval(cam.pos, dir, batch[j].mean, batch[j].R, batch[j].inv_s2, t_lo, t_hi)) continue;
                t_lo = fmaxf(t_lo, 0.0f);
                t_hi = fminf(t_hi, t_far);
                if (t_lo >= t_hi) continue;

                int bin_lo = max(0, (int)floorf(t_lo / t_far * bins));
                int bin_hi = min(bins - 1, (int)floorf(t_hi / t_far * bins));
                for (int bin = bin_lo; bin <= bin_hi; ++bin) {
                    float t_mid = t_far * (bin + 0.5f) / bins;
                    float3 pos = f3add(cam.pos, f3scale(dir, t_mid));
                    float dens = eval_gaussian_density(batch[j].mean, batch[j].R, batch[j].inv_s2, batch[j].amplitude, pos);
                    bin_vals[bin] += dens; // sum-within-bin, same as legacy path's atomicAdd (no race: this thread owns bin_vals exclusively)
                }
            }
        }
        __syncthreads(); // all threads must finish reading `batch` before the next iteration overwrites it
    }

    if (active) {
        float m = 0.0f;
        #pragma unroll
        for (int b = 0; b < RENDER_BINS; ++b) m = fmaxf(m, bin_vals[b]);
        image[py * W + px] = fminf(fmaxf(m, 0.0f), 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Simple raw-binary frame writer: [int32 W][int32 H][float32 * W*H]
// ---------------------------------------------------------------------------
void write_frame(const char* path, const float* h_data, int W, int H) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); exit(1); }
    int32_t w32 = W, h32 = H;
    fwrite(&w32, sizeof(int32_t), 1, f);
    fwrite(&h32, sizeof(int32_t), 1, f);
    fwrite(h_data, sizeof(float), W * H, f);
    fclose(f);
}

// Raw-binary volume writer: [int32 GRID][float32 * GRID^3]. Used for the
// GT/baked volumes so the Python side can compute vol_psnr/vol_ssim/vol_lpips
// (the latter two need skimage/lpips, easier to do there than in CUDA).
void write_volume(const char* path, const float* h_data, int grid) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); exit(1); }
    int32_t g32 = grid;
    fwrite(&g32, sizeof(int32_t), 1, f);
    fwrite(h_data, sizeof(float), (size_t)grid * grid * grid, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Bake: evaluate the TRAINED Gaussian mixture once at every voxel centre of a
// fresh dense GRID^3 array (same density formula forward_kernel uses, just
// without any loss/gradient bookkeeping). This is the "compression at rest,
// fast rendering at display time" step: store only the compact Gaussian
// parameters, then reconstruct a dense grid once and feed it to the existing,
// already-fast DVR kernel -- decoupling storage cost from render cost.
// ---------------------------------------------------------------------------
__global__ void bake_kernel(const GaussianRuntime* rt, int n, float* baked) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= VOXEL_COUNT) return;
    int ix = idx % GRID;
    int iy = (idx / GRID) % GRID;
    int iz = idx / (GRID * GRID);
    float3 x = voxel_center(ix, iy, iz);
    float p = 0.0f;
    for (int k = 0; k < n; ++k) {
        float3 d = f3sub(x, rt[k].mean);
        if (f3dot(d, d) > rt[k].support_radius * rt[k].support_radius) continue;
        p += eval_gaussian_density(rt[k].mean, rt[k].R, rt[k].inv_s2, rt[k].amplitude, x);
    }
    baked[idx] = fminf(fmaxf(p, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// Trained-parameter checkpoint: raw binary, [mean|log_scale|quat|raw_inten]
// arrays back-to-back, N_GAUSSIANS fixed by this build. Lets a re-render at a
// different screen size skip the ~150s training run entirely.
// ---------------------------------------------------------------------------
void save_checkpoint(const char* path, const Params& p, int n) {
    float3* h_mean = new float3[n]; float3* h_logs = new float3[n];
    float4* h_quat = new float4[n]; float* h_int = new float[n];
    CUDA_CHECK(cudaMemcpy(h_mean, p.mean, n * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logs, p.log_scale, n * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_quat, p.quat, n * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_int, p.raw_inten, n * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); exit(1); }
    int32_t n32 = n;
    fwrite(&n32, sizeof(int32_t), 1, f);
    fwrite(h_mean, sizeof(float3), n, f);
    fwrite(h_logs, sizeof(float3), n, f);
    fwrite(h_quat, sizeof(float4), n, f);
    fwrite(h_int, sizeof(float), n, f);
    fclose(f);
    delete[] h_mean; delete[] h_logs; delete[] h_quat; delete[] h_int;
    printf("Saved checkpoint: %s\n", path);
}

bool load_checkpoint(const char* path, Params& p, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int32_t n32 = 0;
    fread(&n32, sizeof(int32_t), 1, f);
    if (n32 != n) { fprintf(stderr, "Checkpoint Gaussian count mismatch (%d vs %d), ignoring.\n", n32, n); fclose(f); return false; }
    float3* h_mean = new float3[n]; float3* h_logs = new float3[n];
    float4* h_quat = new float4[n]; float* h_int = new float[n];
    fread(h_mean, sizeof(float3), n, f);
    fread(h_logs, sizeof(float3), n, f);
    fread(h_quat, sizeof(float4), n, f);
    fread(h_int, sizeof(float), n, f);
    fclose(f);
    CUDA_CHECK(cudaMemcpy(p.mean, h_mean, n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.log_scale, h_logs, n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.quat, h_quat, n * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p.raw_inten, h_int, n * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_mean; delete[] h_logs; delete[] h_quat; delete[] h_int;
    printf("Loaded checkpoint: %s (skipping training)\n", path);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string out_dir = "fafb_pilot/code/renderer/scratch_gs/frames";
    if (argc > 1) out_dir = argv[1];
    int IMAGE_W = 256, IMAGE_H = 256;
    if (argc > 2) IMAGE_W = atoi(argv[2]);
    if (argc > 3) IMAGE_H = atoi(argv[3]);
    std::string checkpoint_path = (argc > 4) ? argv[4] : (out_dir + "/../checkpoint.bin");
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    system(mkdir_cmd.c_str());

    // ---- 1. Synthetic target voxel grid ----
    Blob h_blobs[N_BLOBS];
    make_blobs(h_blobs);
    CUDA_CHECK(cudaMemcpyToSymbol(d_blobs, h_blobs, N_BLOBS * sizeof(Blob)));

    float* d_target;
    CUDA_CHECK(cudaMalloc(&d_target, VOXEL_COUNT * sizeof(float)));
    generate_target_kernel<<<(VOXEL_COUNT + 255) / 256, 256>>>(d_target);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Synthetic %d^3 target voxel grid generated (%d blobs).\n", GRID, N_BLOBS);

    // ---- 2. Trainable Gaussian mixture ----
    Params params; alloc_params(params, N_GAUSSIANS);
    Grads grads; alloc_grads(grads, N_GAUSSIANS);
    AdamState adam; alloc_adam(adam, N_GAUSSIANS);
    init_params_host(params, N_GAUSSIANS);

    GaussianRuntime* d_rt;
    CUDA_CHECK(cudaMalloc(&d_rt, N_GAUSSIANS * sizeof(GaussianRuntime)));
    float* d_pred; CUDA_CHECK(cudaMalloc(&d_pred, VOXEL_COUNT * sizeof(float)));
    float* d_gradout; CUDA_CHECK(cudaMalloc(&d_gradout, VOXEL_COUNT * sizeof(float)));

    int threadsG = 128, blocksG = (N_GAUSSIANS + threadsG - 1) / threadsG;

    if (load_checkpoint(checkpoint_path.c_str(), params, N_GAUSSIANS)) {
        // Re-rendering an already-trained model (e.g. at a new screen size):
        // training + its self-test are skipped entirely.
    } else {
        // ---- 3. Finite-difference self-test BEFORE trusting training ----
        finite_difference_check(params, N_GAUSSIANS, d_target, d_pred, d_gradout);

        // ---- 4. Training loop ----
        // compute_full_loss bundles precompute + forward (L1 grad) + SSIM-slice
        // stats/gradient + regularizer value computation (incl. median/MAD for
        // the outlier term) in one call, matching exactly what backward_kernel
        // differentiates -- so the printed loss is the true composite loss, and
        // median/mad come back ready to feed straight into backward_kernel.
        printf("=== Training: %d Gaussians, %d iterations, composite loss (L1+SSIM+7 regularizers) ===\n",
               N_GAUSSIANS, TRAIN_ITERS);
        LCG slice_rng(987654321ULL);
        auto train_start = std::chrono::high_resolution_clock::now();
        for (int iter = 1; iter <= TRAIN_ITERS; ++iter) {
            int z_slice = (int)(slice_rng.next01() * GRID);
            if (z_slice >= GRID) z_slice = GRID - 1;

            float median, mad;
            double loss = compute_full_loss(params, d_rt, N_GAUSSIANS, d_target, d_pred, d_gradout, z_slice, median, mad);
            backward_kernel<<<blocksG, threadsG>>>(params, d_rt, N_GAUSSIANS, d_gradout, d_target, median, mad, grads);
            adam_update_kernel<<<blocksG, threadsG>>>(params, grads, adam, N_GAUSSIANS, iter);

            if (iter % 200 == 0 || iter == 1) {
                CUDA_CHECK(cudaDeviceSynchronize());
                printf("  iter %4d  loss=%.6f\n", iter, loss);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto train_end = std::chrono::high_resolution_clock::now();
        double train_secs = std::chrono::duration<double>(train_end - train_start).count();
        printf("Training done in %.1f s (%.2f iters/s)\n\n", train_secs, TRAIN_ITERS / train_secs);
        save_checkpoint(checkpoint_path.c_str(), params, N_GAUSSIANS);
    }

    precompute_runtime_kernel<<<blocksG, threadsG>>>(params, d_rt, N_GAUSSIANS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Bake: evaluate the trained Gaussians once onto a dense GRID^3 grid.
    // Compression story: only the Gaussian parameters need to be stored/
    // transmitted; baking happens once at display time, then rendering is
    // pure DVR on the baked grid (same kernel as the GT baseline) -- decouples
    // storage cost from render cost, and (since it evaluates exactly at the
    // training voxel centres) gets the same much-higher fidelity the training
    // loss itself achieves, unlike continuous/off-grid rasterization. ----
    float* d_baked; CUDA_CHECK(cudaMalloc(&d_baked, VOXEL_COUNT * sizeof(float)));
    bake_kernel<<<(VOXEL_COUNT + 255) / 256, 256>>>(d_rt, N_GAUSSIANS, d_baked);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        std::vector<float> h_gt_vol(VOXEL_COUNT), h_baked_vol(VOXEL_COUNT);
        CUDA_CHECK(cudaMemcpy(h_gt_vol.data(), d_target, VOXEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_baked_vol.data(), d_baked, VOXEL_COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        std::string vol_dir = out_dir + "/..";
        write_volume((vol_dir + "/volume_gt.bin").c_str(), h_gt_vol.data(), GRID);
        write_volume((vol_dir + "/volume_baked.bin").c_str(), h_baked_vol.data(), GRID);
        printf("Baked volume written (%s/volume_gt.bin, %s/volume_baked.bin)\n", vol_dir.c_str(), vol_dir.c_str());
    }

    // ---- 5. Render sweep: GT (DVR), Baked (DVR on reconstructed grid), and
    // live Gaussian rasterization, all at the same screen size ----
    printf("Rendering at %dx%d\n", IMAGE_W, IMAGE_H);
    float* d_image_gt; CUDA_CHECK(cudaMalloc(&d_image_gt, IMAGE_W * IMAGE_H * sizeof(float)));
    float* d_image_baked; CUDA_CHECK(cudaMalloc(&d_image_baked, IMAGE_W * IMAGE_H * sizeof(float)));
    float* d_image_rec; CUDA_CHECK(cudaMalloc(&d_image_rec, IMAGE_W * IMAGE_H * sizeof(float)));
    float* h_image = new float[IMAGE_W * IMAGE_H];

    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((IMAGE_W + 15) / 16, (IMAGE_H + 15) / 16);
    float t_far = sqrtf(3.0f) * (BOX_MAX - BOX_MIN); // box diagonal, generous far plane

    int tiles_x = (IMAGE_W + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (IMAGE_H + TILE_SIZE - 1) / TILE_SIZE;
    int n_tiles = tiles_x * tiles_y;
    int* d_tile_counts; CUDA_CHECK(cudaMalloc(&d_tile_counts, n_tiles * sizeof(int)));
    int* d_tile_gaussian_list; CUDA_CHECK(cudaMalloc(&d_tile_gaussian_list, (size_t)n_tiles * N_GAUSSIANS * sizeof(int)));
    dim3 tileBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 tileGridDim(tiles_x, tiles_y);

    // ---- One-time correctness check: tile-based rasterizer vs the legacy
    // one-thread-per-Gaussian rasterizer, both computing the exact same
    // sum-within-bin-then-max-across-bins math, just reorganized for
    // parallelism -- so they should agree to float-rounding precision. ----
    {
        Camera cam0 = make_camera(0.0f);
        float* d_bin_accum; CUDA_CHECK(cudaMalloc(&d_bin_accum, (size_t)IMAGE_W * IMAGE_H * RENDER_BINS * sizeof(float)));
        int threadsPix = 256, blocksPix = (IMAGE_W * IMAGE_H + threadsPix - 1) / threadsPix;
        CUDA_CHECK(cudaMemset(d_bin_accum, 0, (size_t)IMAGE_W * IMAGE_H * RENDER_BINS * sizeof(float)));
        rasterize_gaussians_kernel_legacy<<<blocksG, threadsG>>>(d_rt, N_GAUSSIANS, cam0, d_bin_accum,
                                                                  IMAGE_W, IMAGE_H, RENDER_BINS, t_far);
        CUDA_CHECK(cudaGetLastError());
        float* d_image_legacy; CUDA_CHECK(cudaMalloc(&d_image_legacy, IMAGE_W * IMAGE_H * sizeof(float)));
        reduce_bins_kernel<<<blocksPix, threadsPix>>>(d_bin_accum, d_image_legacy, IMAGE_W, IMAGE_H, RENDER_BINS);

        CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
        build_tile_lists_kernel<<<blocksG, threadsG>>>(d_rt, N_GAUSSIANS, cam0, d_tile_counts, d_tile_gaussian_list,
                                                        IMAGE_W, IMAGE_H, tiles_x, tiles_y);
        CUDA_CHECK(cudaGetLastError());
        render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(d_rt, cam0, d_tile_counts, d_tile_gaussian_list,
                                                            d_image_rec, IMAGE_W, IMAGE_H, tiles_x, N_GAUSSIANS, t_far, RENDER_BINS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_legacy = new float[IMAGE_W * IMAGE_H];
        float* h_tiled = new float[IMAGE_W * IMAGE_H];
        CUDA_CHECK(cudaMemcpy(h_legacy, d_image_legacy, IMAGE_W * IMAGE_H * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tiled, d_image_rec, IMAGE_W * IMAGE_H * sizeof(float), cudaMemcpyDeviceToHost));
        double max_diff = 0.0, sum_diff = 0.0;
        for (int i = 0; i < IMAGE_W * IMAGE_H; ++i) {
            double d = fabs((double)h_legacy[i] - (double)h_tiled[i]);
            max_diff = fmax(max_diff, d);
            sum_diff += d;
        }
        printf("=== Tile-based rasterizer validation vs legacy (frame yaw=0) ===\n");
        printf("  max_abs_diff=%.6e  mean_abs_diff=%.6e\n", max_diff, sum_diff / (IMAGE_W * IMAGE_H));
        if (max_diff > 1e-2) {
            fprintf(stderr, "WARNING: tile-based rasterizer disagrees with legacy oracle by more than expected float-rounding noise!\n");
        }
        delete[] h_legacy; delete[] h_tiled;
        CUDA_CHECK(cudaFree(d_bin_accum));
        CUDA_CHECK(cudaFree(d_image_legacy));
    }

    // ========================================================================
    // 6. STABLE GPU BENCHMARK
    // ========================================================================
    // Why benchmark separately from frame export?
    //
    // Frame export performs device-to-host copies and disk writes. Those are useful
    // application costs, but they are not rasterization costs. Mixing them with the
    // renderer timing produces misleading FPS values.
    //
    // CUDA events are recorded on the GPU timeline. We time many renders inside one
    // event interval so event resolution, clock ramp-up and one-off launch effects do
    // not dominate very small kernels.
    constexpr int BENCH_WARMUP_ROUNDS = 5;
    constexpr int BENCH_MEASURE_ROUNDS = 20;
    const int benchmark_frames = N_FRAMES * BENCH_MEASURE_ROUNDS;

    cudaEvent_t bench_start, bench_stop;
    CUDA_CHECK(cudaEventCreate(&bench_start));
    CUDA_CHECK(cudaEventCreate(&bench_stop));

    auto benchmark_dvr = [&](const float* volume) -> double {
        // Warm-up: exercise all camera directions before measuring.
        for (int round = 0; round < BENCH_WARMUP_ROUNDS; ++round) {
            for (int frame = 0; frame < N_FRAMES; ++frame) {
                Camera cam = make_camera(360.0f * frame / N_FRAMES);
                dvr_render_kernel<<<gridDim2D, blockDim2D>>>(
                    volume, cam, d_image_gt, IMAGE_W, IMAGE_H);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(bench_start));
        for (int round = 0; round < BENCH_MEASURE_ROUNDS; ++round) {
            for (int frame = 0; frame < N_FRAMES; ++frame) {
                Camera cam = make_camera(360.0f * frame / N_FRAMES);
                dvr_render_kernel<<<gridDim2D, blockDim2D>>>(
                    volume, cam, d_image_gt, IMAGE_W, IMAGE_H);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(bench_stop));
        CUDA_CHECK(cudaEventSynchronize(bench_stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, bench_start, bench_stop));
        return benchmark_frames * 1000.0 / static_cast<double>(total_ms);
    };

    auto benchmark_gaussian = [&]() -> double {
        // This is an end-to-end dynamic-camera benchmark. Every render includes:
        //   1) clearing tile counts,
        //   2) projecting Gaussians and rebuilding tile lists,
        //   3) rendering all pixels from those lists.
        // This is the fair cost when the camera changes every frame.
        for (int round = 0; round < BENCH_WARMUP_ROUNDS; ++round) {
            for (int frame = 0; frame < N_FRAMES; ++frame) {
                Camera cam = make_camera(360.0f * frame / N_FRAMES);
                CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
                build_tile_lists_kernel<<<blocksG, threadsG>>>(
                    d_rt, N_GAUSSIANS, cam, d_tile_counts,
                    d_tile_gaussian_list, IMAGE_W, IMAGE_H, tiles_x, tiles_y);
                render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(
                    d_rt, cam, d_tile_counts, d_tile_gaussian_list,
                    d_image_rec, IMAGE_W, IMAGE_H, tiles_x,
                    N_GAUSSIANS, t_far, RENDER_BINS);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(bench_start));
        for (int round = 0; round < BENCH_MEASURE_ROUNDS; ++round) {
            for (int frame = 0; frame < N_FRAMES; ++frame) {
                Camera cam = make_camera(360.0f * frame / N_FRAMES);
                CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
                build_tile_lists_kernel<<<blocksG, threadsG>>>(
                    d_rt, N_GAUSSIANS, cam, d_tile_counts,
                    d_tile_gaussian_list, IMAGE_W, IMAGE_H, tiles_x, tiles_y);
                render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(
                    d_rt, cam, d_tile_counts, d_tile_gaussian_list,
                    d_image_rec, IMAGE_W, IMAGE_H, tiles_x,
                    N_GAUSSIANS, t_far, RENDER_BINS);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(bench_stop));
        CUDA_CHECK(cudaEventSynchronize(bench_stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, bench_start, bench_stop));
        return benchmark_frames * 1000.0 / static_cast<double>(total_ms);
    };

    const double dvr_gpu_fps = benchmark_dvr(d_target);
    const double baked_gpu_fps = benchmark_dvr(d_baked);
    const double rast_gpu_fps = benchmark_gaussian();

    CUDA_CHECK(cudaEventDestroy(bench_start));
    CUDA_CHECK(cudaEventDestroy(bench_stop));

    printf("\n=== Stable GPU-only FPS (CUDA events) ===\n");
    printf("  Warm-up rounds:                       %d\n", BENCH_WARMUP_ROUNDS);
    printf("  Measured renders per method:          %d\n", benchmark_frames);
    printf("  GT DVR (voxel grid):                  %.2f FPS\n", dvr_gpu_fps);
    printf("  Baked + DVR (reconstructed grid):     %.2f FPS\n", baked_gpu_fps);
    printf("  Gaussian rasterizer (end-to-end):     %.2f FPS\n", rast_gpu_fps);

    // ========================================================================
    // 7. EXPORT ONE CAMERA SWEEP
    // ========================================================================
    // These renders are intentionally NOT timed. Their purpose is to produce
    // paired frames for PSNR, SSIM, LPIPS, videos and visual inspection.
    printf("\n=== Exporting %d paired frames at %dx%d ===\n",
           N_FRAMES, IMAGE_W, IMAGE_H);

    for (int frame = 0; frame < N_FRAMES; ++frame) {
        const float yaw = 360.0f * frame / N_FRAMES;
        Camera cam = make_camera(yaw);
        char path[512];

        // A. Ground-truth voxel grid rendered by fixed-step MIP DVR.
        dvr_render_kernel<<<gridDim2D, blockDim2D>>>(
            d_target, cam, d_image_gt, IMAGE_W, IMAGE_H);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_image, d_image_gt,
                              IMAGE_W * IMAGE_H * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(path, sizeof(path), "%s/gt_%04d.bin", out_dir.c_str(), frame);
        write_frame(path, h_image, IMAGE_W, IMAGE_H);

        // B. Gaussian mixture first baked to a voxel grid, then rendered by the
        // same DVR kernel. This isolates representation/baking error because the
        // renderer is identical to the GT renderer.
        dvr_render_kernel<<<gridDim2D, blockDim2D>>>(
            d_baked, cam, d_image_baked, IMAGE_W, IMAGE_H);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_image, d_image_baked,
                              IMAGE_W * IMAGE_H * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(path, sizeof(path), "%s/baked_%04d.bin", out_dir.c_str(), frame);
        write_frame(path, h_image, IMAGE_W, IMAGE_H);

        // C. Live Gaussian rasterization. Tile lists are rebuilt because the
        // camera changes with yaw.
        CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
        build_tile_lists_kernel<<<blocksG, threadsG>>>(
            d_rt, N_GAUSSIANS, cam, d_tile_counts,
            d_tile_gaussian_list, IMAGE_W, IMAGE_H, tiles_x, tiles_y);
        CUDA_CHECK(cudaGetLastError());
        render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(
            d_rt, cam, d_tile_counts, d_tile_gaussian_list,
            d_image_rec, IMAGE_W, IMAGE_H, tiles_x,
            N_GAUSSIANS, t_far, RENDER_BINS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_image, d_image_rec,
                              IMAGE_W * IMAGE_H * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(path, sizeof(path), "%s/rec_%04d.bin", out_dir.c_str(), frame);
        write_frame(path, h_image, IMAGE_W, IMAGE_H);

        if (frame % 10 == 0 || frame == N_FRAMES - 1) {
            printf("  exported frame %2d/%d, yaw=%6.1f degrees\n",
                   frame + 1, N_FRAMES, yaw);
        }
    }

    // Machine-readable benchmark metadata consumed by the Python script.
    {
        std::string fps_path = out_dir + "/fps_summary.txt";
        FILE* f = fopen(fps_path.c_str(), "w");
        if (!f) {
            fprintf(stderr, "Could not open %s for writing.\n", fps_path.c_str());
            return 1;
        }
        fprintf(f, "timing_kind gpu_events_repeated\n");
        fprintf(f, "dvr_gpu_fps %.6f\n", dvr_gpu_fps);
        fprintf(f, "baked_gpu_fps %.6f\n", baked_gpu_fps);
        fprintf(f, "rasterizer_gpu_fps %.6f\n", rast_gpu_fps);
        // Backward-compatible aliases. They intentionally equal GPU FPS now.
        fprintf(f, "dvr_fps %.6f\n", dvr_gpu_fps);
        fprintf(f, "baked_fps %.6f\n", baked_gpu_fps);
        fprintf(f, "rasterizer_fps %.6f\n", rast_gpu_fps);
        fprintf(f, "benchmark_warmup_rounds %d\n", BENCH_WARMUP_ROUNDS);
        fprintf(f, "benchmark_measure_rounds %d\n", BENCH_MEASURE_ROUNDS);
        fprintf(f, "benchmark_render_count %d\n", benchmark_frames);
        fprintf(f, "n_gaussians %d\n", N_GAUSSIANS);
        fprintf(f, "n_frames %d\n", N_FRAMES);
        fprintf(f, "image_w %d\n", IMAGE_W);
        fprintf(f, "image_h %d\n", IMAGE_H);
        fprintf(f, "dvr_samples %d\n", DVR_SAMPLES);
        fprintf(f, "render_bins %d\n", RENDER_BINS);
        fclose(f);
    }

    printf("\nAll frames written to %s\n", out_dir.c_str());
    printf("Run render_outputs_corrected.py to assemble videos, plots, and metric tables.\n");

    delete[] h_image;
    return 0;
}
