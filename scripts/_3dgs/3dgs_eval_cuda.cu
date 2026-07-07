/*
 * 3dgs_eval_cuda.cu — fused inference kernels for Gaussian-cloud evaluation.
 *
 * Exposed functions (via pybind11 / torch.utils.cpp_extension):
 *
 *   reconstruct_volume(means, log_s, quats, inten,
 *                      lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
 *                      D, H, W, scale_min, mahal_clamp)
 *     -> float32 cuda tensor, shape (D*H*W,), values in [0, 1]
 *
 *   splat_mip(means, log_s, quats, inten,
 *             lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
 *             out_h, out_w, depth_samples, view_axis,
 *             scale_min, mahal_clamp)
 *     -> float32 cuda tensor, shape (out_h * out_w,), values in [0, 1]
 *     view_axis: 0 = xy (looking down Z, out_h=H, out_w=W)
 *                1 = xz (looking down Y, out_h=D, out_w=W)
 *                2 = yz (looking down X, out_h=D, out_w=H)
 *
 * Coordinate convention (matches VolumeDataset._indices_to_pts):
 *   flat index  n = iz*(H*W) + ih*W + iw
 *   x = lo_x + (iw / (W-1)) * (hi_x - lo_x)   pts[:,0]
 *   y = lo_y + (ih / (H-1)) * (hi_y - lo_y)   pts[:,1]
 *   z = lo_z + (iz / (D-1)) * (hi_z - lo_z)   pts[:,2]
 *
 * Math (matches GaussianCloud.forward / build_sigma_inv):
 *   s    = exp(log_s).clamp(min=scale_min)
 *   R    = quat_to_rotmat(q)
 *   RD   = R * diag(1/s)          column-wise
 *   SI   = RD @ RD^T              inverse covariance
 *   mah2 = d @ SI @ d             d = pt - mean
 *   out += softplus(inten) * exp(-0.5 * mah2)   if mah2 < mahal_clamp  (else skip)
 */

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ── tuning constants ──────────────────────────────────────────────────────────
// BLOCK / TILE: reconstruct_volume kernel.
// Shared-mem per block = TILE * (3+3+4+1) * 4  =  TILE * 44 bytes.
// With TILE=256: 11 264 B — well within the 48 KB limit.
#define BLOCK     256
#define TILE      256
// BLOCK_MIP / TILE_MIP: splat_mip kernel.
// Shared-mem per tile-block = TILE_GAUSS * (2+5+1) floats.
// TILE_GAUSS=128 keeps the batch small while limiting host-side bucketing cost.
#define TILE_PIX_W 16
#define TILE_PIX_H 16
#define TILE_GAUSS 128

static inline float softplus_stable(float x) {
    return (x > 20.f) ? x : log1pf(expf(x));
}

static inline void quat_to_rotmat_host(
        float qw, float qx, float qy, float qz,
        float& R00, float& R01, float& R02,
        float& R10, float& R11, float& R12,
        float& R20, float& R21, float& R22)
{
    const float n = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    const float s = (n > 0.f) ? (1.f / n) : 1.f;
    qw *= s;  qx *= s;  qy *= s;  qz *= s;

    R00 = 1.f - 2.f*(qy*qy + qz*qz);
    R01 =       2.f*(qx*qy - qw*qz);
    R02 =       2.f*(qx*qz + qw*qy);
    R10 =       2.f*(qx*qy + qw*qz);
    R11 = 1.f - 2.f*(qx*qx + qz*qz);
    R12 =       2.f*(qy*qz - qw*qx);
    R20 =       2.f*(qx*qz - qw*qy);
    R21 =       2.f*(qy*qz + qw*qx);
    R22 = 1.f - 2.f*(qx*qx + qy*qy);
}

static inline void projected_axes_for_view_axis(
        int view_axis, int& u_axis, int& v_axis,
        float& u_lo, float& u_hi, float& v_lo, float& v_hi,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z)
{
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

// ── tiled splat_mip kernel ────────────────────────────────────────────────────
// One CUDA block handles one 16x16 image tile.
// Each tile only iterates over the Gaussians that overlap its projected footprint.
// This is a true tile-bucketed rasterizer: no per-pixel scan over all N Gaussians.
__global__ void splat_mip_tiled_kernel(
        const float* __restrict__ proj_u,
        const float* __restrict__ proj_v,
        const float* __restrict__ proj_inv00,
        const float* __restrict__ proj_inv01,
        const float* __restrict__ proj_inv11,
        const float* __restrict__ inten,
        const int* __restrict__ tile_offsets,
        const int* __restrict__ tile_indices,
        int tiles_x,
        int out_h, int out_w,
        float mahal_clamp,
        float* __restrict__ out)
{
    __shared__ float s_u[TILE_GAUSS];
    __shared__ float s_v[TILE_GAUSS];
    __shared__ float s_i[TILE_GAUSS];
    __shared__ float s_a[TILE_GAUSS];
    __shared__ float s_b[TILE_GAUSS];
    __shared__ float s_c[TILE_GAUSS];

    const int tile_id = blockIdx.x;
    const int tile_y  = tile_id / tiles_x;
    const int tile_x  = tile_id % tiles_x;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int ir = tile_y * TILE_PIX_H + ty;
    const int ic = tile_x * TILE_PIX_W + tx;
    const bool active = (ir < out_h && ic < out_w);

    const int lane = ty * blockDim.x + tx;
    const int start = tile_offsets[tile_id];
    const int end   = tile_offsets[tile_id + 1];

    float max_val = 0.f;

    for (int t0 = start; t0 < end; t0 += TILE_GAUSS) {
        const int tn = min(TILE_GAUSS, end - t0);

        for (int i = lane; i < tn; i += blockDim.x * blockDim.y) {
            const int g = tile_indices[t0 + i];
            s_u[i] = proj_u[g];
            s_v[i] = proj_v[g];
            s_a[i] = proj_inv00[g];
            s_b[i] = proj_inv01[g];
            s_c[i] = proj_inv11[g];
            s_i[i] = inten[g];
        }
        __syncthreads();

        if (active) {
            const float px = static_cast<float>(ic);
            const float py = static_cast<float>(ir);
            for (int i = 0; i < tn; ++i) {
                const float du = px - s_u[i];
                const float dv = py - s_v[i];
                const float mah = du * (s_a[i] * du + s_b[i] * dv)
                                + dv * (s_b[i] * du + s_c[i] * dv);
                if (mah >= mahal_clamp) continue;
                const float val = s_i[i] * __expf(-0.5f * mah);
                max_val = fmaxf(max_val, val);
            }
        }
        __syncthreads();
    }

    if (active) {
        out[ir * out_w + ic] = fminf(fmaxf(max_val, 0.f), 1.f);
    }
}

// ── kernel ────────────────────────────────────────────────────────────────────
__global__ void eval_volume_kernel(
        const float* __restrict__ means,   // (N, 3)  Gaussian centres
        const float* __restrict__ log_s,   // (N, 3)  log per-axis scales
        const float* __restrict__ quats,   // (N, 4)  [w, x, y, z]
        const float* __restrict__ inten,   // (N,)    raw (pre-softplus) intensity
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z,
        int D, int H, int W, int N,
        float scale_min, float mahal_clamp,
        float* __restrict__ out)           // (D*H*W,)
{
    // shared-memory tile for one batch of TILE Gaussians
    __shared__ float s_mu[TILE][3];
    __shared__ float s_ls[TILE][3];
    __shared__ float s_qu[TILE][4];
    __shared__ float s_iv[TILE];       // post-softplus intensity

    const int n   = blockIdx.x * BLOCK + threadIdx.x;
    const int DHW = D * H * W;

    // ── compute this thread's 3-D query coordinate ────────────────────────────
    float cx = 0.f, cy = 0.f, cz = 0.f;
    bool active = (n < DHW);
    if (active) {
        int iz = n / (H * W);
        int ih = (n % (H * W)) / W;
        int iw = n % W;
        // matches VolumeDataset._indices_to_pts: x←iw/W, y←ih/H, z←iz/D
        cx = lo_x + (W > 1 ? (float)iw / (float)(W - 1) : 0.5f) * (hi_x - lo_x);
        cy = lo_y + (H > 1 ? (float)ih / (float)(H - 1) : 0.5f) * (hi_y - lo_y);
        cz = lo_z + (D > 1 ? (float)iz / (float)(D - 1) : 0.5f) * (hi_z - lo_z);
    }

    float acc = 0.f;

    // ── tile loop over all N Gaussians ────────────────────────────────────────
    for (int t0 = 0; t0 < N; t0 += TILE) {
        const int tn = min(TILE, N - t0);   // Gaussians in this tile

        // cooperatively load tile from global → shared memory
        for (int i = threadIdx.x; i < tn; i += BLOCK) {
            const int g = t0 + i;
            s_mu[i][0] = means[g * 3 + 0];
            s_mu[i][1] = means[g * 3 + 1];
            s_mu[i][2] = means[g * 3 + 2];
            s_ls[i][0] = log_s[g * 3 + 0];
            s_ls[i][1] = log_s[g * 3 + 1];
            s_ls[i][2] = log_s[g * 3 + 2];
            s_qu[i][0] = quats[g * 4 + 0];   // w
            s_qu[i][1] = quats[g * 4 + 1];   // x
            s_qu[i][2] = quats[g * 4 + 2];   // y
            s_qu[i][3] = quats[g * 4 + 3];   // z
            // softplus(r) = log(1+exp(r)); numerically stable variant
            float r = inten[g];
            s_iv[i] = (r > 20.f) ? r : log1pf(__expf(r));
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tn; i++) {

                // ── quaternion → rotation matrix ─────────────────────────────
                float qw = s_qu[i][0], qx = s_qu[i][1];
                float qy = s_qu[i][2], qz = s_qu[i][3];
                float ni = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
                qw *= ni;  qx *= ni;  qy *= ni;  qz *= ni;

                float R00 = 1.f - 2.f*(qy*qy + qz*qz);
                float R01 =       2.f*(qx*qy - qw*qz);
                float R02 =       2.f*(qx*qz + qw*qy);
                float R10 =       2.f*(qx*qy + qw*qz);
                float R11 = 1.f - 2.f*(qx*qx + qz*qz);
                float R12 =       2.f*(qy*qz - qw*qx);
                float R20 =       2.f*(qx*qz - qw*qy);
                float R21 =       2.f*(qy*qz + qw*qx);
                float R22 = 1.f - 2.f*(qx*qx + qy*qy);

                // ── per-axis scales ───────────────────────────────────────────
                float sx = fmaxf(__expf(s_ls[i][0]), scale_min);
                float sy = fmaxf(__expf(s_ls[i][1]), scale_min);
                float sz = fmaxf(__expf(s_ls[i][2]), scale_min);
                float isx = 1.f / sx,  isy = 1.f / sy,  isz = 1.f / sz;

                // ── RD = R * diag(1/s)  (scale each column) ──────────────────
                float D00 = R00*isx,  D01 = R01*isy,  D02 = R02*isz;
                float D10 = R10*isx,  D11 = R11*isy,  D12 = R12*isz;
                float D20 = R20*isx,  D21 = R21*isy,  D22 = R22*isz;

                // ── SI = RD @ RD^T  (6 unique elements of symmetric 3×3) ─────
                float S00 = D00*D00 + D01*D01 + D02*D02;
                float S01 = D00*D10 + D01*D11 + D02*D12;
                float S02 = D00*D20 + D01*D21 + D02*D22;
                float S11 = D10*D10 + D11*D11 + D12*D12;
                float S12 = D10*D20 + D11*D21 + D12*D22;
                float S22 = D20*D20 + D21*D21 + D22*D22;

                // ── Mahalanobis²  = d @ SI @ d ────────────────────────────────
                float d0 = cx - s_mu[i][0];
                float d1 = cy - s_mu[i][1];
                float d2 = cz - s_mu[i][2];
                float mah = d0*(S00*d0 + S01*d1 + S02*d2)
                          + d1*(S01*d0 + S11*d1 + S12*d2)
                          + d2*(S02*d0 + S12*d1 + S22*d2);
                if (mah >= mahal_clamp) continue;  // match training kernel: skip, not clamp

                acc += s_iv[i] * __expf(-0.5f * mah);
            }
        }
        __syncthreads();
    }

    if (active)
        out[n] = fminf(fmaxf(acc, 0.f), 1.f);
}

// ── C++ / Python binding ──────────────────────────────────────────────────────
torch::Tensor reconstruct_volume(
        torch::Tensor means,      // (N, 3) float32 cuda contiguous
        torch::Tensor log_s,      // (N, 3)
        torch::Tensor quats,      // (N, 4)
        torch::Tensor inten,      // (N,)
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z,
        int D, int H, int W,
        float scale_min, float mahal_clamp)
{
    TORCH_CHECK(means.is_cuda(),        "means must be a CUDA tensor");
    TORCH_CHECK(means.is_contiguous(),  "means must be contiguous");
    TORCH_CHECK(log_s.is_contiguous(),  "log_s must be contiguous");
    TORCH_CHECK(quats.is_contiguous(),  "quats must be contiguous");
    TORCH_CHECK(inten.is_contiguous(),  "inten must be contiguous");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");

    const int N   = static_cast<int>(means.size(0));
    const int DHW = D * H * W;
    auto out = torch::empty({DHW}, means.options());

    const int blocks = (DHW + BLOCK - 1) / BLOCK;
    eval_volume_kernel<<<blocks, BLOCK>>>(
        means.data_ptr<float>(),
        log_s.data_ptr<float>(),
        quats.data_ptr<float>(),
        inten.data_ptr<float>(),
        lo_x, hi_x,
        lo_y, hi_y,
        lo_z, hi_z,
        D, H, W, N,
        scale_min, mahal_clamp,
        out.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return out;    // (DHW,) float32, values in [0,1]
}

// ── C++ binding: splat_mip ────────────────────────────────────────────────────
torch::Tensor splat_mip(
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor inten,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z,
        int out_h, int out_w, int depth_samples,
        int view_axis,
        float scale_min, float mahal_clamp)
{
    TORCH_CHECK(means.is_cuda(),        "means must be a CUDA tensor");
    TORCH_CHECK(means.is_contiguous(),  "means must be contiguous");
    TORCH_CHECK(log_s.is_contiguous(),  "log_s must be contiguous");
    TORCH_CHECK(quats.is_contiguous(),  "quats must be contiguous");
    TORCH_CHECK(inten.is_contiguous(),  "inten must be contiguous");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(view_axis >= 0 && view_axis <= 2, "view_axis must be 0, 1, or 2");

    (void)depth_samples;  // retained for API compatibility; the tiled rasterizer does not depth-sample

    const int N = static_cast<int>(means.size(0));
    auto out = torch::zeros({out_h * out_w}, means.options());

    int u_axis = 0;
    int v_axis = 1;
    float u_lo = 0.f, u_hi = 1.f, v_lo = 0.f, v_hi = 1.f;
    projected_axes_for_view_axis(view_axis, u_axis, v_axis,
                                 u_lo, u_hi, v_lo, v_hi,
                                 lo_x, hi_x, lo_y, hi_y, lo_z, hi_z);

    const float u_scale = (out_w > 1) ? (static_cast<float>(out_w - 1) / (u_hi - u_lo)) : 1.f;
    const float v_scale = (out_h > 1) ? (static_cast<float>(out_h - 1) / (v_hi - v_lo)) : 1.f;
    const int tiles_x = (out_w + TILE_PIX_W - 1) / TILE_PIX_W;
    const int tiles_y = (out_h + TILE_PIX_H - 1) / TILE_PIX_H;
    const int num_tiles = tiles_x * tiles_y;

    auto means_cpu = means.contiguous().to(torch::kCPU);
    auto log_s_cpu = log_s.contiguous().to(torch::kCPU);
    auto quats_cpu = quats.contiguous().to(torch::kCPU);
    auto inten_cpu = inten.contiguous().to(torch::kCPU);

    const float* m_ptr = means_cpu.data_ptr<float>();
    const float* ls_ptr = log_s_cpu.data_ptr<float>();
    const float* q_ptr = quats_cpu.data_ptr<float>();
    const float* i_ptr = inten_cpu.data_ptr<float>();

    std::vector<float> proj_u(N), proj_v(N), proj_inv00(N), proj_inv01(N), proj_inv11(N), proj_iv(N);
    std::vector<std::vector<int>> tile_lists(num_tiles);

    const float radius_scale = std::sqrt(std::max(mahal_clamp, 0.f));
    for (int g = 0; g < N; ++g) {
        const float qw0 = q_ptr[g * 4 + 0];
        const float qx0 = q_ptr[g * 4 + 1];
        const float qy0 = q_ptr[g * 4 + 2];
        const float qz0 = q_ptr[g * 4 + 3];

        float R00, R01, R02, R10, R11, R12, R20, R21, R22;
        quat_to_rotmat_host(qw0, qx0, qy0, qz0,
                            R00, R01, R02,
                            R10, R11, R12,
                            R20, R21, R22);

        const float sx = fmaxf(expf(ls_ptr[g * 3 + 0]), scale_min);
        const float sy = fmaxf(expf(ls_ptr[g * 3 + 1]), scale_min);
        const float sz = fmaxf(expf(ls_ptr[g * 3 + 2]), scale_min);
        const float sx2 = sx * sx;
        const float sy2 = sy * sy;
        const float sz2 = sz * sz;

        const float C00 = R00*R00*sx2 + R01*R01*sy2 + R02*R02*sz2;
        const float C01 = R00*R10*sx2 + R01*R11*sy2 + R02*R12*sz2;
        const float C02 = R00*R20*sx2 + R01*R21*sy2 + R02*R22*sz2;
        const float C11 = R10*R10*sx2 + R11*R11*sy2 + R12*R12*sz2;
        const float C12 = R10*R20*sx2 + R11*R21*sy2 + R12*R22*sz2;
        const float C22 = R20*R20*sx2 + R21*R21*sy2 + R22*R22*sz2;

        float mu_u = 0.f, mu_v = 0.f;
        float cov_uu = 0.f, cov_uv = 0.f, cov_vv = 0.f;
        if (view_axis == 0) {
            mu_u = (m_ptr[g * 3 + 0] - u_lo) * u_scale;
            mu_v = (m_ptr[g * 3 + 1] - v_lo) * v_scale;
            cov_uu = C00 * u_scale * u_scale;
            cov_uv = C01 * u_scale * v_scale;
            cov_vv = C11 * v_scale * v_scale;
        } else if (view_axis == 1) {
            mu_u = (m_ptr[g * 3 + 0] - u_lo) * u_scale;
            mu_v = (m_ptr[g * 3 + 2] - v_lo) * v_scale;
            cov_uu = C00 * u_scale * u_scale;
            cov_uv = C02 * u_scale * v_scale;
            cov_vv = C22 * v_scale * v_scale;
        } else {
            mu_u = (m_ptr[g * 3 + 1] - u_lo) * u_scale;
            mu_v = (m_ptr[g * 3 + 2] - v_lo) * v_scale;
            cov_uu = C11 * u_scale * u_scale;
            cov_uv = C12 * u_scale * v_scale;
            cov_vv = C22 * v_scale * v_scale;
        }

        float det = cov_uu * cov_vv - cov_uv * cov_uv;
        if (det < 1.0e-12f) det = 1.0e-12f;
        const float inv00 = cov_vv / det;
        const float inv01 = -cov_uv / det;
        const float inv11 = cov_uu / det;

        const float tr = cov_uu + cov_vv;
        const float disc = std::sqrt(std::max(0.f, (cov_uu - cov_vv) * (cov_uu - cov_vv) + 4.f * cov_uv * cov_uv));
        const float lambda_max = 0.5f * (tr + disc);
        const float radius = radius_scale * std::sqrt(std::max(lambda_max, 1.0e-12f));

        int u0 = static_cast<int>(std::floor(mu_u - radius));
        int u1 = static_cast<int>(std::ceil(mu_u + radius));
        int v0 = static_cast<int>(std::floor(mu_v - radius));
        int v1 = static_cast<int>(std::ceil(mu_v + radius));
        if (u1 < 0 || v1 < 0 || u0 >= out_w || v0 >= out_h) {
            continue;
        }
        u0 = std::max(0, u0);
        u1 = std::min(out_w - 1, u1);
        v0 = std::max(0, v0);
        v1 = std::min(out_h - 1, v1);

        const int tx0 = u0 / TILE_PIX_W;
        const int tx1 = u1 / TILE_PIX_W;
        const int ty0 = v0 / TILE_PIX_H;
        const int ty1 = v1 / TILE_PIX_H;
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                tile_lists[ty * tiles_x + tx].push_back(g);
            }
        }

        proj_u[g] = mu_u;
        proj_v[g] = mu_v;
        proj_inv00[g] = inv00;
        proj_inv01[g] = inv01;
        proj_inv11[g] = inv11;
        proj_iv[g] = softplus_stable(i_ptr[g]);
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

    auto float_cpu_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto int_cpu_opts   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    auto t_proj_u = torch::empty({N}, float_cpu_opts);
    auto t_proj_v = torch::empty({N}, float_cpu_opts);
    auto t_inv00  = torch::empty({N}, float_cpu_opts);
    auto t_inv01  = torch::empty({N}, float_cpu_opts);
    auto t_inv11  = torch::empty({N}, float_cpu_opts);
    auto t_iv     = torch::empty({N}, float_cpu_opts);
    std::memcpy(t_proj_u.data_ptr<float>(), proj_u.data(), sizeof(float) * N);
    std::memcpy(t_proj_v.data_ptr<float>(), proj_v.data(), sizeof(float) * N);
    std::memcpy(t_inv00.data_ptr<float>(), proj_inv00.data(), sizeof(float) * N);
    std::memcpy(t_inv01.data_ptr<float>(), proj_inv01.data(), sizeof(float) * N);
    std::memcpy(t_inv11.data_ptr<float>(), proj_inv11.data(), sizeof(float) * N);
    std::memcpy(t_iv.data_ptr<float>(), proj_iv.data(), sizeof(float) * N);

    auto t_offsets = torch::empty({num_tiles + 1}, int_cpu_opts);
    auto t_indices = torch::empty({total}, int_cpu_opts);
    std::memcpy(t_offsets.data_ptr<int>(), tile_offsets.data(), sizeof(int) * (num_tiles + 1));
    if (total > 0) {
        std::memcpy(t_indices.data_ptr<int>(), tile_indices.data(), sizeof(int) * total);
    }

    const auto dev = means.device();
    auto proj_u_d = t_proj_u.to(dev);
    auto proj_v_d = t_proj_v.to(dev);
    auto inv00_d  = t_inv00.to(dev);
    auto inv01_d  = t_inv01.to(dev);
    auto inv11_d  = t_inv11.to(dev);
    auto iv_d     = t_iv.to(dev);
    auto offsets_d = t_offsets.to(dev);
    auto indices_d = t_indices.to(dev);

    const dim3 block(TILE_PIX_W, TILE_PIX_H);
    const dim3 grid(num_tiles);
    splat_mip_tiled_kernel<<<grid, block>>>(
        proj_u_d.data_ptr<float>(),
        proj_v_d.data_ptr<float>(),
        inv00_d.data_ptr<float>(),
        inv01_d.data_ptr<float>(),
        inv11_d.data_ptr<float>(),
        iv_d.data_ptr<float>(),
        offsets_d.data_ptr<int>(),
        indices_d.data_ptr<int>(),
        tiles_x,
        out_h, out_w,
        mahal_clamp,
        out.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reconstruct_volume", &reconstruct_volume,
          "Evaluate Gaussian cloud at every voxel centre. "
          "Returns (D*H*W,) float32 CUDA tensor in [0,1].");
    m.def("splat_mip", &splat_mip,
          "MIP splatting: max-intensity projection along a depth axis. "
          "view_axis: 0=xy(down-Z), 1=xz(down-Y), 2=yz(down-X). "
          "Returns (out_h*out_w,) float32 CUDA tensor in [0,1].");
}
