// ============================================================================
// stitch_experiment.cu
//
// Multi-block stitching comparison, built on top of gaussian_splat_scratch.cu's
// already-trained single-block model (checkpoint.bin, 800 Gaussians fit to a
// 64^3 synthetic scene). No training happens here -- this program only loads
// that one trained block and asks: when several COPIES of it are tiled into a
// larger combined scene (2, 4, or 8 blocks, arranged 2x1x1 / 2x2x1 / 2x2x2),
// which is the better way to render the result?
//
//   BAKE-THEN-STITCH: bake each block INDEPENDENTLY onto its own local 64^3
//   grid (exactly the single-block bake_kernel, unchanged -- hard-gated, no
//   cross-block blending, matching this project's established stitching
//   convention), place the N independent 64^3 grids into the correct
//   sub-region of one combined grid (fixed at 128^3 so 2x2x2=8 blocks always
//   fit; unfilled slots for the 2- and 4-block cases are left at zero), then
//   render the combined grid with the same DVR/MIP kernel used everywhere
//   else in this project.
//
//   GAUSSIAN-STITCH: take all N blocks' Gaussians, shift each block's copy's
//   means to its world position, concatenate into one array of N*800
//   Gaussians, and render that directly with the tile-based rasterizer --
//   no baking, no intermediate grid at all.
//
// Both paths share every piece of Gaussian-density and rasterization math
// with gaussian_splat_scratch.cu verbatim; only the two grid-dependent
// pieces (baking, DVR sampling) need a "local" (per-block) and a "combined"
// (stitched) version.
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
// Configuration
// ---------------------------------------------------------------------------
constexpr int N_GAUSSIANS_PER_BLOCK = 800;
constexpr float MAHALANOBIS_CUTOFF = 20.0f;
constexpr float MIN_SCALE = 1.0e-3f;
constexpr float MAX_SCALE = 1.2f;
constexpr int TILE_SIZE = 16;
constexpr int RENDER_BINS = 48;
constexpr int DVR_SAMPLES = 128;
constexpr float FOV_DEG = 90.0f;
constexpr float NEAR_PLANE = 1.0e-3f;
constexpr int N_FRAMES = 60;

// Per-block LOCAL grid: identical to the single-block convention in
// gaussian_splat_scratch.cu (64^3 voxels spanning [-1,1]^3).
constexpr int LOCAL_GRID = 64;
constexpr float LOCAL_BOX_MIN = -1.0f, LOCAL_BOX_MAX = 1.0f;

// COMBINED (stitched) grid: fixed at 128^3 spanning [-2,2]^3 -- large enough
// for the biggest arrangement tested (2x2x2 = 8 blocks, each 2 units wide),
// same voxel density as the local grid (32 voxels/unit on both). Arrangements
// using fewer blocks (2 or 4) simply leave the unused sub-region empty.
constexpr int COMBINED_GRID = 128;
constexpr float COMBINED_BOX_MIN = -2.0f, COMBINED_BOX_MAX = 2.0f;

// ---------------------------------------------------------------------------
// Vector helpers (identical to gaussian_splat_scratch.cu)
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
__host__ __device__ inline float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__host__ __device__ inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// Local (per-block) voxel centre + trilinear sampling for the LOCAL grid
// ---------------------------------------------------------------------------
__host__ __device__ inline float3 local_voxel_center(int ix, int iy, int iz) {
    float sx = (LOCAL_BOX_MAX - LOCAL_BOX_MIN) / LOCAL_GRID;
    return f3make(LOCAL_BOX_MIN + (ix + 0.5f) * sx, LOCAL_BOX_MIN + (iy + 0.5f) * sx, LOCAL_BOX_MIN + (iz + 0.5f) * sx);
}

// Trilinear sample of the COMBINED (stitched) grid.
__host__ __device__ inline float sample_combined_trilinear(const float* grid, float3 pos) {
    float gx = (pos.x - COMBINED_BOX_MIN) / (COMBINED_BOX_MAX - COMBINED_BOX_MIN) * COMBINED_GRID - 0.5f;
    float gy = (pos.y - COMBINED_BOX_MIN) / (COMBINED_BOX_MAX - COMBINED_BOX_MIN) * COMBINED_GRID - 0.5f;
    float gz = (pos.z - COMBINED_BOX_MIN) / (COMBINED_BOX_MAX - COMBINED_BOX_MIN) * COMBINED_GRID - 0.5f;
    int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
    float fx = gx - x0, fy = gy - y0, fz = gz - z0;
    float acc = 0.0f;
    for (int dz = 0; dz <= 1; ++dz)
        for (int dy = 0; dy <= 1; ++dy)
            for (int dx = 0; dx <= 1; ++dx) {
                int xi = x0 + dx, yi = y0 + dy, zi = z0 + dz;
                if (xi < 0 || xi >= COMBINED_GRID || yi < 0 || yi >= COMBINED_GRID || zi < 0 || zi >= COMBINED_GRID) continue;
                float wx = dx ? fx : (1 - fx);
                float wy = dy ? fy : (1 - fy);
                float wz = dz ? fz : (1 - fz);
                acc += wx * wy * wz * grid[xi + yi * COMBINED_GRID + zi * COMBINED_GRID * COMBINED_GRID];
            }
    return acc;
}

// ---------------------------------------------------------------------------
// Trainable Gaussian parameters (SoA) -- read-only here, loaded from a
// checkpoint written by gaussian_splat_scratch.cu.
// ---------------------------------------------------------------------------
struct Params {
    float3* mean;
    float3* log_scale;
    float4* quat;
    float* raw_inten;
};
void alloc_params(Params& p, int n) {
    CUDA_CHECK(cudaMalloc(&p.mean, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&p.log_scale, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&p.quat, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&p.raw_inten, n * sizeof(float)));
}

bool load_checkpoint_host(const char* path, std::vector<float3>& h_mean, std::vector<float3>& h_logs,
                           std::vector<float4>& h_quat, std::vector<float>& h_int, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int32_t n32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1 || n32 != n) { fclose(f); return false; }
    h_mean.resize(n); h_logs.resize(n); h_quat.resize(n); h_int.resize(n);
    fread(h_mean.data(), sizeof(float3), n, f);
    fread(h_logs.data(), sizeof(float3), n, f);
    fread(h_quat.data(), sizeof(float4), n, f);
    fread(h_int.data(), sizeof(float), n, f);
    fclose(f);
    return true;
}

struct GaussianRuntime {
    float3 mean;
    float R[9];
    float3 inv_s2;
    float3 s2;
    float amplitude;
    float support_radius;
};

__host__ __device__ inline void quat_to_R(float w, float x, float y, float z, float* R) {
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w);     R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);     R[7] = 2 * (y * z + x * w);     R[8] = 1 - 2 * (x * x + y * y);
}

__global__ void precompute_runtime_kernel(const Params p, GaussianRuntime* rt, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float4 q = p.quat[k];
    float qlen = sqrtf(fmaxf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w, 1e-20f));
    float w = q.x / qlen, x = q.y / qlen, y = q.z / qlen, z = q.w / qlen;

    GaussianRuntime r;
    r.mean = p.mean[k];
    quat_to_R(w, x, y, z, r.R);

    float3 ls = p.log_scale[k];
    float sx = fminf(fmaxf(expf(ls.x), MIN_SCALE), MAX_SCALE);
    float sy = fminf(fmaxf(expf(ls.y), MIN_SCALE), MAX_SCALE);
    float sz = fminf(fmaxf(expf(ls.z), MIN_SCALE), MAX_SCALE);
    r.s2 = f3make(sx * sx, sy * sy, sz * sz);
    r.inv_s2 = f3make(1.0f / r.s2.x, 1.0f / r.s2.y, 1.0f / r.s2.z);
    r.amplitude = softplus(p.raw_inten[k]);
    float max_s2 = fmaxf(r.s2.x, fmaxf(r.s2.y, r.s2.z));
    r.support_radius = sqrtf(MAHALANOBIS_CUTOFF * max_s2);
    rt[k] = r;
}

__host__ __device__ inline float3 world_to_local(const float* R, float3 d) {
    return f3make(R[0] * d.x + R[3] * d.y + R[6] * d.z,
                  R[1] * d.x + R[4] * d.y + R[7] * d.z,
                  R[2] * d.x + R[5] * d.y + R[8] * d.z);
}

__host__ __device__ inline float eval_gaussian_density(float3 mean, const float* R, float3 inv_s2, float amplitude, float3 x) {
    float3 d = f3sub(x, mean);
    float3 u = world_to_local(R, d);
    float Q = u.x * u.x * inv_s2.x + u.y * u.y * inv_s2.y + u.z * u.z * inv_s2.z;
    if (Q > MAHALANOBIS_CUTOFF) return 0.0f;
    return amplitude * expf(-0.5f * Q);
}

// ---------------------------------------------------------------------------
// LOCAL per-block bake (identical semantics to gaussian_splat_scratch.cu's
// bake_kernel: one thread per LOCAL voxel, evaluates ONLY this block's own
// Gaussians -- hard-gated, no cross-block blending).
// ---------------------------------------------------------------------------
__global__ void bake_local_kernel(const GaussianRuntime* rt, int n, float* baked_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LOCAL_GRID * LOCAL_GRID * LOCAL_GRID) return;
    int ix = idx % LOCAL_GRID;
    int iy = (idx / LOCAL_GRID) % LOCAL_GRID;
    int iz = idx / (LOCAL_GRID * LOCAL_GRID);
    float3 x = local_voxel_center(ix, iy, iz);
    float p = 0.0f;
    for (int k = 0; k < n; ++k) {
        float3 d = f3sub(x, rt[k].mean);
        if (f3dot(d, d) > rt[k].support_radius * rt[k].support_radius) continue;
        p += eval_gaussian_density(rt[k].mean, rt[k].R, rt[k].inv_s2, rt[k].amplitude, x);
    }
    baked_local[idx] = fminf(fmaxf(p, 0.0f), 1.0f);
}

// Places one block's LOCAL_GRID^3 baked array into its 64-voxel-aligned
// sub-region of the COMBINED_GRID^3 array (voxel offsets 0 or 64 per axis,
// matching that block's world position of -1 or +1 along each axis).
__global__ void place_block_kernel(const float* baked_local, float* combined,
                                    int ox, int oy, int oz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LOCAL_GRID * LOCAL_GRID * LOCAL_GRID) return;
    int ix = idx % LOCAL_GRID;
    int iy = (idx / LOCAL_GRID) % LOCAL_GRID;
    int iz = idx / (LOCAL_GRID * LOCAL_GRID);
    int cx = ox + ix, cy = oy + iy, cz = oz + iz;
    combined[cx + cy * COMBINED_GRID + cz * COMBINED_GRID * COMBINED_GRID] = baked_local[idx];
}

// ---------------------------------------------------------------------------
// Camera (identical to gaussian_splat_scratch.cu)
// ---------------------------------------------------------------------------
struct Camera { float3 pos, forward, right, up; float tan_half_fov; };

Camera make_camera(float yaw_deg) {
    float yaw = yaw_deg * (float)M_PI / 180.0f;
    Camera cam;
    cam.pos = f3make(0, 0, 0);
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

__host__ __device__ inline bool ray_box_combined(float3 origin, float3 dir, float& t0, float& t1) {
    t0 = 0.0f; t1 = 1e9f;
    for (int a = 0; a < 3; ++a) {
        float od = (a == 0) ? origin.x : (a == 1 ? origin.y : origin.z);
        float dd = (a == 0) ? dir.x : (a == 1 ? dir.y : dir.z);
        if (fabsf(dd) < 1e-9f) {
            if (od < COMBINED_BOX_MIN || od > COMBINED_BOX_MAX) return false;
        } else {
            float ta = (COMBINED_BOX_MIN - od) / dd;
            float tb = (COMBINED_BOX_MAX - od) / dd;
            if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
            t0 = fmaxf(t0, ta);
            t1 = fminf(t1, tb);
        }
    }
    return t0 <= t1;
}

__global__ void dvr_render_combined_kernel(const float* grid, Camera cam, float* image, int W, int H) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    float3 dir = camera_ray_dir(cam, px, py, W, H);
    float t0, t1;
    float mip = 0.0f;
    if (ray_box_combined(cam.pos, dir, t0, t1)) {
        float dt = (t1 - t0) / DVR_SAMPLES;
        for (int s = 0; s < DVR_SAMPLES; ++s) {
            float t = t0 + (s + 0.5f) * dt;
            float3 pos = f3add(cam.pos, f3scale(dir, t));
            float v = sample_combined_trilinear(grid, pos);
            mip = fmaxf(mip, v);
        }
    }
    image[py * W + px] = fminf(fmaxf(mip, 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// Tile-based Gaussian rasterizer -- identical to gaussian_splat_scratch.cu,
// operating directly on the COMBINED (offset) Gaussian array, no grid at all.
// ---------------------------------------------------------------------------
__device__ inline bool solve_gaussian_ray_interval(float3 origin, float3 dir, float3 mean, const float* R, float3 inv_s2,
                                                    float& t_lo, float& t_hi) {
    float3 oc = f3sub(origin, mean);
    float3 u0 = world_to_local(R, oc);
    float3 ud = world_to_local(R, dir);
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
    float ang_radius = r.support_radius / dist;
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

struct RenderGaussian { float3 mean; float R[9]; float3 inv_s2; float amplitude; };

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
                    bin_vals[bin] += dens;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        float m = 0.0f;
        #pragma unroll
        for (int b = 0; b < RENDER_BINS; ++b) m = fmaxf(m, bin_vals[b]);
        image[py * W + px] = fminf(fmaxf(m, 0.0f), 1.0f);
    }
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------
void write_frame(const char* path, const float* h_data, int W, int H) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    int32_t w32 = W, h32 = H;
    fwrite(&w32, sizeof(int32_t), 1, f);
    fwrite(&h32, sizeof(int32_t), 1, f);
    fwrite(h_data, sizeof(float), W * H, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Block position tables: world-space offsets (in units of the local box
// half-width, i.e. -1 or +1) for 2/4/8-block arrangements, and the matching
// COMBINED-grid voxel offsets (0 or LOCAL_GRID).
// ---------------------------------------------------------------------------
struct BlockPos { float ox, oy, oz; int vox, voy, voz; };

std::vector<BlockPos> block_positions(int n_blocks) {
    std::vector<BlockPos> out;
    // A block's local range [-1,1], shifted by world offset o, occupies world
    // range [o-1, o+1]. Mapped into the combined [-2,2] box (resolution-
    // matched, LOCAL_GRID voxels per 2 world units on both grids), that
    // range starts at combined voxel index (o+1)*(LOCAL_GRID/2): o=-1 -> 0,
    // o=0 -> LOCAL_GRID/2 (the MIDDLE of the combined grid, not its edge --
    // the bug this fixes: o=0 was being placed at voxel 0, silently
    // shifting every unshifted axis by a full world unit).
    auto axis_offset = [](float o) -> int { return (int)((o + 1.0f) * (LOCAL_GRID / 2)); };
    auto add = [&](float ox, float oy, float oz) {
        BlockPos bp;
        bp.ox = ox; bp.oy = oy; bp.oz = oz;
        bp.vox = axis_offset(ox);
        bp.voy = axis_offset(oy);
        bp.voz = axis_offset(oz);
        out.push_back(bp);
    };
    if (n_blocks == 2) {
        add(-1, 0, 0); add(1, 0, 0);                 // 2x1x1
    } else if (n_blocks == 4) {
        add(-1, -1, 0); add(-1, 1, 0); add(1, -1, 0); add(1, 1, 0);   // 2x2x1
    } else if (n_blocks == 8) {
        for (float ox : {-1.0f, 1.0f})
            for (float oy : {-1.0f, 1.0f})
                for (float oz : {-1.0f, 1.0f})
                    add(ox, oy, oz);                  // 2x2x2
    } else {
        fprintf(stderr, "n_blocks must be 2, 4, or 8\n"); exit(1);
    }
    return out;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <n_blocks: 2|4|8> <checkpoint.bin> <out_dir> [screen_size=512]\n", argv[0]);
        return 1;
    }
    int n_blocks = atoi(argv[1]);
    std::string checkpoint_path = argv[2];
    std::string out_dir = argv[3];
    int SCREEN = (argc > 4) ? atoi(argv[4]) : 512;
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    system(mkdir_cmd.c_str());

    auto positions = block_positions(n_blocks);
    int N_TOTAL = n_blocks * N_GAUSSIANS_PER_BLOCK;
    printf("=== Stitching experiment: %d blocks, %d total Gaussians, screen %dx%d ===\n", n_blocks, N_TOTAL, SCREEN, SCREEN);

    // ---- Load the single trained block once ----
    std::vector<float3> h_mean, h_logs;
    std::vector<float4> h_quat;
    std::vector<float> h_int;
    if (!load_checkpoint_host(checkpoint_path.c_str(), h_mean, h_logs, h_quat, h_int, N_GAUSSIANS_PER_BLOCK)) {
        fprintf(stderr, "Failed to load checkpoint %s (expected %d Gaussians)\n", checkpoint_path.c_str(), N_GAUSSIANS_PER_BLOCK);
        return 1;
    }

    // ---- 1. LOCAL per-block Gaussians (un-offset), for independent baking ----
    Params local_params; alloc_params(local_params, N_GAUSSIANS_PER_BLOCK);
    CUDA_CHECK(cudaMemcpy(local_params.mean, h_mean.data(), N_GAUSSIANS_PER_BLOCK * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(local_params.log_scale, h_logs.data(), N_GAUSSIANS_PER_BLOCK * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(local_params.quat, h_quat.data(), N_GAUSSIANS_PER_BLOCK * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(local_params.raw_inten, h_int.data(), N_GAUSSIANS_PER_BLOCK * sizeof(float), cudaMemcpyHostToDevice));
    GaussianRuntime* local_rt;
    CUDA_CHECK(cudaMalloc(&local_rt, N_GAUSSIANS_PER_BLOCK * sizeof(GaussianRuntime)));
    precompute_runtime_kernel<<<(N_GAUSSIANS_PER_BLOCK + 127) / 128, 128>>>(local_params, local_rt, N_GAUSSIANS_PER_BLOCK);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 2. BAKE-THEN-STITCH: bake each block locally, place into combined grid ----
    float* d_combined_baked;
    CUDA_CHECK(cudaMalloc(&d_combined_baked, (size_t)COMBINED_GRID * COMBINED_GRID * COMBINED_GRID * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_combined_baked, 0, (size_t)COMBINED_GRID * COMBINED_GRID * COMBINED_GRID * sizeof(float)));
    float* d_local_baked;
    CUDA_CHECK(cudaMalloc(&d_local_baked, (size_t)LOCAL_GRID * LOCAL_GRID * LOCAL_GRID * sizeof(float)));

    auto bake_start = std::chrono::high_resolution_clock::now();
    int local_voxels = LOCAL_GRID * LOCAL_GRID * LOCAL_GRID;
    for (const auto& bp : positions) {
        bake_local_kernel<<<(local_voxels + 255) / 256, 256>>>(local_rt, N_GAUSSIANS_PER_BLOCK, d_local_baked);
        CUDA_CHECK(cudaGetLastError());
        place_block_kernel<<<(local_voxels + 255) / 256, 256>>>(d_local_baked, d_combined_baked, bp.vox, bp.voy, bp.voz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto bake_end = std::chrono::high_resolution_clock::now();
    double bake_secs = std::chrono::duration<double>(bake_end - bake_start).count();
    printf("Baked + stitched %d blocks in %.4f s\n", n_blocks, bake_secs);

    // ---- 3. GAUSSIAN-STITCH: combined (offset) Gaussian array ----
    std::vector<float3> combined_mean(N_TOTAL);
    std::vector<float3> combined_logs(N_TOTAL);
    std::vector<float4> combined_quat(N_TOTAL);
    std::vector<float> combined_int(N_TOTAL);
    for (int b = 0; b < n_blocks; ++b) {
        float3 offset = f3make(positions[b].ox, positions[b].oy, positions[b].oz);
        for (int k = 0; k < N_GAUSSIANS_PER_BLOCK; ++k) {
            int idx = b * N_GAUSSIANS_PER_BLOCK + k;
            combined_mean[idx] = f3add(h_mean[k], offset);
            combined_logs[idx] = h_logs[k];
            combined_quat[idx] = h_quat[k];
            combined_int[idx] = h_int[k];
        }
    }
    Params combined_params; alloc_params(combined_params, N_TOTAL);
    CUDA_CHECK(cudaMemcpy(combined_params.mean, combined_mean.data(), N_TOTAL * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(combined_params.log_scale, combined_logs.data(), N_TOTAL * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(combined_params.quat, combined_quat.data(), N_TOTAL * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(combined_params.raw_inten, combined_int.data(), N_TOTAL * sizeof(float), cudaMemcpyHostToDevice));
    GaussianRuntime* combined_rt;
    CUDA_CHECK(cudaMalloc(&combined_rt, N_TOTAL * sizeof(GaussianRuntime)));
    precompute_runtime_kernel<<<(N_TOTAL + 127) / 128, 128>>>(combined_params, combined_rt, N_TOTAL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 4. Render setup ----
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((SCREEN + 15) / 16, (SCREEN + 15) / 16);
    float t_far = sqrtf(3.0f) * (COMBINED_BOX_MAX - COMBINED_BOX_MIN);

    int tiles_x = (SCREEN + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = tiles_x;
    int n_tiles = tiles_x * tiles_y;
    int* d_tile_counts; CUDA_CHECK(cudaMalloc(&d_tile_counts, n_tiles * sizeof(int)));
    int* d_tile_gaussian_list; CUDA_CHECK(cudaMalloc(&d_tile_gaussian_list, (size_t)n_tiles * N_TOTAL * sizeof(int)));
    dim3 tileBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 tileGridDim(tiles_x, tiles_y);

    float* d_image_baked; CUDA_CHECK(cudaMalloc(&d_image_baked, SCREEN * SCREEN * sizeof(float)));
    float* d_image_raster; CUDA_CHECK(cudaMalloc(&d_image_raster, SCREEN * SCREEN * sizeof(float)));
    float* h_image = new float[SCREEN * SCREEN];

    int threadsG = 128, blocksG = (N_TOTAL + threadsG - 1) / threadsG;

    // ---- 5. GPU-only benchmark (CUDA events), warm-up + measured rounds ----
    constexpr int WARMUP = 5, MEASURE = 20;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int r = 0; r < WARMUP; ++r) {
        for (int f = 0; f < N_FRAMES; ++f) {
            Camera cam = make_camera(360.0f * f / N_FRAMES);
            dvr_render_combined_kernel<<<gridDim2D, blockDim2D>>>(d_combined_baked, cam, d_image_baked, SCREEN, SCREEN);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < MEASURE; ++r) {
        for (int f = 0; f < N_FRAMES; ++f) {
            Camera cam = make_camera(360.0f * f / N_FRAMES);
            dvr_render_combined_kernel<<<gridDim2D, blockDim2D>>>(d_combined_baked, cam, d_image_baked, SCREEN, SCREEN);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float baked_ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&baked_ms, ev0, ev1));
    double baked_fps = (double)(MEASURE * N_FRAMES) * 1000.0 / baked_ms;

    for (int r = 0; r < WARMUP; ++r) {
        for (int f = 0; f < N_FRAMES; ++f) {
            Camera cam = make_camera(360.0f * f / N_FRAMES);
            CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
            build_tile_lists_kernel<<<blocksG, threadsG>>>(combined_rt, N_TOTAL, cam, d_tile_counts, d_tile_gaussian_list, SCREEN, SCREEN, tiles_x, tiles_y);
            render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(combined_rt, cam, d_tile_counts, d_tile_gaussian_list, d_image_raster, SCREEN, SCREEN, tiles_x, N_TOTAL, t_far, RENDER_BINS);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int r = 0; r < MEASURE; ++r) {
        for (int f = 0; f < N_FRAMES; ++f) {
            Camera cam = make_camera(360.0f * f / N_FRAMES);
            CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
            build_tile_lists_kernel<<<blocksG, threadsG>>>(combined_rt, N_TOTAL, cam, d_tile_counts, d_tile_gaussian_list, SCREEN, SCREEN, tiles_x, tiles_y);
            render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(combined_rt, cam, d_tile_counts, d_tile_gaussian_list, d_image_raster, SCREEN, SCREEN, tiles_x, N_TOTAL, t_far, RENDER_BINS);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float raster_ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&raster_ms, ev0, ev1));
    double raster_fps = (double)(MEASURE * N_FRAMES) * 1000.0 / raster_ms;

    printf("Baked+DVR (stitched):        %.2f FPS\n", baked_fps);
    printf("Gaussian rasterizer (stitched, %d Gaussians): %.2f FPS\n", N_TOTAL, raster_fps);

    // ---- 6. Export paired frames (untimed) for quality comparison ----
    for (int f = 0; f < N_FRAMES; ++f) {
        Camera cam = make_camera(360.0f * f / N_FRAMES);
        char path[512];

        dvr_render_combined_kernel<<<gridDim2D, blockDim2D>>>(d_combined_baked, cam, d_image_baked, SCREEN, SCREEN);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_image, d_image_baked, SCREEN * SCREEN * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(path, sizeof(path), "%s/baked_%04d.bin", out_dir.c_str(), f);
        write_frame(path, h_image, SCREEN, SCREEN);

        CUDA_CHECK(cudaMemset(d_tile_counts, 0, n_tiles * sizeof(int)));
        build_tile_lists_kernel<<<blocksG, threadsG>>>(combined_rt, N_TOTAL, cam, d_tile_counts, d_tile_gaussian_list, SCREEN, SCREEN, tiles_x, tiles_y);
        render_tiles_kernel<<<tileGridDim, tileBlockDim>>>(combined_rt, cam, d_tile_counts, d_tile_gaussian_list, d_image_raster, SCREEN, SCREEN, tiles_x, N_TOTAL, t_far, RENDER_BINS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_image, d_image_raster, SCREEN * SCREEN * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(path, sizeof(path), "%s/raster_%04d.bin", out_dir.c_str(), f);
        write_frame(path, h_image, SCREEN, SCREEN);
    }

    // ---- 7. Summary file ----
    {
        std::string p = out_dir + "/fps_summary.txt";
        FILE* f = fopen(p.c_str(), "w");
        fprintf(f, "n_blocks %d\n", n_blocks);
        fprintf(f, "n_gaussians_total %d\n", N_TOTAL);
        fprintf(f, "screen %d\n", SCREEN);
        fprintf(f, "n_frames %d\n", N_FRAMES);
        fprintf(f, "warmup_rounds %d\n", WARMUP);
        fprintf(f, "measure_rounds %d\n", MEASURE);
        fprintf(f, "baked_stitch_fps %.4f\n", baked_fps);
        fprintf(f, "gaussian_stitch_fps %.4f\n", raster_fps);
        fprintf(f, "bake_time_seconds %.4f\n", bake_secs);
        fclose(f);
    }

    printf("All outputs written to %s\n", out_dir.c_str());
    delete[] h_image;
    return 0;
}
