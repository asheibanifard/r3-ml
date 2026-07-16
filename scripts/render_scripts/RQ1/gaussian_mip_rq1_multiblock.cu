// gaussian_mip_rq1.cu
//
// RQ1 benchmark:
//   How does active Gaussian payload affect steady-state rendering speed
//   on one fixed GPU?
//
// This program varies only the active Gaussian payload using a deterministic
// prefix subset of one Gaussian binary. Rendering settings are supplied once
// and must remain unchanged across the payload sweep.
//
// It reports:
//   - full and retained Gaussian counts
//   - active payload bytes and MiB
//   - Gaussian-tile pair count
//   - per-frame CUDA render times
//   - mean/median/std/min/max/p5/p95 render time
//   - mean/median/std/min/max/p5/p95 FPS
//   - one CSV row per benchmark run
//
// It intentionally excludes from steady-state timing:
//   - binary loading
//   - subset construction
//   - CUDA allocation
//   - host-to-device upload
//   - Gaussian preprocessing
//   - prefix scan
//   - tile duplication
//   - radix sort
//   - tile-range construction
//   - output download
//   - PFM writing
//   - CSV writing
//
// Build example for RTX 40-series:
//   nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
//     -gencode arch=compute_89,code=sm_89 \
//     gaussian_mip_rq1.cu -o gaussian_mip_rq1
//
// Run example:
//   ./gaussian_mip_rq1 \
//     --input gaussians.bin \
//     --output output.pfm \
//     --width 128 --height 128 \
//     --depth-samples 64 \
//     --warmup-frames 20 \
//     --measured-frames 300 \
//     --retention-ratio 0.50 \
//     --repeat-id 0 \
//     --camera 0 0 0 \
//     --rotation 0 0 0 \
//     --fov-y 90 \
//     --box -1 -1 -1 1 1 1 \
//     --csv-output results/rq1/rq1_payload_fps_raw.csv
//
// Binary format:
//   uint32 magic = 0x47534D50
//   uint32 version = 1
//   uint64 count
//   repeated count times:
//     11 float32 values:
//       mean xyz, scale xyz, quaternion wxyz, intensity
//
// Coordinate convention:
//   right-handed world coordinates
//   zero rotation looks along +Z
//   +X is camera-right and +Y is camera-up
//
// Important:
//   This code assumes checkpoint activations were already applied by the
//   exporter. It does not apply exp(log_scales) or softplus(intensity).

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e__ = (call);                                               \
        if (e__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, cudaGetErrorString(e__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr uint32_t FILE_MAGIC = 0x47534D50u;
constexpr uint32_t FILE_VERSION = 1u;

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int TILE_THREADS = TILE_W * TILE_H;
constexpr int GAUSSIAN_BATCH = 128;

constexpr float MAHAL_CUTOFF = 20.0f;
constexpr float EPS_SCALE = 1e-6f;
constexpr float CAMERA_NEAR = 1e-4f;

struct GaussianDisk {
    float mean[3];
    float scale[3];
    float quat[4];
    float intensity;
};

static_assert(sizeof(GaussianDisk) == 11 * sizeof(float),
              "GaussianDisk must contain exactly 11 float32 values.");

struct GaussianGPU {
    float3 mean;
    float q00, q01, q02, q11, q12, q22;
    float intensity;
    int2 tile_min;
    int2 tile_max;
    int visible;
};

struct Range {
    uint32_t begin;
    uint32_t end;
};

struct FileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t count;
};

struct CameraGPU {
    float3 position;
    float3 right;
    float3 up;
    float3 forward;
    float tan_half_fov_y;
    float aspect;
};

struct BoxGPU {
    float3 minimum;
    float3 maximum;
};

struct Options {
    std::string input_path;
    std::string input_list_path;
    std::string output_path;
    std::string csv_output;

    int width = 128;
    int height = 128;
    int depth_samples = 64;
    int warmup_frames = 20;
    int measured_frames = 300;
    int repeat_id = 0;

    double retention_ratio = 1.0;

    float3 camera_position = make_float3(0.0f, 0.0f, 0.0f);
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    float fov_y = 90.0f;

    BoxGPU box{
        make_float3(-1.0f, -1.0f, -1.0f),
        make_float3( 1.0f,  1.0f,  1.0f)
    };
};

struct SummaryStats {
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double p5 = 0.0;
    double p95 = 0.0;
};

__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 mul3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float3 normalize3(float3 a) {
    const float n2 = dot3(a, a);
    const float inv = rsqrtf(fmaxf(n2, 1e-20f));
    return mul3(a, inv);
}

__device__ inline void quaternion_to_rotation(
    float w, float x, float y, float z,
    float R[9])
{
    const float inv = rsqrtf(fmaxf(w*w + x*x + y*y + z*z, 1e-20f));
    w *= inv;
    x *= inv;
    y *= inv;
    z *= inv;

    R[0] = 1 - 2 * (y*y + z*z);
    R[1] = 2 * (x*y - z*w);
    R[2] = 2 * (x*z + y*w);

    R[3] = 2 * (x*y + z*w);
    R[4] = 1 - 2 * (x*x + z*z);
    R[5] = 2 * (y*z - x*w);

    R[6] = 2 * (x*z - y*w);
    R[7] = 2 * (y*z + x*w);
    R[8] = 1 - 2 * (x*x + y*y);
}

__device__ inline bool inverse_symmetric_3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float& q00, float& q01, float& q02,
    float& q11, float& q12, float& q22)
{
    const float c00 = a11 * a22 - a12 * a12;
    const float c01 = a02 * a12 - a01 * a22;
    const float c02 = a01 * a12 - a02 * a11;
    const float c11 = a00 * a22 - a02 * a02;
    const float c12 = a01 * a02 - a00 * a12;
    const float c22 = a00 * a11 - a01 * a01;
    const float det = a00 * c00 + a01 * c01 + a02 * c02;

    if (!(det > 1e-20f) || !isfinite(det)) {
        return false;
    }

    const float inv = 1.0f / det;
    q00 = c00 * inv;
    q01 = c01 * inv;
    q02 = c02 * inv;
    q11 = c11 * inv;
    q12 = c12 * inv;
    q22 = c22 * inv;
    return true;
}

__device__ inline float largest_eigenvalue_upper_bound(
    float s00, float s01, float s02,
    float s11, float s12, float s22)
{
    const float r0 = fabsf(s01) + fabsf(s02);
    const float r1 = fabsf(s01) + fabsf(s12);
    const float r2 = fabsf(s02) + fabsf(s12);
    return fmaxf(s00 + r0, fmaxf(s11 + r1, s22 + r2));
}

__global__ void preprocess_kernel(
    const GaussianDisk* __restrict__ input,
    GaussianGPU* __restrict__ output,
    uint32_t* __restrict__ tile_counts,
    int P,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    CameraGPU camera,
    float mahal_cutoff)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) {
        return;
    }

    const GaussianDisk in = input[i];
    GaussianGPU out{};
    out.mean = make_float3(in.mean[0], in.mean[1], in.mean[2]);
    out.intensity = fmaxf(in.intensity, 0.0f);

    const float sx = fmaxf(fabsf(in.scale[0]), EPS_SCALE);
    const float sy = fmaxf(fabsf(in.scale[1]), EPS_SCALE);
    const float sz = fmaxf(fabsf(in.scale[2]), EPS_SCALE);

    float R[9];
    quaternion_to_rotation(
        in.quat[0], in.quat[1], in.quat[2], in.quat[3], R
    );

    const float vx = sx * sx;
    const float vy = sy * sy;
    const float vz = sz * sz;

    const float s00 = R[0]*R[0]*vx + R[1]*R[1]*vy + R[2]*R[2]*vz;
    const float s01 = R[0]*R[3]*vx + R[1]*R[4]*vy + R[2]*R[5]*vz;
    const float s02 = R[0]*R[6]*vx + R[1]*R[7]*vy + R[2]*R[8]*vz;
    const float s11 = R[3]*R[3]*vx + R[4]*R[4]*vy + R[5]*R[5]*vz;
    const float s12 = R[3]*R[6]*vx + R[4]*R[7]*vy + R[5]*R[8]*vz;
    const float s22 = R[6]*R[6]*vx + R[7]*R[7]*vy + R[8]*R[8]*vz;

    const bool ok = inverse_symmetric_3x3(
        s00, s01, s02, s11, s12, s22,
        out.q00, out.q01, out.q02,
        out.q11, out.q12, out.q22
    );

    if (!ok || out.intensity <= 0.0f || !isfinite(out.intensity)) {
        tile_counts[i] = 0;
        output[i] = out;
        return;
    }

    const float3 relative = sub3(out.mean, camera.position);
    const float cx = dot3(relative, camera.right);
    const float cy = dot3(relative, camera.up);
    const float cz = dot3(relative, camera.forward);

    const float lambda_bound = fmaxf(
        largest_eigenvalue_upper_bound(s00, s01, s02, s11, s12, s22),
        0.0f
    );
    const float support_radius = sqrtf(mahal_cutoff * lambda_bound);

    int min_tx = 0;
    int max_tx = tiles_x;
    int min_ty = 0;
    int max_ty = tiles_y;

    const float distance_to_camera = sqrtf(
        relative.x * relative.x +
        relative.y * relative.y +
        relative.z * relative.z
    );

    if (distance_to_camera <= support_radius) {
        min_tx = 0;
        max_tx = tiles_x;
        min_ty = 0;
        max_ty = tiles_y;
    } else if (cz > support_radius + CAMERA_NEAR) {
        const float ndc_x =
            cx / (cz * camera.tan_half_fov_y * camera.aspect);
        const float ndc_y =
            cy / (cz * camera.tan_half_fov_y);

        const float denom = fmaxf(cz - support_radius, CAMERA_NEAR);
        const float radius_ndc_y =
            support_radius / (denom * camera.tan_half_fov_y);
        const float radius_ndc_x =
            radius_ndc_y / camera.aspect;

        const float px = (ndc_x * 0.5f + 0.5f) * float(width);
        const float py = (0.5f - ndc_y * 0.5f) * float(height);

        const float radius_px = fmaxf(
            radius_ndc_x * 0.5f * float(width),
            radius_ndc_y * 0.5f * float(height)
        ) + 1.0f;

        const int min_px_x = int(floorf(px - radius_px));
        const int max_px_x = int(ceilf(px + radius_px));
        const int min_px_y = int(floorf(py - radius_px));
        const int max_px_y = int(ceilf(py + radius_px));

        // Use floor division in pixel space to remain conservative for
        // negative coordinates near the left and upper image boundaries.
        const int raw_min_tx = int(floorf(float(min_px_x) / float(TILE_W)));
        const int raw_max_tx = int(floorf(float(max_px_x) / float(TILE_W))) + 1;
        const int raw_min_ty = int(floorf(float(min_px_y) / float(TILE_H)));
        const int raw_max_ty = int(floorf(float(max_px_y) / float(TILE_H))) + 1;

        min_tx = max(0, min(raw_min_tx, tiles_x));
        max_tx = max(0, min(raw_max_tx, tiles_x));
        min_ty = max(0, min(raw_min_ty, tiles_y));
        max_ty = max(0, min(raw_max_ty, tiles_y));

        if (max_px_x < 0 || min_px_x >= width ||
            max_px_y < 0 || min_px_y >= height) {
            min_tx = max_tx = min_ty = max_ty = 0;
        }
    } else if (cz + support_radius <= CAMERA_NEAR) {
        min_tx = max_tx = min_ty = max_ty = 0;
    }

    const int count_x = max_tx - min_tx;
    const int count_y = max_ty - min_ty;
    const uint32_t count =
        (count_x > 0 && count_y > 0)
        ? uint32_t(count_x * count_y)
        : 0u;

    out.tile_min = make_int2(min_tx, min_ty);
    out.tile_max = make_int2(max_tx, max_ty);
    out.visible = (count > 0u);

    tile_counts[i] = count;
    output[i] = out;
}

__global__ void duplicate_kernel(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ values,
    int P,
    int tiles_x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) {
        return;
    }

    const GaussianGPU g = gaussians[i];
    if (!g.visible) {
        return;
    }

    uint32_t write = (i == 0) ? 0u : offsets[i - 1];

    for (int ty = g.tile_min.y; ty < g.tile_max.y; ++ty) {
        for (int tx = g.tile_min.x; tx < g.tile_max.x; ++tx) {
            keys[write] = uint32_t(ty * tiles_x + tx);
            values[write] = uint32_t(i);
            ++write;
        }
    }
}

__global__ void identify_ranges_kernel(
    const uint32_t* __restrict__ keys,
    Range* __restrict__ ranges,
    uint32_t N)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const uint32_t current = keys[i];

    if (i == 0) {
        ranges[current].begin = 0;
    } else {
        const uint32_t previous = keys[i - 1];
        if (current != previous) {
            ranges[previous].end = i;
            ranges[current].begin = i;
        }
    }

    if (i == N - 1) {
        ranges[current].end = N;
    }
}

__device__ inline bool ray_box_exit(
    float3 origin,
    float3 direction,
    BoxGPU box,
    float& t_enter,
    float& t_exit)
{
    float near_t = -CUDART_INF_F;
    float far_t = CUDART_INF_F;

    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {direction.x, direction.y, direction.z};
    const float lo[3] = {box.minimum.x, box.minimum.y, box.minimum.z};
    const float hi[3] = {box.maximum.x, box.maximum.y, box.maximum.z};

    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1e-12f) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) {
                return false;
            }
        } else {
            const float inv = 1.0f / d[axis];
            float t0 = (lo[axis] - o[axis]) * inv;
            float t1 = (hi[axis] - o[axis]) * inv;

            if (t0 > t1) {
                const float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }

            near_t = fmaxf(near_t, t0);
            far_t = fminf(far_t, t1);

            if (far_t < near_t) {
                return false;
            }
        }
    }

    t_enter = fmaxf(near_t, 0.0f);
    t_exit = far_t;
    return t_exit > t_enter;
}

__device__ inline float mahalanobis(
    const GaussianGPU& g,
    float3 p)
{
    const float dx = p.x - g.mean.x;
    const float dy = p.y - g.mean.y;
    const float dz = p.z - g.mean.z;

    return
        g.q00 * dx * dx +
        2.0f * g.q01 * dx * dy +
        2.0f * g.q02 * dx * dz +
        g.q11 * dy * dy +
        2.0f * g.q12 * dy * dz +
        g.q22 * dz * dz;
}

__global__ void render_kernel(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ sorted_values,
    const Range* __restrict__ ranges,
    float* __restrict__ output,
    int width,
    int height,
    int tiles_x,
    int depth_samples,
    CameraGPU camera,
    BoxGPU box,
    float mahal_cutoff)
{
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int linear_tid = ly * TILE_W + lx;

    const int px = blockIdx.x * TILE_W + lx;
    const int py = blockIdx.y * TILE_H + ly;
    const bool inside_image = (px < width && py < height);

    const int tile_id = blockIdx.y * tiles_x + blockIdx.x;
    const Range range = ranges[tile_id];

    float3 direction = make_float3(0, 0, 1);
    float t0 = 0.0f;
    float t1 = 0.0f;
    bool valid_ray = false;

    if (inside_image) {
        const float ndc_x =
            2.0f * (float(px) + 0.5f) / float(width) - 1.0f;
        const float ndc_y =
            1.0f - 2.0f * (float(py) + 0.5f) / float(height);

        const float camera_x =
            ndc_x * camera.aspect * camera.tan_half_fov_y;
        const float camera_y =
            ndc_y * camera.tan_half_fov_y;

        direction = normalize3(add3(
            camera.forward,
            add3(
                mul3(camera.right, camera_x),
                mul3(camera.up, camera_y)
            )
        ));

        valid_ray = ray_box_exit(
            camera.position, direction, box, t0, t1
        );
        t0 = fmaxf(t0, CAMERA_NEAR);
    }

    float best = 0.0f;
    __shared__ GaussianGPU shared_g[GAUSSIAN_BATCH];

    for (int sample = 0; sample < depth_samples; ++sample) {
        float density = 0.0f;
        float3 point = make_float3(0, 0, 0);

        if (inside_image && valid_ray) {
            const float u =
                (depth_samples > 1)
                ? float(sample) / float(depth_samples - 1)
                : 0.5f;
            const float t = t0 + (t1 - t0) * u;
            point = add3(camera.position, mul3(direction, t));
        }

        for (uint32_t begin = range.begin;
             begin < range.end;
             begin += GAUSSIAN_BATCH)
        {
            const uint32_t count = min(
                uint32_t(GAUSSIAN_BATCH),
                range.end - begin
            );

            if (linear_tid < int(count)) {
                shared_g[linear_tid] =
                    gaussians[sorted_values[begin + linear_tid]];
            }

            __syncthreads();

            if (inside_image && valid_ray) {
                #pragma unroll 4
                for (uint32_t j = 0; j < count; ++j) {
                    const GaussianGPU g = shared_g[j];
                    const float m = mahalanobis(g, point);

                    if (m >= 0.0f && m <= mahal_cutoff) {
                        density +=
                            g.intensity * __expf(-0.5f * m);
                    }
                }
            }

            __syncthreads();
        }

        if (inside_image && valid_ray) {
            best = fmaxf(best, density);
        }
    }

    if (inside_image) {
        output[py * width + px] = best;
    }
}

static std::vector<GaussianDisk> read_gaussians(
    const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open " + path);
    }

    FileHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!stream ||
        header.magic != FILE_MAGIC ||
        header.version != FILE_VERSION)
    {
        throw std::runtime_error("Invalid Gaussian binary header.");
    }

    if (header.count == 0 ||
        header.count > uint64_t(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Invalid Gaussian count.");
    }

    std::vector<GaussianDisk> data(size_t(header.count));

    stream.read(
        reinterpret_cast<char*>(data.data()),
        std::streamsize(data.size() * sizeof(GaussianDisk))
    );

    if (!stream) {
        throw std::runtime_error("Truncated Gaussian binary.");
    }

    return data;
}

static std::vector<size_t> make_fixed_shuffled_order(
    size_t gaussian_count,
    uint32_t seed)
{
    if (gaussian_count == 0) {
        throw std::runtime_error(
            "Cannot create an ordering for zero Gaussians."
        );
    }

    std::vector<size_t> indices(gaussian_count);
    std::iota(indices.begin(), indices.end(), size_t{0});

    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);

    return indices;
}

static std::vector<GaussianDisk> make_nested_subset(
    const std::vector<GaussianDisk>& full,
    const std::vector<size_t>& fixed_order,
    double retention_ratio)
{
    if (full.empty()) {
        throw std::runtime_error(
            "Cannot create a subset from an empty Gaussian set."
        );
    }

    if (fixed_order.size() != full.size()) {
        throw std::runtime_error(
            "Fixed shuffled order does not match the Gaussian count."
        );
    }

    if (!(retention_ratio > 0.0 && retention_ratio <= 1.0)) {
        throw std::runtime_error(
            "Retention ratio must be in the interval (0, 1]."
        );
    }

    size_t retained = static_cast<size_t>(
        std::floor(retention_ratio * double(full.size()))
    );

    retained = std::max<size_t>(1, retained);
    retained = std::min(retained, full.size());

    std::vector<GaussianDisk> subset;
    subset.reserve(retained);

    // The same fixed shuffled ordering is used for every ratio. Therefore,
    // all payload levels are nested:
    // G_10% subset G_20% subset ... subset G_100%.
    for (size_t i = 0; i < retained; ++i) {
        const size_t source_index = fixed_order[i];

        if (source_index >= full.size()) {
            throw std::runtime_error(
                "Fixed shuffled order contains an invalid index."
            );
        }

        subset.push_back(full[source_index]);
    }

    return subset;
}

static void write_pfm(
    const std::string& path,
    const std::vector<float>& image,
    int width,
    int height)
{
    std::filesystem::path output_path(path);

    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(
            output_path.parent_path()
        );
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot create " + path);
    }

    stream << "Pf\n" << width << " " << height << "\n-1.0\n";

    for (int y = height - 1; y >= 0; --y) {
        stream.write(
            reinterpret_cast<const char*>(
                image.data() + size_t(y) * width
            ),
            std::streamsize(width * sizeof(float))
        );
    }
}

static float deg_to_rad(float degrees) {
    return degrees * 3.14159265358979323846f / 180.0f;
}

static CameraGPU make_camera(
    float3 position,
    float yaw_deg,
    float pitch_deg,
    float roll_deg,
    float fov_y_deg,
    int width,
    int height)
{
    // Avoid the world-up singularity at exactly +/-90 degrees pitch.
    pitch_deg = std::max(-89.9f, std::min(89.9f, pitch_deg));

    const float yaw = deg_to_rad(yaw_deg);
    const float pitch = deg_to_rad(pitch_deg);
    const float roll = deg_to_rad(roll_deg);

    float3 forward = make_float3(
        sinf(yaw) * cosf(pitch),
        sinf(pitch),
        cosf(yaw) * cosf(pitch)
    );
    forward = normalize3(forward);

    float3 reference_up = make_float3(0, 1, 0);

    if (fabsf(dot3(reference_up, forward)) > 0.999f) {
        reference_up = make_float3(0, 0, 1);
    }

    float3 right = normalize3(cross3(reference_up, forward));
    float3 up = normalize3(cross3(forward, right));

    const float cr = cosf(roll);
    const float sr = sinf(roll);

    const float3 rolled_right =
        add3(mul3(right, cr), mul3(up, sr));
    const float3 rolled_up =
        add3(mul3(up, cr), mul3(right, -sr));

    CameraGPU camera{};
    camera.position = position;
    camera.right = normalize3(rolled_right);
    camera.up = normalize3(rolled_up);
    camera.forward = forward;
    camera.tan_half_fov_y =
        tanf(0.5f * deg_to_rad(fov_y_deg));
    camera.aspect = float(width) / float(height);

    return camera;
}

static double percentile_sorted(
    const std::vector<double>& sorted,
    double percentile)
{
    if (sorted.empty()) {
        throw std::runtime_error(
            "Cannot calculate percentile of empty data."
        );
    }

    if (sorted.size() == 1) {
        return sorted.front();
    }

    const double position =
        percentile * double(sorted.size() - 1);

    const size_t lower =
        static_cast<size_t>(std::floor(position));
    const size_t upper =
        static_cast<size_t>(std::ceil(position));

    const double fraction = position - double(lower);

    return
        sorted[lower] * (1.0 - fraction) +
        sorted[upper] * fraction;
}

static SummaryStats calculate_stats(
    const std::vector<double>& values)
{
    if (values.empty()) {
        throw std::runtime_error(
            "Cannot calculate statistics from empty values."
        );
    }

    SummaryStats stats{};

    stats.mean =
        std::accumulate(values.begin(), values.end(), 0.0) /
        double(values.size());

    double sum_squared = 0.0;
    for (const double value : values) {
        const double delta = value - stats.mean;
        sum_squared += delta * delta;
    }

    stats.stddev =
        std::sqrt(sum_squared / double(values.size()));

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    stats.minimum = sorted.front();
    stats.maximum = sorted.back();
    stats.median = percentile_sorted(sorted, 0.50);
    stats.p5 = percentile_sorted(sorted, 0.05);
    stats.p95 = percentile_sorted(sorted, 0.95);

    return stats;
}

static bool file_is_empty_or_missing(
    const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return true;
    }

    return std::filesystem::file_size(path) == 0;
}


static std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += "\"";
    return escaped;
}

static std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::vector<std::string> read_input_paths(const Options& options) {
    std::vector<std::string> paths;

    if (!options.input_path.empty()) {
        paths.push_back(options.input_path);
    }

    if (!options.input_list_path.empty()) {
        std::ifstream stream(options.input_list_path);
        if (!stream) {
            throw std::runtime_error(
                "Cannot open input list: " + options.input_list_path
            );
        }

        std::string line;
        while (std::getline(stream, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;
            paths.push_back(line);
        }
    }

    if (paths.size() < 5) {
        throw std::runtime_error(
            "RQ1 multi-block evaluation requires at least five input blocks. "
            "Provide --input-list with one .bin path per line."
        );
    }

    for (const auto& path : paths) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Input block does not exist: " + path);
        }
    }

    return paths;
}

static std::string block_identifier(
    const std::string& input_path,
    size_t block_index)
{
    std::string stem = std::filesystem::path(input_path).stem().string();
    if (stem.empty()) stem = "block";
    std::ostringstream out;
    out << "b" << std::setw(3) << std::setfill('0') << block_index
        << "_" << stem;
    return out.str();
}

static void append_csv(
    const Options& options,
    const std::string& block_id,
    const std::string& block_path,
    const cudaDeviceProp& device,
    int driver_version,
    int runtime_version,
    size_t full_count,
    size_t active_count,
    size_t active_payload_bytes,
    uint32_t pair_count,
    const SummaryStats& time_stats,
    const SummaryStats& fps_stats,
    float output_min,
    float output_max)
{
    if (options.csv_output.empty()) {
        return;
    }

    const std::filesystem::path csv_path(options.csv_output);

    if (csv_path.has_parent_path()) {
        std::filesystem::create_directories(
            csv_path.parent_path()
        );
    }

    const bool write_header =
        file_is_empty_or_missing(csv_path);

    std::ofstream csv(
        options.csv_output,
        std::ios::app
    );

    if (!csv) {
        throw std::runtime_error(
            "Cannot open CSV output: " + options.csv_output
        );
    }

    if (write_header) {
        csv
            << "block_id,"
            << "block_path,"
            << "repeat_id,"
            << "retention_ratio,"
            << "full_active_gaussians,"
            << "active_gaussians,"
            << "bytes_per_gaussian,"
            << "active_payload_bytes,"
            << "active_payload_mib,"
            << "gaussian_tile_pairs,"
            << "width,"
            << "height,"
            << "depth_samples,"
            << "fov_degrees,"
            << "camera_x,"
            << "camera_y,"
            << "camera_z,"
            << "yaw_degrees,"
            << "pitch_degrees,"
            << "roll_degrees,"
            << "box_min_x,"
            << "box_min_y,"
            << "box_min_z,"
            << "box_max_x,"
            << "box_max_y,"
            << "box_max_z,"
            << "tile_width,"
            << "tile_height,"
            << "mahalanobis_cutoff,"
            << "warmup_frames,"
            << "measured_frames,"
            << "mean_render_ms,"
            << "median_render_ms,"
            << "std_render_ms,"
            << "min_render_ms,"
            << "max_render_ms,"
            << "p5_render_ms,"
            << "p95_render_ms,"
            << "mean_fps,"
            << "median_fps,"
            << "std_fps,"
            << "min_fps,"
            << "max_fps,"
            << "p5_fps,"
            << "p95_fps,"
            << "output_min,"
            << "output_max,"
            << "gpu_name,"
            << "gpu_total_vram_bytes,"
            << "compute_major,"
            << "compute_minor,"
            << "cuda_driver_version,"
            << "cuda_runtime_version"
            << "\n";
    }

    const double payload_mib =
        double(active_payload_bytes) / (1024.0 * 1024.0);

    csv << std::setprecision(12)
        << csv_escape(block_id) << ","
        << csv_escape(block_path) << ","
        << options.repeat_id << ","
        << options.retention_ratio << ","
        << full_count << ","
        << active_count << ","
        << sizeof(GaussianDisk) << ","
        << active_payload_bytes << ","
        << payload_mib << ","
        << pair_count << ","
        << options.width << ","
        << options.height << ","
        << options.depth_samples << ","
        << options.fov_y << ","
        << options.camera_position.x << ","
        << options.camera_position.y << ","
        << options.camera_position.z << ","
        << options.yaw << ","
        << options.pitch << ","
        << options.roll << ","
        << options.box.minimum.x << ","
        << options.box.minimum.y << ","
        << options.box.minimum.z << ","
        << options.box.maximum.x << ","
        << options.box.maximum.y << ","
        << options.box.maximum.z << ","
        << TILE_W << ","
        << TILE_H << ","
        << MAHAL_CUTOFF << ","
        << options.warmup_frames << ","
        << options.measured_frames << ","
        << time_stats.mean << ","
        << time_stats.median << ","
        << time_stats.stddev << ","
        << time_stats.minimum << ","
        << time_stats.maximum << ","
        << time_stats.p5 << ","
        << time_stats.p95 << ","
        << fps_stats.mean << ","
        << fps_stats.median << ","
        << fps_stats.stddev << ","
        << fps_stats.minimum << ","
        << fps_stats.maximum << ","
        << fps_stats.p5 << ","
        << fps_stats.p95 << ","
        << output_min << ","
        << output_max << ","
        << "\"" << device.name << "\"" << ","
        << device.totalGlobalMem << ","
        << device.major << ","
        << device.minor << ","
        << driver_version << ","
        << runtime_version
        << "\n";
}

static void print_help(const char* executable) {
    std::cout
        << "Usage:\n"
        << "  " << executable << " [options]\n\n"
        << "Required:\n"
        << "  --input-list PATH          text file with at least five .bin paths\n"
        << "  --input PATH               optional additional single block\n"
        << "  --output PATH\n\n"
        << "RQ1 payload control:\n"
        << "  --retention-ratio FLOAT    default 1.0, range (0,1]\n"
        << "  --repeat-id INT            default 0\n"
        << "  --csv-output PATH          optional\n\n"
        << "Fixed rendering controls:\n"
        << "  --width INT                default 128\n"
        << "  --height INT               default 128\n"
        << "  --depth-samples INT        default 64\n"
        << "  --warmup-frames INT        default 20\n"
        << "  --measured-frames INT      default 300\n"
        << "  --camera X Y Z             default 0 0 0\n"
        << "  --rotation YAW PITCH ROLL  default 0 0 0\n"
        << "  --fov-y FLOAT              default 90\n"
        << "  --box MINX MINY MINZ MAXX MAXY MAXZ\n"
        << "                             default -1 -1 -1 1 1 1\n"
        << "  --help\n";
}

static Options parse_options(int argc, char** argv) {
    Options options{};

    auto require_values = [&](int index, int count, const char* flag) {
        if (index + count >= argc) {
            throw std::runtime_error(
                std::string("Missing value(s) after ") + flag
            );
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help") {
            print_help(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--input") {
            require_values(i, 1, "--input");
            options.input_path = argv[++i];
        } else if (arg == "--input-list") {
            require_values(i, 1, "--input-list");
            options.input_list_path = argv[++i];
        } else if (arg == "--output") {
            require_values(i, 1, "--output");
            options.output_path = argv[++i];
        } else if (arg == "--csv-output") {
            require_values(i, 1, "--csv-output");
            options.csv_output = argv[++i];
        } else if (arg == "--width") {
            require_values(i, 1, "--width");
            options.width = std::stoi(argv[++i]);
        } else if (arg == "--height") {
            require_values(i, 1, "--height");
            options.height = std::stoi(argv[++i]);
        } else if (arg == "--depth-samples") {
            require_values(i, 1, "--depth-samples");
            options.depth_samples = std::stoi(argv[++i]);
        } else if (arg == "--warmup-frames") {
            require_values(i, 1, "--warmup-frames");
            options.warmup_frames = std::stoi(argv[++i]);
        } else if (arg == "--measured-frames") {
            require_values(i, 1, "--measured-frames");
            options.measured_frames = std::stoi(argv[++i]);
        } else if (arg == "--repeat-id") {
            require_values(i, 1, "--repeat-id");
            options.repeat_id = std::stoi(argv[++i]);
        } else if (arg == "--retention-ratio") {
            require_values(i, 1, "--retention-ratio");
            options.retention_ratio = std::stod(argv[++i]);
        } else if (arg == "--camera") {
            require_values(i, 3, "--camera");
            options.camera_position = make_float3(
                std::stof(argv[++i]),
                std::stof(argv[++i]),
                std::stof(argv[++i])
            );
        } else if (arg == "--rotation") {
            require_values(i, 3, "--rotation");
            options.yaw = std::stof(argv[++i]);
            options.pitch = std::stof(argv[++i]);
            options.roll = std::stof(argv[++i]);
        } else if (arg == "--fov-y") {
            require_values(i, 1, "--fov-y");
            options.fov_y = std::stof(argv[++i]);
        } else if (arg == "--box") {
            require_values(i, 6, "--box");
            options.box.minimum = make_float3(
                std::stof(argv[++i]),
                std::stof(argv[++i]),
                std::stof(argv[++i])
            );
            options.box.maximum = make_float3(
                std::stof(argv[++i]),
                std::stof(argv[++i]),
                std::stof(argv[++i])
            );
        } else {
            throw std::runtime_error(
                "Unknown argument: " + arg
            );
        }
    }

    if (options.input_path.empty() && options.input_list_path.empty()) {
        throw std::runtime_error("--input-list or --input is required.");
    }

    if (options.output_path.empty()) {
        throw std::runtime_error("--output is required.");
    }

    if (options.width <= 0 ||
        options.height <= 0 ||
        options.depth_samples <= 0 ||
        options.warmup_frames < 0 ||
        options.measured_frames <= 0)
    {
        throw std::runtime_error(
            "Invalid image, sample, warm-up, or measured-frame setting."
        );
    }

    if (!(options.retention_ratio > 0.0 &&
          options.retention_ratio <= 1.0))
    {
        throw std::runtime_error(
            "--retention-ratio must be in (0,1]."
        );
    }

    if (!(options.fov_y > 1.0f &&
          options.fov_y < 179.0f))
    {
        throw std::runtime_error(
            "--fov-y must be between 1 and 179 degrees."
        );
    }

    if (!(options.box.maximum.x > options.box.minimum.x &&
          options.box.maximum.y > options.box.minimum.y &&
          options.box.maximum.z > options.box.minimum.z))
    {
        throw std::runtime_error(
            "Invalid box bounds."
        );
    }

    if (options.camera_position.x < options.box.minimum.x ||
        options.camera_position.x > options.box.maximum.x ||
        options.camera_position.y < options.box.minimum.y ||
        options.camera_position.y > options.box.maximum.y ||
        options.camera_position.z < options.box.minimum.z ||
        options.camera_position.z > options.box.maximum.z)
    {
        throw std::runtime_error(
            "Camera position is outside the render box."
        );
    }

    return options;
}

class Renderer {
public:
    Renderer(
        const std::vector<GaussianDisk>& host,
        int width,
        int height,
        int depth_samples,
        CameraGPU camera,
        BoxGPU box)
        : P_(int(host.size())),
          width_(width),
          height_(height),
          depth_samples_(depth_samples),
          tiles_x_(div_up(width, TILE_W)),
          tiles_y_(div_up(height, TILE_H)),
          tile_count_(tiles_x_ * tiles_y_),
          camera_(camera),
          box_(box)
    {
        if (P_ <= 0) {
            throw std::runtime_error(
                "Renderer received zero Gaussians."
            );
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream_,
            cudaStreamNonBlocking
        ));

        CUDA_CHECK(cudaMalloc(
            &d_input_,
            size_t(P_) * sizeof(GaussianDisk)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_gaussians_,
            size_t(P_) * sizeof(GaussianGPU)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_counts_,
            size_t(P_) * sizeof(uint32_t)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_offsets_,
            size_t(P_) * sizeof(uint32_t)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_ranges_,
            size_t(tile_count_) * sizeof(Range)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_output_,
            size_t(width_) * height_ * sizeof(float)
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            d_input_,
            host.data(),
            size_t(P_) * sizeof(GaussianDisk),
            cudaMemcpyHostToDevice,
            stream_
        ));

        rebuild_bins();
    }

    ~Renderer() {
        cudaFree(d_input_);
        cudaFree(d_gaussians_);
        cudaFree(d_counts_);
        cudaFree(d_offsets_);
        cudaFree(d_keys_in_);
        cudaFree(d_keys_out_);
        cudaFree(d_values_in_);
        cudaFree(d_values_out_);
        cudaFree(d_ranges_);
        cudaFree(d_output_);
        cudaFree(d_scan_temp_);
        cudaFree(d_sort_temp_);
        cudaStreamDestroy(stream_);
    }

    void render() {
        const dim3 block(TILE_W, TILE_H);
        const dim3 grid(tiles_x_, tiles_y_);

        render_kernel<<<grid, block, 0, stream_>>>(
            d_gaussians_,
            d_values_out_,
            d_ranges_,
            d_output_,
            width_,
            height_,
            tiles_x_,
            depth_samples_,
            camera_,
            box_,
            MAHAL_CUTOFF
        );

        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t stream() const {
        return stream_;
    }

    uint32_t pair_count() const {
        return pair_count_;
    }

    std::vector<float> download() {
        std::vector<float> output(
            size_t(width_) * height_
        );

        CUDA_CHECK(cudaMemcpyAsync(
            output.data(),
            d_output_,
            output.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_
        ));

        synchronize();
        return output;
    }

private:
    void rebuild_bins() {
        const int threads = 256;
        const int blocks = div_up(P_, threads);

        preprocess_kernel<<<blocks, threads, 0, stream_>>>(
            d_input_,
            d_gaussians_,
            d_counts_,
            P_,
            width_,
            height_,
            tiles_x_,
            tiles_y_,
            camera_,
            MAHAL_CUTOFF
        );

        CUDA_CHECK(cudaGetLastError());

        size_t scan_bytes = 0;

        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr,
            scan_bytes,
            d_counts_,
            d_offsets_,
            P_,
            stream_
        ));

        CUDA_CHECK(cudaMalloc(
            &d_scan_temp_,
            scan_bytes
        ));

        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            d_scan_temp_,
            scan_bytes,
            d_counts_,
            d_offsets_,
            P_,
            stream_
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            &pair_count_,
            d_offsets_ + (P_ - 1),
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream_
        ));

        synchronize();

        if (pair_count_ == 0) {
            throw std::runtime_error(
                "No Gaussian overlaps the camera frustum."
            );
        }

        CUDA_CHECK(cudaMalloc(
            &d_keys_in_,
            size_t(pair_count_) * sizeof(uint32_t)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_keys_out_,
            size_t(pair_count_) * sizeof(uint32_t)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_values_in_,
            size_t(pair_count_) * sizeof(uint32_t)
        ));
        CUDA_CHECK(cudaMalloc(
            &d_values_out_,
            size_t(pair_count_) * sizeof(uint32_t)
        ));

        duplicate_kernel<<<blocks, threads, 0, stream_>>>(
            d_gaussians_,
            d_offsets_,
            d_keys_in_,
            d_values_in_,
            P_,
            tiles_x_
        );

        CUDA_CHECK(cudaGetLastError());

        size_t sort_bytes = 0;

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_bytes,
            d_keys_in_,
            d_keys_out_,
            d_values_in_,
            d_values_out_,
            pair_count_,
            0,
            32,
            stream_
        ));

        CUDA_CHECK(cudaMalloc(
            &d_sort_temp_,
            sort_bytes
        ));

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_temp_,
            sort_bytes,
            d_keys_in_,
            d_keys_out_,
            d_values_in_,
            d_values_out_,
            pair_count_,
            0,
            32,
            stream_
        ));

        CUDA_CHECK(cudaMemsetAsync(
            d_ranges_,
            0,
            size_t(tile_count_) * sizeof(Range),
            stream_
        ));

        identify_ranges_kernel<<<
            div_up(int(pair_count_), threads),
            threads,
            0,
            stream_
        >>>(
            d_keys_out_,
            d_ranges_,
            pair_count_
        );

        CUDA_CHECK(cudaGetLastError());

        synchronize();
    }

    int P_;
    int width_;
    int height_;
    int depth_samples_;
    int tiles_x_;
    int tiles_y_;
    int tile_count_;

    uint32_t pair_count_ = 0;

    CameraGPU camera_;
    BoxGPU box_;
    cudaStream_t stream_{};

    GaussianDisk* d_input_ = nullptr;
    GaussianGPU* d_gaussians_ = nullptr;
    uint32_t* d_counts_ = nullptr;
    uint32_t* d_offsets_ = nullptr;
    uint32_t* d_keys_in_ = nullptr;
    uint32_t* d_keys_out_ = nullptr;
    uint32_t* d_values_in_ = nullptr;
    uint32_t* d_values_out_ = nullptr;
    Range* d_ranges_ = nullptr;
    float* d_output_ = nullptr;
    void* d_scan_temp_ = nullptr;
    void* d_sort_temp_ = nullptr;
};

static std::string make_run_output_path(
    const std::string& base_output_path,
    const std::string& block_id,
    double retention_ratio,
    int repeat_id)
{
    const std::filesystem::path base(base_output_path);
    const std::filesystem::path parent =
        base.has_parent_path() ? base.parent_path() : std::filesystem::path(".");

    std::string stem = base.stem().string();
    if (stem.empty()) stem = "rq1_render";

    std::string extension = base.extension().string();
    if (extension.empty()) extension = ".pfm";

    const int ratio_percent =
        int(std::lround(retention_ratio * 100.0));

    std::ostringstream filename;
    filename
        << stem << "_" << block_id
        << "_ratio_" << std::setw(3) << std::setfill('0') << ratio_percent
        << "_repeat_" << repeat_id
        << extension;

    return (parent / filename.str()).string();
}

int main(int argc, char** argv) {
    try {
        Options base_options = parse_options(argc, argv);

        const std::vector<double> retention_ratios = {
            0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00
        };
        constexpr int repeat_count = 5;
        constexpr uint32_t subset_seed = 42u;

        const std::vector<std::string> input_paths =
            read_input_paths(base_options);

        const int expected_run_count =
            static_cast<int>(input_paths.size()) *
            static_cast<int>(retention_ratios.size()) *
            repeat_count;

        if (!base_options.csv_output.empty()) {
            const std::filesystem::path csv_path(base_options.csv_output);
            if (std::filesystem::exists(csv_path) &&
                std::filesystem::file_size(csv_path) > 0) {
                throw std::runtime_error(
                    "CSV output already exists and is non-empty: " +
                    base_options.csv_output +
                    ". Move or delete it before starting a new sweep."
                );
            }
        }

        int device_index = 0;
        CUDA_CHECK(cudaGetDevice(&device_index));

        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, device_index));

        int driver_version = 0;
        int runtime_version = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        const CameraGPU camera = make_camera(
            base_options.camera_position,
            base_options.yaw,
            base_options.pitch,
            base_options.roll,
            base_options.fov_y,
            base_options.width,
            base_options.height
        );

        std::cout << std::fixed << std::setprecision(6);
        std::cout
            << "RQ1 multi-block payload-scalability benchmark\n"
            << "Blocks: " << input_paths.size() << "\n"
            << "Ratios per block: " << retention_ratios.size() << "\n"
            << "Repeats per ratio: " << repeat_count << "\n"
            << "Total runs: " << expected_run_count << "\n"
            << "Subset method: fixed shuffled nested subsets\n"
            << "Subset seed: " << subset_seed << "\n"
            << "GPU: " << device.name << "\n"
            << "Resolution: " << base_options.width << " x "
            << base_options.height << "\n"
            << "Depth samples: " << base_options.depth_samples << "\n\n";

        int completed_runs = 0;

        for (size_t block_index = 0;
             block_index < input_paths.size();
             ++block_index)
        {
            const std::string& block_path = input_paths[block_index];
            const std::string block_id =
                block_identifier(block_path, block_index);

            const std::vector<GaussianDisk> full =
                read_gaussians(block_path);

            const std::vector<size_t> fixed_order =
                make_fixed_shuffled_order(full.size(), subset_seed);

            std::cout
                << "============================================================\n"
                << "Block " << (block_index + 1)
                << "/" << input_paths.size() << "\n"
                << "Block ID: " << block_id << "\n"
                << "Path: " << block_path << "\n"
                << "Full Gaussian count: " << full.size() << "\n"
                << "============================================================\n";

            for (const double retention_ratio : retention_ratios) {
                const std::vector<GaussianDisk> active =
                    make_nested_subset(
                        full,
                        fixed_order,
                        retention_ratio
                    );

                const size_t active_payload_bytes =
                    active.size() * sizeof(GaussianDisk);

                for (int repeat_id = 0;
                     repeat_id < repeat_count;
                     ++repeat_id)
                {
                    Options run_options = base_options;
                    run_options.input_path = block_path;
                    run_options.retention_ratio = retention_ratio;
                    run_options.repeat_id = repeat_id;
                    run_options.output_path = make_run_output_path(
                        base_options.output_path,
                        block_id,
                        retention_ratio,
                        repeat_id
                    );

                    std::cout
                        << "\nRun " << (completed_runs + 1)
                        << "/" << expected_run_count
                        << " | block=" << block_id
                        << " | ratio=" << retention_ratio
                        << " | repeat=" << repeat_id << "\n";

                    Renderer renderer(
                        active,
                        run_options.width,
                        run_options.height,
                        run_options.depth_samples,
                        camera,
                        run_options.box
                    );

                    for (int frame = 0;
                         frame < run_options.warmup_frames;
                         ++frame) {
                        renderer.render();
                    }
                    renderer.synchronize();

                    cudaEvent_t start{}, stop{};
                    CUDA_CHECK(cudaEventCreate(&start));
                    CUDA_CHECK(cudaEventCreate(&stop));

                    std::vector<double> frame_times_ms;
                    frame_times_ms.reserve(
                        size_t(run_options.measured_frames)
                    );

                    for (int frame = 0;
                         frame < run_options.measured_frames;
                         ++frame)
                    {
                        CUDA_CHECK(cudaEventRecord(
                            start, renderer.stream()
                        ));
                        renderer.render();
                        CUDA_CHECK(cudaEventRecord(
                            stop, renderer.stream()
                        ));
                        CUDA_CHECK(cudaEventSynchronize(stop));

                        float elapsed_ms = 0.0f;
                        CUDA_CHECK(cudaEventElapsedTime(
                            &elapsed_ms, start, stop
                        ));

                        if (!(elapsed_ms > 0.0f) ||
                            !std::isfinite(elapsed_ms)) {
                            throw std::runtime_error(
                                "Encountered invalid CUDA frame time."
                            );
                        }
                        frame_times_ms.push_back(double(elapsed_ms));
                    }

                    CUDA_CHECK(cudaEventDestroy(start));
                    CUDA_CHECK(cudaEventDestroy(stop));

                    std::vector<double> fps_values;
                    fps_values.reserve(frame_times_ms.size());
                    for (double frame_ms : frame_times_ms) {
                        fps_values.push_back(1000.0 / frame_ms);
                    }

                    const SummaryStats time_stats =
                        calculate_stats(frame_times_ms);
                    const SummaryStats fps_stats =
                        calculate_stats(fps_values);

                    renderer.render();
                    const std::vector<float> output =
                        renderer.download();

                    write_pfm(
                        run_options.output_path,
                        output,
                        run_options.width,
                        run_options.height
                    );

                    const auto minmax = std::minmax_element(
                        output.begin(), output.end()
                    );
                    const float output_min = *minmax.first;
                    const float output_max = *minmax.second;

                    append_csv(
                        run_options,
                        block_id,
                        block_path,
                        device,
                        driver_version,
                        runtime_version,
                        full.size(),
                        active.size(),
                        active_payload_bytes,
                        renderer.pair_count(),
                        time_stats,
                        fps_stats,
                        output_min,
                        output_max
                    );

                    std::cout
                        << "Active Gaussians: " << active.size() << "\n"
                        << "Gaussian-tile pairs: "
                        << renderer.pair_count() << "\n"
                        << "Median render time: "
                        << time_stats.median << " ms\n"
                        << "Median FPS: "
                        << fps_stats.median << "\n";

                    ++completed_runs;
                }
            }
        }

        if (completed_runs != expected_run_count) {
            throw std::runtime_error(
                "Unexpected number of completed benchmark runs."
            );
        }

        std::cout
            << "\nRQ1 multi-block sweep completed successfully.\n"
            << "Completed runs: " << completed_runs << "\n"
            << "CSV: " << base_options.csv_output << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
