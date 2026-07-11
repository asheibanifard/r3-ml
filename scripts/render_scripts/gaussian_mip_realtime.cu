
// gaussian_mip_realtime.cu
//
// Standalone CUDA renderer for orthographic Z-axis MIP of a SUMMED 3D
// Gaussian mixture:
//
//   M(x,y) = max_z sum_i intensity_i *
//            exp(-0.5 * (p-mu_i)^T Q_i (p-mu_i))
//
// Pipeline:
//   1. Read compact binary Gaussian checkpoint.
//   2. Build covariance precision matrices on the GPU.
//   3. Conservatively bin each Gaussian into overlapping 16x16 image tiles.
//   4. Prefix-scan tile counts.
//   5. Emit (tile_id, gaussian_id) pairs.
//   6. Radix-sort pairs by tile.
//   7. Identify contiguous Gaussian ranges for every tile.
//   8. Render one CUDA block per tile. Each thread renders one pixel,
//      sums Gaussian density at every Z sample, and keeps the maximum.
//
// Runtime dependencies: CUDA Toolkit only.
// Benchmarking: CUDA events are recorded on the renderer stream, with
// an independent CPU wall-clock sanity check.
// Build:
//   nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
//        gaussian_mip_realtime.cu -o gaussian_mip_realtime
//
// Example:
//   ./gaussian_mip_realtime gaussians.bin output.pfm 128 128 50 200
//
// Arguments:
//   argv[1] gaussian binary input
//   argv[2] output PFM path
//   argv[3] width
//   argv[4] height
//   argv[5] number of Z samples
//   argv[6] benchmark frames, optional (default 100)
//
// Coordinate convention:
//   Gaussian means/scales are in normalized AABB coordinates [-1,1]^3.
//   Pixel centres map exactly to x,y coordinates spanning [-1,1].
//   Depth samples span [-1,1].
//
// Binary format:
//   uint32 magic = 0x47534D50 ("GSMP")
//   uint32 version = 1
//   uint64 P
//   repeated P times:
//       float mean_x, mean_y, mean_z
//       float scale_x, scale_y, scale_z
//       float quat_w, quat_x, quat_y, quat_z
//       float intensity
//
// Notes:
//   * Pure MIP of a Gaussian mixture requires SUM at each depth, then MAX
//     across depth. It is not max over independent projected splats.
//   * Tile binning uses a conservative XY marginal support. Exact 3D
//     Mahalanobis rejection happens in the render kernel.
//   * This reference renderer is optimized but intentionally readable.
//   * For very high Gaussian overlap, use smaller tiles, occupancy hierarchy,
//     or adaptive depth sampling.

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, cudaGetErrorString(err__));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr uint32_t FILE_MAGIC = 0x47534D50u;
constexpr uint32_t FILE_VERSION = 1u;

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int BLOCK_THREADS = TILE_W * TILE_H;
constexpr float MAHAL_CUTOFF = 20.0f;
constexpr float EPS_SCALE = 1e-6f;
constexpr int GAUSSIAN_BATCH = 128;

struct GaussianDisk {
    float mean[3];
    float scale[3];
    float quat[4];  // w, x, y, z
    float intensity;
};

struct GaussianGPU {
    float3 mean;
    // Symmetric precision Q:
    // [q00 q01 q02]
    // [q01 q11 q12]
    // [q02 q12 q22]
    float q00, q01, q02, q11, q12, q22;
    float intensity;

    // Conservative image-space support in pixel coordinates.
    float2 mean_px;
    int radius_px;
    int2 tile_min;
    int2 tile_max; // half-open
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

__host__ __device__ inline int div_up_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ inline float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ inline void quat_to_rotation(
    float w, float x, float y, float z,
    float R[9])
{
    float n2 = w*w + x*x + y*y + z*z;
    float inv_n = rsqrtf(fmaxf(n2, 1e-20f));
    w *= inv_n;
    x *= inv_n;
    y *= inv_n;
    z *= inv_n;

    R[0] = 1.0f - 2.0f*(y*y + z*z);
    R[1] = 2.0f*(x*y - z*w);
    R[2] = 2.0f*(x*z + y*w);

    R[3] = 2.0f*(x*y + z*w);
    R[4] = 1.0f - 2.0f*(x*x + z*z);
    R[5] = 2.0f*(y*z - x*w);

    R[6] = 2.0f*(x*z - y*w);
    R[7] = 2.0f*(y*z + x*w);
    R[8] = 1.0f - 2.0f*(x*x + y*y);
}

__device__ inline bool invert_symmetric_3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float& q00, float& q01, float& q02,
    float& q11, float& q12, float& q22)
{
    const float c00 = a11*a22 - a12*a12;
    const float c01 = a02*a12 - a01*a22;
    const float c02 = a01*a12 - a02*a11;
    const float c11 = a00*a22 - a02*a02;
    const float c12 = a01*a02 - a00*a12;
    const float c22 = a00*a11 - a01*a01;

    const float det = a00*c00 + a01*c01 + a02*c02;
    if (!(det > 1e-20f) || !isfinite(det)) {
        return false;
    }

    const float inv_det = 1.0f / det;
    q00 = c00 * inv_det;
    q01 = c01 * inv_det;
    q02 = c02 * inv_det;
    q11 = c11 * inv_det;
    q12 = c12 * inv_det;
    q22 = c22 * inv_det;
    return true;
}

__global__ void preprocess_gaussians_kernel(
    const GaussianDisk* __restrict__ input,
    GaussianGPU* __restrict__ output,
    uint32_t* __restrict__ tiles_touched,
    int P,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    float mahal_cutoff)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;

    const GaussianDisk g = input[i];

    GaussianGPU out{};
    out.mean = make_float3(g.mean[0], g.mean[1], g.mean[2]);
    out.intensity = fmaxf(g.intensity, 0.0f);

    const float sx = fmaxf(fabsf(g.scale[0]), EPS_SCALE);
    const float sy = fmaxf(fabsf(g.scale[1]), EPS_SCALE);
    const float sz = fmaxf(fabsf(g.scale[2]), EPS_SCALE);

    float R[9];
    quat_to_rotation(g.quat[0], g.quat[1], g.quat[2], g.quat[3], R);

    const float vx = sx*sx;
    const float vy = sy*sy;
    const float vz = sz*sz;

    // Sigma = R diag(v) R^T.
    const float s00 = R[0]*R[0]*vx + R[1]*R[1]*vy + R[2]*R[2]*vz;
    const float s01 = R[0]*R[3]*vx + R[1]*R[4]*vy + R[2]*R[5]*vz;
    const float s02 = R[0]*R[6]*vx + R[1]*R[7]*vy + R[2]*R[8]*vz;
    const float s11 = R[3]*R[3]*vx + R[4]*R[4]*vy + R[5]*R[5]*vz;
    const float s12 = R[3]*R[6]*vx + R[4]*R[7]*vy + R[5]*R[8]*vz;
    const float s22 = R[6]*R[6]*vx + R[7]*R[7]*vy + R[8]*R[8]*vz;

    bool ok = invert_symmetric_3x3(
        s00, s01, s02, s11, s12, s22,
        out.q00, out.q01, out.q02,
        out.q11, out.q12, out.q22
    );

    if (!ok || !isfinite(out.intensity) || out.intensity <= 0.0f) {
        tiles_touched[i] = 0;
        out.radius_px = 0;
        output[i] = out;
        return;
    }

    // Orthographic mapping [-1,1] -> [0, W-1] and [0, H-1].
    const float scale_x = 0.5f * float(max(width - 1, 1));
    const float scale_y = 0.5f * float(max(height - 1, 1));
    const float px = (g.mean[0] + 1.0f) * scale_x;
    const float py = (g.mean[1] + 1.0f) * scale_y;
    out.mean_px = make_float2(px, py);

    // XY marginal covariance is top-left 2x2 of Sigma.
    // Convert from normalized-world variance to pixel variance.
    const float c00 = s00 * scale_x * scale_x;
    const float c01 = s01 * scale_x * scale_y;
    const float c11 = s11 * scale_y * scale_y;

    const float tr_half = 0.5f * (c00 + c11);
    const float disc = sqrtf(fmaxf(
        0.25f * (c00 - c11) * (c00 - c11) + c01*c01,
        0.0f
    ));
    const float lambda_max = fmaxf(tr_half + disc, 0.0f);

    const int radius = int(ceilf(sqrtf(mahal_cutoff * lambda_max))) + 1;
    out.radius_px = radius;

    int min_px_x = int(floorf(px - float(radius)));
    int max_px_x = int(ceilf (px + float(radius)));
    int min_px_y = int(floorf(py - float(radius)));
    int max_px_y = int(ceilf (py + float(radius)));

    int min_tx = min_px_x / TILE_W;
    int max_tx = max_px_x / TILE_W + 1;
    int min_ty = min_px_y / TILE_H;
    int max_ty = max_px_y / TILE_H + 1;

    if (min_px_x < 0) min_tx = -div_up_int(-min_px_x, TILE_W);
    if (min_px_y < 0) min_ty = -div_up_int(-min_px_y, TILE_H);

    min_tx = max(0, min(min_tx, tiles_x));
    max_tx = max(0, min(max_tx, tiles_x));
    min_ty = max(0, min(min_ty, tiles_y));
    max_ty = max(0, min(max_ty, tiles_y));

    out.tile_min = make_int2(min_tx, min_ty);
    out.tile_max = make_int2(max_tx, max_ty);

    const int count_x = max_tx - min_tx;
    const int count_y = max_ty - min_ty;
    const uint32_t count = (count_x > 0 && count_y > 0)
        ? uint32_t(count_x * count_y)
        : 0u;

    if (count == 0u) {
        out.radius_px = 0;
    }

    tiles_touched[i] = count;
    output[i] = out;
}

__global__ void duplicate_with_tile_keys_kernel(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ inclusive_offsets,
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ values,
    int P)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;

    const GaussianGPU g = gaussians[i];
    if (g.radius_px <= 0) return;

    uint32_t write = (i == 0) ? 0u : inclusive_offsets[i - 1];

    for (int ty = g.tile_min.y; ty < g.tile_max.y; ++ty) {
        for (int tx = g.tile_min.x; tx < g.tile_max.x; ++tx) {
            const uint32_t tile_id = uint32_t(ty * gridDim.y + tx);
            // gridDim.y is not the tile-grid width here, so this kernel receives
            // a corrected tile ID in the wrapper variant below.
            keys[write] = tile_id;
            values[write] = uint32_t(i);
            ++write;
        }
    }
}

__global__ void duplicate_with_tile_keys_kernel_fixed(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ inclusive_offsets,
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ values,
    int P,
    int tiles_x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;

    const GaussianGPU g = gaussians[i];
    if (g.radius_px <= 0) return;

    uint32_t write = (i == 0) ? 0u : inclusive_offsets[i - 1];

    for (int ty = g.tile_min.y; ty < g.tile_max.y; ++ty) {
        for (int tx = g.tile_min.x; tx < g.tile_max.x; ++tx) {
            keys[write] = uint32_t(ty * tiles_x + tx);
            values[write] = uint32_t(i);
            ++write;
        }
    }
}

__global__ void identify_ranges_kernel(
    const uint32_t* __restrict__ sorted_keys,
    Range* __restrict__ ranges,
    uint32_t N)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const uint32_t current = sorted_keys[i];

    if (i == 0) {
        ranges[current].begin = 0;
    } else {
        const uint32_t previous = sorted_keys[i - 1];
        if (current != previous) {
            ranges[previous].end = i;
            ranges[current].begin = i;
        }
    }

    if (i == N - 1) {
        ranges[current].end = N;
    }
}

__device__ inline float mahalanobis3(
    const GaussianGPU& g,
    float x, float y, float z)
{
    const float dx = x - g.mean.x;
    const float dy = y - g.mean.y;
    const float dz = z - g.mean.z;

    return
        g.q00*dx*dx +
        2.0f*g.q01*dx*dy +
        2.0f*g.q02*dx*dz +
        g.q11*dy*dy +
        2.0f*g.q12*dy*dz +
        g.q22*dz*dz;
}

__global__ void render_mip_kernel(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ sorted_values,
    const Range* __restrict__ ranges,
    float* __restrict__ output,
    int width,
    int height,
    int tiles_x,
    int num_depth_samples,
    float z_min,
    float z_max,
    float mahal_cutoff)
{
    static_assert(BLOCK_THREADS == 256, "Expected 16x16 tile.");

    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;
    const int tile_id = tile_y * tiles_x + tile_x;

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int linear_tid = local_y * TILE_W + local_x;

    const int pixel_x = tile_x * TILE_W + local_x;
    const int pixel_y = tile_y * TILE_H + local_y;
    const bool inside = pixel_x < width && pixel_y < height;

    const Range range = ranges[tile_id];

    float x_world = 0.0f;
    float y_world = 0.0f;

    if (inside) {
        x_world = (width > 1)
            ? -1.0f + 2.0f * float(pixel_x) / float(width - 1)
            : 0.0f;

        y_world = (height > 1)
            ? -1.0f + 2.0f * float(pixel_y) / float(height - 1)
            : 0.0f;
    }

    float best = 0.0f;

    __shared__ GaussianGPU shared_g[GAUSSIAN_BATCH];

    const float z_step = (num_depth_samples > 1)
        ? (z_max - z_min) / float(num_depth_samples - 1)
        : 0.0f;

    // All threads must execute every __syncthreads(), including threads in
    // partially covered edge tiles. Only arithmetic and output are guarded
    // by `inside`.
    for (int zi = 0; zi < num_depth_samples; ++zi) {
        const float z_world = z_min + float(zi) * z_step;
        float density = 0.0f;

        for (uint32_t batch_begin = range.begin;
             batch_begin < range.end;
             batch_begin += GAUSSIAN_BATCH)
        {
            const uint32_t batch_count = min(
                uint32_t(GAUSSIAN_BATCH),
                range.end - batch_begin
            );

            if (linear_tid < int(batch_count)) {
                const uint32_t gaussian_id =
                    sorted_values[batch_begin + uint32_t(linear_tid)];
                shared_g[linear_tid] = gaussians[gaussian_id];
            }

            __syncthreads();

            if (inside) {
                #pragma unroll 4
                for (uint32_t j = 0; j < batch_count; ++j) {
                    const GaussianGPU g = shared_g[j];
                    const float m = mahalanobis3(
                        g,
                        x_world,
                        y_world,
                        z_world
                    );

                    if (m >= 0.0f && m <= mahal_cutoff) {
                        density += g.intensity * __expf(-0.5f * m);
                    }
                }
            }

            __syncthreads();
        }

        if (inside) {
            best = fmaxf(best, density);
        }
    }

    if (inside) {
        output[pixel_y * width + pixel_x] = best;
    }
}

static std::vector<GaussianDisk> read_gaussians(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not open Gaussian file: " + path);
    }

    FileHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream) {
        throw std::runtime_error("Failed to read Gaussian file header.");
    }
    if (header.magic != FILE_MAGIC) {
        throw std::runtime_error("Invalid Gaussian file magic.");
    }
    if (header.version != FILE_VERSION) {
        throw std::runtime_error("Unsupported Gaussian file version.");
    }
    if (header.count == 0 || header.count > uint64_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Invalid Gaussian count.");
    }

    std::vector<GaussianDisk> data(size_t(header.count));
    stream.read(
        reinterpret_cast<char*>(data.data()),
        std::streamsize(data.size() * sizeof(GaussianDisk))
    );
    if (!stream) {
        throw std::runtime_error("Gaussian file is truncated.");
    }
    return data;
}

static void write_pfm(
    const std::string& path,
    const std::vector<float>& image,
    int width,
    int height)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Could not create output file: " + path);
    }

    stream << "Pf\n" << width << " " << height << "\n-1.0\n";

    // PFM stores rows bottom-to-top.
    for (int y = height - 1; y >= 0; --y) {
        stream.write(
            reinterpret_cast<const char*>(image.data() + size_t(y) * width),
            std::streamsize(width * sizeof(float))
        );
    }
}

class Renderer {
public:
    Renderer(
        const std::vector<GaussianDisk>& host_gaussians,
        int width,
        int height,
        int depth_samples)
        : P_(int(host_gaussians.size())),
          width_(width),
          height_(height),
          depth_samples_(depth_samples),
          tiles_x_(div_up_int(width, TILE_W)),
          tiles_y_(div_up_int(height, TILE_H)),
          tile_count_(tiles_x_ * tiles_y_)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        CUDA_CHECK(cudaMalloc(&d_input_, size_t(P_) * sizeof(GaussianDisk)));
        CUDA_CHECK(cudaMalloc(&d_gaussians_, size_t(P_) * sizeof(GaussianGPU)));
        CUDA_CHECK(cudaMalloc(&d_tiles_touched_, size_t(P_) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_offsets_, size_t(P_) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ranges_, size_t(tile_count_) * sizeof(Range)));
        CUDA_CHECK(cudaMalloc(&d_output_, size_t(width_) * height_ * sizeof(float)));

        CUDA_CHECK(cudaMemcpyAsync(
            d_input_,
            host_gaussians.data(),
            size_t(P_) * sizeof(GaussianDisk),
            cudaMemcpyHostToDevice,
            stream_
        ));

        preprocess_and_bin();
    }

    ~Renderer() {
        cudaFree(d_input_);
        cudaFree(d_gaussians_);
        cudaFree(d_tiles_touched_);
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
        dim3 block(TILE_W, TILE_H);
        dim3 grid(tiles_x_, tiles_y_);

        render_mip_kernel<<<grid, block, 0, stream_>>>(
            d_gaussians_,
            d_values_out_,
            d_ranges_,
            d_output_,
            width_,
            height_,
            tiles_x_,
            depth_samples_,
            -1.0f,
            1.0f,
            MAHAL_CUTOFF
        );
        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    std::vector<float> download() {
        std::vector<float> host(size_t(width_) * height_);
        CUDA_CHECK(cudaMemcpyAsync(
            host.data(),
            d_output_,
            host.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_
        ));
        synchronize();
        return host;
    }

    uint32_t pair_count() const { return pair_count_; }

    cudaStream_t stream() const {
        return stream_;
    }

private:
    void preprocess_and_bin() {
        const int threads = 256;
        const int blocks = div_up_int(P_, threads);

        preprocess_gaussians_kernel<<<blocks, threads, 0, stream_>>>(
            d_input_,
            d_gaussians_,
            d_tiles_touched_,
            P_,
            width_,
            height_,
            tiles_x_,
            tiles_y_,
            MAHAL_CUTOFF
        );
        CUDA_CHECK(cudaGetLastError());

        size_t scan_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr,
            scan_bytes,
            d_tiles_touched_,
            d_offsets_,
            P_,
            stream_
        ));
        CUDA_CHECK(cudaMalloc(&d_scan_temp_, scan_bytes));
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            d_scan_temp_,
            scan_bytes,
            d_tiles_touched_,
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
            throw std::runtime_error("No Gaussian overlaps the image.");
        }

        CUDA_CHECK(cudaMalloc(&d_keys_in_, size_t(pair_count_) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_out_, size_t(pair_count_) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_in_, size_t(pair_count_) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_out_, size_t(pair_count_) * sizeof(uint32_t)));

        duplicate_with_tile_keys_kernel_fixed<<<blocks, threads, 0, stream_>>>(
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
        CUDA_CHECK(cudaMalloc(&d_sort_temp_, sort_bytes));
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

        const int range_blocks = div_up_int(int(pair_count_), threads);
        identify_ranges_kernel<<<range_blocks, threads, 0, stream_>>>(
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

    cudaStream_t stream_{};

    GaussianDisk* d_input_ = nullptr;
    GaussianGPU* d_gaussians_ = nullptr;
    uint32_t* d_tiles_touched_ = nullptr;
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

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 7) {
            std::cerr
                << "Usage: " << argv[0]
                << " gaussians.bin output.pfm width height depth_samples"
                << " [benchmark_frames]\n";
            return EXIT_FAILURE;
        }

        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const int width = std::stoi(argv[3]);
        const int height = std::stoi(argv[4]);
        const int depth_samples = std::stoi(argv[5]);
        const int benchmark_frames = (argc == 7) ? std::stoi(argv[6]) : 100;

        if (width <= 0 || height <= 0 || depth_samples <= 0 ||
            benchmark_frames <= 0) {
            throw std::runtime_error("Dimensions and frame count must be positive.");
        }

        const auto host_gaussians = read_gaussians(input_path);
        std::cout << "Gaussians: " << host_gaussians.size() << "\n";
        std::cout << "Resolution: " << width << "x" << height << "\n";
        std::cout << "Depth samples: " << depth_samples << "\n";

        Renderer renderer(host_gaussians, width, height, depth_samples);

        std::cout << "Gaussian-tile pairs: " << renderer.pair_count() << "\n";

        // Warm-up.
        for (int i = 0; i < 10; ++i) {
            renderer.render();
        }
        renderer.synchronize();

        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Record both events on the same non-blocking stream used by render().
        CUDA_CHECK(cudaEventRecord(start, renderer.stream()));

        for (int i = 0; i < benchmark_frames; ++i) {
            renderer.render();
        }

        CUDA_CHECK(cudaEventRecord(stop, renderer.stream()));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        const float frame_ms =
            total_ms / static_cast<float>(benchmark_frames);

        const float fps =
            (frame_ms > 0.0f)
                ? 1000.0f / frame_ms
                : std::numeric_limits<float>::infinity();

        std::cout << "CUDA event render time: "
                  << frame_ms << " ms\n";

        std::cout << "CUDA event FPS: "
                  << fps << "\n";

        // Independent CPU wall-clock sanity check.
        renderer.synchronize();

        const auto cpu_begin =
            std::chrono::high_resolution_clock::now();

        for (int i = 0; i < benchmark_frames; ++i) {
            renderer.render();
        }

        renderer.synchronize();

        const auto cpu_end =
            std::chrono::high_resolution_clock::now();

        const double cpu_total_ms =
            std::chrono::duration<double, std::milli>(
                cpu_end - cpu_begin
            ).count();

        const double cpu_frame_ms =
            cpu_total_ms / static_cast<double>(benchmark_frames);

        const double cpu_fps =
            (cpu_frame_ms > 0.0)
                ? 1000.0 / cpu_frame_ms
                : std::numeric_limits<double>::infinity();

        std::cout << "CPU wall-clock render time: "
                  << cpu_frame_ms << " ms\n";

        std::cout << "CPU wall-clock FPS: "
                  << cpu_fps << "\n";

        renderer.render();
        auto output = renderer.download();
        write_pfm(output_path, output, width, height);

        auto [minimum, maximum] = std::minmax_element(output.begin(), output.end());
        std::cout << "Output range: [" << *minimum << ", " << *maximum << "]\n";
        std::cout << "Saved: " << output_path << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
