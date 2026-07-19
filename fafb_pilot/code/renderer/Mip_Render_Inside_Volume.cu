// ============================================================================
// simple_gaussian_mip_renderer.cu
// ============================================================================
// A deliberately simple CUDA implementation of a perspective MIP renderer,
// with two interchangeable input representations sharing one camera/box
// convention so they can be rendered and compared directly:
//
//   dense_voxel        - trilinear MIP through a dense voxel grid (ground
//                         truth), via hardware texture sampling.
//   pretrained_gaussian - ray-marched MIP through a mixture of 3D anisotropic
//                         Gaussians (reconstruction).
//
// IMPORTANT:
// The Gaussian path is not the conventional screen-space 3D Gaussian
// Splatting algorithm. It is a ray-marched Gaussian-field renderer:
//
//   1. Build one ray for each output pixel.
//   2. Sample several positions along that ray.
//   3. At every sample position, sum all relevant 3D Gaussian values.
//   4. Keep the maximum summed value along the ray.
//
// Mathematical field:
//
//   F(x) = sum_i amplitude_i * exp(-0.5 * Mahalanobis_i(x))
//
//   Mahalanobis_i(x) = (x - mean_i)^T * inverse_covariance_i * (x - mean_i)
//
// Output image:
//
//   image(pixel) = max_t F(ray_origin + t * ray_direction)
//
// The Gaussian path uses image tiles only as an acceleration structure. Each
// Gaussian is conservatively assigned to every tile that its projected
// support may overlap.
//
// Build example for an RTX 40-series GPU:
//
//   nvcc -O3 -std=c++17 --use_fast_math \
//       -gencode arch=compute_89,code=sm_89 \
//       simple_gaussian_mip_renderer.cu \
//       -o simple_gaussian_mip_renderer
//
// Run:
//
//   ./simple_gaussian_mip_renderer \
//       <dense_voxel|pretrained_gaussian> \
//       input.bin output.pfm \
//       512 512 256 200 \
//       0 0 0 \
//       90 \
//       -1 -1 -1 1 1 1
//
// Arguments:
//   1  Representation: dense_voxel or pretrained_gaussian
//   2  Input binary file (VOXL dense volume, or GSMP Gaussian mixture)
//   3  Output PFM file
//   4  Output width
//   5  Output height
//   6  Number of samples along each ray
//   7  Number of benchmark frames
//   8  Camera yaw in degrees
//   9  Camera pitch in degrees
//   10 Camera roll in degrees
//   11 Vertical field of view in degrees
//   12-14 Box minimum x y z
//   15-17 Box maximum x y z
//
// The camera is placed at the centre of the box.
// ============================================================================

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cfloat>
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

// ============================================================================
// 1. Error checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error_code = (call);                                       \
        if (error_code != cudaSuccess) {                                       \
            std::fprintf(                                                      \
                stderr,                                                        \
                "CUDA error at %s:%d: %s\n",                                  \
                __FILE__,                                                      \
                __LINE__,                                                      \
                cudaGetErrorString(error_code));                               \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// 2. Renderer constants
// ============================================================================

// Each CUDA block renders one 16 x 16 image tile.
constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 16;
constexpr int THREADS_PER_TILE = TILE_WIDTH * TILE_HEIGHT;

// Number of Gaussians copied to shared memory at once.
constexpr int GAUSSIANS_PER_BATCH = 128;

// We ignore Gaussian contributions outside this Mahalanobis distance.
// sqrt(20) is approximately 4.47 standard deviations.
constexpr float MAHALANOBIS_CUTOFF = 20.0f;

// Avoid zero Gaussian scales and singular covariance matrices.
constexpr float MINIMUM_SCALE = 1.0e-6f;

// Avoid projection and ray-start singularities at t = 0.
constexpr float CAMERA_NEAR_DISTANCE = 1.0e-4f;

// Binary file header.
constexpr uint32_t GAUSSIAN_FILE_MAGIC = 0x47534D50u;
constexpr uint32_t GAUSSIAN_FILE_VERSION = 1u;

// ============================================================================
// 3. Input and GPU data structures
// ============================================================================

// One record exactly as it appears in the binary input file.
struct GaussianDisk {
    float mean[3];       // Gaussian centre: x, y, z
    float scale[3];      // Standard deviations along local axes
    float quaternion[4]; // Rotation quaternion in w, x, y, z order
    float amplitude;     // Non-negative Gaussian amplitude
};

// Header stored before the Gaussian records.
struct GaussianFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t count;
};

// Data used during rendering.
struct GaussianGPU {
    float3 mean;

    // Symmetric inverse covariance matrix Q = Sigma^{-1}.
    // Only six unique values are required.
    float q00;
    float q01;
    float q02;
    float q11;
    float q12;
    float q22;

    float amplitude;

    // Half-open tile rectangle:
    // [tile_min.x, tile_max.x) x [tile_min.y, tile_max.y)
    int2 tile_min;
    int2 tile_max;

    int visible;
};

// The sorted Gaussian list belonging to one tile is stored in this range.
struct TileRange {
    uint32_t begin;
    uint32_t end;
};

struct Camera {
    float3 position;
    float3 right;
    float3 up;
    float3 forward;

    float tangent_half_vertical_fov;
    float aspect_ratio;
};

struct AxisAlignedBox {
    float3 minimum;
    float3 maximum;
};

// ============================================================================
// 4. Small vector helpers
// ============================================================================

__host__ __device__ inline int divide_round_up(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

__host__ __device__ inline float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 multiply(float3 vector, float scalar) {
    return make_float3(
        vector.x * scalar,
        vector.y * scalar,
        vector.z * scalar);
}

__host__ __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float3 normalize(float3 vector) {
    const float squared_length = dot(vector, vector);
    const float inverse_length = rsqrtf(fmaxf(squared_length, 1.0e-20f));
    return multiply(vector, inverse_length);
}

// ============================================================================
// 5. Quaternion, covariance, and matrix helpers
// ============================================================================

// Convert quaternion w,x,y,z to a row-major 3 x 3 rotation matrix.
__device__ inline void quaternion_to_rotation_matrix(
    float w,
    float x,
    float y,
    float z,
    float rotation[9])
{
    // Normalise the quaternion first.
    const float inverse_length = rsqrtf(
        fmaxf(w * w + x * x + y * y + z * z, 1.0e-20f));

    w *= inverse_length;
    x *= inverse_length;
    y *= inverse_length;
    z *= inverse_length;

    rotation[0] = 1.0f - 2.0f * (y * y + z * z);
    rotation[1] = 2.0f * (x * y - z * w);
    rotation[2] = 2.0f * (x * z + y * w);

    rotation[3] = 2.0f * (x * y + z * w);
    rotation[4] = 1.0f - 2.0f * (x * x + z * z);
    rotation[5] = 2.0f * (y * z - x * w);

    rotation[6] = 2.0f * (x * z - y * w);
    rotation[7] = 2.0f * (y * z + x * w);
    rotation[8] = 1.0f - 2.0f * (x * x + y * y);
}

// Invert a symmetric 3 x 3 matrix.
//
// Input matrix:
//
//   [ a00 a01 a02 ]
//   [ a01 a11 a12 ]
//   [ a02 a12 a22 ]
//
// Output matrix uses the same six-value representation.
__device__ inline bool invert_symmetric_3x3(
    float a00,
    float a01,
    float a02,
    float a11,
    float a12,
    float a22,
    float& inverse00,
    float& inverse01,
    float& inverse02,
    float& inverse11,
    float& inverse12,
    float& inverse22)
{
    const float cofactor00 = a11 * a22 - a12 * a12;
    const float cofactor01 = a02 * a12 - a01 * a22;
    const float cofactor02 = a01 * a12 - a02 * a11;
    const float cofactor11 = a00 * a22 - a02 * a02;
    const float cofactor12 = a01 * a02 - a00 * a12;
    const float cofactor22 = a00 * a11 - a01 * a01;

    const float determinant =
        a00 * cofactor00 +
        a01 * cofactor01 +
        a02 * cofactor02;

    if (!(determinant > 1.0e-20f) || !isfinite(determinant)) {
        return false;
    }

    const float inverse_determinant = 1.0f / determinant;

    inverse00 = cofactor00 * inverse_determinant;
    inverse01 = cofactor01 * inverse_determinant;
    inverse02 = cofactor02 * inverse_determinant;
    inverse11 = cofactor11 * inverse_determinant;
    inverse12 = cofactor12 * inverse_determinant;
    inverse22 = cofactor22 * inverse_determinant;

    return true;
}

// Conservative upper bound for the largest covariance eigenvalue.
// This uses the Gershgorin circle theorem.
__device__ inline float largest_eigenvalue_upper_bound(
    float covariance00,
    float covariance01,
    float covariance02,
    float covariance11,
    float covariance12,
    float covariance22)
{
    const float row0 =
        covariance00 + fabsf(covariance01) + fabsf(covariance02);

    const float row1 =
        covariance11 + fabsf(covariance01) + fabsf(covariance12);

    const float row2 =
        covariance22 + fabsf(covariance02) + fabsf(covariance12);

    return fmaxf(row0, fmaxf(row1, row2));
}

// ============================================================================
// 6. Gaussian preprocessing kernel
// ============================================================================
//
// For every Gaussian, this kernel:
//
//   1. Builds its covariance matrix.
//   2. Inverts the covariance matrix.
//   3. Computes a conservative support sphere.
//   4. Projects that sphere into the image.
//   5. Stores the rectangle of overlapping image tiles.
//   6. Stores how many Gaussian-tile pairs must later be created.
// ============================================================================

__global__ void preprocess_gaussians_kernel(
    const GaussianDisk* input_gaussians,
    GaussianGPU* output_gaussians,
    uint32_t* tile_pair_counts,
    int gaussian_count,
    int image_width,
    int image_height,
    int tiles_x,
    int tiles_y,
    Camera camera,
    bool hard_gate)
{
    const int gaussian_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (gaussian_index >= gaussian_count) {
        return;
    }

    const GaussianDisk input = input_gaussians[gaussian_index];

    GaussianGPU gaussian{};
    gaussian.mean = make_float3(
        input.mean[0],
        input.mean[1],
        input.mean[2]);

    // This implementation assumes non-negative density/intensity amplitudes.
    gaussian.amplitude = fmaxf(input.amplitude, 0.0f);

    const float scale_x =
        fmaxf(fabsf(input.scale[0]), MINIMUM_SCALE);
    const float scale_y =
        fmaxf(fabsf(input.scale[1]), MINIMUM_SCALE);
    const float scale_z =
        fmaxf(fabsf(input.scale[2]), MINIMUM_SCALE);

    float rotation[9];
    quaternion_to_rotation_matrix(
        input.quaternion[0],
        input.quaternion[1],
        input.quaternion[2],
        input.quaternion[3],
        rotation);

    // Local diagonal covariance values.
    const float variance_x = scale_x * scale_x;
    const float variance_y = scale_y * scale_y;
    const float variance_z = scale_z * scale_z;

    // Build world-space covariance:
    // Sigma = R * diag(sx^2, sy^2, sz^2) * R^T
    const float covariance00 =
        rotation[0] * rotation[0] * variance_x +
        rotation[1] * rotation[1] * variance_y +
        rotation[2] * rotation[2] * variance_z;

    const float covariance01 =
        rotation[0] * rotation[3] * variance_x +
        rotation[1] * rotation[4] * variance_y +
        rotation[2] * rotation[5] * variance_z;

    const float covariance02 =
        rotation[0] * rotation[6] * variance_x +
        rotation[1] * rotation[7] * variance_y +
        rotation[2] * rotation[8] * variance_z;

    const float covariance11 =
        rotation[3] * rotation[3] * variance_x +
        rotation[4] * rotation[4] * variance_y +
        rotation[5] * rotation[5] * variance_z;

    const float covariance12 =
        rotation[3] * rotation[6] * variance_x +
        rotation[4] * rotation[7] * variance_y +
        rotation[5] * rotation[8] * variance_z;

    const float covariance22 =
        rotation[6] * rotation[6] * variance_x +
        rotation[7] * rotation[7] * variance_y +
        rotation[8] * rotation[8] * variance_z;

    const bool inverse_is_valid = invert_symmetric_3x3(
        covariance00,
        covariance01,
        covariance02,
        covariance11,
        covariance12,
        covariance22,
        gaussian.q00,
        gaussian.q01,
        gaussian.q02,
        gaussian.q11,
        gaussian.q12,
        gaussian.q22);

    if (!inverse_is_valid ||
        gaussian.amplitude <= 0.0f ||
        !isfinite(gaussian.amplitude)) {
        gaussian.visible = 0;
        tile_pair_counts[gaussian_index] = 0;
        output_gaussians[gaussian_index] = gaussian;
        return;
    }

    // Express the Gaussian centre in camera coordinates.
    const float3 relative_to_camera =
        subtract(gaussian.mean, camera.position);

    const float camera_x = dot(relative_to_camera, camera.right);
    const float camera_y = dot(relative_to_camera, camera.up);
    const float camera_z = dot(relative_to_camera, camera.forward);

    // Convert the ellipsoidal support to a conservative containing sphere.
    const float maximum_eigenvalue_bound = fmaxf(
        largest_eigenvalue_upper_bound(
            covariance00,
            covariance01,
            covariance02,
            covariance11,
            covariance12,
            covariance22),
        0.0f);

    const float support_radius = sqrtf(
        MAHALANOBIS_CUTOFF * maximum_eigenvalue_bound);

    // Default rectangle is empty.
    int minimum_tile_x = 0;
    int maximum_tile_x = 0;
    int minimum_tile_y = 0;
    int maximum_tile_y = 0;

    // If the complete support lies behind the camera, discard it.
    if (camera_z + support_radius <= CAMERA_NEAR_DISTANCE) {
        // Keep the empty rectangle.
    }
    // If the support intersects the near plane, use all tiles.
    // This is expensive, but avoids false-negative culling.
    else if (camera_z - support_radius <= CAMERA_NEAR_DISTANCE) {
        minimum_tile_x = 0;
        maximum_tile_x = tiles_x;
        minimum_tile_y = 0;
        maximum_tile_y = tiles_y;

        // Hard-gated multi-block scenes: the camera sits at the box centre,
        // so a ray's world-space octant is fixed by its own direction sign
        // for its whole length (see same_octant() and the hard-gating
        // comments above render_gaussian_mip_kernel). With an axis-aligned
        // camera (camera.right/up equal to two world axes -- true whenever
        // this renderer is invoked with yaw=pitch=roll=0, its only actual
        // use), camera_x/camera_y ARE the Gaussian's world X/Y coordinates,
        // so only the matching screen half can ever reach it: restrict to
        // that half here too, same as the normal-case branch below (pixel
        // half-boundary, converted via the same /TILE_WIDTH,/TILE_HEIGHT
        // truncation as everywhere else, rather than assuming tiles_x/y are
        // even).
        if (hard_gate) {
            const int half_pixel_x = image_width / 2;
            const int half_pixel_y = image_height / 2;
            const int half_tile_x = half_pixel_x / TILE_WIDTH;
            const int half_tile_y = half_pixel_y / TILE_HEIGHT;
            if (camera_x < 0.0f) {
                maximum_tile_x = min(maximum_tile_x, half_tile_x + 1);
            } else {
                minimum_tile_x = max(minimum_tile_x, half_tile_x);
            }
            if (camera_y >= 0.0f) {
                maximum_tile_y = min(maximum_tile_y, half_tile_y + 1);
            } else {
                minimum_tile_y = max(minimum_tile_y, half_tile_y);
            }
        }
    }
    else {
        // Conservative perspective bounds for the containing sphere.
        //
        // The sphere's screen-space footprint is bounded by projecting the
        // axis-aligned box [camera_x ± r, camera_y ± r, nearest_z, farthest_z]
        // that contains it (a superset of the sphere, so still conservative).
        // Projecting only at nearest_z is NOT safe in general: for an offset
        // extreme that keeps the same sign as the centre coordinate (e.g.
        // camera_y - r and camera_y + r both negative), the true worst-case
        // ratio for the *other* bound is achieved at farthest_z, not
        // nearest_z -- pairing every offset extreme with nearest_z alone can
        // under-estimate that bound and silently drop real coverage (visible
        // as missing/dim Gaussians near tile edges). Evaluating both depths
        // for every offset extreme and taking the true min/max fixes this.
        const float nearest_z =
            fmaxf(camera_z - support_radius, CAMERA_NEAR_DISTANCE);
        const float farthest_z =
            fmaxf(camera_z + support_radius, CAMERA_NEAR_DISTANCE);

        const float horizontal_denominator_near =
            nearest_z *
            camera.tangent_half_vertical_fov *
            camera.aspect_ratio;
        const float horizontal_denominator_far =
            farthest_z *
            camera.tangent_half_vertical_fov *
            camera.aspect_ratio;

        const float vertical_denominator_near =
            nearest_z *
            camera.tangent_half_vertical_fov;
        const float vertical_denominator_far =
            farthest_z *
            camera.tangent_half_vertical_fov;

        const float x_offset_low = camera_x - support_radius;
        const float x_offset_high = camera_x + support_radius;
        const float y_offset_low = camera_y - support_radius;
        const float y_offset_high = camera_y + support_radius;

        const float minimum_ndc_x = fminf(
            fminf(x_offset_low / horizontal_denominator_near, x_offset_low / horizontal_denominator_far),
            fminf(x_offset_high / horizontal_denominator_near, x_offset_high / horizontal_denominator_far));
        const float maximum_ndc_x = fmaxf(
            fmaxf(x_offset_low / horizontal_denominator_near, x_offset_low / horizontal_denominator_far),
            fmaxf(x_offset_high / horizontal_denominator_near, x_offset_high / horizontal_denominator_far));

        const float minimum_ndc_y = fminf(
            fminf(y_offset_low / vertical_denominator_near, y_offset_low / vertical_denominator_far),
            fminf(y_offset_high / vertical_denominator_near, y_offset_high / vertical_denominator_far));
        const float maximum_ndc_y = fmaxf(
            fmaxf(y_offset_low / vertical_denominator_near, y_offset_low / vertical_denominator_far),
            fmaxf(y_offset_high / vertical_denominator_near, y_offset_high / vertical_denominator_far));

        // Convert NDC bounds to pixel-space bounds.
        // Image y increases downward, so y formulas are reversed.
        const float minimum_pixel_x_float =
            (minimum_ndc_x * 0.5f + 0.5f) * float(image_width);
        const float maximum_pixel_x_float =
            (maximum_ndc_x * 0.5f + 0.5f) * float(image_width);

        const float minimum_pixel_y_float =
            (0.5f - maximum_ndc_y * 0.5f) * float(image_height);
        const float maximum_pixel_y_float =
            (0.5f - minimum_ndc_y * 0.5f) * float(image_height);

        int minimum_pixel_x =
            int(floorf(minimum_pixel_x_float)) - 1;
        int maximum_pixel_x =
            int(ceilf(maximum_pixel_x_float)) + 1;
        int minimum_pixel_y =
            int(floorf(minimum_pixel_y_float)) - 1;
        int maximum_pixel_y =
            int(ceilf(maximum_pixel_y_float)) + 1;

        // Hard-gated multi-block scenes: restrict to the screen half that
        // can actually reach this Gaussian's own world-space X/Y octant --
        // see the identical comment on the near-plane-straddle branch above
        // for the full reasoning. This is what turns "stage every candidate
        // Gaussian from all blocks, then discard 7/8 of them per sample"
        // into "only stage Gaussians whose octant this tile's rays can
        // reach in the first place."
        if (hard_gate) {
            const int half_pixel_x = image_width / 2;
            const int half_pixel_y = image_height / 2;
            if (camera_x < 0.0f) {
                maximum_pixel_x = min(maximum_pixel_x, half_pixel_x);
            } else {
                minimum_pixel_x = max(minimum_pixel_x, half_pixel_x);
            }
            if (camera_y >= 0.0f) {
                maximum_pixel_y = min(maximum_pixel_y, half_pixel_y);
            } else {
                minimum_pixel_y = max(minimum_pixel_y, half_pixel_y);
            }
        }

        // Reject the Gaussian if the full projected bound misses the image.
        const bool overlaps_image =
            maximum_pixel_x >= 0 &&
            minimum_pixel_x < image_width &&
            maximum_pixel_y >= 0 &&
            minimum_pixel_y < image_height;

        if (overlaps_image) {
            // Convert inclusive pixel bounds to a half-open tile rectangle.
            minimum_tile_x = max(
                0,
                min(minimum_pixel_x / TILE_WIDTH, tiles_x));

            maximum_tile_x = max(
                0,
                min(maximum_pixel_x / TILE_WIDTH + 1, tiles_x));

            minimum_tile_y = max(
                0,
                min(minimum_pixel_y / TILE_HEIGHT, tiles_y));

            maximum_tile_y = max(
                0,
                min(maximum_pixel_y / TILE_HEIGHT + 1, tiles_y));
        }
    }

    gaussian.tile_min = make_int2(minimum_tile_x, minimum_tile_y);
    gaussian.tile_max = make_int2(maximum_tile_x, maximum_tile_y);

    const int covered_tiles_x = maximum_tile_x - minimum_tile_x;
    const int covered_tiles_y = maximum_tile_y - minimum_tile_y;

    const uint32_t pair_count =
        covered_tiles_x > 0 && covered_tiles_y > 0
            ? uint32_t(covered_tiles_x * covered_tiles_y)
            : 0u;

    gaussian.visible = pair_count > 0 ? 1 : 0;

    tile_pair_counts[gaussian_index] = pair_count;
    output_gaussians[gaussian_index] = gaussian;
}

// ============================================================================
// 7. Duplicate every Gaussian once for every overlapping tile
// ============================================================================
//
// Example:
//
//   Gaussian 7 overlaps tiles 3, 4, and 8.
//
// We write:
//
//   keys   = [3, 4, 8]
//   values = [7, 7, 7]
//
// The key/value pairs are sorted by tile ID afterward.
// ============================================================================

__global__ void create_gaussian_tile_pairs_kernel(
    const GaussianGPU* gaussians,
    const uint32_t* inclusive_offsets,
    uint32_t* tile_keys,
    uint32_t* gaussian_indices,
    int gaussian_count,
    int tiles_x)
{
    const int gaussian_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (gaussian_index >= gaussian_count) {
        return;
    }

    const GaussianGPU gaussian = gaussians[gaussian_index];

    if (!gaussian.visible) {
        return;
    }

    // Inclusive scan stores the end offset for each Gaussian.
    // Therefore, the beginning is the previous Gaussian's end.
    uint32_t write_index =
        gaussian_index == 0
            ? 0u
            : inclusive_offsets[gaussian_index - 1];

    for (int tile_y = gaussian.tile_min.y;
         tile_y < gaussian.tile_max.y;
         ++tile_y) {
        for (int tile_x = gaussian.tile_min.x;
             tile_x < gaussian.tile_max.x;
             ++tile_x) {
            const uint32_t tile_id =
                uint32_t(tile_y * tiles_x + tile_x);

            tile_keys[write_index] = tile_id;
            gaussian_indices[write_index] = uint32_t(gaussian_index);
            ++write_index;
        }
    }
}

// ============================================================================
// 8. Identify the sorted list range belonging to each tile
// ============================================================================

__global__ void identify_tile_ranges_kernel(
    const uint32_t* sorted_tile_keys,
    TileRange* tile_ranges,
    uint32_t pair_count)
{
    const uint32_t pair_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (pair_index >= pair_count) {
        return;
    }

    const uint32_t current_tile = sorted_tile_keys[pair_index];

    if (pair_index == 0) {
        tile_ranges[current_tile].begin = 0;
    }
    else {
        const uint32_t previous_tile =
            sorted_tile_keys[pair_index - 1];

        if (current_tile != previous_tile) {
            tile_ranges[previous_tile].end = pair_index;
            tile_ranges[current_tile].begin = pair_index;
        }
    }

    if (pair_index == pair_count - 1) {
        tile_ranges[current_tile].end = pair_count;
    }
}

// ============================================================================
// 9. Ray-box intersection
// ============================================================================

__device__ inline bool intersect_ray_with_box(
    float3 ray_origin,
    float3 ray_direction,
    AxisAlignedBox box,
    float& entry_distance,
    float& exit_distance)
{
    float near_distance = -FLT_MAX;
    float far_distance = FLT_MAX;

    const float origin[3] = {
        ray_origin.x,
        ray_origin.y,
        ray_origin.z
    };

    const float direction[3] = {
        ray_direction.x,
        ray_direction.y,
        ray_direction.z
    };

    const float minimum[3] = {
        box.minimum.x,
        box.minimum.y,
        box.minimum.z
    };

    const float maximum[3] = {
        box.maximum.x,
        box.maximum.y,
        box.maximum.z
    };

    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(direction[axis]) < 1.0e-12f) {
            // Ray is parallel to this pair of box planes.
            if (origin[axis] < minimum[axis] ||
                origin[axis] > maximum[axis]) {
                return false;
            }
        }
        else {
            const float inverse_direction = 1.0f / direction[axis];

            float distance0 =
                (minimum[axis] - origin[axis]) * inverse_direction;
            float distance1 =
                (maximum[axis] - origin[axis]) * inverse_direction;

            if (distance0 > distance1) {
                const float temporary = distance0;
                distance0 = distance1;
                distance1 = temporary;
            }

            near_distance = fmaxf(near_distance, distance0);
            far_distance = fminf(far_distance, distance1);

            if (far_distance < near_distance) {
                return false;
            }
        }
    }

    entry_distance = fmaxf(near_distance, 0.0f);
    exit_distance = far_distance;

    return exit_distance > entry_distance;
}

// ============================================================================
// 10. Evaluate one Gaussian at one 3D point
// ============================================================================

__device__ inline float gaussian_value_at_point(
    const GaussianGPU& gaussian,
    float3 point)
{
    const float dx = point.x - gaussian.mean.x;
    const float dy = point.y - gaussian.mean.y;
    const float dz = point.z - gaussian.mean.z;

    const float mahalanobis_distance =
        gaussian.q00 * dx * dx +
        2.0f * gaussian.q01 * dx * dy +
        2.0f * gaussian.q02 * dx * dz +
        gaussian.q11 * dy * dy +
        2.0f * gaussian.q12 * dy * dz +
        gaussian.q22 * dz * dz;

    if (mahalanobis_distance < 0.0f ||
        mahalanobis_distance > MAHALANOBIS_CUTOFF) {
        return 0.0f;
    }

    return gaussian.amplitude *
           __expf(-0.5f * mahalanobis_distance);
}

// ============================================================================
// 11. Main rendering kernel
// ============================================================================
//
// One CUDA block renders one image tile.
// One CUDA thread renders one pixel.
//
// Every tile reads only the Gaussians that preprocessing associated with it.
// Gaussians are copied to shared memory in small batches.
// ============================================================================

// Hard position gating for multi-block stitched scenes: a Gaussian only
// contributes to a sample point if both lie in the same octant relative to
// the box centre (matching the [-1,0]/[0,1] per-axis convention used to
// remap each block into the shared frame). Both inputs are continuous
// floats (a Gaussian's fitted mean, a ray-marched sample position), never
// snapped to a discrete grid, so an exact tie at 0.0 is a measure-zero event
// -- unlike gating a discrete voxel-index grid (where the low octant's last
// index and the high octant's first index collide at exactly 0.0), this is
// safe without any extra bookkeeping.
__device__ inline bool same_octant(float3 a, float3 b) {
    return (a.x >= 0.0f) == (b.x >= 0.0f) &&
           (a.y >= 0.0f) == (b.y >= 0.0f) &&
           (a.z >= 0.0f) == (b.z >= 0.0f);
}

__global__ void render_gaussian_mip_kernel(
    const GaussianGPU* gaussians,
    const uint32_t* sorted_gaussian_indices,
    const TileRange* tile_ranges,
    float* output_image,
    int image_width,
    int image_height,
    int tiles_x,
    int ray_sample_count,
    Camera camera,
    AxisAlignedBox box,
    bool hard_gate)
{
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;

    const int thread_linear_index =
        local_y * TILE_WIDTH + local_x;

    const int pixel_x =
        blockIdx.x * TILE_WIDTH + local_x;

    const int pixel_y =
        blockIdx.y * TILE_HEIGHT + local_y;

    const bool pixel_is_inside_image =
        pixel_x < image_width && pixel_y < image_height;

    const int tile_id =
        blockIdx.y * tiles_x + blockIdx.x;

    const TileRange tile_range = tile_ranges[tile_id];

    float3 ray_direction = make_float3(0.0f, 0.0f, 1.0f);
    float ray_entry_distance = 0.0f;
    float ray_exit_distance = 0.0f;
    bool ray_is_valid = false;

    // ------------------------------------------------------------------------
    // Step 1: construct the perspective ray for this pixel.
    // ------------------------------------------------------------------------
    if (pixel_is_inside_image) {
        const float ndc_x =
            2.0f * (float(pixel_x) + 0.5f) /
            float(image_width) - 1.0f;

        const float ndc_y =
            1.0f -
            2.0f * (float(pixel_y) + 0.5f) /
            float(image_height);

        const float camera_plane_x =
            ndc_x *
            camera.aspect_ratio *
            camera.tangent_half_vertical_fov;

        const float camera_plane_y =
            ndc_y * camera.tangent_half_vertical_fov;

        ray_direction = normalize(
            add(
                camera.forward,
                add(
                    multiply(camera.right, camera_plane_x),
                    multiply(camera.up, camera_plane_y))));

        ray_is_valid = intersect_ray_with_box(
            camera.position,
            ray_direction,
            box,
            ray_entry_distance,
            ray_exit_distance);

        ray_entry_distance = fmaxf(
            ray_entry_distance,
            CAMERA_NEAR_DISTANCE);
    }

    // Shared memory holds one Gaussian batch for the whole tile.
    __shared__ GaussianGPU shared_gaussians[GAUSSIANS_PER_BATCH];

    float maximum_density = 0.0f;

    // ------------------------------------------------------------------------
    // Step 2: sample positions along the ray.
    // ------------------------------------------------------------------------
    for (int sample_index = 0;
         sample_index < ray_sample_count;
         ++sample_index) {
        float3 sample_point = make_float3(0.0f, 0.0f, 0.0f);

        if (pixel_is_inside_image && ray_is_valid) {
            const float interpolation =
                ray_sample_count > 1
                    ? float(sample_index) /
                      float(ray_sample_count - 1)
                    : 0.5f;

            const float distance =
                ray_entry_distance +
                (ray_exit_distance - ray_entry_distance) * interpolation;

            sample_point = add(
                camera.position,
                multiply(ray_direction, distance));
        }

        float density_at_sample = 0.0f;

        // --------------------------------------------------------------------
        // Step 3: sum every relevant Gaussian at this sample point.
        // --------------------------------------------------------------------
        for (uint32_t batch_begin = tile_range.begin;
             batch_begin < tile_range.end;
             batch_begin += GAUSSIANS_PER_BATCH) {
            const uint32_t batch_count = min(
                uint32_t(GAUSSIANS_PER_BATCH),
                tile_range.end - batch_begin);

            // Cooperatively copy one batch to shared memory.
            if (thread_linear_index < int(batch_count)) {
                const uint32_t gaussian_index =
                    sorted_gaussian_indices[
                        batch_begin + thread_linear_index];

                shared_gaussians[thread_linear_index] =
                    gaussians[gaussian_index];
            }

            __syncthreads();

            if (pixel_is_inside_image && ray_is_valid) {
                for (uint32_t batch_index = 0;
                     batch_index < batch_count;
                     ++batch_index) {
                    if (hard_gate &&
                        !same_octant(
                            sample_point,
                            shared_gaussians[batch_index].mean)) {
                        continue;
                    }
                    density_at_sample += gaussian_value_at_point(
                        shared_gaussians[batch_index],
                        sample_point);
                }
            }

            __syncthreads();
        }

        // --------------------------------------------------------------------
        // Step 4: MIP keeps the largest summed density along the ray.
        // --------------------------------------------------------------------
        if (pixel_is_inside_image && ray_is_valid) {
            maximum_density = fmaxf(
                maximum_density,
                density_at_sample);
        }
    }

    if (pixel_is_inside_image) {
        // Unlike the dense-voxel texture (normalised to [0, 1] by
        // construction), this field is an unbounded sum of overlapping
        // Gaussians and can locally exceed 1. Clamp to the training-time
        // intensity range so the output is on the same scale as the
        // dense-voxel MIP it gets compared against.
        output_image[pixel_y * image_width + pixel_x] =
            fminf(fmaxf(maximum_density, 0.0f), 1.0f);
    }
}

// ============================================================================
// 12. File input and output
// ============================================================================

static std::vector<GaussianDisk> read_gaussian_file(
    const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);

    if (!stream) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    GaussianFileHeader header{};
    stream.read(
        reinterpret_cast<char*>(&header),
        sizeof(header));

    if (!stream ||
        header.magic != GAUSSIAN_FILE_MAGIC ||
        header.version != GAUSSIAN_FILE_VERSION) {
        throw std::runtime_error("Invalid Gaussian binary header.");
    }

    if (header.count == 0 ||
        header.count > uint64_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Invalid Gaussian count.");
    }

    std::vector<GaussianDisk> gaussians(
        static_cast<size_t>(header.count));

    stream.read(
        reinterpret_cast<char*>(gaussians.data()),
        std::streamsize(gaussians.size() * sizeof(GaussianDisk)));

    if (!stream) {
        throw std::runtime_error("Truncated Gaussian binary file.");
    }

    return gaussians;
}

static void write_pfm_file(
    const std::string& path,
    const std::vector<float>& image,
    int width,
    int height)
{
    std::ofstream stream(path, std::ios::binary);

    if (!stream) {
        throw std::runtime_error("Cannot create output file: " + path);
    }

    // "Pf" means one-channel floating-point image.
    // Negative scale means little-endian data.
    stream << "Pf\n" << width << " " << height << "\n-1.0\n";

    // PFM convention stores rows from bottom to top.
    for (int y = height - 1; y >= 0; --y) {
        stream.write(
            reinterpret_cast<const char*>(
                image.data() + size_t(y) * width),
            std::streamsize(width * sizeof(float)));
    }
}

// ============================================================================
// 13. Camera construction
// ============================================================================

static float degrees_to_radians(float degrees) {
    return degrees * 3.14159265358979323846f / 180.0f;
}

static Camera create_camera(
    float3 position,
    float yaw_degrees,
    float pitch_degrees,
    float roll_degrees,
    float vertical_fov_degrees,
    int image_width,
    int image_height)
{
    const float yaw = degrees_to_radians(yaw_degrees);
    const float pitch = degrees_to_radians(pitch_degrees);
    const float roll = degrees_to_radians(roll_degrees);

    // At zero rotation, the camera looks toward +Z.
    float3 forward = make_float3(
        sinf(yaw) * cosf(pitch),
        sinf(pitch),
        cosf(yaw) * cosf(pitch));

    forward = normalize(forward);

    const float3 world_up = make_float3(0.0f, 1.0f, 0.0f);

    float3 right = normalize(cross(world_up, forward));
    float3 up = normalize(cross(forward, right));

    // Apply roll around the forward axis.
    const float cosine_roll = cosf(roll);
    const float sine_roll = sinf(roll);

    const float3 rolled_right = add(
        multiply(right, cosine_roll),
        multiply(up, sine_roll));

    const float3 rolled_up = add(
        multiply(up, cosine_roll),
        multiply(right, -sine_roll));

    Camera camera{};
    camera.position = position;
    camera.right = normalize(rolled_right);
    camera.up = normalize(rolled_up);
    camera.forward = forward;
    camera.tangent_half_vertical_fov = tanf(
        0.5f * degrees_to_radians(vertical_fov_degrees));
    camera.aspect_ratio =
        float(image_width) / float(image_height);

    return camera;
}

static float3 box_centre(const AxisAlignedBox& box) {
    return multiply(add(box.minimum, box.maximum), 0.5f);
}

// ============================================================================
// 14. Renderer class
// ============================================================================

class GaussianMIPRenderer {
public:
    GaussianMIPRenderer(
        const std::vector<GaussianDisk>& host_gaussians,
        int image_width,
        int image_height,
        int ray_sample_count,
        Camera camera,
        AxisAlignedBox box,
        bool hard_gate = false)
        : gaussian_count_(int(host_gaussians.size())),
          image_width_(image_width),
          image_height_(image_height),
          ray_sample_count_(ray_sample_count),
          tiles_x_(divide_round_up(image_width, TILE_WIDTH)),
          tiles_y_(divide_round_up(image_height, TILE_HEIGHT)),
          tile_count_(tiles_x_ * tiles_y_),
          camera_(camera),
          box_(box),
          hard_gate_(hard_gate)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream_,
            cudaStreamNonBlocking));

        CUDA_CHECK(cudaMalloc(
            &device_input_gaussians_,
            size_t(gaussian_count_) * sizeof(GaussianDisk)));

        CUDA_CHECK(cudaMalloc(
            &device_gaussians_,
            size_t(gaussian_count_) * sizeof(GaussianGPU)));

        CUDA_CHECK(cudaMalloc(
            &device_pair_counts_,
            size_t(gaussian_count_) * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(
            &device_pair_offsets_,
            size_t(gaussian_count_) * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(
            &device_tile_ranges_,
            size_t(tile_count_) * sizeof(TileRange)));

        CUDA_CHECK(cudaMalloc(
            &device_output_image_,
            size_t(image_width_) *
            size_t(image_height_) *
            sizeof(float)));

        CUDA_CHECK(cudaMemcpyAsync(
            device_input_gaussians_,
            host_gaussians.data(),
            size_t(gaussian_count_) * sizeof(GaussianDisk),
            cudaMemcpyHostToDevice,
            stream_));

        build_tile_lists();
    }

    ~GaussianMIPRenderer() {
        cudaFree(device_input_gaussians_);
        cudaFree(device_gaussians_);
        cudaFree(device_pair_counts_);
        cudaFree(device_pair_offsets_);
        cudaFree(device_tile_keys_unsorted_);
        cudaFree(device_tile_keys_sorted_);
        cudaFree(device_gaussian_indices_unsorted_);
        cudaFree(device_gaussian_indices_sorted_);
        cudaFree(device_tile_ranges_);
        cudaFree(device_output_image_);
        cudaFree(device_scan_workspace_);
        cudaFree(device_sort_workspace_);

        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    GaussianMIPRenderer(const GaussianMIPRenderer&) = delete;
    GaussianMIPRenderer& operator=(const GaussianMIPRenderer&) = delete;

    void render() {
        const dim3 block(TILE_WIDTH, TILE_HEIGHT);
        const dim3 grid(tiles_x_, tiles_y_);

        render_gaussian_mip_kernel<<<grid, block, 0, stream_>>>(
            device_gaussians_,
            device_gaussian_indices_sorted_,
            device_tile_ranges_,
            device_output_image_,
            image_width_,
            image_height_,
            tiles_x_,
            ray_sample_count_,
            camera_,
            box_,
            hard_gate_);

        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t stream() const {
        return stream_;
    }

    uint32_t gaussian_tile_pair_count() const {
        return pair_count_;
    }

    std::vector<float> download_image() {
        std::vector<float> image(
            size_t(image_width_) * size_t(image_height_));

        CUDA_CHECK(cudaMemcpyAsync(
            image.data(),
            device_output_image_,
            image.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_));

        synchronize();
        return image;
    }

private:
    void build_tile_lists() {
        const int threads = 256;
        const int blocks = divide_round_up(gaussian_count_, threads);

        // --------------------------------------------------------------------
        // Stage A: preprocess Gaussians and count required tile pairs.
        // --------------------------------------------------------------------
        preprocess_gaussians_kernel<<<blocks, threads, 0, stream_>>>(
            device_input_gaussians_,
            device_gaussians_,
            device_pair_counts_,
            gaussian_count_,
            image_width_,
            image_height_,
            tiles_x_,
            tiles_y_,
            camera_,
            hard_gate_);

        CUDA_CHECK(cudaGetLastError());

        // --------------------------------------------------------------------
        // Stage B: inclusive prefix sum over pair counts.
        // The last prefix value gives the total number of pairs.
        // --------------------------------------------------------------------
        size_t scan_workspace_bytes = 0;

        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr,
            scan_workspace_bytes,
            device_pair_counts_,
            device_pair_offsets_,
            gaussian_count_,
            stream_));

        CUDA_CHECK(cudaMalloc(
            &device_scan_workspace_,
            scan_workspace_bytes));

        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            device_scan_workspace_,
            scan_workspace_bytes,
            device_pair_counts_,
            device_pair_offsets_,
            gaussian_count_,
            stream_));

        CUDA_CHECK(cudaMemcpyAsync(
            &pair_count_,
            device_pair_offsets_ + gaussian_count_ - 1,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream_));

        synchronize();

        if (pair_count_ == 0) {
            throw std::runtime_error(
                "No Gaussian overlaps the camera view.");
        }

        // --------------------------------------------------------------------
        // Stage C: allocate key/value arrays for all Gaussian-tile pairs.
        // --------------------------------------------------------------------
        CUDA_CHECK(cudaMalloc(
            &device_tile_keys_unsorted_,
            size_t(pair_count_) * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(
            &device_tile_keys_sorted_,
            size_t(pair_count_) * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(
            &device_gaussian_indices_unsorted_,
            size_t(pair_count_) * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(
            &device_gaussian_indices_sorted_,
            size_t(pair_count_) * sizeof(uint32_t)));

        // --------------------------------------------------------------------
        // Stage D: create one key/value pair per Gaussian-tile overlap.
        // --------------------------------------------------------------------
        create_gaussian_tile_pairs_kernel<<<blocks, threads, 0, stream_>>>(
            device_gaussians_,
            device_pair_offsets_,
            device_tile_keys_unsorted_,
            device_gaussian_indices_unsorted_,
            gaussian_count_,
            tiles_x_);

        CUDA_CHECK(cudaGetLastError());

        // --------------------------------------------------------------------
        // Stage E: sort pairs by tile ID.
        // --------------------------------------------------------------------
        size_t sort_workspace_bytes = 0;

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_workspace_bytes,
            device_tile_keys_unsorted_,
            device_tile_keys_sorted_,
            device_gaussian_indices_unsorted_,
            device_gaussian_indices_sorted_,
            pair_count_,
            0,
            32,
            stream_));

        CUDA_CHECK(cudaMalloc(
            &device_sort_workspace_,
            sort_workspace_bytes));

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            device_sort_workspace_,
            sort_workspace_bytes,
            device_tile_keys_unsorted_,
            device_tile_keys_sorted_,
            device_gaussian_indices_unsorted_,
            device_gaussian_indices_sorted_,
            pair_count_,
            0,
            32,
            stream_));

        // --------------------------------------------------------------------
        // Stage F: build begin/end ranges for every tile.
        // --------------------------------------------------------------------
        CUDA_CHECK(cudaMemsetAsync(
            device_tile_ranges_,
            0,
            size_t(tile_count_) * sizeof(TileRange),
            stream_));

        identify_tile_ranges_kernel<<<
            divide_round_up(int(pair_count_), threads),
            threads,
            0,
            stream_>>>(
                device_tile_keys_sorted_,
                device_tile_ranges_,
                pair_count_);

        CUDA_CHECK(cudaGetLastError());
        synchronize();
    }

    int gaussian_count_ = 0;
    int image_width_ = 0;
    int image_height_ = 0;
    int ray_sample_count_ = 0;

    int tiles_x_ = 0;
    int tiles_y_ = 0;
    int tile_count_ = 0;

    uint32_t pair_count_ = 0;

    Camera camera_{};
    AxisAlignedBox box_{};
    bool hard_gate_ = false;

    cudaStream_t stream_{};

    GaussianDisk* device_input_gaussians_ = nullptr;
    GaussianGPU* device_gaussians_ = nullptr;

    uint32_t* device_pair_counts_ = nullptr;
    uint32_t* device_pair_offsets_ = nullptr;

    uint32_t* device_tile_keys_unsorted_ = nullptr;
    uint32_t* device_tile_keys_sorted_ = nullptr;

    uint32_t* device_gaussian_indices_unsorted_ = nullptr;
    uint32_t* device_gaussian_indices_sorted_ = nullptr;

    TileRange* device_tile_ranges_ = nullptr;
    float* device_output_image_ = nullptr;

    void* device_scan_workspace_ = nullptr;
    void* device_sort_workspace_ = nullptr;
};

// ============================================================================
// 15. Dense voxel input path (ground-truth MIP rendering)
// ============================================================================
//
// Renders a MIP of a dense voxel grid (e.g. the raw EM block, minmax-
// normalised to [0, 1]) using hardware-accelerated trilinear texture
// sampling, so it can be compared directly against the Gaussian-mixture
// reconstruction rendered by GaussianMIPRenderer above with identical
// camera/box parameters.
// ============================================================================

constexpr uint32_t DENSE_FILE_MAGIC = 0x564F584Cu; // 'VOXL'
constexpr uint32_t DENSE_FILE_VERSION = 1u;

struct DenseFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
};

static_assert(
    sizeof(DenseFileHeader) == 20,
    "Unexpected DenseFileHeader size.");

__device__ inline float3 dense_world_to_texture(
    float3 point,
    AxisAlignedBox box)
{
    const float3 extent = subtract(box.maximum, box.minimum);

    return make_float3(
        (point.x - box.minimum.x) / extent.x,
        (point.y - box.minimum.y) / extent.y,
        (point.z - box.minimum.z) / extent.z);
}

__global__ void dense_mip_kernel(
    cudaTextureObject_t texture,
    float* __restrict__ output,
    int image_width,
    int image_height,
    int ray_sample_count,
    Camera camera,
    AxisAlignedBox box)
{
    const int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= image_width || pixel_y >= image_height) {
        return;
    }

    const float ndc_x =
        2.0f * (float(pixel_x) + 0.5f) / float(image_width) - 1.0f;
    const float ndc_y =
        1.0f - 2.0f * (float(pixel_y) + 0.5f) / float(image_height);

    const float camera_plane_x =
        ndc_x * camera.aspect_ratio * camera.tangent_half_vertical_fov;
    const float camera_plane_y =
        ndc_y * camera.tangent_half_vertical_fov;

    const float3 ray_direction = normalize(
        add(
            camera.forward,
            add(
                multiply(camera.right, camera_plane_x),
                multiply(camera.up, camera_plane_y))));

    float entry_distance = 0.0f;
    float exit_distance = 0.0f;

    if (!intersect_ray_with_box(
            camera.position,
            ray_direction,
            box,
            entry_distance,
            exit_distance)) {
        output[pixel_y * image_width + pixel_x] = 0.0f;
        return;
    }

    entry_distance = fmaxf(entry_distance, CAMERA_NEAR_DISTANCE);

    float maximum_value = -FLT_MAX;

    for (int sample_index = 0; sample_index < ray_sample_count; ++sample_index) {
        const float interpolation =
            ray_sample_count > 1
                ? float(sample_index) / float(ray_sample_count - 1)
                : 0.5f;

        const float distance =
            entry_distance + (exit_distance - entry_distance) * interpolation;

        const float3 sample_point = add(
            camera.position,
            multiply(ray_direction, distance));

        const float3 texture_position =
            dense_world_to_texture(sample_point, box);

        const float value = tex3D<float>(
            texture,
            texture_position.x,
            texture_position.y,
            texture_position.z);

        maximum_value = fmaxf(maximum_value, value);
    }

    output[pixel_y * image_width + pixel_x] =
        isfinite(maximum_value) ? maximum_value : 0.0f;
}

static std::vector<float> read_dense_volume(
    const std::string& path,
    uint32_t& depth,
    uint32_t& height,
    uint32_t& width)
{
    std::ifstream stream(path, std::ios::binary);

    if (!stream) {
        throw std::runtime_error("Cannot open dense-volume binary: " + path);
    }

    DenseFileHeader header{};
    stream.read(
        reinterpret_cast<char*>(&header),
        sizeof(header));

    if (!stream ||
        header.magic != DENSE_FILE_MAGIC ||
        header.version != DENSE_FILE_VERSION) {
        throw std::runtime_error("Invalid dense-volume binary header.");
    }

    if (header.depth == 0 || header.height == 0 || header.width == 0) {
        throw std::runtime_error("Dense-volume dimensions must be positive.");
    }

    const uint64_t voxel_count =
        uint64_t(header.depth) *
        uint64_t(header.height) *
        uint64_t(header.width);

    if (voxel_count >
        uint64_t(std::numeric_limits<size_t>::max() / sizeof(float))) {
        throw std::runtime_error("Dense volume is too large.");
    }

    std::vector<float> volume(static_cast<size_t>(voxel_count));

    stream.read(
        reinterpret_cast<char*>(volume.data()),
        static_cast<std::streamsize>(volume.size() * sizeof(float)));

    if (!stream) {
        throw std::runtime_error("Truncated dense-volume binary.");
    }

    depth = header.depth;
    height = header.height;
    width = header.width;

    return volume;
}

class DenseVoxelRenderer {
public:
    DenseVoxelRenderer(
        const std::vector<float>& host_volume,
        uint32_t volume_depth,
        uint32_t volume_height,
        uint32_t volume_width,
        int output_width,
        int output_height,
        int ray_sample_count,
        Camera camera,
        AxisAlignedBox box)
        : output_width_(output_width),
          output_height_(output_height),
          ray_sample_count_(ray_sample_count),
          camera_(camera),
          box_(box)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream_,
            cudaStreamNonBlocking));

        const cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();

        const cudaExtent extent = make_cudaExtent(
            volume_width,
            volume_height,
            volume_depth);

        CUDA_CHECK(cudaMalloc3DArray(&volume_array_, &channel, extent));

        cudaMemcpy3DParms copy{};
        copy.srcPtr = make_cudaPitchedPtr(
            const_cast<float*>(host_volume.data()),
            size_t(volume_width) * sizeof(float),
            volume_width,
            volume_height);
        copy.dstArray = volume_array_;
        copy.extent = extent;
        copy.kind = cudaMemcpyHostToDevice;

        CUDA_CHECK(cudaMemcpy3DAsync(&copy, stream_));

        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = volume_array_;

        cudaTextureDesc texture{};
        texture.addressMode[0] = cudaAddressModeClamp;
        texture.addressMode[1] = cudaAddressModeClamp;
        texture.addressMode[2] = cudaAddressModeClamp;
        texture.filterMode = cudaFilterModeLinear;
        texture.readMode = cudaReadModeElementType;
        texture.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(
            &texture_,
            &resource,
            &texture,
            nullptr));

        CUDA_CHECK(cudaMalloc(
            &device_output_image_,
            size_t(output_width_) * size_t(output_height_) * sizeof(float)));

        synchronize();
    }

    ~DenseVoxelRenderer() {
        if (device_output_image_) {
            cudaFree(device_output_image_);
        }
        if (texture_) {
            cudaDestroyTextureObject(texture_);
        }
        if (volume_array_) {
            cudaFreeArray(volume_array_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    DenseVoxelRenderer(const DenseVoxelRenderer&) = delete;
    DenseVoxelRenderer& operator=(const DenseVoxelRenderer&) = delete;

    void render() {
        const dim3 block(TILE_WIDTH, TILE_HEIGHT);
        const dim3 grid(
            divide_round_up(output_width_, TILE_WIDTH),
            divide_round_up(output_height_, TILE_HEIGHT));

        dense_mip_kernel<<<grid, block, 0, stream_>>>(
            texture_,
            device_output_image_,
            output_width_,
            output_height_,
            ray_sample_count_,
            camera_,
            box_);

        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t stream() const {
        return stream_;
    }

    std::vector<float> download_image() {
        std::vector<float> image(
            size_t(output_width_) * size_t(output_height_));

        CUDA_CHECK(cudaMemcpyAsync(
            image.data(),
            device_output_image_,
            image.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_));

        synchronize();
        return image;
    }

private:
    int output_width_{};
    int output_height_{};
    int ray_sample_count_{};

    Camera camera_{};
    AxisAlignedBox box_{};

    cudaStream_t stream_{};
    cudaArray_t volume_array_{};
    cudaTextureObject_t texture_{};
    float* device_output_image_{};
};

// ============================================================================
// 16. Benchmark helper
// ============================================================================

template <typename Renderer>
static std::vector<float> benchmark_renderer(
    Renderer& renderer,
    int warmup_frame_count,
    int measured_frame_count,
    float& average_render_time_ms,
    float& frames_per_second)
{
    for (int frame = 0; frame < warmup_frame_count; ++frame) {
        renderer.render();
    }

    renderer.synchronize();

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, renderer.stream()));

    for (int frame = 0; frame < measured_frame_count; ++frame) {
        renderer.render();
    }

    CUDA_CHECK(cudaEventRecord(stop_event, renderer.stream()));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float total_time_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_time_ms,
        start_event,
        stop_event));

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    average_render_time_ms =
        total_time_ms / float(measured_frame_count);

    frames_per_second =
        average_render_time_ms > 0.0f
            ? 1000.0f / average_render_time_ms
            : 0.0f;

    // Render once more so that the downloaded image is the final clean frame.
    renderer.render();
    return renderer.download_image();
}

// ============================================================================
// 17. Main program
// ============================================================================

enum class RepresentationType {
    DenseVoxel,
    PretrainedGaussian,
    PretrainedGaussianHardGated
};

static RepresentationType parse_representation_type(const std::string& value) {
    if (value == "dense_voxel") {
        return RepresentationType::DenseVoxel;
    }

    if (value == "pretrained_gaussian") {
        return RepresentationType::PretrainedGaussian;
    }

    if (value == "pretrained_gaussian_hard_gated") {
        return RepresentationType::PretrainedGaussianHardGated;
    }

    throw std::runtime_error(
        "Invalid representation type '" + value +
        "'. Expected dense_voxel, pretrained_gaussian, or "
        "pretrained_gaussian_hard_gated.");
}

int main(int argument_count, char** arguments) {
    try {
        if (argument_count != 18) {
            std::cerr
                << "Usage:\n  "
                << arguments[0]
                << " <dense_voxel|pretrained_gaussian|pretrained_gaussian_hard_gated>"
                << " input.bin output.pfm"
                << " width height ray_samples benchmark_frames"
                << " yaw pitch roll fov_y"
                << " min_x min_y min_z"
                << " max_x max_y max_z\n\n"
                << "Examples:\n  "
                << arguments[0]
                << " dense_voxel volume.bin voxel.pfm"
                << " 128 128 64 200"
                << " 0 0 0 90"
                << " -1 -1 -1 1 1 1\n\n  "
                << arguments[0]
                << " pretrained_gaussian gaussians.bin gaussian.pfm"
                << " 128 128 64 200"
                << " 0 0 0 90"
                << " -1 -1 -1 1 1 1\n";

            return EXIT_FAILURE;
        }

        const RepresentationType representation =
            parse_representation_type(arguments[1]);

        const std::string input_path = arguments[2];
        const std::string output_path = arguments[3];

        const int image_width = std::stoi(arguments[4]);
        const int image_height = std::stoi(arguments[5]);
        const int ray_sample_count = std::stoi(arguments[6]);
        const int benchmark_frame_count = std::stoi(arguments[7]);

        const float yaw = std::stof(arguments[8]);
        const float pitch = std::stof(arguments[9]);
        const float roll = std::stof(arguments[10]);
        const float vertical_fov = std::stof(arguments[11]);

        AxisAlignedBox box{};

        box.minimum = make_float3(
            std::stof(arguments[12]),
            std::stof(arguments[13]),
            std::stof(arguments[14]));

        box.maximum = make_float3(
            std::stof(arguments[15]),
            std::stof(arguments[16]),
            std::stof(arguments[17]));

        if (image_width <= 0 ||
            image_height <= 0 ||
            ray_sample_count <= 0 ||
            benchmark_frame_count <= 0) {
            throw std::runtime_error(
                "Image dimensions, ray samples, and frame count must be positive.");
        }

        if (!(vertical_fov > 1.0f && vertical_fov < 179.0f)) {
            throw std::runtime_error(
                "Vertical FOV must be between 1 and 179 degrees.");
        }

        if (!(box.maximum.x > box.minimum.x &&
              box.maximum.y > box.minimum.y &&
              box.maximum.z > box.minimum.z)) {
            throw std::runtime_error("Invalid box bounds.");
        }

        const float3 camera_position = box_centre(box);

        const Camera camera = create_camera(
            camera_position,
            yaw,
            pitch,
            roll,
            vertical_fov,
            image_width,
            image_height);

        std::cout
            << "Representation: "
            << (representation == RepresentationType::DenseVoxel
                    ? "dense_voxel"
                    : representation == RepresentationType::PretrainedGaussian
                        ? "pretrained_gaussian"
                        : "pretrained_gaussian_hard_gated")
            << "\n"
            << "Output: " << image_width << " x " << image_height << "\n"
            << "Ray samples: " << ray_sample_count << "\n"
            << "Camera position: "
            << camera_position.x << " "
            << camera_position.y << " "
            << camera_position.z << "\n"
            << "Yaw/pitch/roll: "
            << yaw << " " << pitch << " " << roll << "\n"
            << "Vertical FOV: " << vertical_fov << "\n";

        constexpr int warmup_frame_count = 20;

        float average_render_time_ms = 0.0f;
        float frames_per_second = 0.0f;
        std::vector<float> output_image;

        if (representation == RepresentationType::DenseVoxel) {
            uint32_t volume_depth = 0;
            uint32_t volume_height = 0;
            uint32_t volume_width = 0;

            const std::vector<float> volume = read_dense_volume(
                input_path,
                volume_depth,
                volume_height,
                volume_width);

            std::cout
                << "Dense volume: "
                << volume_width << " x "
                << volume_height << " x "
                << volume_depth << "\n";

            DenseVoxelRenderer renderer(
                volume,
                volume_depth,
                volume_height,
                volume_width,
                image_width,
                image_height,
                ray_sample_count,
                camera,
                box);

            output_image = benchmark_renderer(
                renderer,
                warmup_frame_count,
                benchmark_frame_count,
                average_render_time_ms,
                frames_per_second);
        }
        else {
            const std::vector<GaussianDisk> gaussians =
                read_gaussian_file(input_path);

            std::cout << "Gaussians: " << gaussians.size() << "\n";

            const bool hard_gate =
                representation ==
                RepresentationType::PretrainedGaussianHardGated;

            GaussianMIPRenderer renderer(
                gaussians,
                image_width,
                image_height,
                ray_sample_count,
                camera,
                box,
                hard_gate);

            std::cout
                << "Gaussian-tile pairs: "
                << renderer.gaussian_tile_pair_count()
                << "\n";

            output_image = benchmark_renderer(
                renderer,
                warmup_frame_count,
                benchmark_frame_count,
                average_render_time_ms,
                frames_per_second);
        }

        write_pfm_file(
            output_path,
            output_image,
            image_width,
            image_height);

        const auto minimum_and_maximum = std::minmax_element(
            output_image.begin(),
            output_image.end());

        std::cout
            << "Average render time: "
            << average_render_time_ms << " ms\n"
            << "FPS: " << frames_per_second << "\n"
            << "Output range: ["
            << *minimum_and_maximum.first << ", "
            << *minimum_and_maximum.second << "]\n"
            << "Saved image: " << output_path << "\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}