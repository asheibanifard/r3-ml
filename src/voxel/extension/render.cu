// voxel/extension/render.cu
//
// Synthetic ground-truth volume generation and the two comparison
// renderers: trilinear direct-volume ray marching (the "GT" renderer) and
// front-to-back dense-grid 3D-DDA voxel rasterization (the "predicted"
// renderer) — a compact teaching analogue of SVRaster's sparse
// sorted-voxel renderer.
#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace {

// ── 1. Synthetic ground-truth scalar volume ────────────────────────────────
__global__ void create_ground_truth_volume_kernel(
        float* __restrict__ volume, int64_t grid_size) {
    const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t voxel_count = grid_size * grid_size * grid_size;
    if (index >= voxel_count) return;

    const int64_t x = index % grid_size;
    const int64_t y = (index / grid_size) % grid_size;
    const int64_t z = index / (grid_size * grid_size);

    // Voxel-centre coordinates in [-1, 1]^3.
    const float px = (static_cast<float>(x) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float py = (static_cast<float>(y) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float pz = (static_cast<float>(z) + 0.5f) / grid_size * 2.0f - 1.0f;

    const float sphere = expf(-18.0f * (px * px + py * py + pz * pz));

    const float radial_xy = sqrtf(px * px + py * py);
    const float torus_distance =
        (radial_xy - 0.48f) * (radial_xy - 0.48f) + pz * pz;
    const float torus = expf(-90.0f * torus_distance);

    const float bx = px + 0.35f;
    const float by = py - 0.23f;
    const float bz = pz + 0.10f;
    const float blob = expf(-55.0f * (bx * bx + by * by + bz * bz));

    volume[index] = clamp01(0.85f * sphere + 0.70f * torus + 0.75f * blob);
}

// ── 2. Ray construction and volume sampling ─────────────────────────────────
__device__ inline void generate_camera_ray(
        int pixel_x, int pixel_y, int64_t image_width, int64_t image_height,
        Camera camera, Vec3& ray_origin, Vec3& ray_direction) {
    const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);

    const float screen_x =
        (2.0f * (pixel_x + 0.5f) / image_width - 1.0f) *
        camera.tan_half_fov * aspect;

    const float screen_y =
        (1.0f - 2.0f * (pixel_y + 0.5f) / image_height) *
        camera.tan_half_fov;

    ray_origin = camera.origin;
    ray_direction = normalize(
        camera.forward + camera.right * screen_x + camera.up * screen_y
    );
}

// Intersect ray with the volume box [-1,1]^3.
__device__ inline bool intersect_volume_box(
        Vec3 ray_origin, Vec3 ray_direction, float& entry_t, float& exit_t) {
    entry_t = 0.0f;
    exit_t = 1.0e30f;

    const float origin[3] = {ray_origin.x, ray_origin.y, ray_origin.z};
    const float direction[3] = {ray_direction.x, ray_direction.y, ray_direction.z};

    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(direction[axis]) < 1.0e-12f) {
            if (origin[axis] < -1.0f || origin[axis] > 1.0f) return false;
            continue;
        }

        const float inverse_direction = 1.0f / direction[axis];
        float near_t = (-1.0f - origin[axis]) * inverse_direction;
        float far_t = (1.0f - origin[axis]) * inverse_direction;

        if (near_t > far_t) {
            const float temporary = near_t;
            near_t = far_t;
            far_t = temporary;
        }

        entry_t = fmaxf(entry_t, near_t);
        exit_t = fminf(exit_t, far_t);
        if (entry_t > exit_t) return false;
    }

    return true;
}

__device__ inline float sample_trilinear(
        const float* __restrict__ volume, Vec3 point, int64_t grid_size) {
    const float gx = clamp01((point.x + 1.0f) * 0.5f) * (grid_size - 1);
    const float gy = clamp01((point.y + 1.0f) * 0.5f) * (grid_size - 1);
    const float gz = clamp01((point.z + 1.0f) * 0.5f) * (grid_size - 1);

    const int64_t x0 = static_cast<int64_t>(floorf(gx));
    const int64_t y0 = static_cast<int64_t>(floorf(gy));
    const int64_t z0 = static_cast<int64_t>(floorf(gz));
    const int64_t x1 = min(x0 + 1, grid_size - 1);
    const int64_t y1 = min(y0 + 1, grid_size - 1);
    const int64_t z1 = min(z0 + 1, grid_size - 1);

    const float tx = gx - x0;
    const float ty = gy - y0;
    const float tz = gz - z0;

    const float c00 =
        volume[voxel_index(x0, y0, z0, grid_size)] * (1.0f - tx) +
        volume[voxel_index(x1, y0, z0, grid_size)] * tx;
    const float c10 =
        volume[voxel_index(x0, y1, z0, grid_size)] * (1.0f - tx) +
        volume[voxel_index(x1, y1, z0, grid_size)] * tx;
    const float c01 =
        volume[voxel_index(x0, y0, z1, grid_size)] * (1.0f - tx) +
        volume[voxel_index(x1, y0, z1, grid_size)] * tx;
    const float c11 =
        volume[voxel_index(x0, y1, z1, grid_size)] * (1.0f - tx) +
        volume[voxel_index(x1, y1, z1, grid_size)] * tx;

    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

// ── 3A. Ground-truth renderer: direct volume ray marching ──────────────────
__global__ void render_direct_volume_kernel(
        const float* __restrict__ volume,
        float* __restrict__ image,
        const float* __restrict__ camera_ptr,
        int64_t grid_size, int64_t image_width, int64_t image_height,
        int dvr_steps, float density_scale) {
    const int64_t pixel_index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t pixel_count = image_width * image_height;
    if (pixel_index >= pixel_count) return;

    const Camera camera = load_camera(camera_ptr);
    const int pixel_x = static_cast<int>(pixel_index % image_width);
    const int pixel_y = static_cast<int>(pixel_index / image_width);

    Vec3 ray_origin, ray_direction;
    generate_camera_ray(pixel_x, pixel_y, image_width, image_height, camera, ray_origin, ray_direction);

    float entry_t, exit_t;
    if (!intersect_volume_box(ray_origin, ray_direction, entry_t, exit_t)) {
        image[pixel_index] = 0.0f;
        return;
    }

    const float step_length = (exit_t - entry_t) / dvr_steps;
    float transmittance = 1.0f;
    float accumulated_intensity = 0.0f;

    for (int step = 0; step < dvr_steps && transmittance > 1.0e-4f; ++step) {
        const float sample_t = entry_t + (step + 0.5f) * step_length;
        const Vec3 sample_position = ray_origin + ray_direction * sample_t;

        const float density = sample_trilinear(volume, sample_position, grid_size) * density_scale;
        const float alpha = 1.0f - expf(-density * step_length);

        accumulated_intensity += transmittance * alpha;
        transmittance *= (1.0f - alpha);
    }

    image[pixel_index] = clamp01(accumulated_intensity);
}

// ── 3B. Predicted renderer: front-to-back dense voxel DDA rasterisation ────
__global__ void render_voxel_rasterizer_kernel(
        const float* __restrict__ volume,
        float* __restrict__ image,
        const float* __restrict__ camera_ptr,
        int64_t grid_size, int64_t image_width, int64_t image_height,
        float density_scale) {
    const int64_t pixel_index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t pixel_count = image_width * image_height;
    if (pixel_index >= pixel_count) return;

    const Camera camera = load_camera(camera_ptr);
    const int pixel_x = static_cast<int>(pixel_index % image_width);
    const int pixel_y = static_cast<int>(pixel_index / image_width);

    Vec3 ray_origin, ray_direction;
    generate_camera_ray(pixel_x, pixel_y, image_width, image_height, camera, ray_origin, ray_direction);

    float entry_t, exit_t;
    if (!intersect_volume_box(ray_origin, ray_direction, entry_t, exit_t)) {
        image[pixel_index] = 0.0f;
        return;
    }

    const float cell_size = 2.0f / grid_size;
    const Vec3 entry_point = ray_origin + ray_direction * (entry_t + 1.0e-5f);

    int64_t voxel_x = min(grid_size - 1, max(int64_t{0}, static_cast<int64_t>((entry_point.x + 1.0f) / cell_size)));
    int64_t voxel_y = min(grid_size - 1, max(int64_t{0}, static_cast<int64_t>((entry_point.y + 1.0f) / cell_size)));
    int64_t voxel_z = min(grid_size - 1, max(int64_t{0}, static_cast<int64_t>((entry_point.z + 1.0f) / cell_size)));

    const int step_x = (ray_direction.x >= 0.0f) ? 1 : -1;
    const int step_y = (ray_direction.y >= 0.0f) ? 1 : -1;
    const int step_z = (ray_direction.z >= 0.0f) ? 1 : -1;

    const float next_boundary_x = -1.0f + (voxel_x + (step_x > 0 ? 1 : 0)) * cell_size;
    const float next_boundary_y = -1.0f + (voxel_y + (step_y > 0 ? 1 : 0)) * cell_size;
    const float next_boundary_z = -1.0f + (voxel_z + (step_z > 0 ? 1 : 0)) * cell_size;

    float next_t_x = (fabsf(ray_direction.x) < 1.0e-12f) ? 1.0e30f : (next_boundary_x - ray_origin.x) / ray_direction.x;
    float next_t_y = (fabsf(ray_direction.y) < 1.0e-12f) ? 1.0e30f : (next_boundary_y - ray_origin.y) / ray_direction.y;
    float next_t_z = (fabsf(ray_direction.z) < 1.0e-12f) ? 1.0e30f : (next_boundary_z - ray_origin.z) / ray_direction.z;

    const float delta_t_x = (fabsf(ray_direction.x) < 1.0e-12f) ? 1.0e30f : fabsf(cell_size / ray_direction.x);
    const float delta_t_y = (fabsf(ray_direction.y) < 1.0e-12f) ? 1.0e30f : fabsf(cell_size / ray_direction.y);
    const float delta_t_z = (fabsf(ray_direction.z) < 1.0e-12f) ? 1.0e30f : fabsf(cell_size / ray_direction.z);

    float current_t = entry_t;
    float transmittance = 1.0f;
    float accumulated_intensity = 0.0f;

    // A ray can cross at most 3*N cells in an N^3 regular grid.
    const int64_t max_iterations = 3 * grid_size;
    for (int64_t iteration = 0;
         iteration < max_iterations && current_t < exit_t && transmittance > 1.0e-4f;
         ++iteration) {
        float segment_end_t = fminf(next_t_x, fminf(next_t_y, next_t_z));
        segment_end_t = fminf(segment_end_t, exit_t);

        const float segment_length = fmaxf(0.0f, segment_end_t - current_t);
        const float density = volume[voxel_index(voxel_x, voxel_y, voxel_z, grid_size)] * density_scale;
        const float alpha = 1.0f - expf(-density * segment_length);

        accumulated_intensity += transmittance * alpha;
        transmittance *= (1.0f - alpha);
        current_t = segment_end_t;

        // Advance across the closest next grid boundary.
        if (next_t_x <= next_t_y && next_t_x <= next_t_z) {
            voxel_x += step_x;
            next_t_x += delta_t_x;
            if (voxel_x < 0 || voxel_x >= grid_size) break;
        } else if (next_t_y <= next_t_z) {
            voxel_y += step_y;
            next_t_y += delta_t_y;
            if (voxel_y < 0 || voxel_y >= grid_size) break;
        } else {
            voxel_z += step_z;
            next_t_z += delta_t_z;
            if (voxel_z < 0 || voxel_z >= grid_size) break;
        }
    }

    image[pixel_index] = clamp01(accumulated_intensity);
}

}  // namespace

// ── Python-visible entry points ─────────────────────────────────────────────

torch::Tensor create_ground_truth_volume_cuda(int64_t grid_size, torch::Device device) {
    TORCH_CHECK(grid_size > 0, "grid_size must be positive");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto volume = torch::empty({grid_size, grid_size, grid_size}, options);
    const int64_t voxel_count = grid_size * grid_size * grid_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    create_ground_truth_volume_kernel<<<num_blocks(voxel_count), kThreadsPerBlock, 0, stream>>>(
        volume.data_ptr<float>(), grid_size
    );

    return volume;
}

namespace {
void validate_render_inputs(
        const torch::Tensor& volume, const torch::Tensor& camera) {
    CHECK_INPUT(volume);
    CHECK_INPUT(camera);
    TORCH_CHECK(volume.dim() == 3, "volume must have shape [G,G,G]");
    TORCH_CHECK(volume.size(0) == volume.size(1) && volume.size(1) == volume.size(2),
                "volume must be a cube [G,G,G]");
    TORCH_CHECK(camera.numel() == kCameraParamCount,
                "camera must have ", kCameraParamCount, " elements, got ", camera.numel());
}
}  // namespace

torch::Tensor render_direct_volume_cuda(
        const torch::Tensor& volume,
        const torch::Tensor& camera,
        int64_t image_width,
        int64_t image_height,
        int64_t dvr_steps,
        double density_scale) {
    validate_render_inputs(volume, camera);
    TORCH_CHECK(image_width > 0 && image_height > 0, "image dimensions must be positive");
    TORCH_CHECK(dvr_steps > 0, "dvr_steps must be positive");

    const int64_t grid_size = volume.size(0);
    const int64_t pixel_count = image_width * image_height;
    auto image = torch::empty({image_height, image_width}, volume.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    render_direct_volume_kernel<<<num_blocks(pixel_count), kThreadsPerBlock, 0, stream>>>(
        volume.data_ptr<float>(),
        image.data_ptr<float>(),
        camera.data_ptr<float>(),
        grid_size, image_width, image_height,
        static_cast<int>(dvr_steps), static_cast<float>(density_scale)
    );

    return image;
}

torch::Tensor render_voxel_rasterizer_cuda(
        const torch::Tensor& volume,
        const torch::Tensor& camera,
        int64_t image_width,
        int64_t image_height,
        double density_scale) {
    validate_render_inputs(volume, camera);
    TORCH_CHECK(image_width > 0 && image_height > 0, "image dimensions must be positive");

    const int64_t grid_size = volume.size(0);
    const int64_t pixel_count = image_width * image_height;
    auto image = torch::empty({image_height, image_width}, volume.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    render_voxel_rasterizer_kernel<<<num_blocks(pixel_count), kThreadsPerBlock, 0, stream>>>(
        volume.data_ptr<float>(),
        image.data_ptr<float>(),
        camera.data_ptr<float>(),
        grid_size, image_width, image_height,
        static_cast<float>(density_scale)
    );

    return image;
}
