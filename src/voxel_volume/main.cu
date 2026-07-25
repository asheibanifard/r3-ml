#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error__));        \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

// =============================================================================
// Experiment configuration
// =============================================================================
constexpr int GRID_SIZE = 512;
constexpr int VOXEL_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;
constexpr int IMAGE_WIDTH = 1024;
constexpr int IMAGE_HEIGHT = 1024;
constexpr int PIXEL_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr int INPUT_SIZE = 3;
constexpr int HIDDEN_SIZE = 128;
constexpr int HIDDEN_LAYERS = 4;
constexpr int TRAIN_BATCH = 4096;
constexpr int DVR_STEPS = 384;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float DENSITY_SCALE = 8.0f;

// SIREN architecture is controlled by HIDDEN_SIZE and HIDDEN_LAYERS.
// =============================================================================
// Small vector type shared by host and device
// =============================================================================
struct Vec3 {
    float x, y, z;
};

__host__ __device__ Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vec3 operator*(Vec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__host__ __device__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ Vec3 normalize(Vec3 v) {
    const float length_squared = fmaxf(dot(v, v), 1.0e-20f);
    return v * (1.0f / sqrtf(length_squared));
}

__host__ __device__ int voxel_index(int x, int y, int z) {
    return (z * GRID_SIZE + y) * GRID_SIZE + x;
}

__host__ __device__ float clamp01(float value) {
    return fminf(1.0f, fmaxf(0.0f, value));
}

// Camera basis is computed once on the CPU and passed to both renderers.
struct Camera {
    Vec3 origin;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float tan_half_fov;
};

Camera make_camera(float azimuth_radians,
                   float elevation_radians,
                   float radius,
                   float vertical_fov_degrees) {
    Camera camera{};

    camera.origin = {
        radius * std::cos(elevation_radians) * std::cos(azimuth_radians),
        radius * std::sin(elevation_radians),
        radius * std::cos(elevation_radians) * std::sin(azimuth_radians)
    };

    const Vec3 target{0.0f, 0.0f, 0.0f};
    const Vec3 world_up{0.0f, 1.0f, 0.0f};
    camera.forward = normalize(target - camera.origin);
    camera.right = normalize(cross(camera.forward, world_up));
    camera.up = normalize(cross(camera.right, camera.forward));
    camera.tan_half_fov = std::tan(vertical_fov_degrees * PI_F / 360.0f);
    return camera;
}

// =============================================================================
// 1. Create a synthetic GRID_SIZE^3 ground-truth scalar volume
// =============================================================================
__global__ void create_ground_truth_volume(float* volume) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= VOXEL_COUNT) return;

    const int x = index % GRID_SIZE;
    const int y = (index / GRID_SIZE) % GRID_SIZE;
    const int z = index / (GRID_SIZE * GRID_SIZE);

    // Voxel-centre coordinates in [-1, 1]^3.
    const float px = (x + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float py = (y + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float pz = (z + 0.5f) / GRID_SIZE * 2.0f - 1.0f;

    const float sphere = expf(-18.0f * (px * px + py * py + pz * pz));

    const float radial_xy = sqrtf(px * px + py * py);
    const float torus_distance =
        (radial_xy - 0.48f) * (radial_xy - 0.48f) + pz * pz;
    const float torus = expf(-90.0f * torus_distance);

    const float bx = px + 0.35f;
    const float by = py - 0.23f;
    const float bz = pz + 0.10f;
    const float blob = expf(-55.0f * (bx * bx + by * by + bz * bz));

    volume[index] = clamp01(
        0.85f * sphere +
        0.70f * torus +
        0.75f * blob
    );
}

// =============================================================================
// 2. Configurable SIREN continuous function F_theta(x,y,z)
// =============================================================================
constexpr float FIRST_OMEGA_0 = 30.0f;
constexpr float HIDDEN_OMEGA_0 = 1.0f;

struct SirenLayout {
    static_assert(INPUT_SIZE == 3, "This program expects x, y, z input coordinates.");
    static_assert(HIDDEN_SIZE >= 1, "HIDDEN_SIZE must be positive.");
    static_assert(HIDDEN_LAYERS >= 1, "HIDDEN_LAYERS must be at least one.");

    static constexpr int FIRST_WEIGHT_COUNT = INPUT_SIZE * HIDDEN_SIZE;
    static constexpr int FIRST_BIAS_COUNT = HIDDEN_SIZE;
    static constexpr int HIDDEN_WEIGHT_COUNT = HIDDEN_SIZE * HIDDEN_SIZE;
    static constexpr int HIDDEN_BIAS_COUNT = HIDDEN_SIZE;

    static constexpr int FIRST_W = 0;
    static constexpr int FIRST_B = FIRST_W + FIRST_WEIGHT_COUNT;
    static constexpr int HIDDEN_START = FIRST_B + FIRST_BIAS_COUNT;

    // hidden_layer is the activation-layer index and lies in [1, HIDDEN_LAYERS-1].
    __host__ __device__ static constexpr int hidden_weight_offset(int hidden_layer) {
        return HIDDEN_START +
               (hidden_layer - 1) *
                   (HIDDEN_WEIGHT_COUNT + HIDDEN_BIAS_COUNT);
    }

    __host__ __device__ static constexpr int hidden_bias_offset(int hidden_layer) {
        return hidden_weight_offset(hidden_layer) + HIDDEN_WEIGHT_COUNT;
    }

    static constexpr int OUTPUT_W =
        HIDDEN_START +
        (HIDDEN_LAYERS - 1) *
            (HIDDEN_WEIGHT_COUNT + HIDDEN_BIAS_COUNT);

    static constexpr int OUTPUT_B = OUTPUT_W + HIDDEN_SIZE;
    static constexpr int PARAM_COUNT = OUTPUT_B + 1;
};

constexpr int PARAM_COUNT = SirenLayout::PARAM_COUNT;

__device__ float deterministic_uniform_signed(int index, unsigned int seed) {
    unsigned int state =
        (static_cast<unsigned int>(index) + 1u) * 747796405u +
        seed * 2891336453u;
    state = (state ^ (state >> 16)) * 2246822519u;
    state ^= state >> 13;
    return ((state & 0x00ffffffu) / 16777216.0f) * 2.0f - 1.0f;
}

__global__ void initialize_siren(float* parameters, unsigned int seed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= PARAM_COUNT) return;

    const float random_value = deterministic_uniform_signed(index, seed);
    float bound = 0.0f;

    if (index < SirenLayout::FIRST_B) {
        // First-layer SIREN initialization: U(-1/fan_in, 1/fan_in).
        bound = 1.0f / static_cast<float>(INPUT_SIZE);
    } else if (index < SirenLayout::HIDDEN_START) {
        // First-layer biases start at zero.
        parameters[index] = 0.0f;
        return;
    } else if (index < SirenLayout::OUTPUT_W) {
        // Hidden-layer region contains alternating weights and biases.
        const int relative = index - SirenLayout::HIDDEN_START;
        const int block_size =
            SirenLayout::HIDDEN_WEIGHT_COUNT + SirenLayout::HIDDEN_BIAS_COUNT;
        const int within_block = relative % block_size;

        if (within_block >= SirenLayout::HIDDEN_WEIGHT_COUNT) {
            parameters[index] = 0.0f;
            return;
        }

        bound = sqrtf(6.0f / static_cast<float>(HIDDEN_SIZE)) /
                HIDDEN_OMEGA_0;
    } else if (index < SirenLayout::OUTPUT_B) {
        // Linear output-layer weights.
        bound = sqrtf(6.0f / static_cast<float>(HIDDEN_SIZE));
    } else {
        // Output bias.
        parameters[index] = 0.0f;
        return;
    }

    parameters[index] = random_value * bound;
}

__device__ float siren_forward(
    const float* parameters,
    float x,
    float y,
    float z,
    float activations[HIDDEN_LAYERS][HIDDEN_SIZE]
) {
    const float input[INPUT_SIZE] = {x, y, z};

    // First hidden layer: INPUT_SIZE -> HIDDEN_SIZE.
    for (int output = 0; output < HIDDEN_SIZE; ++output) {
        float preactivation = parameters[SirenLayout::FIRST_B + output];

        for (int input_index = 0; input_index < INPUT_SIZE; ++input_index) {
            preactivation += parameters[
                SirenLayout::FIRST_W + output * INPUT_SIZE + input_index
            ] * input[input_index];
        }

        activations[0][output] = sinf(FIRST_OMEGA_0 * preactivation);
    }

    // Remaining sine-activated hidden layers.
    for (int layer = 1; layer < HIDDEN_LAYERS; ++layer) {
        const int weight_offset = SirenLayout::hidden_weight_offset(layer);
        const int bias_offset = SirenLayout::hidden_bias_offset(layer);

        for (int output = 0; output < HIDDEN_SIZE; ++output) {
            float preactivation = parameters[bias_offset + output];

            for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
                preactivation += parameters[
                    weight_offset + output * HIDDEN_SIZE + input_index
                ] * activations[layer - 1][input_index];
            }

            activations[layer][output] =
                sinf(HIDDEN_OMEGA_0 * preactivation);
        }
    }

    // Linear scalar output followed by sigmoid to constrain density to [0,1].
    float output_value = parameters[SirenLayout::OUTPUT_B];

    for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
        output_value += parameters[SirenLayout::OUTPUT_W + hidden] *
                        activations[HIDDEN_LAYERS - 1][hidden];
    }

    return 1.0f / (1.0f + expf(-output_value));
}

// One CUDA thread processes one randomly selected training voxel.
// The implementation is intentionally direct and uses atomic gradient accumulation.
__global__ void siren_training_step(
    const float* target_volume,
    const float* parameters,
    float* gradients,
    float* batch_loss,
    int iteration
) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= TRAIN_BATCH) return;

    unsigned int hash =
        (static_cast<unsigned int>(sample) + 1u) * 747796405u +
        (static_cast<unsigned int>(iteration) + 1u) * 2891336453u;
    hash = (hash ^ (hash >> 16)) * 2246822519u;
    hash = (hash ^ (hash >> 13)) * 3266489917u;
    hash ^= hash >> 16;

    const int index = static_cast<int>(hash % VOXEL_COUNT);
    const int vx = index % GRID_SIZE;
    const int vy = (index / GRID_SIZE) % GRID_SIZE;
    const int vz = index / (GRID_SIZE * GRID_SIZE);

    const float x = (vx + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float y = (vy + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float z = (vz + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float input[INPUT_SIZE] = {x, y, z};

    float activations[HIDDEN_LAYERS][HIDDEN_SIZE];
    const float prediction = siren_forward(
        parameters, x, y, z, activations
    );

    const float error = prediction - target_volume[index];
    atomicAdd(batch_loss, error * error / static_cast<float>(TRAIN_BATCH));

    // d(MSE)/d(pre-sigmoid output).
    const float output_gradient =
        2.0f * error * prediction * (1.0f - prediction) /
        static_cast<float>(TRAIN_BATCH);

    // Output layer gradients and gradient entering the last hidden activation.
    atomicAdd(&gradients[SirenLayout::OUTPUT_B], output_gradient);

    float current_activation_gradient[HIDDEN_SIZE];
    float previous_activation_gradient[HIDDEN_SIZE];

    for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
        atomicAdd(
            &gradients[SirenLayout::OUTPUT_W + hidden],
            output_gradient * activations[HIDDEN_LAYERS - 1][hidden]
        );

        current_activation_gradient[hidden] =
            output_gradient * parameters[SirenLayout::OUTPUT_W + hidden];
    }

    // Backpropagate through hidden layers HIDDEN_LAYERS-1 down to 1.
    for (int layer = HIDDEN_LAYERS - 1; layer >= 1; --layer) {
        const int weight_offset = SirenLayout::hidden_weight_offset(layer);
        const int bias_offset = SirenLayout::hidden_bias_offset(layer);

        for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
            previous_activation_gradient[input_index] = 0.0f;
        }

        for (int output = 0; output < HIDDEN_SIZE; ++output) {
            float preactivation = parameters[bias_offset + output];

            for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
                preactivation += parameters[
                    weight_offset + output * HIDDEN_SIZE + input_index
                ] * activations[layer - 1][input_index];
            }

            const float preactivation_gradient =
                current_activation_gradient[output] *
                HIDDEN_OMEGA_0 *
                cosf(HIDDEN_OMEGA_0 * preactivation);

            atomicAdd(
                &gradients[bias_offset + output],
                preactivation_gradient
            );

            for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
                const int weight_index =
                    weight_offset + output * HIDDEN_SIZE + input_index;

                atomicAdd(
                    &gradients[weight_index],
                    preactivation_gradient * activations[layer - 1][input_index]
                );

                previous_activation_gradient[input_index] +=
                    preactivation_gradient * parameters[weight_index];
            }
        }

        for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
            current_activation_gradient[hidden] =
                previous_activation_gradient[hidden];
        }
    }

    // Backpropagate through the first hidden layer.
    for (int output = 0; output < HIDDEN_SIZE; ++output) {
        float preactivation = parameters[SirenLayout::FIRST_B + output];

        for (int input_index = 0; input_index < INPUT_SIZE; ++input_index) {
            preactivation += parameters[
                SirenLayout::FIRST_W + output * INPUT_SIZE + input_index
            ] * input[input_index];
        }

        const float preactivation_gradient =
            current_activation_gradient[output] *
            FIRST_OMEGA_0 *
            cosf(FIRST_OMEGA_0 * preactivation);

        atomicAdd(
            &gradients[SirenLayout::FIRST_B + output],
            preactivation_gradient
        );

        for (int input_index = 0; input_index < INPUT_SIZE; ++input_index) {
            atomicAdd(
                &gradients[
                    SirenLayout::FIRST_W + output * INPUT_SIZE + input_index
                ],
                preactivation_gradient * input[input_index]
            );
        }
    }
}

__global__ void adam_update(float* parameters,
                            const float* gradients,
                            float* first_moment,
                            float* second_moment,
                            int iteration,
                            float learning_rate) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= PARAM_COUNT) return;

    const float gradient = gradients[index];
    first_moment[index] = 0.9f * first_moment[index] + 0.1f * gradient;
    second_moment[index] =
        0.999f * second_moment[index] + 0.001f * gradient * gradient;

    const float corrected_first =
        first_moment[index] / (1.0f - powf(0.9f, static_cast<float>(iteration)));
    const float corrected_second =
        second_moment[index] / (1.0f - powf(0.999f, static_cast<float>(iteration)));

    parameters[index] -=
        learning_rate * corrected_first /
        (sqrtf(corrected_second) + 1.0e-8f);
}

__global__ void reconstruct_voxel_grid(const float* parameters,
                                       float* reconstructed_volume) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= VOXEL_COUNT) return;

    const int x = index % GRID_SIZE;
    const int y = (index / GRID_SIZE) % GRID_SIZE;
    const int z = index / (GRID_SIZE * GRID_SIZE);

    const float px = (x + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float py = (y + 0.5f) / GRID_SIZE * 2.0f - 1.0f;
    const float pz = (z + 0.5f) / GRID_SIZE * 2.0f - 1.0f;

    float activations[HIDDEN_LAYERS][HIDDEN_SIZE];

    reconstructed_volume[index] = siren_forward(
        parameters,
        px,
        py,
        pz,
        activations
    );
}

// =============================================================================
// 3. Ray construction and volume sampling
// =============================================================================
__device__ void generate_camera_ray(int pixel_x,
                                    int pixel_y,
                                    Camera camera,
                                    Vec3& ray_origin,
                                    Vec3& ray_direction) {
    const float aspect =
        static_cast<float>(IMAGE_WIDTH) / static_cast<float>(IMAGE_HEIGHT);

    const float screen_x =
        (2.0f * (pixel_x + 0.5f) / IMAGE_WIDTH - 1.0f) *
        camera.tan_half_fov * aspect;

    const float screen_y =
        (1.0f - 2.0f * (pixel_y + 0.5f) / IMAGE_HEIGHT) *
        camera.tan_half_fov;

    ray_origin = camera.origin;
    ray_direction = normalize(
        camera.forward +
        camera.right * screen_x +
        camera.up * screen_y
    );
}

// Intersect ray with the volume box [-1,1]^3.
__device__ bool intersect_volume_box(Vec3 ray_origin,
                                     Vec3 ray_direction,
                                     float& entry_t,
                                     float& exit_t) {
    entry_t = 0.0f;
    exit_t = 1.0e30f;

    const float origin[3] = {ray_origin.x, ray_origin.y, ray_origin.z};
    const float direction[3] = {
        ray_direction.x, ray_direction.y, ray_direction.z
    };

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

__device__ float sample_trilinear(const float* volume, Vec3 point) {
    const float gx = clamp01((point.x + 1.0f) * 0.5f) * (GRID_SIZE - 1);
    const float gy = clamp01((point.y + 1.0f) * 0.5f) * (GRID_SIZE - 1);
    const float gz = clamp01((point.z + 1.0f) * 0.5f) * (GRID_SIZE - 1);

    const int x0 = static_cast<int>(floorf(gx));
    const int y0 = static_cast<int>(floorf(gy));
    const int z0 = static_cast<int>(floorf(gz));
    const int x1 = min(x0 + 1, GRID_SIZE - 1);
    const int y1 = min(y0 + 1, GRID_SIZE - 1);
    const int z1 = min(z0 + 1, GRID_SIZE - 1);

    const float tx = gx - x0;
    const float ty = gy - y0;
    const float tz = gz - z0;

    const float c00 =
        volume[voxel_index(x0, y0, z0)] * (1.0f - tx) +
        volume[voxel_index(x1, y0, z0)] * tx;
    const float c10 =
        volume[voxel_index(x0, y1, z0)] * (1.0f - tx) +
        volume[voxel_index(x1, y1, z0)] * tx;
    const float c01 =
        volume[voxel_index(x0, y0, z1)] * (1.0f - tx) +
        volume[voxel_index(x1, y0, z1)] * tx;
    const float c11 =
        volume[voxel_index(x0, y1, z1)] * (1.0f - tx) +
        volume[voxel_index(x1, y1, z1)] * tx;

    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

// =============================================================================
// 4A. Ground-truth renderer: direct volume ray marching
// =============================================================================
__global__ void render_direct_volume(const float* volume,
                                     float* image,
                                     Camera camera) {
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_index >= PIXEL_COUNT) return;

    const int pixel_x = pixel_index % IMAGE_WIDTH;
    const int pixel_y = pixel_index / IMAGE_WIDTH;

    Vec3 ray_origin, ray_direction;
    generate_camera_ray(
        pixel_x, pixel_y, camera, ray_origin, ray_direction
    );

    float entry_t, exit_t;
    if (!intersect_volume_box(
            ray_origin, ray_direction, entry_t, exit_t)) {
        image[pixel_index] = 0.0f;
        return;
    }

    const float step_length = (exit_t - entry_t) / DVR_STEPS;
    float transmittance = 1.0f;
    float accumulated_intensity = 0.0f;

    for (int step = 0;
         step < DVR_STEPS && transmittance > 1.0e-4f;
         ++step) {
        const float sample_t = entry_t + (step + 0.5f) * step_length;
        const Vec3 sample_position =
            ray_origin + ray_direction * sample_t;

        const float density =
            sample_trilinear(volume, sample_position) * DENSITY_SCALE;
        const float alpha = 1.0f - expf(-density * step_length);

        accumulated_intensity += transmittance * alpha;
        transmittance *= (1.0f - alpha);
    }

    image[pixel_index] = clamp01(accumulated_intensity);
}

// =============================================================================
// 4B. Predicted renderer: front-to-back dense voxel DDA rasterisation
// =============================================================================
__global__ void render_voxel_rasterizer(const float* volume,
                                        float* image,
                                        Camera camera) {
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_index >= PIXEL_COUNT) return;

    const int pixel_x = pixel_index % IMAGE_WIDTH;
    const int pixel_y = pixel_index / IMAGE_WIDTH;

    Vec3 ray_origin, ray_direction;
    generate_camera_ray(
        pixel_x, pixel_y, camera, ray_origin, ray_direction
    );

    float entry_t, exit_t;
    if (!intersect_volume_box(
            ray_origin, ray_direction, entry_t, exit_t)) {
        image[pixel_index] = 0.0f;
        return;
    }

    const float cell_size = 2.0f / GRID_SIZE;
    const Vec3 entry_point =
        ray_origin + ray_direction * (entry_t + 1.0e-5f);

    int voxel_x = min(
        GRID_SIZE - 1,
        max(0, static_cast<int>((entry_point.x + 1.0f) / cell_size))
    );
    int voxel_y = min(
        GRID_SIZE - 1,
        max(0, static_cast<int>((entry_point.y + 1.0f) / cell_size))
    );
    int voxel_z = min(
        GRID_SIZE - 1,
        max(0, static_cast<int>((entry_point.z + 1.0f) / cell_size))
    );

    const int step_x = (ray_direction.x >= 0.0f) ? 1 : -1;
    const int step_y = (ray_direction.y >= 0.0f) ? 1 : -1;
    const int step_z = (ray_direction.z >= 0.0f) ? 1 : -1;

    const float next_boundary_x =
        -1.0f + (voxel_x + (step_x > 0 ? 1 : 0)) * cell_size;
    const float next_boundary_y =
        -1.0f + (voxel_y + (step_y > 0 ? 1 : 0)) * cell_size;
    const float next_boundary_z =
        -1.0f + (voxel_z + (step_z > 0 ? 1 : 0)) * cell_size;

    float next_t_x =
        (fabsf(ray_direction.x) < 1.0e-12f)
            ? 1.0e30f
            : (next_boundary_x - ray_origin.x) / ray_direction.x;
    float next_t_y =
        (fabsf(ray_direction.y) < 1.0e-12f)
            ? 1.0e30f
            : (next_boundary_y - ray_origin.y) / ray_direction.y;
    float next_t_z =
        (fabsf(ray_direction.z) < 1.0e-12f)
            ? 1.0e30f
            : (next_boundary_z - ray_origin.z) / ray_direction.z;

    const float delta_t_x =
        (fabsf(ray_direction.x) < 1.0e-12f)
            ? 1.0e30f
            : fabsf(cell_size / ray_direction.x);
    const float delta_t_y =
        (fabsf(ray_direction.y) < 1.0e-12f)
            ? 1.0e30f
            : fabsf(cell_size / ray_direction.y);
    const float delta_t_z =
        (fabsf(ray_direction.z) < 1.0e-12f)
            ? 1.0e30f
            : fabsf(cell_size / ray_direction.z);

    float current_t = entry_t;
    float transmittance = 1.0f;
    float accumulated_intensity = 0.0f;

    // A ray can cross at most 3*N cells in an N^3 regular grid.
    for (int iteration = 0;
         iteration < 3 * GRID_SIZE &&
         current_t < exit_t &&
         transmittance > 1.0e-4f;
         ++iteration) {
        float segment_end_t = fminf(next_t_x, fminf(next_t_y, next_t_z));
        segment_end_t = fminf(segment_end_t, exit_t);

        const float segment_length =
            fmaxf(0.0f, segment_end_t - current_t);
        const float density =
            volume[voxel_index(voxel_x, voxel_y, voxel_z)] *
            DENSITY_SCALE;
        const float alpha =
            1.0f - expf(-density * segment_length);

        accumulated_intensity += transmittance * alpha;
        transmittance *= (1.0f - alpha);
        current_t = segment_end_t;

        // Advance across the closest next grid boundary.
        if (next_t_x <= next_t_y && next_t_x <= next_t_z) {
            voxel_x += step_x;
            next_t_x += delta_t_x;
            if (voxel_x < 0 || voxel_x >= GRID_SIZE) break;
        } else if (next_t_y <= next_t_z) {
            voxel_y += step_y;
            next_t_y += delta_t_y;
            if (voxel_y < 0 || voxel_y >= GRID_SIZE) break;
        } else {
            voxel_z += step_z;
            next_t_z += delta_t_z;
            if (voxel_z < 0 || voxel_z >= GRID_SIZE) break;
        }
    }

    image[pixel_index] = clamp01(accumulated_intensity);
}

// =============================================================================
// 5. Image metrics
// =============================================================================
__global__ void accumulate_squared_error(const float* image_a,
                                         const float* image_b,
                                         float* squared_error_sum) {
    __shared__ float block_sum[256];

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float squared_error = 0.0f;
    if (index < PIXEL_COUNT) {
        const float difference = image_a[index] - image_b[index];
        squared_error = difference * difference;
    }

    block_sum[threadIdx.x] = squared_error;
    __syncthreads();

    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(squared_error_sum, block_sum[0]);
}

// Same reduction as accumulate_squared_error, sized for the full VOXEL_COUNT
// grid instead of a rendered PIXEL_COUNT image — used to track full-volume
// PSNR during training, for checkpoint "best" selection.
__global__ void accumulate_volume_squared_error(const float* volume_a,
                                                const float* volume_b,
                                                float* squared_error_sum) {
    __shared__ float block_sum[256];

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float squared_error = 0.0f;
    if (index < VOXEL_COUNT) {
        const float difference = volume_a[index] - volume_b[index];
        squared_error = difference * difference;
    }

    block_sum[threadIdx.x] = squared_error;
    __syncthreads();

    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(squared_error_sum, block_sum[0]);
}

__global__ void compute_ssim_map(const float* image_a,
                                 const float* image_b,
                                 float* ssim_map) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= PIXEL_COUNT) return;

    const int x = index % IMAGE_WIDTH;
    const int y = index / IMAGE_WIDTH;

    float mean_a = 0.0f;
    float mean_b = 0.0f;
    int count = 0;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            const int sample_x = min(IMAGE_WIDTH - 1, max(0, x + dx));
            const int sample_y = min(IMAGE_HEIGHT - 1, max(0, y + dy));
            mean_a += image_a[sample_y * IMAGE_WIDTH + sample_x];
            mean_b += image_b[sample_y * IMAGE_WIDTH + sample_x];
            ++count;
        }
    }

    mean_a /= count;
    mean_b /= count;

    float variance_a = 0.0f;
    float variance_b = 0.0f;
    float covariance = 0.0f;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            const int sample_x = min(IMAGE_WIDTH - 1, max(0, x + dx));
            const int sample_y = min(IMAGE_HEIGHT - 1, max(0, y + dy));
            const float deviation_a =
                image_a[sample_y * IMAGE_WIDTH + sample_x] - mean_a;
            const float deviation_b =
                image_b[sample_y * IMAGE_WIDTH + sample_x] - mean_b;

            variance_a += deviation_a * deviation_a;
            variance_b += deviation_b * deviation_b;
            covariance += deviation_a * deviation_b;
        }
    }

    variance_a /= (count - 1);
    variance_b /= (count - 1);
    covariance /= (count - 1);

    constexpr float C1 = 0.01f * 0.01f;
    constexpr float C2 = 0.03f * 0.03f;

    ssim_map[index] =
        ((2.0f * mean_a * mean_b + C1) *
         (2.0f * covariance + C2)) /
        ((mean_a * mean_a + mean_b * mean_b + C1) *
         (variance_a + variance_b + C2));
}

// =============================================================================
// 6. Host-side utilities
// =============================================================================
void save_pgm(const std::string& path,
              const std::vector<float>& image,
              int width = IMAGE_WIDTH,
              int height = IMAGE_HEIGHT) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "Could not write %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }

    file << "P5\n" << width << " " << height << "\n255\n";
    for (float value : image) {
        const auto byte = static_cast<unsigned char>(
            std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f)
        );
        file.write(reinterpret_cast<const char*>(&byte), 1);
    }
}

// Save GT | prediction | amplified difference as one RGB PPM image.
void save_comparison_ppm(const std::string& path,
                         const std::vector<float>& ground_truth,
                         const std::vector<float>& prediction,
                         const std::vector<float>& difference) {
    const int output_width = IMAGE_WIDTH * 3;
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "Could not write %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }

    file << "P6\n" << output_width << " " << IMAGE_HEIGHT << "\n255\n";

    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int panel = 0; panel < 3; ++panel) {
            const std::vector<float>* source = nullptr;
            if (panel == 0) source = &ground_truth;
            if (panel == 1) source = &prediction;
            if (panel == 2) source = &difference;

            for (int x = 0; x < IMAGE_WIDTH; ++x) {
                float value = (*source)[y * IMAGE_WIDTH + x];
                if (panel == 2) value = clamp01(value * 4.0f);

                // GT and prediction are grayscale. Difference uses a simple
                // red-yellow heat-map to make small errors visible.
                unsigned char rgb[3];
                if (panel < 2) {
                    const auto gray = static_cast<unsigned char>(
                        std::lround(clamp01(value) * 255.0f)
                    );
                    rgb[0] = gray;
                    rgb[1] = gray;
                    rgb[2] = gray;
                } else {
                    rgb[0] = static_cast<unsigned char>(
                        std::lround(clamp01(2.0f * value) * 255.0f)
                    );
                    rgb[1] = static_cast<unsigned char>(
                        std::lround(clamp01(2.0f * value - 0.5f) * 255.0f)
                    );
                    rgb[2] = 0;
                }
                file.write(reinterpret_cast<const char*>(rgb), 3);
            }
        }
    }
}

struct Metrics {
    float mse;
    float psnr;
    float ssim;
};

Metrics evaluate_images(const float* device_ground_truth,
                        const float* device_prediction,
                        float* device_ssim_map,
                        float* device_squared_error_sum) {
    CUDA_CHECK(cudaMemset(device_squared_error_sum, 0, sizeof(float)));

    accumulate_squared_error<<<(PIXEL_COUNT + 255) / 256, 256>>>(
        device_ground_truth,
        device_prediction,
        device_squared_error_sum
    );
    compute_ssim_map<<<(PIXEL_COUNT + 255) / 256, 256>>>(
        device_ground_truth,
        device_prediction,
        device_ssim_map
    );
    CUDA_CHECK(cudaGetLastError());

    float squared_error_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(
        &squared_error_sum,
        device_squared_error_sum,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::vector<float> host_ssim(PIXEL_COUNT);
    CUDA_CHECK(cudaMemcpy(
        host_ssim.data(),
        device_ssim_map,
        PIXEL_COUNT * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float mean_ssim = 0.0f;
    for (float value : host_ssim) mean_ssim += value;
    mean_ssim /= PIXEL_COUNT;

    const float mse = squared_error_sum / PIXEL_COUNT;
    const float psnr = -10.0f * std::log10(std::max(mse, 1.0e-12f));
    return {mse, psnr, mean_ssim};
}

// Full-grid PSNR between the reconstructed and ground-truth voxel volumes
// (as opposed to evaluate_images's rendered-projection PSNR). Reuses the
// caller's device_squared_error_sum scratch buffer.
float compute_volume_psnr(
    const float* device_ground_truth_volume,
    const float* device_reconstructed_volume,
    float* device_squared_error_sum
) {
    CUDA_CHECK(cudaMemset(device_squared_error_sum, 0, sizeof(float)));

    accumulate_volume_squared_error<<<(VOXEL_COUNT + 255) / 256, 256>>>(
        device_ground_truth_volume,
        device_reconstructed_volume,
        device_squared_error_sum
    );
    CUDA_CHECK(cudaGetLastError());

    float squared_error_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(
        &squared_error_sum,
        device_squared_error_sum,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const float mse = squared_error_sum / VOXEL_COUNT;
    return -10.0f * std::log10(std::max(mse, 1.0e-12f));
}

template <typename LaunchFunction>
float benchmark_renderer(const char* name,
                         LaunchFunction launch,
                         int warmup_frames,
                         int benchmark_frames) {
    for (int frame = 0; frame < warmup_frames; ++frame) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event));
    for (int frame = 0; frame < benchmark_frames; ++frame) launch();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float total_milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &total_milliseconds, start_event, stop_event
    ));

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    const float milliseconds_per_frame =
        total_milliseconds / benchmark_frames;
    const float fps = 1000.0f / milliseconds_per_frame;

    std::printf(
        "%-28s %9.4f ms/frame   %10.2f FPS\n",
        name,
        milliseconds_per_frame,
        fps
    );
    return fps;
}

void copy_image_to_host(const float* device_image,
                        std::vector<float>& host_image) {
    CUDA_CHECK(cudaMemcpy(
        host_image.data(),
        device_image,
        PIXEL_COUNT * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

void print_usage(const char* program) {
    std::printf(
        "Usage: %s [iterations] [learning_rate] [orbit_frames] "
        "[benchmark_frames]\n\n"
        "Example:\n"
        "  %s 2000 0.001 120 1000\n\n"
        "Arguments:\n"
        "  iterations        SIREN training iterations (default 2000)\n"
        "  learning_rate     Adam learning rate (default 0.001)\n"
        "  orbit_frames      Number of saved orbit views; 0 disables "
        "orbit output (default 120)\n"
        "  benchmark_frames  Timed GPU frames per renderer (default 1000)\n",
        program,
        program
    );
}

// =============================================================================
// 7. save synthetice volume to disk
// =============================================================================
void save_raw_volume(
    const std::string& path,
    const float* device_volume
) {
    std::vector<float> host_volume(VOXEL_COUNT);

    CUDA_CHECK(cudaMemcpy(
        host_volume.data(),
        device_volume,
        VOXEL_COUNT * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::ofstream file(path, std::ios::binary);

    if (!file) {
        std::fprintf(
            stderr,
            "Could not write volume: %s\n",
            path.c_str()
        );
        std::exit(EXIT_FAILURE);
    }

    file.write(
        reinterpret_cast<const char*>(host_volume.data()),
        static_cast<std::streamsize>(
            VOXEL_COUNT * sizeof(float)
        )
    );
}

// Dumps the flat SIREN parameter buffer (PARAM_COUNT floats: all layer
// weights and biases, see SirenLayout) to a raw binary checkpoint file.
// Used for init.bin (pre-training), best.bin (highest volume PSNR seen so
// far), the periodic <iteration>.bin snapshots, and last.bin (final state).
void save_checkpoint(
    const std::string& path,
    const float* device_parameters
) {
    std::vector<float> host_parameters(PARAM_COUNT);

    CUDA_CHECK(cudaMemcpy(
        host_parameters.data(),
        device_parameters,
        PARAM_COUNT * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::ofstream file(path, std::ios::binary);

    if (!file) {
        std::fprintf(
            stderr,
            "Could not write checkpoint: %s\n",
            path.c_str()
        );
        std::exit(EXIT_FAILURE);
    }

    file.write(
        reinterpret_cast<const char*>(host_parameters.data()),
        static_cast<std::streamsize>(
            PARAM_COUNT * sizeof(float)
        )
    );
}

// =============================================================================
// 7. Main experiment
// =============================================================================
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    const int training_iterations =
        (argc > 1) ? std::max(0, std::atoi(argv[1])) : 2000;
    const float learning_rate =
        (argc > 2) ? std::atof(argv[2]) : 1.0e-3f;
    const int orbit_frames =
        (argc > 3) ? std::max(0, std::atoi(argv[3])) : 120;
    const int benchmark_frames =
        (argc > 4) ? std::max(1, std::atoi(argv[4])) : 1000;

    std::filesystem::create_directories("output");
    if (orbit_frames > 0) {
        std::filesystem::create_directories("output/orbit_gt");
        std::filesystem::create_directories("output/orbit_pred");
        std::filesystem::create_directories("output/orbit_diff");
        std::filesystem::create_directories("output/orbit_comparison");
    }

    float* ground_truth_volume = nullptr;
    float* reconstructed_volume = nullptr;
    float* parameters = nullptr;
    float* gradients = nullptr;
    float* first_moment = nullptr;
    float* second_moment = nullptr;
    float* batch_loss = nullptr;
    float* ground_truth_image = nullptr;
    float* predicted_image = nullptr;
    float* ssim_map = nullptr;
    float* squared_error_sum = nullptr;

    CUDA_CHECK(cudaMalloc(
        &ground_truth_volume, VOXEL_COUNT * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(
        &reconstructed_volume, VOXEL_COUNT * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(&parameters, PARAM_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gradients, PARAM_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&first_moment, PARAM_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&second_moment, PARAM_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &ground_truth_image, PIXEL_COUNT * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(
        &predicted_image, PIXEL_COUNT * sizeof(float)
    ));
    CUDA_CHECK(cudaMalloc(&ssim_map, PIXEL_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&squared_error_sum, sizeof(float)));

    create_ground_truth_volume<<<(VOXEL_COUNT + 255) / 256, 256>>>(
        ground_truth_volume
    );
    initialize_siren<<<(PARAM_COUNT + 255) / 256, 256>>>(
        parameters, 1234u
    );
    CUDA_CHECK(cudaMemset(
        first_moment, 0, PARAM_COUNT * sizeof(float)
    ));
    CUDA_CHECK(cudaMemset(
        second_moment, 0, PARAM_COUNT * sizeof(float)
    ));
    CUDA_CHECK(cudaGetLastError());

    // Pre-training snapshot (randomly-initialized SIREN weights).
    save_checkpoint("output/init.bin", parameters);

    std::printf("\nTraining SIREN continuous volume function\n");
    std::printf("Grid: %d^3, hidden width: %d, batch: %d\n",
                GRID_SIZE, HIDDEN_SIZE, TRAIN_BATCH);

    // Tracks the highest full-volume PSNR seen so far, for best.bin.
    float best_volume_psnr = -1.0e30f;

    for (int iteration = 1;
         iteration <= training_iterations;
         ++iteration) {
        CUDA_CHECK(cudaMemset(
            gradients, 0, PARAM_COUNT * sizeof(float)
        ));
        CUDA_CHECK(cudaMemset(batch_loss, 0, sizeof(float)));

        siren_training_step<<<(TRAIN_BATCH + 255) / 256, 256>>>(
            ground_truth_volume,
            parameters,
            gradients,
            batch_loss,
            iteration
        );
        adam_update<<<(PARAM_COUNT + 255) / 256, 256>>>(
            parameters,
            gradients,
            first_moment,
            second_moment,
            iteration,
            learning_rate
        );
        CUDA_CHECK(cudaGetLastError());

        if (iteration == 1 ||
            iteration % 100 == 0 ||
            iteration == training_iterations) {
            float host_loss = 0.0f;
            CUDA_CHECK(cudaMemcpy(
                &host_loss,
                batch_loss,
                sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            // Full-grid reconstruction + PSNR, for checkpointing (this is
            // exact and volume-wide, unlike the noisy minibatch loss above).
            reconstruct_voxel_grid<<<(VOXEL_COUNT + 255) / 256, 256>>>(
                parameters, reconstructed_volume
            );
            CUDA_CHECK(cudaGetLastError());

            const float volume_psnr = compute_volume_psnr(
                ground_truth_volume,
                reconstructed_volume,
                squared_error_sum
            );

            std::printf(
                "iteration %5d   minibatch voxel MSE %.8f   "
                "volume PSNR %.3f dB\n",
                iteration,
                host_loss,
                volume_psnr
            );

            char checkpoint_path[64];
            std::snprintf(
                checkpoint_path,
                sizeof(checkpoint_path),
                "output/%06d.bin",
                iteration
            );
            save_checkpoint(checkpoint_path, parameters);

            if (volume_psnr > best_volume_psnr) {
                best_volume_psnr = volume_psnr;
                save_checkpoint("output/best.bin", parameters);
            }
        }
    }

    // Final trained state.
    save_checkpoint("output/last.bin", parameters);

    reconstruct_voxel_grid<<<(VOXEL_COUNT + 255) / 256, 256>>>(
        parameters, reconstructed_volume
    );
    CUDA_CHECK(cudaGetLastError());

    // Static evaluation camera.
    const Camera evaluation_camera = make_camera(
        0.72f,
        0.57f,
        4.15f,
        45.0f
    );

    render_direct_volume<<<(PIXEL_COUNT + 255) / 256, 256>>>(
        ground_truth_volume,
        ground_truth_image,
        evaluation_camera
    );
    render_voxel_rasterizer<<<(PIXEL_COUNT + 255) / 256, 256>>>(
        reconstructed_volume,
        predicted_image,
        evaluation_camera
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    save_raw_volume(
    "output/ground_truth_volume_f32.raw",
    ground_truth_volume
);

    const Metrics metrics = evaluate_images(
        ground_truth_image,
        predicted_image,
        ssim_map,
        squared_error_sum
    );

    std::vector<float> host_ground_truth(PIXEL_COUNT);
    std::vector<float> host_prediction(PIXEL_COUNT);
    std::vector<float> host_difference(PIXEL_COUNT);

    copy_image_to_host(ground_truth_image, host_ground_truth);
    copy_image_to_host(predicted_image, host_prediction);
    for (int index = 0; index < PIXEL_COUNT; ++index) {
        host_difference[index] =
            std::fabs(host_ground_truth[index] - host_prediction[index]);
    }

    save_pgm("output/gt_projection.pgm", host_ground_truth);
    save_pgm("output/pred_projection.pgm", host_prediction);
    save_pgm("output/diff.pgm", host_difference);
    save_comparison_ppm(
        "output/comparison.ppm",
        host_ground_truth,
        host_prediction,
        host_difference
    );

    std::printf("\nStatic-view quality metrics\n");
    std::printf("MSE   = %.8g\n", metrics.mse);
    std::printf("PSNR  = %.4f dB\n", metrics.psnr);
    std::printf("SSIM  = %.6f\n", metrics.ssim);

    // Kernel-only timing. File I/O, metrics, SIREN training, and reconstruction
    // are deliberately excluded from these FPS numbers.
    std::printf("\nGPU rendering performance at %dx%d\n",
                IMAGE_WIDTH, IMAGE_HEIGHT);
    constexpr int WARMUP_FRAMES = 100;

    benchmark_renderer(
        "GT direct volume renderer",
        [&]() {
            render_direct_volume<<<(PIXEL_COUNT + 255) / 256, 256>>>(
                ground_truth_volume,
                ground_truth_image,
                evaluation_camera
            );
        },
        WARMUP_FRAMES,
        benchmark_frames
    );

    benchmark_renderer(
        "Pred voxel rasterizer",
        [&]() {
            render_voxel_rasterizer<<<(PIXEL_COUNT + 255) / 256, 256>>>(
                reconstructed_volume,
                predicted_image,
                evaluation_camera
            );
        },
        WARMUP_FRAMES,
        benchmark_frames
    );

    // Save a rotating multi-view visualization.
    if (orbit_frames > 0) {
        std::printf("\nSaving %d orbit views...\n", orbit_frames);

        for (int frame = 0; frame < orbit_frames; ++frame) {
            const float azimuth =
                2.0f * PI_F * frame / static_cast<float>(orbit_frames);
            const float elevation =
                0.34f + 0.10f * std::sin(2.0f * azimuth);
            const Camera orbit_camera = make_camera(
                azimuth,
                elevation,
                4.15f,
                45.0f
            );

            render_direct_volume<<<(PIXEL_COUNT + 255) / 256, 256>>>(
                ground_truth_volume,
                ground_truth_image,
                orbit_camera
            );
            render_voxel_rasterizer<<<(PIXEL_COUNT + 255) / 256, 256>>>(
                reconstructed_volume,
                predicted_image,
                orbit_camera
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            copy_image_to_host(ground_truth_image, host_ground_truth);
            copy_image_to_host(predicted_image, host_prediction);
            for (int index = 0; index < PIXEL_COUNT; ++index) {
                host_difference[index] = std::fabs(
                    host_ground_truth[index] - host_prediction[index]
                );
            }

            char frame_name[32];
            std::snprintf(frame_name, sizeof(frame_name), "%04d", frame);

            save_pgm(
                std::string("output/orbit_gt/gt_") + frame_name + ".pgm",
                host_ground_truth
            );
            save_pgm(
                std::string("output/orbit_pred/pred_") + frame_name + ".pgm",
                host_prediction
            );
            save_pgm(
                std::string("output/orbit_diff/diff_") + frame_name + ".pgm",
                host_difference
            );
            save_comparison_ppm(
                std::string("output/orbit_comparison/comparison_") +
                    frame_name + ".ppm",
                host_ground_truth,
                host_prediction,
                host_difference
            );

            if ((frame + 1) % 20 == 0 || frame + 1 == orbit_frames) {
                std::printf("  saved %d/%d views\n", frame + 1, orbit_frames);
            }
        }
    }
    save_raw_volume(
    "output/reconstructed_volume_f32.raw",
    reconstructed_volume
);
    std::printf("\nOutputs\n");
    std::printf("  output/gt_projection.pgm\n");
    std::printf("  output/pred_projection.pgm\n");
    std::printf("  output/diff.pgm\n");
    std::printf("  output/comparison.ppm\n");
    if (orbit_frames > 0) {
        std::printf("  output/orbit_gt/\n");
        std::printf("  output/orbit_pred/\n");
        std::printf("  output/orbit_diff/\n");
        std::printf("  output/orbit_comparison/\n");
    }
    std::printf(
        "Run: python3 visualize.py --make-video --show\n"
        "Run LPIPS: python3 lpips_metric.py "
        "output/gt_projection.pgm output/pred_projection.pgm\n"
    );

    CUDA_CHECK(cudaFree(ground_truth_volume));
    CUDA_CHECK(cudaFree(reconstructed_volume));
    CUDA_CHECK(cudaFree(parameters));
    CUDA_CHECK(cudaFree(gradients));
    CUDA_CHECK(cudaFree(first_moment));
    CUDA_CHECK(cudaFree(second_moment));
    CUDA_CHECK(cudaFree(batch_loss));
    CUDA_CHECK(cudaFree(ground_truth_image));
    CUDA_CHECK(cudaFree(predicted_image));
    CUDA_CHECK(cudaFree(ssim_map));
    CUDA_CHECK(cudaFree(squared_error_sum));

    return EXIT_SUCCESS;
}