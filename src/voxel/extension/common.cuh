// voxel/extension/common.cuh
//
// Shared device helpers for the SIREN-voxel extension: Vec3 math, the
// pinhole camera basis, and the fixed-size SIREN architecture layout.
//
// HIDDEN_SIZE/HIDDEN_LAYERS must be compile-time constants (siren_forward
// keeps its activations in a fixed-size local array), so they are injected
// as preprocessor defines by extension/loader.py at JIT-compile time —
// changing them triggers a real recompile, they are not runtime kernel
// arguments like grid_size/image_width/train_batch below.
#pragma once

#include <torch/extension.h>

#include <cmath>

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 128
#endif

#ifndef HIDDEN_LAYERS
#define HIDDEN_LAYERS 4
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int kThreadsPerBlock = 256;
constexpr int kInputSize = 3;
constexpr float kPiF = 3.14159265358979323846f;

inline int64_t num_blocks(int64_t n, int threads = kThreadsPerBlock) {
    return (n + threads - 1) / threads;
}

// ── Small vector type shared by host and device ────────────────────────────
struct Vec3 {
    float x, y, z;
};

__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline Vec3 operator*(Vec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__host__ __device__ inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ inline Vec3 normalize(Vec3 v) {
    const float length_squared = fmaxf(dot(v, v), 1.0e-20f);
    return v * (1.0f / sqrtf(length_squared));
}

__host__ __device__ inline float clamp01(float value) {
    return fminf(1.0f, fmaxf(0.0f, value));
}

__host__ __device__ inline int64_t voxel_index(
        int64_t x, int64_t y, int64_t z, int64_t grid_size) {
    return (z * grid_size + y) * grid_size + x;
}

// Camera basis, packed as 13 floats: origin(3) forward(3) right(3) up(3)
// tan_half_fov(1). Built host-side in Python (rendering.py's make_camera)
// and passed to the render kernels as a small CUDA tensor.
struct Camera {
    Vec3 origin;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float tan_half_fov;
};

__device__ inline Camera load_camera(const float* __restrict__ camera_ptr) {
    Camera camera{};
    camera.origin = {camera_ptr[0], camera_ptr[1], camera_ptr[2]};
    camera.forward = {camera_ptr[3], camera_ptr[4], camera_ptr[5]};
    camera.right = {camera_ptr[6], camera_ptr[7], camera_ptr[8]};
    camera.up = {camera_ptr[9], camera_ptr[10], camera_ptr[11]};
    camera.tan_half_fov = camera_ptr[12];
    return camera;
}

constexpr int kCameraParamCount = 13;

// ── SIREN parameter layout ──────────────────────────────────────────────────
// Flat parameter buffer layout: first layer (INPUT_SIZE -> HIDDEN_SIZE)
// weights+biases, then (HIDDEN_LAYERS - 1) hidden layers (HIDDEN_SIZE ->
// HIDDEN_SIZE) weights+biases, then a linear (HIDDEN_SIZE -> 1) output.
struct SirenLayout {
    static_assert(HIDDEN_SIZE >= 1, "HIDDEN_SIZE must be positive.");
    static_assert(HIDDEN_LAYERS >= 1, "HIDDEN_LAYERS must be at least one.");

    static constexpr int FIRST_WEIGHT_COUNT = kInputSize * HIDDEN_SIZE;
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
