// Pure CUDA kernels for Gaussian field rendering.
// No PyTorch headers - compiles independently with nvcc.

#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// Debug helper.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
        } \
    } while (0)

__device__ __forceinline__ void quat_to_rot(float w, float x, float y, float z, float* R) {
    R[0] = 1.0f - 2.0f * (y * y + z * z);
    R[1] = 2.0f * (x * y - w * z);
    R[2] = 2.0f * (x * z + w * y);
    R[3] = 2.0f * (x * y + w * z);
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z - w * x);
    R[6] = 2.0f * (x * z - w * y);
    R[7] = 2.0f * (y * z + w * x);
    R[8] = 1.0f - 2.0f * (x * x + y * y);
}

__device__ __forceinline__ void normalize_quaternion(float* w, float* x, float* y, float* z) {
    float norm = sqrtf((*w) * (*w) + (*x) * (*x) + (*y) * (*y) + (*z) * (*z));
    if (norm > 1e-8f) {
        float inv = 1.0f / norm;
        *w *= inv;
        *x *= inv;
        *y *= inv;
        *z *= inv;
    } else {
        *w = 1.0f;
        *x = 0.0f;
        *y = 0.0f;
        *z = 0.0f;
    }
}

__device__ __forceinline__ float gaussian_eval_rotated(
    float dx,
    float dy,
    float dz,
    const float* log_scale,
    const float* quat,
    float intensity) {

    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];
    normalize_quaternion(&w, &x, &y, &z);

    float sx = fmaxf(0.001f, fminf(20.0f, expf(log_scale[0])));
    float sy = fmaxf(0.001f, fminf(20.0f, expf(log_scale[1])));
    float sz = fmaxf(0.001f, fminf(20.0f, expf(log_scale[2])));

    float R[9];
    quat_to_rot(w, x, y, z, R);

    // Rotate the query into the Gaussian local frame: local = R^T * diff.
    float lx = R[0] * dx + R[3] * dy + R[6] * dz;
    float ly = R[1] * dx + R[4] * dy + R[7] * dz;
    float lz = R[2] * dx + R[5] * dy + R[8] * dz;

    float mahal_sq = (lx / sx) * (lx / sx) + (ly / sy) * (ly / sy) + (lz / sz) * (lz / sz);
    mahal_sq = fminf(10.0f, fmaxf(0.0f, mahal_sq));

    return intensity * expf(-0.5f * mahal_sq);
}

// Point-sampling kernel for arbitrary query points.
extern "C" __global__ void render_gaussian_kernel(
    const float* pts,         // (N_pts, 3) query points
    const float* means,       // (M, 3) Gaussian centers
    const float* log_scales,  // (M, 3) log scales
    const float* quats,       // (M, 4) quaternions (w, x, y, z)
    const float* intensities, // (M,) sigmoid(alpha)
    float* output,            // (N_pts,) output intensities
    int N_pts,
    int M) {

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N_pts) return;

    float px = pts[pt_idx * 3 + 0];
    float py = pts[pt_idx * 3 + 1];
    float pz = pts[pt_idx * 3 + 2];

    float intensity = 0.0f;
    for (int m = 0; m < M; m++) {
        float dx = px - means[m * 3 + 0];
        float dy = py - means[m * 3 + 1];
        float dz = pz - means[m * 3 + 2];

        intensity += gaussian_eval_rotated(
            dx,
            dy,
            dz,
            &log_scales[m * 3],
            &quats[m * 4],
            intensities[m]);
    }

    output[pt_idx] += intensity;
}

// Dense voxel reconstruction kernel for a normalized block-local grid [-1, 1]^3.
extern "C" __global__ void reconstruct_gaussian_volume_kernel(
    const float* means,       // (M, 3) Gaussian centers in block-local coords
    const float* log_scales,  // (M, 3) log scales
    const float* quats,       // (M, 4) quaternions (w, x, y, z)
    const float* intensities, // (M,) sigmoid(alpha)
    float* output,            // (Dz * Dy * Dx,) output volume
    int Dz,
    int Dy,
    int Dx,
    int M) {

    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = Dz * Dy * Dx;
    if (voxel_idx >= total_voxels) return;

    int z = voxel_idx / (Dy * Dx);
    int rem = voxel_idx - z * (Dy * Dx);
    int y = rem / Dx;
    int x = rem - y * Dx;

    float px = (Dx > 1) ? (-1.0f + 2.0f * ((float)x / (float)(Dx - 1))) : 0.0f;
    float py = (Dy > 1) ? (-1.0f + 2.0f * ((float)y / (float)(Dy - 1))) : 0.0f;
    float pz = (Dz > 1) ? (-1.0f + 2.0f * ((float)z / (float)(Dz - 1))) : 0.0f;

    float value = 0.0f;
    for (int m = 0; m < M; m++) {
        float dx = px - means[m * 3 + 0];
        float dy = py - means[m * 3 + 1];
        float dz = pz - means[m * 3 + 2];

        value += gaussian_eval_rotated(
            dx,
            dy,
            dz,
            &log_scales[m * 3],
            &quats[m * 4],
            intensities[m]);
    }

    output[voxel_idx] += value;
}

// Wrapper functions for Python.
extern "C" {
    cudaError_t render_gaussian_cuda(
        const float* d_pts,
        const float* d_means,
        const float* d_log_scales,
        const float* d_quats,
        const float* d_intensities,
        float* d_output,
        int N_pts,
        int M) {

        if (N_pts <= 0 || M <= 0) return cudaErrorInvalidValue;
        if (!d_pts || !d_means || !d_log_scales || !d_quats || !d_intensities || !d_output)
            return cudaErrorInvalidDevicePointer;

        int blocks = (N_pts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        render_gaussian_kernel<<<blocks, BLOCK_SIZE>>>(
            d_pts,
            d_means,
            d_log_scales,
            d_quats,
            d_intensities,
            d_output,
            N_pts,
            M);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        err = cudaDeviceSynchronize();
        return err;
    }

    cudaError_t reconstruct_gaussian_volume_cuda(
        const float* d_means,
        const float* d_log_scales,
        const float* d_quats,
        const float* d_intensities,
        float* d_output,
        int Dz,
        int Dy,
        int Dx,
        int M) {

        if (Dz <= 0 || Dy <= 0 || Dx <= 0 || M <= 0) return cudaErrorInvalidValue;
        if (!d_means || !d_log_scales || !d_quats || !d_intensities || !d_output)
            return cudaErrorInvalidDevicePointer;

        int total_voxels = Dz * Dy * Dx;
        int blocks = (total_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reconstruct_gaussian_volume_kernel<<<blocks, BLOCK_SIZE>>>(
            d_means,
            d_log_scales,
            d_quats,
            d_intensities,
            d_output,
            Dz,
            Dy,
            Dx,
            M);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        err = cudaDeviceSynchronize();
        return err;
    }
}
