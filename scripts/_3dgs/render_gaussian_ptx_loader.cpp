#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef CUBIN_PATH
#define CUBIN_PATH "src/_3dgs/render_gaussian_kernel.cubin"
#endif

#define BLOCK_SIZE 256
#define GAUSSIAN_CHUNK_SIZE 1024

static CUmodule g_module = nullptr;
static CUfunction g_render_kernel = nullptr;
static CUfunction g_volume_kernel = nullptr;
static bool g_loaded = false;

static cudaError_t map_cu_error(CUresult result) {
    switch (result) {
        case CUDA_SUCCESS:
            return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_NOT_INITIALIZED:
            return cudaErrorInitializationError;
        case CUDA_ERROR_INVALID_CONTEXT:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_INVALID_DEVICE:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return cudaErrorMemoryAllocation;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return cudaErrorNoKernelImageForDevice;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return cudaErrorLaunchOutOfResources;
        default:
            return cudaErrorUnknown;
    }
}

static cudaError_t ensure_module_loaded() {
    if (g_loaded) {
        return cudaSuccess;
    }

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        return map_cu_error(result);
    }

    CUcontext context = nullptr;
    result = cuCtxGetCurrent(&context);
    if (result != CUDA_SUCCESS) {
        return map_cu_error(result);
    }

    if (context == nullptr) {
        int device_idx = 0;
        cudaError_t runtime_err = cudaGetDevice(&device_idx);
        if (runtime_err != cudaSuccess) {
            return runtime_err;
        }

        CUdevice device = 0;
        result = cuDeviceGet(&device, device_idx);
        if (result != CUDA_SUCCESS) {
            return map_cu_error(result);
        }

        result = cuDevicePrimaryCtxRetain(&context, device);
        if (result != CUDA_SUCCESS) {
            return map_cu_error(result);
        }

        result = cuCtxSetCurrent(context);
        if (result != CUDA_SUCCESS) {
            return map_cu_error(result);
        }
    }

    result = cuModuleLoad(&g_module, CUBIN_PATH);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to load cubin module %s: %d\n", CUBIN_PATH, result);
        return map_cu_error(result);
    }

    result = cuModuleGetFunction(&g_render_kernel, g_module, "render_gaussian_kernel");
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to find kernel render_gaussian_kernel: %d\n", result);
        return map_cu_error(result);
    }

    result = cuModuleGetFunction(&g_volume_kernel, g_module, "reconstruct_gaussian_volume_kernel");
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to find kernel reconstruct_gaussian_volume_kernel: %d\n", result);
        return map_cu_error(result);
    }

    g_loaded = true;
    return cudaSuccess;
}

static cudaError_t launch_render_chunk(
    const float* d_pts,
    const float* d_means,
    const float* d_log_scales,
    const float* d_quats,
    const float* d_intensities,
    float* d_output,
    int N_pts,
    int M) {

    int blocks = (N_pts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = {
        (void*)&d_pts,
        (void*)&d_means,
        (void*)&d_log_scales,
        (void*)&d_quats,
        (void*)&d_intensities,
        (void*)&d_output,
        (void*)&N_pts,
        (void*)&M,
    };

    CUresult result = cuLaunchKernel(
        g_render_kernel,
        blocks, 1, 1,
        BLOCK_SIZE, 1, 1,
        0,
        nullptr,
        args,
        nullptr);
    if (result != CUDA_SUCCESS) {
        return map_cu_error(result);
    }

    result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        return map_cu_error(result);
    }

    return cudaSuccess;
}

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
        if (!d_pts || !d_means || !d_log_scales || !d_quats || !d_intensities || !d_output) {
            return cudaErrorInvalidDevicePointer;
        }

        cudaError_t load_err = ensure_module_loaded();
        if (load_err != cudaSuccess) {
            return load_err;
        }

        for (int start = 0; start < M; start += GAUSSIAN_CHUNK_SIZE) {
            int chunk = M - start;
            if (chunk > GAUSSIAN_CHUNK_SIZE) {
                chunk = GAUSSIAN_CHUNK_SIZE;
            }

            cudaError_t chunk_err = launch_render_chunk(
                d_pts,
                d_means + start * 3,
                d_log_scales + start * 3,
                d_quats + start * 4,
                d_intensities + start,
                d_output,
                N_pts,
                chunk);
            if (chunk_err != cudaSuccess) {
                return chunk_err;
            }
        }

        return cudaSuccess;
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
        if (!d_means || !d_log_scales || !d_quats || !d_intensities || !d_output) {
            return cudaErrorInvalidDevicePointer;
        }

        cudaError_t load_err = ensure_module_loaded();
        if (load_err != cudaSuccess) {
            return load_err;
        }

        int total_voxels = Dz * Dy * Dx;
        for (int start = 0; start < M; start += GAUSSIAN_CHUNK_SIZE) {
            int chunk = M - start;
            if (chunk > GAUSSIAN_CHUNK_SIZE) {
                chunk = GAUSSIAN_CHUNK_SIZE;
            }

            int blocks = (total_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
            void* args[] = {
                (void*)&d_means,
                (void*)&d_log_scales,
                (void*)&d_quats,
                (void*)&d_intensities,
                (void*)&d_output,
                (void*)&Dz,
                (void*)&Dy,
                (void*)&Dx,
                (void*)&chunk,
            };

            CUresult result = cuLaunchKernel(
                g_volume_kernel,
                blocks, 1, 1,
                BLOCK_SIZE, 1, 1,
                0,
                nullptr,
                args,
                nullptr);
            if (result != CUDA_SUCCESS) {
                return map_cu_error(result);
            }

            result = cuCtxSynchronize();
            if (result != CUDA_SUCCESS) {
                return map_cu_error(result);
            }

            d_means += chunk * 3;
            d_log_scales += chunk * 3;
            d_quats += chunk * 4;
            d_intensities += chunk;
        }

        return cudaSuccess;
    }
}
