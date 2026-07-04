// Simple test kernel to verify memory access works

#include <cuda_runtime.h>

__global__ void test_fill_kernel(float* output, int N, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        output[idx] = value;
    }
}

extern "C" {
    cudaError_t test_fill(float* d_output, int N, float value) {
        int blocks = (N + 255) / 256;
        test_fill_kernel<<<blocks, 256>>>(d_output, N, value);
        cudaError_t err = cudaGetLastError();
        if(err == cudaSuccess) {
            err = cudaDeviceSynchronize();
        }
        return err;
    }
}
