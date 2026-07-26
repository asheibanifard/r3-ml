// voxel/extension/metrics.cu
//
// Image/volume comparison metrics: MSE + PSNR via a block-reduced squared
// error accumulator, and a windowed SSIM map (7x7 window, matching the
// classic SSIM formulation).
#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace {

__global__ void accumulate_squared_error_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ squared_error_sum,
        int64_t count) {
    __shared__ float block_sum[kThreadsPerBlock];

    const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    float squared_error = 0.0f;
    if (index < count) {
        const float difference = a[index] - b[index];
        squared_error = difference * difference;
    }

    block_sum[threadIdx.x] = squared_error;
    __syncthreads();

    for (int offset = kThreadsPerBlock / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(squared_error_sum, block_sum[0]);
}

__global__ void compute_ssim_map_kernel(
        const float* __restrict__ image_a,
        const float* __restrict__ image_b,
        float* __restrict__ ssim_map,
        int64_t image_width, int64_t image_height) {
    const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t pixel_count = image_width * image_height;
    if (index >= pixel_count) return;

    const int64_t x = index % image_width;
    const int64_t y = index / image_width;

    float mean_a = 0.0f;
    float mean_b = 0.0f;
    int count = 0;

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            const int64_t sample_x = min(image_width - 1, max(int64_t{0}, x + dx));
            const int64_t sample_y = min(image_height - 1, max(int64_t{0}, y + dy));
            mean_a += image_a[sample_y * image_width + sample_x];
            mean_b += image_b[sample_y * image_width + sample_x];
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
            const int64_t sample_x = min(image_width - 1, max(int64_t{0}, x + dx));
            const int64_t sample_y = min(image_height - 1, max(int64_t{0}, y + dy));
            const float deviation_a = image_a[sample_y * image_width + sample_x] - mean_a;
            const float deviation_b = image_b[sample_y * image_width + sample_x] - mean_b;

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
        ((2.0f * mean_a * mean_b + C1) * (2.0f * covariance + C2)) /
        ((mean_a * mean_a + mean_b * mean_b + C1) * (variance_a + variance_b + C2));
}

}  // namespace

// ── Python-visible entry points ─────────────────────────────────────────────

// Returns (mse, psnr, ssim) for two same-shape [H,W] images in [0,1].
std::tuple<double, double, double> evaluate_images_cuda(
        const torch::Tensor& image_a,
        const torch::Tensor& image_b) {
    CHECK_INPUT(image_a);
    CHECK_INPUT(image_b);
    TORCH_CHECK(image_a.sizes() == image_b.sizes(), "image_a and image_b must have the same shape");
    TORCH_CHECK(image_a.dim() == 2, "images must have shape [H,W]");

    const int64_t image_height = image_a.size(0);
    const int64_t image_width = image_a.size(1);
    const int64_t pixel_count = image_width * image_height;

    auto squared_error_sum = torch::zeros({1}, image_a.options());
    auto ssim_map = torch::empty({pixel_count}, image_a.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    accumulate_squared_error_kernel<<<num_blocks(pixel_count), kThreadsPerBlock, 0, stream>>>(
        image_a.data_ptr<float>(), image_b.data_ptr<float>(),
        squared_error_sum.data_ptr<float>(), pixel_count
    );
    compute_ssim_map_kernel<<<num_blocks(pixel_count), kThreadsPerBlock, 0, stream>>>(
        image_a.data_ptr<float>(), image_b.data_ptr<float>(),
        ssim_map.data_ptr<float>(), image_width, image_height
    );

    const double mse = squared_error_sum.item<float>() / static_cast<double>(pixel_count);
    const double psnr = -10.0 * std::log10(std::max(mse, 1.0e-12));
    const double ssim = ssim_map.mean().item<float>();

    return {mse, psnr, ssim};
}

// Full-grid PSNR between two same-shape [G,G,G] volumes (as opposed to
// evaluate_images_cuda's rendered-projection PSNR).
double compute_volume_psnr_cuda(
        const torch::Tensor& volume_a,
        const torch::Tensor& volume_b) {
    CHECK_INPUT(volume_a);
    CHECK_INPUT(volume_b);
    TORCH_CHECK(volume_a.sizes() == volume_b.sizes(), "volume_a and volume_b must have the same shape");

    const int64_t voxel_count = volume_a.numel();
    auto squared_error_sum = torch::zeros({1}, volume_a.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    accumulate_squared_error_kernel<<<num_blocks(voxel_count), kThreadsPerBlock, 0, stream>>>(
        volume_a.data_ptr<float>(), volume_b.data_ptr<float>(),
        squared_error_sum.data_ptr<float>(), voxel_count
    );

    const double mse = squared_error_sum.item<float>() / static_cast<double>(voxel_count);
    return -10.0 * std::log10(std::max(mse, 1.0e-12));
}
