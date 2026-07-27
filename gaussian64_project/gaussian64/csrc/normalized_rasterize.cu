#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void normalized_rasterize_kernel(
    const float* __restrict__ means,
    const float* __restrict__ precision6,
    const float* __restrict__ confidence,
    const float* __restrict__ features,
    int n,
    int depth,
    int height,
    int width,
    float epsilon,
    float cutoff2,
    float* __restrict__ output) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int voxels = depth * height * width;
    if (index >= voxels) {
        return;
    }

    const int x_index = index % width;
    const int y_index = (index / width) % height;
    const int z_index = index / (width * height);

    // Model coordinates follow the voxel-centre convention i + 0.5.
    const float x = static_cast<float>(x_index) + 0.5f;
    const float y = static_cast<float>(y_index) + 0.5f;
    const float z = static_cast<float>(z_index) + 0.5f;

    float numerator = 0.0f;
    float denominator = 0.0f;

    for (int g = 0; g < n; ++g) {
        const float dx = x - means[g * 3 + 0];
        const float dy = y - means[g * 3 + 1];
        const float dz = z - means[g * 3 + 2];

        const float* p = precision6 + g * 6;
        const float mahal =
            p[0] * dx * dx
            + 2.0f * p[1] * dx * dy
            + 2.0f * p[2] * dx * dz
            + p[3] * dy * dy
            + 2.0f * p[4] * dy * dz
            + p[5] * dz * dz;

        if (mahal < cutoff2) {
            const float weight = confidence[g] * __expf(-0.5f * mahal);
            numerator += weight * features[g];
            denominator += weight;
        }
    }

    output[index] = numerator / (denominator + epsilon);
}

void check_input(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

}  // namespace

torch::Tensor normalized_rasterize_cuda(
    torch::Tensor means,
    torch::Tensor precision6,
    torch::Tensor confidence,
    torch::Tensor features,
    int64_t depth,
    int64_t height,
    int64_t width,
    double epsilon,
    double cutoff_sigma) {

    check_input(means, "means");
    check_input(precision6, "precision6");
    check_input(confidence, "confidence");
    check_input(features, "features");

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must have shape [N,3]");
    TORCH_CHECK(
        precision6.dim() == 2 && precision6.size(1) == 6,
        "precision6 must have shape [N,6]"
    );
    const int64_t n = means.size(0);
    TORCH_CHECK(precision6.size(0) == n, "precision6 population mismatch");
    TORCH_CHECK(confidence.numel() == n, "confidence population mismatch");
    TORCH_CHECK(features.numel() == n, "features population mismatch");
    TORCH_CHECK(
        means.device() == precision6.device()
            && means.device() == confidence.device()
            && means.device() == features.device(),
        "all inputs must be on the same CUDA device"
    );
    TORCH_CHECK(depth > 0 && height > 0 && width > 0, "volume dimensions must be positive");
    TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
    TORCH_CHECK(cutoff_sigma > 0.0, "cutoff_sigma must be positive");

    const c10::cuda::CUDAGuard device_guard(means.device());
    auto output = torch::empty({depth * height * width}, means.options());

    constexpr int threads = 256;
    const int64_t voxels = depth * height * width;
    const int blocks = static_cast<int>((voxels + threads - 1) / threads);

    normalized_rasterize_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        means.data_ptr<float>(),
        precision6.data_ptr<float>(),
        confidence.data_ptr<float>(),
        features.data_ptr<float>(),
        static_cast<int>(n),
        static_cast<int>(depth),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<float>(epsilon),
        static_cast<float>(cutoff_sigma * cutoff_sigma),
        output.data_ptr<float>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.reshape({depth, height, width});
}
