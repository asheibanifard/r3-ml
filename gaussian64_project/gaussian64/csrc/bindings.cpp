#include <torch/extension.h>

torch::Tensor normalized_rasterize_cuda(
    torch::Tensor means,
    torch::Tensor precision6,
    torch::Tensor confidence,
    torch::Tensor features,
    int64_t depth,
    int64_t height,
    int64_t width,
    double epsilon,
    double cutoff_sigma);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "normalized_rasterize",
        &normalized_rasterize_cuda,
        "Normalized 3D Gaussian volume rasterisation (CUDA)"
    );
}
