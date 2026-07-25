#include <torch/extension.h>
#include <vector>

torch::Tensor gaussian_forward_cuda(
    const torch::Tensor& means,
    const torch::Tensor& precision,
    const torch::Tensor& confidences,
    const torch::Tensor& features,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const double cutoff_squared,
    const double epsilon);

std::vector<torch::Tensor> gaussian_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& means,
    const torch::Tensor& precision,
    const torch::Tensor& confidences,
    const torch::Tensor& features,
    const double cutoff_squared,
    const double epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gaussian_forward_cuda, "3D Gaussian volume forward (CUDA)");
    m.def("backward", &gaussian_backward_cuda, "3D Gaussian volume backward (CUDA)");
}
