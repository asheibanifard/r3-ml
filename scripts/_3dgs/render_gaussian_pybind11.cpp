// pybind11 wrapper for CUDA Gaussian rendering kernel
// No PyTorch headers - uses pybind11 to interface with Python

#include <pybind11/pybind11.h>
#include <torch/extension.h>  // Only for torch::Tensor type, not compilation
#include <cuda_runtime.h>

namespace py = pybind11;

// Forward declare the kernel wrapper from .cu file
extern "C" cudaError_t render_gaussian_cuda(
    const float* d_pts,
    const float* d_means,
    const float* d_log_scales,
    const float* d_quats,
    const float* d_intensities,
    float* d_output,
    int N_pts,
    int M);

// pybind11 wrapper
torch::Tensor render_gaussian_pybind(
    torch::Tensor pts,
    torch::Tensor means,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor intensities) {

    // Validate inputs
    TORCH_CHECK(pts.is_cuda() && pts.dtype() == torch::kFloat32, "pts must be float32 on CUDA");
    TORCH_CHECK(means.is_cuda() && means.dtype() == torch::kFloat32, "means must be float32 on CUDA");
    TORCH_CHECK(log_scales.is_cuda() && log_scales.dtype() == torch::kFloat32, "log_scales must be float32 on CUDA");
    TORCH_CHECK(quats.is_cuda() && quats.dtype() == torch::kFloat32, "quats must be float32 on CUDA");
    TORCH_CHECK(intensities.is_cuda() && intensities.dtype() == torch::kFloat32, "intensities must be float32 on CUDA");

    TORCH_CHECK(pts.size(1) == 3, "pts must have shape (N, 3)");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (M, 3)");
    TORCH_CHECK(log_scales.size(1) == 3, "log_scales must have shape (M, 3)");
    TORCH_CHECK(quats.size(1) == 4, "quats must have shape (M, 4)");
    TORCH_CHECK(intensities.size(0) == means.size(0), "All Gaussian arrays must have same first dimension");

    // Ensure contiguous and on same device
    auto pts_c = pts.contiguous();
    auto means_c = means.contiguous();
    auto log_scales_c = log_scales.contiguous();
    auto quats_c = quats.contiguous();
    auto intensities_c = intensities.contiguous();

    int N_pts = pts_c.size(0);
    int M = means_c.size(0);

    // Create output tensor
    auto output = torch::zeros({N_pts}, pts_c.options());

    // Get pointers
    const float* d_pts = pts_c.data_ptr<float>();
    const float* d_means = means_c.data_ptr<float>();
    const float* d_log_scales = log_scales_c.data_ptr<float>();
    const float* d_quats = quats_c.data_ptr<float>();
    const float* d_intensities = intensities_c.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    // Call CUDA kernel
    cudaError_t err = render_gaussian_cuda(
        d_pts, d_means, d_log_scales, d_quats, d_intensities,
        d_output, N_pts, M);

    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed");

    return output;
}

// pybind11 module
PYBIND11_MODULE(render_gaussian_cuda, m) {
    m.def("render_gaussian_cuda", &render_gaussian_pybind,
          "Render Gaussian mixture field",
          py::arg("pts"), py::arg("means"), py::arg("log_scales"),
          py::arg("quats"), py::arg("intensities"));
}
