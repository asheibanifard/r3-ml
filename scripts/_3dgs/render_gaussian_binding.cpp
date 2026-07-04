// Minimal binding for precompiled CUDA kernel
// Uses only pybind11, links with nvcc-compiled kernel object

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace py = pybind11;

// Forward declare the precompiled kernel wrapper from .cu file
extern "C" cudaError_t render_gaussian_cuda(
    const float* d_pts,
    const float* d_means,
    const float* d_log_scales,
    const float* d_quats,
    const float* d_intensities,
    float* d_output,
    int N_pts,
    int M);

// Binding function
py::object render_gaussian_kernel_binding(
    py::object pts_obj,
    py::object means_obj,
    py::object log_scales_obj,
    py::object quats_obj,
    py::object intensities_obj) {

  // Extract device pointers from torch tensors (using Python object introspection)
  // We expect torch tensors, but pybind11 doesn't know about torch types
  // So we use Python API to extract data_ptr() directly

  uintptr_t d_pts = py::cast<uintptr_t>(pts_obj.attr("data_ptr")());
  uintptr_t d_means = py::cast<uintptr_t>(means_obj.attr("data_ptr")());
  uintptr_t d_log_scales = py::cast<uintptr_t>(log_scales_obj.attr("data_ptr")());
  uintptr_t d_quats = py::cast<uintptr_t>(quats_obj.attr("data_ptr")());
  uintptr_t d_intensities = py::cast<uintptr_t>(intensities_obj.attr("data_ptr")());

  int N_pts = py::cast<int>(pts_obj.attr("shape")[0]);
  int M = py::cast<int>(means_obj.attr("shape")[0]);

  // Get output tensor object and extract pointer
  py::object torch_module = py::module_::import("torch");
  py::list shape = py::list();
  shape.append(N_pts);
  py::object output = torch_module.attr("zeros")(
      shape,
      py::arg("device") = pts_obj.attr("device"),
      py::arg("dtype") = torch_module.attr("float32"));
  uintptr_t d_output = py::cast<uintptr_t>(output.attr("data_ptr")());

  // Call the precompiled kernel
  cudaError_t err = render_gaussian_cuda(
      (const float*)d_pts,
      (const float*)d_means,
      (const float*)d_log_scales,
      (const float*)d_quats,
      (const float*)d_intensities,
      (float*)d_output,
      N_pts, M);

  if(err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed with error ") +
                             std::to_string((int)err));
  }

  return output;
}

PYBIND11_MODULE(render_gaussian_cuda, m) {
  m.def("render_gaussian_cuda", &render_gaussian_kernel_binding,
        "Render Gaussian mixture field",
        py::arg("pts"), py::arg("means"), py::arg("log_scales"),
        py::arg("quats"), py::arg("intensities"));
}
