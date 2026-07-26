/*
 * bindings.cpp — pybind11 module definition for the eval (reconstruct_volume
 * / splat_mip) CUDA extension. Contains no kernel logic of its own; see
 * reconstruct_volume.cu, splat_mip.cu.
 */

#include <torch/extension.h>

#include "ext.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reconstruct_volume", &reconstruct_volume,
          py::arg("means"), py::arg("log_s"), py::arg("quats"), py::arg("inten"),
          py::arg("lo_x"), py::arg("hi_x"), py::arg("lo_y"), py::arg("hi_y"),
          py::arg("lo_z"), py::arg("hi_z"),
          py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("scale_min"), py::arg("mahal_clamp"),
          "Evaluate Gaussian cloud at every voxel centre. Returns a flat float32 CUDA tensor.");

    m.def("splat_mip", &splat_mip,
          py::arg("means"), py::arg("log_s"), py::arg("quats"), py::arg("inten"),
          py::arg("lo_x"), py::arg("hi_x"), py::arg("lo_y"), py::arg("hi_y"),
          py::arg("lo_z"), py::arg("hi_z"),
          py::arg("out_h"), py::arg("out_w"), py::arg("depth_samples"), py::arg("view_axis"),
          py::arg("scale_min"), py::arg("mahal_clamp"),
          py::arg("max_gauss_per_tile") = 0,
          py::arg("print_stats") = false,
          py::arg("clamp_output") = true,
          "Tiled true Maximum Intensity Projection of a Gaussian mixture with conservative culling.");
}
