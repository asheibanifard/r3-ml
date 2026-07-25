/*
 * bindings.cpp — pybind11 module definition for the training (forward /
 * backward / regularisation) CUDA extension. Contains no kernel logic of
 * its own; see forward.cu, backward.cu, regularization.cu.
 */

#include "ext.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &gaussian_forward,  "Gaussian splatting forward (CUDA)");
    m.def("backward", &gaussian_backward, "Gaussian splatting backward (CUDA)");
    m.def("gaussian_reg", &gaussian_reg,  "Fused regularisation loss + grad (CUDA)");
}
