/*
 * ext.h — declarations of the Python-visible entry points implemented in
 * forward.cu, backward.cu, and regularization.cu. Included only by
 * bindings.cpp, so the pybind11 module definition stays decoupled from the
 * kernel implementations themselves.
 */
#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

torch::Tensor gaussian_forward(
        torch::Tensor pts,
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor gain,
        torch::Tensor inten,
        float scale_min,
        float mahal_clamp);

py::tuple gaussian_backward(
        torch::Tensor grad_out,
        torch::Tensor pts,
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor gain,
        torch::Tensor inten,
        float scale_min,
        float mahal_clamp);

py::tuple gaussian_reg(
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor inten,
        torch::Tensor volume,
        int64_t N, int64_t D, int64_t H, int64_t W,
        float w_scale, float w_ceil,  float cap,
        float w_out,   float out_thresh,
        float w_aniso,
        float w_count, float w_L1,
        float w_cov,   float s_ref,   float cap_over_sref,
        float w_spar,
        float inv_N);
