/*
 * ext.h — declarations of the Python-visible entry points implemented in
 * reconstruct_volume.cu and splat_mip.cu. Included only by bindings.cpp.
 */
#pragma once

#include <torch/extension.h>

torch::Tensor reconstruct_volume(
        torch::Tensor means, torch::Tensor log_s, torch::Tensor quats, torch::Tensor inten,
        float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z,
        int D, int H, int W, float scale_min, float mahal_clamp);

torch::Tensor splat_mip(
        torch::Tensor means, torch::Tensor log_s, torch::Tensor quats, torch::Tensor inten,
        float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z,
        int out_h, int out_w, int depth_samples, int view_axis,
        float scale_min, float mahal_clamp,
        int max_gauss_per_tile,
        bool print_stats,
        bool clamp_output);
