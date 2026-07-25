/*
 * common.cuh — shared constants and host/device helpers for the eval
 * (reconstruct_volume / splat_mip) extension.
 *
 * quat_to_rotmat_host / projected_axes_for_view_axis / depth_bounds_for_view_axis
 * are plain `static inline` host functions (not __device__), so — like the
 * device-side helpers in the training extension's common.cuh — they are
 * safe to #include from multiple .cu translation units without any
 * cross-TU linkage requirement.
 */
#pragma once

#include <algorithm>
#include <cmath>

static constexpr int kBlockSize = 256;
static constexpr int kTilePixW = 16;
static constexpr int kTilePixH = 16;
#define K_TILE_GAUSS 128

__host__ __device__ static inline float softplus_stable(float x) {
    return (x > 20.f) ? x : std::log1pf(std::exp(x));
}

static inline void quat_to_rotmat_host(
        float qw, float qx, float qy, float qz,
        float& R00, float& R01, float& R02,
        float& R10, float& R11, float& R12,
        float& R20, float& R21, float& R22) {
    const float n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    const float inv_n = (n > 0.f) ? (1.f / n) : 1.f;
    qw *= inv_n;
    qx *= inv_n;
    qy *= inv_n;
    qz *= inv_n;

    R00 = 1.f - 2.f * (qy * qy + qz * qz);
    R01 = 2.f * (qx * qy - qw * qz);
    R02 = 2.f * (qx * qz + qw * qy);
    R10 = 2.f * (qx * qy + qw * qz);
    R11 = 1.f - 2.f * (qx * qx + qz * qz);
    R12 = 2.f * (qy * qz - qw * qx);
    R20 = 2.f * (qx * qz - qw * qy);
    R21 = 2.f * (qy * qz + qw * qx);
    R22 = 1.f - 2.f * (qx * qx + qy * qy);
}

static inline void projected_axes_for_view_axis(
        int view_axis,
        int& u_axis, int& v_axis,
        float& u_lo, float& u_hi,
        float& v_lo, float& v_hi,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z) {
    if (view_axis == 0) {
        u_axis = 0; v_axis = 1;
        u_lo = lo_x; u_hi = hi_x;
        v_lo = lo_y; v_hi = hi_y;
    } else if (view_axis == 1) {
        u_axis = 0; v_axis = 2;
        u_lo = lo_x; u_hi = hi_x;
        v_lo = lo_z; v_hi = hi_z;
    } else {
        u_axis = 1; v_axis = 2;
        u_lo = lo_y; u_hi = hi_y;
        v_lo = lo_z; v_hi = hi_z;
    }
}

static inline void depth_bounds_for_view_axis(
        int view_axis,
        float& depth_lo, float& depth_hi,
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z) {
    if (view_axis == 0) {
        depth_lo = lo_z; depth_hi = hi_z;
    } else if (view_axis == 1) {
        depth_lo = lo_y; depth_hi = hi_y;
    } else {
        depth_lo = lo_x; depth_hi = hi_x;
    }
}
