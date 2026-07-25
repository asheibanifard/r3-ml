/*
 * common.cuh — shared device-side quaternion/rotation math for the training
 * (forward/backward) kernels.
 *
 * Every function here is __device__ __forceinline__, so this header can be
 * #include'd from multiple .cu translation units (forward.cu, backward.cu)
 * without any cross-TU device-linkage requirement (no -rdc=true needed):
 * each TU gets its own inlined copy.
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/* Normalize a raw quaternion q → qn and store inv_norm = 1/||q||.
 * inv_norm is saved so that the gradient through normalisation can be
 * computed without a second rsqrt in the backward pass. */
__device__ __forceinline__ void normalize_quat(
        const float* __restrict__ q,
        float* qn,
        float& inv_norm)
{
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    inv_norm  = rsqrtf(fmaxf(n2, 1e-12f));
    qn[0] = q[0] * inv_norm;
    qn[1] = q[1] * inv_norm;
    qn[2] = q[2] * inv_norm;
    qn[3] = q[3] * inv_norm;
}

/* Closed-form Rodrigues formula: unit quaternion [w,x,y,z] → 3×3 rotation.
 * Row-major flat layout: R[i*3+j] = R_{ij}. */
__device__ __forceinline__ void quat_to_rotmat(const float* qn, float* R)
{
    const float w=qn[0], x=qn[1], y=qn[2], z=qn[3];
    R[0] = 1.f - 2.f*(y*y + z*z);
    R[1] = 2.f*(x*y - w*z);
    R[2] = 2.f*(x*z + w*y);
    R[3] = 2.f*(x*y + w*z);
    R[4] = 1.f - 2.f*(x*x + z*z);
    R[5] = 2.f*(y*z - w*x);
    R[6] = 2.f*(x*z - w*y);
    R[7] = 2.f*(y*z + w*x);
    R[8] = 1.f - 2.f*(x*x + y*y);
}

/* u = Rᵀ v   (transposed matrix–vector product) */
__device__ __forceinline__ void mat_t_vec(
        const float* __restrict__ R,
        const float* __restrict__ v,
        float* u)
{
    u[0] = R[0]*v[0] + R[3]*v[1] + R[6]*v[2];
    u[1] = R[1]*v[0] + R[4]*v[1] + R[7]*v[2];
    u[2] = R[2]*v[0] + R[5]*v[1] + R[8]*v[2];
}

/* u = R v */
__device__ __forceinline__ void mat_vec(
        const float* __restrict__ R,
        const float* __restrict__ v,
        float* u)
{
    u[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    u[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    u[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

/* Backprop through quat_to_rotmat (Rodrigues) and through normalize_quat.
 *
 * Given:
 *   grad_R[9]  — ∂L/∂R in flat row-major layout
 *   qn[4]      — normalised quaternion [w,x,y,z]
 *   inv_norm   — 1/||q_raw|| from normalize_quat
 *
 * Computes:
 *   grad_qraw[4] — ∂L/∂q_raw
 *
 * Two steps:
 *   1. ∂L/∂qn  by summing ∂R[i,j]/∂qn · grad_R[i,j] over all (i,j).
 *   2. Backprop through normalisation: qn = q_raw·inv_norm
 *      ∂L/∂q_raw = inv_norm · (∂L/∂qn  −  qn · (qnᵀ · ∂L/∂qn))
 */
__device__ __forceinline__ void quat_grad_from_rot_grad(
        const float* __restrict__ grad_R,
        const float* __restrict__ qn,
        float inv_norm,
        float* grad_qraw)
{
    const float w=qn[0], x=qn[1], y=qn[2], z=qn[3];
    /* Index helpers: g(i,j) = grad_R[i*3+j] */
    #define G(i,j) grad_R[(i)*3+(j)]

    float gw =  2.f*( -z*G(0,1) + y*G(0,2) + z*G(1,0) - x*G(1,2) - y*G(2,0) + x*G(2,1) );
    float gx =  2.f*(  y*G(0,1) + z*G(0,2) + y*G(1,0) - 2.f*x*G(1,1) - w*G(1,2) + z*G(2,0) + w*G(2,1) - 2.f*x*G(2,2) );
    float gy =  2.f*( -2.f*y*G(0,0) + x*G(0,1) + w*G(0,2) + x*G(1,0) + z*G(1,2) - w*G(2,0) + z*G(2,1) - 2.f*y*G(2,2) );
    float gz =  2.f*( -2.f*z*G(0,0) - w*G(0,1) + x*G(0,2) + w*G(1,0) - 2.f*z*G(1,1) + y*G(1,2) + x*G(2,0) + y*G(2,1) );
    #undef G

    /* Backprop through normalisation: remove the component along qn. */
    float dot = gw*w + gx*x + gy*y + gz*z;
    grad_qraw[0] = inv_norm * (gw - w*dot);
    grad_qraw[1] = inv_norm * (gx - x*dot);
    grad_qraw[2] = inv_norm * (gy - y*dot);
    grad_qraw[3] = inv_norm * (gz - z*dot);
}
