#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Compute quaternion to rotation matrix
__device__ void quat_to_rot(float w, float x, float y, float z, float *R) {
    // R is 9 floats arranged as [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    R[0] = 1.0f - 2.0f * (y*y + z*z);
    R[1] = 2.0f * (x*y - w*z);
    R[2] = 2.0f * (x*z + w*y);
    R[3] = 2.0f * (x*y + w*z);
    R[4] = 1.0f - 2.0f * (x*x + z*z);
    R[5] = 2.0f * (y*z - w*x);
    R[6] = 2.0f * (x*z - w*y);
    R[7] = 2.0f * (y*z + w*x);
    R[8] = 1.0f - 2.0f * (x*x + y*y);
}

// Matrix-vector product: out = A @ v, A is 3x3, v is 3
__device__ void matvec3(float *A, float *v, float *out) {
    out[0] = A[0]*v[0] + A[1]*v[1] + A[2]*v[2];
    out[1] = A[3]*v[0] + A[4]*v[1] + A[5]*v[2];
    out[2] = A[6]*v[0] + A[7]*v[1] + A[8]*v[2];
}

// Compute Σ⁻¹ from log-scales and quaternion
// Returns the inverse covariance matrix as 9 floats
__device__ void compute_sigma_inv(float *log_s, float w, float x, float y, float z, float *sigma_inv) {
    // Clamp scales
    float s0 = fmax(0.001f, fmin(20.0f, expf(log_s[0])));
    float s1 = fmax(0.001f, fmin(20.0f, expf(log_s[1])));
    float s2 = fmax(0.001f, fmin(20.0f, expf(log_s[2])));

    // Compute rotation matrix
    float R[9];
    quat_to_rot(w, x, y, z, R);

    // S = diag(s0, s1, s2)
    // Σ = R S S^T R^T = R S² R^T
    // First compute S² R^T
    float SR[9];
    SR[0] = s0*s0 * R[0];
    SR[1] = s1*s1 * R[1];
    SR[2] = s2*s2 * R[2];
    SR[3] = s0*s0 * R[3];
    SR[4] = s1*s1 * R[4];
    SR[5] = s2*s2 * R[5];
    SR[6] = s0*s0 * R[6];
    SR[7] = s1*s1 * R[7];
    SR[8] = s2*s2 * R[8];

    // Σ = R @ SR
    float Sigma[9];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            Sigma[i*3 + j] = R[i*3 + 0]*SR[0*3 + j] + R[i*3 + 1]*SR[1*3 + j] + R[i*3 + 2]*SR[2*3 + j];
        }
    }

    // Add small regularization to diagonal
    Sigma[0] += 1e-5f;
    Sigma[4] += 1e-5f;
    Sigma[8] += 1e-5f;

    // Invert 3x3 matrix using analytical formula
    float det = Sigma[0]*(Sigma[4]*Sigma[8] - Sigma[5]*Sigma[7])
              - Sigma[1]*(Sigma[3]*Sigma[8] - Sigma[5]*Sigma[6])
              + Sigma[2]*(Sigma[3]*Sigma[7] - Sigma[4]*Sigma[6]);

    if(fabsf(det) < 1e-10f) det = 1e-10f;
    float inv_det = 1.0f / det;

    sigma_inv[0] = (Sigma[4]*Sigma[8] - Sigma[5]*Sigma[7]) * inv_det;
    sigma_inv[1] = (Sigma[2]*Sigma[7] - Sigma[1]*Sigma[8]) * inv_det;
    sigma_inv[2] = (Sigma[1]*Sigma[5] - Sigma[2]*Sigma[4]) * inv_det;
    sigma_inv[3] = (Sigma[5]*Sigma[6] - Sigma[3]*Sigma[8]) * inv_det;
    sigma_inv[4] = (Sigma[0]*Sigma[8] - Sigma[2]*Sigma[6]) * inv_det;
    sigma_inv[5] = (Sigma[2]*Sigma[3] - Sigma[0]*Sigma[5]) * inv_det;
    sigma_inv[6] = (Sigma[3]*Sigma[7] - Sigma[4]*Sigma[6]) * inv_det;
    sigma_inv[7] = (Sigma[1]*Sigma[6] - Sigma[0]*Sigma[7]) * inv_det;
    sigma_inv[8] = (Sigma[0]*Sigma[4] - Sigma[1]*Sigma[3]) * inv_det;
}

// Main kernel: render Gaussian mixture
// grid: (N_pts + BLOCK_SIZE - 1) / BLOCK_SIZE threads
// block: BLOCK_SIZE threads per block
__global__ void render_gaussian_kernel(
    const float* pts,          // (N_pts, 3) query points
    const float* means,        // (M, 3) Gaussian centers
    const float* log_scales,   // (M, 3) log scales
    const float* quats,        // (M, 4) quaternions (w, x, y, z)
    const float* intensities,  // (M,) sigmoid(alpha)
    float* output,             // (N_pts,) output intensities
    int N_pts,
    int M) {

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= N_pts) return;

    float px = pts[pt_idx * 3 + 0];
    float py = pts[pt_idx * 3 + 1];
    float pz = pts[pt_idx * 3 + 2];

    float intensity = 0.0f;

    // Sum over all Gaussians
    for(int m = 0; m < M; m++) {
        float mx = means[m * 3 + 0];
        float my = means[m * 3 + 1];
        float mz = means[m * 3 + 2];

        // Displacement vector
        float dx = px - mx;
        float dy = py - my;
        float dz = pz - mz;

        // Quaternion components
        float w = quats[m * 4 + 0];
        float x = quats[m * 4 + 1];
        float y = quats[m * 4 + 2];
        float z = quats[m * 4 + 3];

        // Compute Σ⁻¹
        float sigma_inv[9];
        compute_sigma_inv(&log_scales[m * 3], w, x, y, z, sigma_inv);

        // Compute Mahalanobis distance: diff^T Σ⁻¹ diff
        float v[3] = {dx, dy, dz};
        float Sv[3];
        matvec3(sigma_inv, v, Sv);
        float mahal_sq = Sv[0]*v[0] + Sv[1]*v[1] + Sv[2]*v[2];

        // Clamp to avoid overflow
        mahal_sq = fmin(10.0f, fmax(0.0f, mahal_sq));

        // Gaussian contribution
        float contrib = intensities[m] * expf(-0.5f * mahal_sq);
        intensity += contrib;
    }

    output[pt_idx] = fmin(1.0f, fmax(0.0f, intensity));
}

// PyTorch C++ binding
torch::Tensor render_gaussian_cuda_impl(
    torch::Tensor pts,
    torch::Tensor means,
    torch::Tensor log_scales,
    torch::Tensor quats,
    torch::Tensor intensities) {

    auto N_pts = pts.size(0);
    auto M = means.size(0);

    auto output = torch::zeros({N_pts}, pts.options());

    const int block_size = BLOCK_SIZE;
    const int num_blocks = (N_pts + block_size - 1) / block_size;

    render_gaussian_kernel<<<num_blocks, block_size>>>(
        pts.data_ptr<float>(),
        means.data_ptr<float>(),
        log_scales.data_ptr<float>(),
        quats.data_ptr<float>(),
        intensities.data_ptr<float>(),
        output.data_ptr<float>(),
        N_pts,
        M);

    return output;
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_gaussian_cuda", &render_gaussian_cuda_impl, "Render Gaussian mixture field");
}
