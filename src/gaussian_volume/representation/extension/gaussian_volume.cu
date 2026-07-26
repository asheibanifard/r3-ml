/*
 * gaussian_volume.cu — CUDA forward + backward for normalized 3D Gaussian
 * volume reconstruction.
 *
 * Reimplemented to match the exact interface declared in bindings.cpp and
 * the documented formula in model.py (the original source was lost when
 * this package was deleted earlier; this is a faithful reimplementation of
 * the known spec, following the same tiled-shared-memory forward /
 * one-block-per-Gaussian backward pattern already proven in the sibling
 * extension/training kernels of src/gaussian_volume's other CUDA pipeline).
 *
 * Reconstruction:
 *
 *                sum_i w_i(x) * f_i
 *   V(x) = --------------------------------
 *                sum_i w_i(x) + epsilon
 *
 * where:
 *
 *   w_i(x) =
 *       confidence_i
 *       * exp(-0.5 * (x - mean_i)^T Q_i (x - mean_i))
 *
 * Q_i is the compact six-value symmetric precision matrix
 * [qxx, qyy, qzz, qxy, qxz, qyz] (see math3d.py); it is already
 * differentiable PyTorch-side (built from scale + rotation), so the CUDA
 * kernels below only need gradients w.r.t. Q_i directly, not w.r.t.
 * quaternion/scale — no quaternion backprop lives in this file.
 *
 * Voxel convention: means are in voxel coordinates [x, y, z]; voxel index i
 * along an axis corresponds to coordinate i + 0.5 (matching model.py).
 *
 * Neither pass caches intermediate per-voxel state across the Python-visible
 * forward/backward boundary (autograd.Function.forward here does not save
 * the output tensor), so gaussian_backward_cuda recomputes the forward
 * reconstruction and denominator internally before accumulating gradients.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kGaussianTile = 256;

void validate_gaussian_inputs(
        const torch::Tensor& means,
        const torch::Tensor& precision,
        const torch::Tensor& confidences,
        const torch::Tensor& features) {
    CHECK_INPUT(means);
    CHECK_INPUT(precision);
    CHECK_INPUT(confidences);
    CHECK_INPUT(features);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(precision.scalar_type() == torch::kFloat32, "precision must be float32");
    TORCH_CHECK(confidences.scalar_type() == torch::kFloat32, "confidences must be float32");
    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "features must be float32");

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must have shape [N,3]");
    TORCH_CHECK(precision.dim() == 2 && precision.size(1) == 6, "precision must have shape [N,6]");
    TORCH_CHECK(confidences.dim() == 1, "confidences must have shape [N]");
    TORCH_CHECK(features.dim() == 1, "features must have shape [N]");

    const int64_t N = means.size(0);
    TORCH_CHECK(precision.size(0) == N, "precision must have the same N as means");
    TORCH_CHECK(confidences.size(0) == N, "confidences must have the same N as means");
    TORCH_CHECK(features.size(0) == N, "features must have the same N as means");
}

void validate_volume_shape(int64_t depth, int64_t height, int64_t width) {
    TORCH_CHECK(
        depth > 0 && height > 0 && width > 0,
        "depth, height, and width must all be positive"
    );
}

__device__ __forceinline__ float mahalanobis_squared(
        float dx, float dy, float dz,
        float qxx, float qyy, float qzz,
        float qxy, float qxz, float qyz) {
    return qxx * dx * dx + qyy * dy * dy + qzz * dz * dz
         + 2.f * qxy * dx * dy + 2.f * qxz * dx * dz + 2.f * qyz * dy * dz;
}

// ── Forward kernel (tiled shared memory) ──────────────────────────────────
// One thread per output voxel; Gaussians are loaded cooperatively into
// shared memory kGaussianTile at a time. Writes both the normalized
// reconstruction (output) and the per-voxel denominator (sum_i w_i +
// epsilon) — the denominator is not exposed to Python, but is reused by
// gaussian_backward_cuda's internal recompute pass below.
__global__ void gaussian_forward_kernel(
        const float* __restrict__ means,
        const float* __restrict__ precision,
        const float* __restrict__ confidences,
        const float* __restrict__ features,
        int64_t depth, int64_t height, int64_t width,
        int64_t N,
        float cutoff_squared, float epsilon,
        float* __restrict__ output,
        float* __restrict__ denom_out) {
    __shared__ float s_mean[kGaussianTile][3];
    __shared__ float s_prec[kGaussianTile][6];
    __shared__ float s_conf[kGaussianTile];
    __shared__ float s_feat[kGaussianTile];

    const int64_t voxel = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t voxel_count = depth * height * width;
    const bool active = voxel < voxel_count;

    float vx = 0.f, vy = 0.f, vz = 0.f;
    if (active) {
        const int64_t hw = height * width;
        const int64_t iz = voxel / hw;
        const int64_t iy = (voxel % hw) / width;
        const int64_t ix = voxel % width;
        vx = static_cast<float>(ix) + 0.5f;
        vy = static_cast<float>(iy) + 0.5f;
        vz = static_cast<float>(iz) + 0.5f;
    }

    float numerator = 0.f;
    float denom = 0.f;

    for (int64_t tile_start = 0; tile_start < N; tile_start += kGaussianTile) {
        const int tile_n = static_cast<int>(
            std::min<int64_t>(kGaussianTile, N - tile_start)
        );

        for (int i = threadIdx.x; i < tile_n; i += blockDim.x) {
            const int64_t g = tile_start + i;
            s_mean[i][0] = means[g * 3 + 0];
            s_mean[i][1] = means[g * 3 + 1];
            s_mean[i][2] = means[g * 3 + 2];
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                s_prec[i][k] = precision[g * 6 + k];
            }
            s_conf[i] = confidences[g];
            s_feat[i] = features[g];
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tile_n; ++i) {
                const float dx = vx - s_mean[i][0];
                const float dy = vy - s_mean[i][1];
                const float dz = vz - s_mean[i][2];

                const float mahal = mahalanobis_squared(
                    dx, dy, dz,
                    s_prec[i][0], s_prec[i][1], s_prec[i][2],
                    s_prec[i][3], s_prec[i][4], s_prec[i][5]
                );

                if (mahal >= cutoff_squared) continue;

                const float weight = s_conf[i] * __expf(-0.5f * mahal);
                numerator += weight * s_feat[i];
                denom += weight;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float total_denom = denom + epsilon;
        output[voxel] = numerator / total_denom;
        denom_out[voxel] = total_denom;
    }
}

// ── Backward kernel (one block per Gaussian) ──────────────────────────────
// Each block recomputes, for its one Gaussian, every voxel's mahalanobis
// distance / weight (using the reconstruction and denominator recomputed by
// gaussian_forward_kernel just above) and accumulates the four analytic
// gradients via warp-shuffle + shared-memory reduction — no atomicAdd
// contention, since each Gaussian's gradients are only ever written by its
// own block.
__global__ void gaussian_backward_kernel(
        const float* __restrict__ grad_output,
        const float* __restrict__ output,
        const float* __restrict__ denom,
        const float* __restrict__ means,
        const float* __restrict__ precision,
        const float* __restrict__ confidences,
        const float* __restrict__ features,
        int64_t depth, int64_t height, int64_t width,
        int64_t N,
        float cutoff_squared,
        float* __restrict__ grad_means,
        float* __restrict__ grad_precision,
        float* __restrict__ grad_confidences,
        float* __restrict__ grad_features) {
    const int64_t g = blockIdx.x;
    if (g >= N) return;

    const float mean_x = means[g * 3 + 0];
    const float mean_y = means[g * 3 + 1];
    const float mean_z = means[g * 3 + 2];
    const float qxx = precision[g * 6 + 0];
    const float qyy = precision[g * 6 + 1];
    const float qzz = precision[g * 6 + 2];
    const float qxy = precision[g * 6 + 3];
    const float qxz = precision[g * 6 + 4];
    const float qyz = precision[g * 6 + 5];
    const float confidence = confidences[g];
    const float feature = features[g];

    const int64_t voxel_count = depth * height * width;
    const int64_t hw = height * width;

    float grad_mx = 0.f, grad_my = 0.f, grad_mz = 0.f;
    float grad_qxx = 0.f, grad_qyy = 0.f, grad_qzz = 0.f;
    float grad_qxy = 0.f, grad_qxz = 0.f, grad_qyz = 0.f;
    float grad_conf = 0.f, grad_feat = 0.f;

    for (int64_t voxel = threadIdx.x; voxel < voxel_count; voxel += blockDim.x) {
        const int64_t iz = voxel / hw;
        const int64_t iy = (voxel % hw) / width;
        const int64_t ix = voxel % width;
        const float vx = static_cast<float>(ix) + 0.5f;
        const float vy = static_cast<float>(iy) + 0.5f;
        const float vz = static_cast<float>(iz) + 0.5f;

        const float dx = vx - mean_x;
        const float dy = vy - mean_y;
        const float dz = vz - mean_z;

        const float mahal = mahalanobis_squared(dx, dy, dz, qxx, qyy, qzz, qxy, qxz, qyz);
        if (mahal >= cutoff_squared) continue;

        const float weight = confidence * __expf(-0.5f * mahal);
        const float total_denom = denom[voxel];
        const float reconstructed = output[voxel];
        const float g_out = grad_output[voxel];

        // V = (sum_i w_i f_i) / (sum_i w_i + eps)
        // dV/dw_i    = (f_i - V) / denom
        // dV/df_i    = w_i / denom
        const float grad_weight = g_out * (feature - reconstructed) / total_denom;
        grad_feat += g_out * weight / total_denom;

        // w_i = confidence_i * exp(-0.5 * mahal)
        grad_conf += grad_weight * __expf(-0.5f * mahal);
        const float grad_mahal = grad_weight * (-0.5f * weight);

        // mahal = qxx dx^2 + qyy dy^2 + qzz dz^2 + 2 qxy dx dy + 2 qxz dx dz + 2 qyz dy dz
        grad_qxx += grad_mahal * dx * dx;
        grad_qyy += grad_mahal * dy * dy;
        grad_qzz += grad_mahal * dz * dz;
        grad_qxy += grad_mahal * 2.f * dx * dy;
        grad_qxz += grad_mahal * 2.f * dx * dz;
        grad_qyz += grad_mahal * 2.f * dy * dz;

        // d(mahal)/d(dx,dy,dz) = 2 * Q * delta ; delta = x - mean, so
        // d(mahal)/d(mean) = -d(mahal)/d(delta).
        const float dmahal_ddx = 2.f * (qxx * dx + qxy * dy + qxz * dz);
        const float dmahal_ddy = 2.f * (qxy * dx + qyy * dy + qyz * dz);
        const float dmahal_ddz = 2.f * (qxz * dx + qyz * dy + qzz * dz);

        grad_mx -= grad_mahal * dmahal_ddx;
        grad_my -= grad_mahal * dmahal_ddy;
        grad_mz -= grad_mahal * dmahal_ddz;
    }

    // ── Warp-level reduction (shuffle down) ───────────────────────────────
    #define WRED(x) \
        x += __shfl_down_sync(0xffffffffu, x, 16); \
        x += __shfl_down_sync(0xffffffffu, x,  8); \
        x += __shfl_down_sync(0xffffffffu, x,  4); \
        x += __shfl_down_sync(0xffffffffu, x,  2); \
        x += __shfl_down_sync(0xffffffffu, x,  1);
    WRED(grad_mx)  WRED(grad_my)  WRED(grad_mz)
    WRED(grad_qxx) WRED(grad_qyy) WRED(grad_qzz)
    WRED(grad_qxy) WRED(grad_qxz) WRED(grad_qyz)
    WRED(grad_conf) WRED(grad_feat)
    #undef WRED

    // ── Inter-warp reduction via shared memory ────────────────────────────
    constexpr int kWarps = kThreadsPerBlock / 32;
    __shared__ float sm[11][kWarps];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    if (lane_id == 0) {
        sm[0][warp_id] = grad_mx;   sm[1][warp_id] = grad_my;   sm[2][warp_id] = grad_mz;
        sm[3][warp_id] = grad_qxx;  sm[4][warp_id] = grad_qyy;  sm[5][warp_id] = grad_qzz;
        sm[6][warp_id] = grad_qxy;  sm[7][warp_id] = grad_qxz;  sm[8][warp_id] = grad_qyz;
        sm[9][warp_id] = grad_conf; sm[10][warp_id] = grad_feat;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total[11] = {0.f,0.f,0.f, 0.f,0.f,0.f, 0.f,0.f,0.f, 0.f,0.f};
        for (int w = 0; w < kWarps; ++w) {
            for (int i = 0; i < 11; ++i) total[i] += sm[i][w];
        }

        grad_means[g * 3 + 0] = total[0];
        grad_means[g * 3 + 1] = total[1];
        grad_means[g * 3 + 2] = total[2];

        grad_precision[g * 6 + 0] = total[3];
        grad_precision[g * 6 + 1] = total[4];
        grad_precision[g * 6 + 2] = total[5];
        grad_precision[g * 6 + 3] = total[6];
        grad_precision[g * 6 + 4] = total[7];
        grad_precision[g * 6 + 5] = total[8];

        grad_confidences[g] = total[9];
        grad_features[g] = total[10];
    }
}

}  // namespace


// ── Python-visible entry points ────────────────────────────────────────────

torch::Tensor gaussian_forward_cuda(
        const torch::Tensor& means,
        const torch::Tensor& precision,
        const torch::Tensor& confidences,
        const torch::Tensor& features,
        const int64_t depth,
        const int64_t height,
        const int64_t width,
        const double cutoff_squared,
        const double epsilon) {
    validate_gaussian_inputs(means, precision, confidences, features);
    validate_volume_shape(depth, height, width);

    const int64_t N = means.size(0);
    const int64_t voxel_count = depth * height * width;

    auto output = torch::empty({depth, height, width}, means.options());
    auto denom = torch::empty({voxel_count}, means.options());

    const int64_t blocks = (voxel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_forward_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        means.data_ptr<float>(),
        precision.data_ptr<float>(),
        confidences.data_ptr<float>(),
        features.data_ptr<float>(),
        depth, height, width, N,
        static_cast<float>(cutoff_squared), static_cast<float>(epsilon),
        output.data_ptr<float>(),
        denom.data_ptr<float>()
    );

    return output;
}

std::vector<torch::Tensor> gaussian_backward_cuda(
        const torch::Tensor& grad_output,
        const torch::Tensor& means,
        const torch::Tensor& precision,
        const torch::Tensor& confidences,
        const torch::Tensor& features,
        const double cutoff_squared,
        const double epsilon) {
    CHECK_INPUT(grad_output);
    validate_gaussian_inputs(means, precision, confidences, features);

    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [D, H, W]");
    TORCH_CHECK(
        grad_output.device() == means.device(),
        "grad_output and Gaussian tensors must be on the same CUDA device"
    );

    const int64_t depth = grad_output.size(0);
    const int64_t height = grad_output.size(1);
    const int64_t width = grad_output.size(2);
    const int64_t N = means.size(0);
    const int64_t voxel_count = depth * height * width;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Recompute the forward reconstruction and denominator: neither is
    // cached from the original forward() call (autograd.Function.forward
    // only saves the raw parameters, not the output), so backward derives
    // them itself using the same kernel the public forward uses.
    auto recomputed_output = torch::empty({depth, height, width}, means.options());
    auto recomputed_denom = torch::empty({voxel_count}, means.options());

    const int64_t forward_blocks = (voxel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    gaussian_forward_kernel<<<forward_blocks, kThreadsPerBlock, 0, stream>>>(
        means.data_ptr<float>(),
        precision.data_ptr<float>(),
        confidences.data_ptr<float>(),
        features.data_ptr<float>(),
        depth, height, width, N,
        static_cast<float>(cutoff_squared), static_cast<float>(epsilon),
        recomputed_output.data_ptr<float>(),
        recomputed_denom.data_ptr<float>()
    );

    auto grad_means = torch::zeros_like(means);
    auto grad_precision = torch::zeros_like(precision);
    auto grad_confidences = torch::zeros_like(confidences);
    auto grad_features = torch::zeros_like(features);

    gaussian_backward_kernel<<<N, kThreadsPerBlock, 0, stream>>>(
        grad_output.contiguous().data_ptr<float>(),
        recomputed_output.data_ptr<float>(),
        recomputed_denom.data_ptr<float>(),
        means.data_ptr<float>(),
        precision.data_ptr<float>(),
        confidences.data_ptr<float>(),
        features.data_ptr<float>(),
        depth, height, width, N,
        static_cast<float>(cutoff_squared),
        grad_means.data_ptr<float>(),
        grad_precision.data_ptr<float>(),
        grad_confidences.data_ptr<float>(),
        grad_features.data_ptr<float>()
    );

    return {grad_means, grad_precision, grad_confidences, grad_features};
}
