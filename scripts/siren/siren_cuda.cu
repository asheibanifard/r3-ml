/*
 * siren_cuda.cu — Hand-written CUDA forward + backward for a SIREN MLP
 *
 * Implements the implicit field of Sitzmann et al., "Implicit Neural
 * Representations with Periodic Activation Functions" (NeurIPS 2020):
 *
 *   a_0(x)   = x                                          (3-D query point)
 *   z_l      = W_l a_{l-1} + b_l                           l = 0 .. L-2
 *   a_l      = sin(w0_l · z_l)                              (sine layers)
 *   z_{L-1}  = W_{L-1} a_{L-2} + b_{L-1}                   (output, no sine)
 *   y        = sigmoid(z_{L-1})                             prediction in [0,1]
 *
 * with w0_0 = w0_first (first layer) and w0_l = w0_hidden for l = 1 .. L-2.
 * L (number of layers, i.e. hidden_layers + 1) is read from the length of the
 * weights/biases vectors, so the same kernel serves any depth/width.
 *
 * Every linear layer is evaluated with a hand-rolled tiled shared-memory GEMM
 * (no cuBLAS) so the entire forward and analytic backward pass — matmuls,
 * bias add, sine/sigmoid activations and their derivatives — run as raw CUDA
 * kernels, matching the project's existing 3dgs_cuda.cu philosophy of a
 * hand-written training kernel rather than calling into a DL framework's
 * built-in linear algebra.
 *
 * Forward saves the pre-activations Z_l (needed for the sin'(z)=w0·cos(w0 z)
 * derivative) and the per-layer inputs A_l (needed for the dW_l = dZ_l^T A_l
 * outer product) so backward never has to re-run the forward pass.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

#define TILE 16
#define EW_BLOCK 256
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

// ─── Tiled GEMM: C[Mk,Nk] = opA(A) @ opB(B) ──────────────────────────────────
// opA(A) is logically (Mk,Kk):  A physically (Mk,Kk) if !transA, else (Kk,Mk).
// opB(B) is logically (Kk,Nk):  B physically (Kk,Nk) if !transB, else (Nk,Kk).
// Standard shared-memory tiled matmul; reused for every linear layer in both
// the forward pass (A_prev @ W^T) and the backward pass (dZ^T @ A_prev for
// weight grads, dZ @ W for the upstream activation grad).
__global__ void matmul_kernel(
        const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C,
        int Mk, int Nk, int Kk, bool transA, bool transB)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;   // index into Mk
    const int col = blockIdx.x * TILE + threadIdx.x;   // index into Nk

    float acc = 0.0f;
    const int numTiles = CEILDIV(Kk, TILE);

    for (int t = 0; t < numTiles; ++t) {
        const int kA = t * TILE + threadIdx.x;
        const int kB = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < Mk && kA < Kk)
            ? (transA ? A[kA * Mk + row] : A[row * Kk + kA])
            : 0.0f;

        Bs[threadIdx.y][threadIdx.x] = (col < Nk && kB < Kk)
            ? (transB ? B[col * Kk + kB] : B[kB * Nk + col])
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < Mk && col < Nk) {
        C[row * Nk + col] = acc;
    }
}

// ─── Elementwise kernels ──────────────────────────────────────────────────

__global__ void bias_add_kernel(float* __restrict__ Z, const float* __restrict__ bias, int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        Z[idx] += bias[idx % N];
    }
}

__global__ void sine_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size, float w0)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = sinf(w0 * Z[idx]);
}

// dZ = dA · w0·cos(w0·Z)   — analytic derivative of sin(w0·z)
__global__ void sine_backward_kernel(
        const float* __restrict__ dA, const float* __restrict__ Z,
        float* __restrict__ dZ, int size, float w0)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dZ[idx] = dA[idx] * w0 * cosf(w0 * Z[idx]);
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = 1.0f / (1.0f + expf(-Z[idx]));
}

// dZ = dA · y·(1-y), using the saved sigmoid output y (no re-exp needed)
__global__ void sigmoid_backward_kernel(
        const float* __restrict__ dA, const float* __restrict__ Y,
        float* __restrict__ dZ, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float y = Y[idx];
        dZ[idx] = dA[idx] * y * (1.0f - y);
    }
}

// db[n] = sum_m dZ[m,n] — one block per output column, tree reduction in shared mem.
__global__ void col_sum_kernel(const float* __restrict__ dZ, float* __restrict__ db, int M, int N)
{
    extern __shared__ float sdata[];
    const int n = blockIdx.x;

    float sum = 0.0f;
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        sum += dZ[m * N + n];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) db[n] = sdata[0];
}

// ─── Host-side launch helper ──────────────────────────────────────────────

static void launch_matmul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
                          int Mk, int Nk, int Kk, bool transA, bool transB, cudaStream_t stream)
{
    const dim3 block(TILE, TILE);
    const dim3 grid(CEILDIV(Nk, TILE), CEILDIV(Mk, TILE));
    matmul_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        Mk, Nk, Kk, transA, transB);
}

// ─── Forward ───────────────────────────────────────────────────────────────
// Returns [pred, Z_0, ..., Z_{L-2}, A_0(=pts), A_1, ..., A_{L-1}]
//   - pred         : (M,)        sigmoid output
//   - Z_0..Z_{L-2} : (M,out_l)   pre-activations of the L-1 sine layers
//   - A_0..A_{L-1} : (M,in_l)    input fed into each of the L layers (A_0 = pts)
std::vector<torch::Tensor> siren_forward(
        torch::Tensor pts,
        std::vector<torch::Tensor> weights,
        std::vector<torch::Tensor> biases,
        float w0_first,
        float w0_hidden)
{
    TORCH_CHECK(pts.is_cuda() && pts.is_contiguous(), "pts must be contiguous CUDA float32");
    TORCH_CHECK(pts.scalar_type() == torch::kFloat32, "pts must be float32");
    TORCH_CHECK(pts.dim() == 2 && pts.size(1) == 3, "pts must be (M, 3)");

    const int L = static_cast<int>(weights.size());
    TORCH_CHECK(L >= 1, "weights must have at least one layer");
    TORCH_CHECK(static_cast<int>(biases.size()) == L, "weights/biases length mismatch");

    const int M = static_cast<int>(pts.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts = pts.options();

    std::vector<torch::Tensor> acts;   // A_0 .. A_{L-1}
    std::vector<torch::Tensor> Zs;     // Z_0 .. Z_{L-2}
    acts.push_back(pts);

    torch::Tensor cur = pts;
    int in_dim = 3;

    for (int l = 0; l < L - 1; ++l) {
        TORCH_CHECK(weights[l].is_cuda() && weights[l].is_contiguous() &&
                    weights[l].scalar_type() == torch::kFloat32,
                    "weights[", l, "] must be contiguous CUDA float32");
        TORCH_CHECK(weights[l].dim() == 2 && weights[l].size(1) == in_dim,
                    "weights[", l, "] expected in_features=", in_dim);
        const int out_dim = static_cast<int>(weights[l].size(0));
        TORCH_CHECK(biases[l].dim() == 1 && biases[l].size(0) == out_dim,
                    "biases[", l, "] shape mismatch");

        auto Z = torch::empty({M, out_dim}, opts);
        launch_matmul(cur, weights[l], Z, M, out_dim, in_dim, false, true, stream);

        const int total = M * out_dim;
        const int bblocks = CEILDIV(total, EW_BLOCK);
        bias_add_kernel<<<bblocks, EW_BLOCK, 0, stream>>>(
            Z.data_ptr<float>(), biases[l].data_ptr<float>(), M, out_dim);

        const float w0 = (l == 0) ? w0_first : w0_hidden;
        auto A = torch::empty({M, out_dim}, opts);
        sine_forward_kernel<<<bblocks, EW_BLOCK, 0, stream>>>(
            Z.data_ptr<float>(), A.data_ptr<float>(), total, w0);

        Zs.push_back(Z);
        acts.push_back(A);
        cur = A;
        in_dim = out_dim;
    }

    // Output layer: linear (out_features must be 1) + sigmoid, no sine.
    const int l = L - 1;
    TORCH_CHECK(weights[l].dim() == 2 && weights[l].size(1) == in_dim,
                "output layer expected in_features=", in_dim);
    TORCH_CHECK(weights[l].size(0) == 1, "output layer must have out_features == 1");
    TORCH_CHECK(biases[l].dim() == 1 && biases[l].size(0) == 1, "output bias must be (1,)");

    auto Zout = torch::empty({M, 1}, opts);
    launch_matmul(cur, weights[l], Zout, M, 1, in_dim, false, true, stream);

    const int bblocks_out = CEILDIV(M, EW_BLOCK);
    bias_add_kernel<<<bblocks_out, EW_BLOCK, 0, stream>>>(
        Zout.data_ptr<float>(), biases[l].data_ptr<float>(), M, 1);

    auto pred = torch::empty({M, 1}, opts);
    sigmoid_forward_kernel<<<bblocks_out, EW_BLOCK, 0, stream>>>(
        Zout.data_ptr<float>(), pred.data_ptr<float>(), M);

    std::vector<torch::Tensor> result;
    result.push_back(pred.view({M}));
    for (auto& z : Zs)   result.push_back(z);
    for (auto& a : acts) result.push_back(a);
    return result;
}

// ─── Backward ──────────────────────────────────────────────────────────────
// Returns [dW_0, db_0, dW_1, db_1, ..., dW_{L-1}, db_{L-1}].
// No gradient w.r.t. pts is computed — query coordinates are fixed samples,
// never a learned parameter.
std::vector<torch::Tensor> siren_backward(
        torch::Tensor grad_pred,
        torch::Tensor pred,
        std::vector<torch::Tensor> Zs,
        std::vector<torch::Tensor> acts,
        std::vector<torch::Tensor> weights,
        float w0_first,
        float w0_hidden)
{
    const int L = static_cast<int>(weights.size());
    TORCH_CHECK(static_cast<int>(Zs.size())   == L - 1, "Zs length mismatch");
    TORCH_CHECK(static_cast<int>(acts.size()) == L,     "acts length mismatch");
    TORCH_CHECK(grad_pred.is_cuda() && grad_pred.is_contiguous(), "grad_pred must be contiguous CUDA");

    const int M = static_cast<int>(grad_pred.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts = grad_pred.options();

    std::vector<torch::Tensor> dWs(L), dbs(L);

    // Sigmoid backward → dZ for the output layer, shape (M,1).
    auto dZ = torch::empty({M, 1}, opts);
    {
        const int bblocks = CEILDIV(M, EW_BLOCK);
        sigmoid_backward_kernel<<<bblocks, EW_BLOCK, 0, stream>>>(
            grad_pred.data_ptr<float>(), pred.data_ptr<float>(), dZ.data_ptr<float>(), M);
    }

    for (int l = L - 1; l >= 0; --l) {
        const int out_dim = static_cast<int>(weights[l].size(0));
        const int in_dim  = static_cast<int>(weights[l].size(1));
        const torch::Tensor& A_prev = acts[l];

        // dW_l (out,in) = dZ_l^T (out,M) @ A_prev (M,in)
        auto dW = torch::empty({out_dim, in_dim}, opts);
        launch_matmul(dZ, A_prev, dW, out_dim, in_dim, M, /*transA=*/true, /*transB=*/false, stream);

        // db_l (out,) = column-sum of dZ_l over the batch
        auto db = torch::empty({out_dim}, opts);
        col_sum_kernel<<<out_dim, EW_BLOCK, EW_BLOCK * sizeof(float), stream>>>(
            dZ.data_ptr<float>(), db.data_ptr<float>(), M, out_dim);

        dWs[l] = dW;
        dbs[l] = db;

        if (l > 0) {
            // dA_prev (M,in) = dZ_l (M,out) @ W_l (out,in) — no transpose.
            auto dA_prev = torch::empty({M, in_dim}, opts);
            launch_matmul(dZ, weights[l], dA_prev, M, in_dim, out_dim, false, false, stream);

            // Backprop through sine layer (l-1): dZ_{l-1} = dA_prev · w0·cos(w0·Z_{l-1})
            const float w0 = (l - 1 == 0) ? w0_first : w0_hidden;
            const auto& Z_prev = Zs[l - 1];
            const int total = M * in_dim;
            const int bblocks = CEILDIV(total, EW_BLOCK);
            auto dZ_next = torch::empty({M, in_dim}, opts);
            sine_backward_kernel<<<bblocks, EW_BLOCK, 0, stream>>>(
                dA_prev.data_ptr<float>(), Z_prev.data_ptr<float>(), dZ_next.data_ptr<float>(), total, w0);

            dZ = dZ_next;
        }
    }

    std::vector<torch::Tensor> result;
    for (int l = 0; l < L; ++l) {
        result.push_back(dWs[l]);
        result.push_back(dbs[l]);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &siren_forward,  "SIREN MLP forward (CUDA)");
    m.def("backward", &siren_backward, "SIREN MLP backward (CUDA)");
}
