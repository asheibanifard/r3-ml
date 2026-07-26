// voxel/extension/siren.cu
//
// SIREN continuous field F_theta(x,y,z): initialization, fused forward +
// analytic backward training step (one CUDA thread per sampled voxel,
// atomicAdd gradient accumulation), Adam parameter update, and full-grid
// reconstruction. A hand-written from-scratch CUDA training kernel — no
// torch.autograd involved, mirroring the project's other CUDA pipelines.
#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace {

__device__ inline float deterministic_uniform_signed(int index, unsigned int seed) {
    unsigned int state =
        (static_cast<unsigned int>(index) + 1u) * 747796405u +
        seed * 2891336453u;
    state = (state ^ (state >> 16)) * 2246822519u;
    state ^= state >> 13;
    return ((state & 0x00ffffffu) / 16777216.0f) * 2.0f - 1.0f;
}

__global__ void initialize_siren_kernel(
        float* __restrict__ parameters, unsigned int seed, float hidden_omega_0) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SirenLayout::PARAM_COUNT) return;

    const float random_value = deterministic_uniform_signed(index, seed);
    float bound = 0.0f;

    if (index < SirenLayout::FIRST_B) {
        // First-layer SIREN initialization: U(-1/fan_in, 1/fan_in).
        bound = 1.0f / static_cast<float>(kInputSize);
    } else if (index < SirenLayout::HIDDEN_START) {
        // First-layer biases start at zero.
        parameters[index] = 0.0f;
        return;
    } else if (index < SirenLayout::OUTPUT_W) {
        // Hidden-layer region contains alternating weights and biases.
        const int relative = index - SirenLayout::HIDDEN_START;
        const int block_size =
            SirenLayout::HIDDEN_WEIGHT_COUNT + SirenLayout::HIDDEN_BIAS_COUNT;
        const int within_block = relative % block_size;

        if (within_block >= SirenLayout::HIDDEN_WEIGHT_COUNT) {
            parameters[index] = 0.0f;
            return;
        }

        bound = sqrtf(6.0f / static_cast<float>(HIDDEN_SIZE)) / hidden_omega_0;
    } else if (index < SirenLayout::OUTPUT_B) {
        // Linear output-layer weights.
        bound = sqrtf(6.0f / static_cast<float>(HIDDEN_SIZE));
    } else {
        // Output bias.
        parameters[index] = 0.0f;
        return;
    }

    parameters[index] = random_value * bound;
}

__device__ inline float siren_forward(
        const float* __restrict__ parameters,
        float x, float y, float z,
        float activations[HIDDEN_LAYERS][HIDDEN_SIZE],
        float first_omega_0, float hidden_omega_0) {
    const float input[kInputSize] = {x, y, z};

    // First hidden layer: kInputSize -> HIDDEN_SIZE.
    for (int output = 0; output < HIDDEN_SIZE; ++output) {
        float preactivation = parameters[SirenLayout::FIRST_B + output];

        for (int input_index = 0; input_index < kInputSize; ++input_index) {
            preactivation += parameters[
                SirenLayout::FIRST_W + output * kInputSize + input_index
            ] * input[input_index];
        }

        activations[0][output] = sinf(first_omega_0 * preactivation);
    }

    // Remaining sine-activated hidden layers.
    for (int layer = 1; layer < HIDDEN_LAYERS; ++layer) {
        const int weight_offset = SirenLayout::hidden_weight_offset(layer);
        const int bias_offset = SirenLayout::hidden_bias_offset(layer);

        for (int output = 0; output < HIDDEN_SIZE; ++output) {
            float preactivation = parameters[bias_offset + output];

            for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
                preactivation += parameters[
                    weight_offset + output * HIDDEN_SIZE + input_index
                ] * activations[layer - 1][input_index];
            }

            activations[layer][output] = sinf(hidden_omega_0 * preactivation);
        }
    }

    // Linear scalar output followed by sigmoid to constrain density to [0,1].
    float output_value = parameters[SirenLayout::OUTPUT_B];

    for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
        output_value += parameters[SirenLayout::OUTPUT_W + hidden] *
                        activations[HIDDEN_LAYERS - 1][hidden];
    }

    return 1.0f / (1.0f + expf(-output_value));
}

// One CUDA thread processes one randomly selected training voxel.
__global__ void siren_training_step_kernel(
        const float* __restrict__ target_volume,
        const float* __restrict__ parameters,
        float* __restrict__ gradients,
        float* __restrict__ batch_loss,
        int64_t grid_size,
        int64_t train_batch,
        int iteration,
        float first_omega_0,
        float hidden_omega_0) {
    const int64_t sample = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (sample >= train_batch) return;

    const int64_t voxel_count = grid_size * grid_size * grid_size;

    unsigned int hash =
        (static_cast<unsigned int>(sample) + 1u) * 747796405u +
        (static_cast<unsigned int>(iteration) + 1u) * 2891336453u;
    hash = (hash ^ (hash >> 16)) * 2246822519u;
    hash = (hash ^ (hash >> 13)) * 3266489917u;
    hash ^= hash >> 16;

    const int64_t index = static_cast<int64_t>(hash % static_cast<unsigned int>(voxel_count));
    const int64_t vx = index % grid_size;
    const int64_t vy = (index / grid_size) % grid_size;
    const int64_t vz = index / (grid_size * grid_size);

    const float x = (static_cast<float>(vx) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float y = (static_cast<float>(vy) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float z = (static_cast<float>(vz) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float input[kInputSize] = {x, y, z};

    float activations[HIDDEN_LAYERS][HIDDEN_SIZE];
    const float prediction = siren_forward(
        parameters, x, y, z, activations, first_omega_0, hidden_omega_0
    );

    const float error = prediction - target_volume[index];
    atomicAdd(batch_loss, error * error / static_cast<float>(train_batch));

    // d(MSE)/d(pre-sigmoid output).
    const float output_gradient =
        2.0f * error * prediction * (1.0f - prediction) /
        static_cast<float>(train_batch);

    // Output layer gradients and gradient entering the last hidden activation.
    atomicAdd(&gradients[SirenLayout::OUTPUT_B], output_gradient);

    float current_activation_gradient[HIDDEN_SIZE];
    float previous_activation_gradient[HIDDEN_SIZE];

    for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
        atomicAdd(
            &gradients[SirenLayout::OUTPUT_W + hidden],
            output_gradient * activations[HIDDEN_LAYERS - 1][hidden]
        );

        current_activation_gradient[hidden] =
            output_gradient * parameters[SirenLayout::OUTPUT_W + hidden];
    }

    // Backpropagate through hidden layers HIDDEN_LAYERS-1 down to 1.
    for (int layer = HIDDEN_LAYERS - 1; layer >= 1; --layer) {
        const int weight_offset = SirenLayout::hidden_weight_offset(layer);
        const int bias_offset = SirenLayout::hidden_bias_offset(layer);

        for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
            previous_activation_gradient[input_index] = 0.0f;
        }

        for (int output = 0; output < HIDDEN_SIZE; ++output) {
            float preactivation = parameters[bias_offset + output];

            for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
                preactivation += parameters[
                    weight_offset + output * HIDDEN_SIZE + input_index
                ] * activations[layer - 1][input_index];
            }

            const float preactivation_gradient =
                current_activation_gradient[output] *
                hidden_omega_0 *
                cosf(hidden_omega_0 * preactivation);

            atomicAdd(&gradients[bias_offset + output], preactivation_gradient);

            for (int input_index = 0; input_index < HIDDEN_SIZE; ++input_index) {
                const int weight_index =
                    weight_offset + output * HIDDEN_SIZE + input_index;

                atomicAdd(
                    &gradients[weight_index],
                    preactivation_gradient * activations[layer - 1][input_index]
                );

                previous_activation_gradient[input_index] +=
                    preactivation_gradient * parameters[weight_index];
            }
        }

        for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
            current_activation_gradient[hidden] = previous_activation_gradient[hidden];
        }
    }

    // Backpropagate through the first hidden layer.
    for (int output = 0; output < HIDDEN_SIZE; ++output) {
        float preactivation = parameters[SirenLayout::FIRST_B + output];

        for (int input_index = 0; input_index < kInputSize; ++input_index) {
            preactivation += parameters[
                SirenLayout::FIRST_W + output * kInputSize + input_index
            ] * input[input_index];
        }

        const float preactivation_gradient =
            current_activation_gradient[output] *
            first_omega_0 *
            cosf(first_omega_0 * preactivation);

        atomicAdd(&gradients[SirenLayout::FIRST_B + output], preactivation_gradient);

        for (int input_index = 0; input_index < kInputSize; ++input_index) {
            atomicAdd(
                &gradients[SirenLayout::FIRST_W + output * kInputSize + input_index],
                preactivation_gradient * input[input_index]
            );
        }
    }
}

__global__ void adam_update_kernel(
        float* __restrict__ parameters,
        const float* __restrict__ gradients,
        float* __restrict__ first_moment,
        float* __restrict__ second_moment,
        int iteration,
        float learning_rate) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SirenLayout::PARAM_COUNT) return;

    const float gradient = gradients[index];
    first_moment[index] = 0.9f * first_moment[index] + 0.1f * gradient;
    second_moment[index] =
        0.999f * second_moment[index] + 0.001f * gradient * gradient;

    const float corrected_first =
        first_moment[index] / (1.0f - powf(0.9f, static_cast<float>(iteration)));
    const float corrected_second =
        second_moment[index] / (1.0f - powf(0.999f, static_cast<float>(iteration)));

    parameters[index] -=
        learning_rate * corrected_first / (sqrtf(corrected_second) + 1.0e-8f);
}

__global__ void reconstruct_voxel_grid_kernel(
        const float* __restrict__ parameters,
        float* __restrict__ reconstructed_volume,
        int64_t grid_size,
        float first_omega_0,
        float hidden_omega_0) {
    const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t voxel_count = grid_size * grid_size * grid_size;
    if (index >= voxel_count) return;

    const int64_t x = index % grid_size;
    const int64_t y = (index / grid_size) % grid_size;
    const int64_t z = index / (grid_size * grid_size);

    const float px = (static_cast<float>(x) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float py = (static_cast<float>(y) + 0.5f) / grid_size * 2.0f - 1.0f;
    const float pz = (static_cast<float>(z) + 0.5f) / grid_size * 2.0f - 1.0f;

    float activations[HIDDEN_LAYERS][HIDDEN_SIZE];
    reconstructed_volume[index] = siren_forward(
        parameters, px, py, pz, activations, first_omega_0, hidden_omega_0
    );
}

}  // namespace

// ── Python-visible entry points ─────────────────────────────────────────────

int64_t siren_param_count() {
    return SirenLayout::PARAM_COUNT;
}

torch::Tensor initialize_siren_cuda(int64_t seed, torch::Device device, double hidden_omega_0) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto parameters = torch::empty({SirenLayout::PARAM_COUNT}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    initialize_siren_kernel<<<num_blocks(SirenLayout::PARAM_COUNT), kThreadsPerBlock, 0, stream>>>(
        parameters.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(hidden_omega_0)
    );

    return parameters;
}

std::vector<torch::Tensor> siren_training_step_cuda(
        const torch::Tensor& target_volume,
        const torch::Tensor& parameters,
        int64_t grid_size,
        int64_t train_batch,
        int64_t iteration,
        double first_omega_0,
        double hidden_omega_0) {
    CHECK_INPUT(target_volume);
    CHECK_INPUT(parameters);
    TORCH_CHECK(
        parameters.numel() == SirenLayout::PARAM_COUNT,
        "parameters must have ", SirenLayout::PARAM_COUNT, " elements, got ",
        parameters.numel()
    );

    auto gradients = torch::zeros_like(parameters);
    auto batch_loss = torch::zeros({1}, parameters.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    siren_training_step_kernel<<<num_blocks(train_batch), kThreadsPerBlock, 0, stream>>>(
        target_volume.data_ptr<float>(),
        parameters.data_ptr<float>(),
        gradients.data_ptr<float>(),
        batch_loss.data_ptr<float>(),
        grid_size, train_batch,
        static_cast<int>(iteration),
        static_cast<float>(first_omega_0),
        static_cast<float>(hidden_omega_0)
    );

    return {gradients, batch_loss};
}

void adam_update_cuda(
        torch::Tensor& parameters,
        const torch::Tensor& gradients,
        torch::Tensor& first_moment,
        torch::Tensor& second_moment,
        int64_t iteration,
        double learning_rate) {
    CHECK_INPUT(parameters);
    CHECK_INPUT(gradients);
    CHECK_INPUT(first_moment);
    CHECK_INPUT(second_moment);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    adam_update_kernel<<<num_blocks(SirenLayout::PARAM_COUNT), kThreadsPerBlock, 0, stream>>>(
        parameters.data_ptr<float>(),
        gradients.data_ptr<float>(),
        first_moment.data_ptr<float>(),
        second_moment.data_ptr<float>(),
        static_cast<int>(iteration),
        static_cast<float>(learning_rate)
    );
}

torch::Tensor reconstruct_voxel_grid_cuda(
        const torch::Tensor& parameters,
        int64_t grid_size,
        double first_omega_0,
        double hidden_omega_0) {
    CHECK_INPUT(parameters);
    TORCH_CHECK(
        parameters.numel() == SirenLayout::PARAM_COUNT,
        "parameters must have ", SirenLayout::PARAM_COUNT, " elements, got ",
        parameters.numel()
    );

    const int64_t voxel_count = grid_size * grid_size * grid_size;
    auto volume = torch::empty({grid_size, grid_size, grid_size}, parameters.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reconstruct_voxel_grid_kernel<<<num_blocks(voxel_count), kThreadsPerBlock, 0, stream>>>(
        parameters.data_ptr<float>(),
        volume.data_ptr<float>(),
        grid_size,
        static_cast<float>(first_omega_0),
        static_cast<float>(hidden_omega_0)
    );

    return volume;
}
