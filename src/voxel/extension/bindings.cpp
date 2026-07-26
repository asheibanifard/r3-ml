// voxel/extension/bindings.cpp
#include <torch/extension.h>
#include <tuple>
#include <vector>

// siren.cu
int64_t siren_param_count();
torch::Tensor initialize_siren_cuda(int64_t seed, torch::Device device, double hidden_omega_0);
std::vector<torch::Tensor> siren_training_step_cuda(
    const torch::Tensor& target_volume,
    const torch::Tensor& parameters,
    int64_t grid_size,
    int64_t train_batch,
    int64_t iteration,
    double first_omega_0,
    double hidden_omega_0);
void adam_update_cuda(
    torch::Tensor& parameters,
    const torch::Tensor& gradients,
    torch::Tensor& first_moment,
    torch::Tensor& second_moment,
    int64_t iteration,
    double learning_rate);
torch::Tensor reconstruct_voxel_grid_cuda(
    const torch::Tensor& parameters,
    int64_t grid_size,
    double first_omega_0,
    double hidden_omega_0);

// render.cu
torch::Tensor create_ground_truth_volume_cuda(int64_t grid_size, torch::Device device);
torch::Tensor render_direct_volume_cuda(
    const torch::Tensor& volume,
    const torch::Tensor& camera,
    int64_t image_width,
    int64_t image_height,
    int64_t dvr_steps,
    double density_scale);
torch::Tensor render_voxel_rasterizer_cuda(
    const torch::Tensor& volume,
    const torch::Tensor& camera,
    int64_t image_width,
    int64_t image_height,
    double density_scale);

// metrics.cu
std::tuple<double, double, double> evaluate_images_cuda(
    const torch::Tensor& image_a,
    const torch::Tensor& image_b);
double compute_volume_psnr_cuda(const torch::Tensor& volume_a, const torch::Tensor& volume_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("siren_param_count", &siren_param_count, "Flat SIREN parameter count for the compiled architecture");
    m.def("initialize_siren", &initialize_siren_cuda, "Randomly initialize SIREN parameters (CUDA)");
    m.def("siren_training_step", &siren_training_step_cuda,
          "Fused SIREN forward + analytic backward training step (CUDA)");
    m.def("adam_update", &adam_update_cuda, "In-place Adam parameter update (CUDA)");
    m.def("reconstruct_voxel_grid", &reconstruct_voxel_grid_cuda,
          "Query the SIREN field at every voxel centre (CUDA)");

    m.def("create_ground_truth_volume", &create_ground_truth_volume_cuda,
          "Synthetic sphere+torus+blob scalar volume (CUDA)");
    m.def("render_direct_volume", &render_direct_volume_cuda,
          "Trilinear direct-volume ray marching renderer (CUDA)");
    m.def("render_voxel_rasterizer", &render_voxel_rasterizer_cuda,
          "Front-to-back dense-grid 3D-DDA voxel rasterizer (CUDA)");

    m.def("evaluate_images", &evaluate_images_cuda, "MSE/PSNR/SSIM between two [H,W] images (CUDA)");
    m.def("compute_volume_psnr", &compute_volume_psnr_cuda, "Full-grid PSNR between two [G,G,G] volumes (CUDA)");
}
