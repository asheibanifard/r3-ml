#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

__device__ __forceinline__ void normalize_quat(const float* q, float* qn) {
    float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    float inv_norm = rsqrtf(fmaxf(n2, 1e-12f));
    qn[0] = q[0] * inv_norm;
    qn[1] = q[1] * inv_norm;
    qn[2] = q[2] * inv_norm;
    qn[3] = q[3] * inv_norm;
}

// Convention: q = (w, x, y, z). If the training checkpoint stores (x, y, z, w)
// (scipy / PLY convention), reorder in the driver before calling in -- a
// scrambled rotation is invisible on near-isotropic Gaussians and only shows
// up on filaments.
__device__ __forceinline__ void quat_to_rotmat(const float* q, float* R) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1.f - 2.f * (y * y + z * z);
    R[1] = 2.f * (x * y - w * z);
    R[2] = 2.f * (x * z + w * y);
    R[3] = 2.f * (x * y + w * z);
    R[4] = 1.f - 2.f * (x * x + z * z);
    R[5] = 2.f * (y * z - w * x);
    R[6] = 2.f * (x * z - w * y);
    R[7] = 2.f * (y * z + w * x);
    R[8] = 1.f - 2.f * (x * x + y * y);
}

__device__ __forceinline__ void mat3_mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c]
                         + A[r * 3 + 1] * B[1 * 3 + c]
                         + A[r * 3 + 2] * B[2 * 3 + c];
        }
    }
}

__device__ __forceinline__ void mat3_transpose(const float* A, float* AT) {
    AT[0] = A[0]; AT[1] = A[3]; AT[2] = A[6];
    AT[3] = A[1]; AT[4] = A[4]; AT[5] = A[7];
    AT[6] = A[2]; AT[7] = A[5]; AT[8] = A[8];
}

// Returns false on (near-)singular input. Callers must treat the fallback
// identity as INVALID -- it is only written so downstream reads are defined,
// and the corresponding Gaussian must be dropped via valid_mask.
__device__ __forceinline__ bool invert_2x2(const float* A, float* invA) {
    float det = A[0] * A[3] - A[1] * A[2];
    if (fabsf(det) < 1e-12f) {
        invA[0] = 1.f;
        invA[1] = 0.f;
        invA[2] = 0.f;
        invA[3] = 1.f;
        return false;
    }
    float inv_det = 1.f / det;
    invA[0] =  A[3] * inv_det;
    invA[1] = -A[1] * inv_det;
    invA[2] = -A[2] * inv_det;
    invA[3] =  A[0] * inv_det;
    return true;
}

__global__ void transform_kernel(
    const float* means,
    const float* log_scales,
    const float* quats,
    const float* view,
    int n,
    float* means_cam,
    float* cov_cam,
    float* depths
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* m = means + i * 3;
    const float* ls = log_scales + i * 3;
    const float* q = quats + i * 4;

    float homo[4] = {m[0], m[1], m[2], 1.f};
    float cam[4];
    for (int r = 0; r < 4; ++r) {
        cam[r] = view[r * 4 + 0] * homo[0]
               + view[r * 4 + 1] * homo[1]
               + view[r * 4 + 2] * homo[2]
               + view[r * 4 + 3] * homo[3];
    }
    means_cam[i * 3 + 0] = cam[0];
    means_cam[i * 3 + 1] = cam[1];
    means_cam[i * 3 + 2] = cam[2];
    // Camera-space z. This -- not the NDC depth from project_kernel -- is the
    // depth the driver should feed to rasterize_scene: it is linear in
    // physical distance, so depth slabs are uniform in world units and the
    // depth cue decays proportionally to distance. NDC z is a hyperbolic
    // remap under perspective and would make both nonuniform.
    depths[i] = cam[2];

    float qn[4];
    float Rq[9];
    normalize_quat(q, qn);
    quat_to_rotmat(qn, Rq);

    float scales[3] = {expf(ls[0]), expf(ls[1]), expf(ls[2])};
    float S[9] = {
        scales[0] * scales[0], 0.f, 0.f,
        0.f, scales[1] * scales[1], 0.f,
        0.f, 0.f, scales[2] * scales[2],
    };

    float tmp[9], Rt[9], cov_world[9];
    mat3_mul(Rq, S, tmp);
    mat3_transpose(Rq, Rt);
    mat3_mul(tmp, Rt, cov_world);

    const float* V = view;
    float Rv[9] = {
        V[0], V[1], V[2],
        V[4], V[5], V[6],
        V[8], V[9], V[10],
    };
    float Rv_t[9], tmp2[9], cov_cam_i[9];
    mat3_transpose(Rv, Rv_t);
    mat3_mul(Rv, cov_world, tmp2);
    mat3_mul(tmp2, Rv_t, cov_cam_i);

    for (int j = 0; j < 9; ++j) {
        cov_cam[i * 9 + j] = cov_cam_i[j];
    }
}

__global__ void project_kernel(
    const float* means_cam,
    const float* cov_cam,
    const float* projection,
    int n,
    int image_h,
    int image_w,
    float eps,
    float* u_norm,
    float* v_norm,
    float* u_px,
    float* v_px,
    float* ndc_depths,
    float* inv_screen_covariances,
    uint8_t* valid_mask
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* m = means_cam + i * 3;
    const float* C = cov_cam + i * 9;

    float homo[4] = {m[0], m[1], m[2], 1.f};
    float clip[4];
    for (int r = 0; r < 4; ++r) {
        clip[r] = projection[r * 4 + 0] * homo[0]
                + projection[r * 4 + 1] * homo[1]
                + projection[r * 4 + 2] * homo[2]
                + projection[r * 4 + 3] * homo[3];
    }

    // Sign-preserving clamp: behind-camera points project to a mirrored but
    // finite NDC position rather than diverging. They are NOT rejected here;
    // cull_kernel's z bounds / z_forward_sign are responsible for that.
    float w = clip[3];
    if (fabsf(w) < 1e-6f) {
        w = copysignf(1e-6f, (w == 0.f) ? 1.f : w);
    }
    float inv_w = 1.f / w;
    float inv_w2 = inv_w * inv_w;
    float px = clip[0] * inv_w;
    float py = clip[1] * inv_w;
    float pz = clip[2] * inv_w;

    u_norm[i] = px;
    v_norm[i] = py;
    ndc_depths[i] = pz;
    u_px[i] = (px + 1.f) * 0.5f * float(image_w - 1);
    v_px[i] = (py + 1.f) * 0.5f * float(image_h - 1);

    // Viewport scale folded into the Jacobian so the screen covariance (and
    // its inverse Q) is in pixel^-2 units, matching the pixel-space offsets
    // used in bin_kernel and raster_kernel.
    float su = 0.5f * float(image_w - 1);
    float sv = 0.5f * float(image_h - 1);

    float J[6];
    for (int c = 0; c < 3; ++c) {
        J[c]     = su * (projection[0 * 4 + c] * inv_w - clip[0] * inv_w2 * projection[3 * 4 + c]);
        J[3 + c] = sv * (projection[1 * 4 + c] * inv_w - clip[1] * inv_w2 * projection[3 * 4 + c]);
    }

    float JC[6];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            JC[r * 3 + c] = J[r * 3 + 0] * C[0 * 3 + c]
                          + J[r * 3 + 1] * C[1 * 3 + c]
                          + J[r * 3 + 2] * C[2 * 3 + c];
        }
    }

    float JT[6] = {J[0], J[3], J[1], J[4], J[2], J[5]};
    float screen[4];
    screen[0] = JC[0] * JT[0] + JC[1] * JT[2] + JC[2] * JT[4];
    screen[1] = JC[0] * JT[1] + JC[1] * JT[3] + JC[2] * JT[5];
    screen[2] = JC[3] * JT[0] + JC[4] * JT[2] + JC[5] * JT[4];
    screen[3] = JC[3] * JT[1] + JC[4] * JT[3] + JC[5] * JT[5];
    screen[0] += eps;
    screen[3] += eps;

    float inv_screen[4];
    bool ok = invert_2x2(screen, inv_screen);
    // Also flag non-finite Q entries: a NaN that sneaks through the inversion
    // would otherwise fail every mahal test silently (contrib NaN -> compares
    // false), or worse, poison the accumulator.
    ok = ok && isfinite(inv_screen[0]) && isfinite(inv_screen[1])
            && isfinite(inv_screen[2]) && isfinite(inv_screen[3]);
    valid_mask[i] = ok ? 1 : 0;
    inv_screen_covariances[i * 4 + 0] = inv_screen[0];
    inv_screen_covariances[i * 4 + 1] = inv_screen[1];
    inv_screen_covariances[i * 4 + 2] = inv_screen[2];
    inv_screen_covariances[i * 4 + 3] = inv_screen[3];
}

__global__ void cull_kernel(
    const float* means_cam,
    const float* depths,
    const float* inten,
    const uint8_t* valid_mask,
    int n,
    float x_min,
    float x_max,
    float y_min,
    float y_max,
    float z_min,
    float z_max,
    float opacity_threshold,
    float z_forward_sign,
    uint8_t* keep_mask
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* m = means_cam + i * 3;

    // Camera-forward test, parameterized instead of hard-coded: the sign of
    // "in front of the camera" depends on the projection convention
    // (OpenGL/3DGS look down -z -> pass -1; DirectX/COLMAP +z forward ->
    // pass +1). Pass 0 to disable, e.g. for the identity-matrix smoke test
    // where camera space is just the checkpoint's native coordinates and a
    // sign test would silently cull half the volume.
    bool forward_ok = (z_forward_sign == 0.f) || (m[2] * z_forward_sign > 0.f);

    bool keep = (valid_mask[i] != 0 &&
                 forward_ok &&
                 m[0] >= x_min && m[0] <= x_max &&
                 m[1] >= y_min && m[1] <= y_max &&
                 m[2] >= z_min && m[2] <= z_max &&
                 inten[i] >= opacity_threshold &&
                 isfinite(m[0]) && isfinite(m[1]) && isfinite(m[2]) &&
                 isfinite(depths[i]) && isfinite(inten[i]));

    keep_mask[i] = keep ? 1 : 0;
}

__global__ void bin_kernel(
    const float* u_px,
    const float* v_px,
    const float* inv_screen_covariances,
    const uint8_t* keep_mask,
    int n,
    int image_h,
    int image_w,
    int tile_size,
    uint8_t* tile_mask
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || keep_mask[i] == 0) return;

    int tiles_x = (image_w + tile_size - 1) / tile_size;
    int tiles_y = (image_h + tile_size - 1) / tile_size;

    // Marginal sigmas from the inverse covariance: Sigma_00 = Q11/det(Q),
    // Sigma_11 = Q00/det(Q). The conditional sigma rsqrt(Q00) underestimates
    // the footprint of tilted anisotropic Gaussians. fabsf(det): a negative
    // det (numerical noise / corrupted Q) must not clamp to +1e-8 and bin the
    // Gaussian into every tile.
    const float* Q = inv_screen_covariances + i * 4;
    float det = Q[0] * Q[3] - Q[1] * Q[2];
    float inv_det = 1.f / fmaxf(fabsf(det), 1e-8f);
    float sigma_u = sqrtf(fmaxf(Q[3] * inv_det, 0.f));
    float sigma_v = sqrtf(fmaxf(Q[0] * inv_det, 0.f));

    // 3-sigma binning radius, matched to raster_kernel's mahal > 9 cutoff.
    int u_min = static_cast<int>(floorf((u_px[i] - 3.f * sigma_u) / float(tile_size)));
    int u_max = static_cast<int>(floorf((u_px[i] + 3.f * sigma_u) / float(tile_size)));
    int v_min = static_cast<int>(floorf((v_px[i] - 3.f * sigma_v) / float(tile_size)));
    int v_max = static_cast<int>(floorf((v_px[i] + 3.f * sigma_v) / float(tile_size)));

    // Early-out before clamping: an entirely off-screen Gaussian would
    // otherwise clamp both endpoints onto the same edge tile and force every
    // pixel there to pay inner-loop cost for a guaranteed mahal miss.
    if (u_max < 0 || u_min > tiles_x - 1 || v_max < 0 || v_min > tiles_y - 1) return;

    u_min = max(0, min(u_min, tiles_x - 1));
    u_max = max(0, min(u_max, tiles_x - 1));
    v_min = max(0, min(v_min, tiles_y - 1));
    v_max = max(0, min(v_max, tiles_y - 1));

    for (int ty = v_min; ty <= v_max; ++ty) {
        for (int tx = u_min; tx <= u_max; ++tx) {
            tile_mask[(i * tiles_y + ty) * tiles_x + tx] = 1;
        }
    }
}

constexpr int MAX_DEPTH_SLABS = 64;

// Depth-binned MIP with the depth cue folded into the max.
//
// The density field is a *sum* of overlapping Gaussians -- no single one
// carries structure alone -- so a correct MIP needs two nested aggregation
// stages: (1) sum all contributions within a depth slab (recovers the smooth
// local density at that depth), then (2) max across slabs, which is the
// actual MIP. A single running scalar can only do one or the other: summing
// everything washes out contrast (no occlusion); maxing over individuals
// gives a speckled mosaic of isolated blobs.
//
// The depth cue exp(-lambda * (t_k - d_ref)) is applied per-slab BEFORE the
// max, so it is a genuinely depth-attenuated MIP (nearer structure wins
// ties), and the previous post-hoc composite stage -- with its broken
// depth<=0 background sentinel that collided with legitimate negative camera
// depths -- is gone. Empty pixels write depth_map = +inf; consumers should
// test isfinite(), not sign.
//
// Note on slab_sum[]: the scatter index is data-dependent (computed from
// depths[i]), so this array lives in local memory regardless of how it is
// sized or templated -- the compiler cannot register-allocate dynamically
// indexed storage. Irrelevant at preview resolution; if this ever needs to
// scale, the fix is the sorted-Gaussian single-pass variant (sort by depth
// once on the host, then per pixel keep a running slab sum that resets at
// slab boundaries and a running max), which needs no per-thread array at all
// and is compatible with mean-depth binning (though not with longitudinal
// slab weighting).
__global__ void raster_kernel(
    const float* u_px,
    const float* v_px,
    const float* depths,
    const float* inten,
    const float* inv_screen_covariances,
    const uint8_t* keep_mask,
    const uint8_t* tile_mask,
    int n,
    int image_h,
    int image_w,
    int tile_size,
    int tiles_x,
    int tiles_y,
    float depth_min,
    float depth_max,
    int num_depth_slabs,
    float depth_cue_lambda,
    float d_ref,
    float* image,
    float* depth_map
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= image_h * image_w) return;

    int y = pixel / image_w;
    int x = pixel - y * image_w;
    int tile_x = x / tile_size;
    int tile_y = y / tile_size;
    float px = float(x) + 0.5f;
    float py = float(y) + 0.5f;

    int slabs = min(max(num_depth_slabs, 1), MAX_DEPTH_SLABS);
    float slab_sum[MAX_DEPTH_SLABS];
    for (int s = 0; s < slabs; ++s) slab_sum[s] = 0.f;

    float depth_span = fmaxf(depth_max - depth_min, 1e-6f);
    float slab_width = depth_span / float(slabs);

    for (int i = 0; i < n; ++i) {
        if (keep_mask[i] == 0) continue;
        if (tile_mask[(i * tiles_y + tile_y) * tiles_x + tile_x] == 0) continue;

        float dx = px - u_px[i];
        float dy = py - v_px[i];
        const float* Q = inv_screen_covariances + i * 4;
        float mahal = dx * (Q[0] * dx + Q[1] * dy) + dy * (Q[2] * dx + Q[3] * dy);
        if (mahal > 9.f) continue;

        float contrib = inten[i] * __expf(-0.5f * mahal);
        int slab = int((depths[i] - depth_min) / depth_span * float(slabs));
        slab = min(max(slab, 0), slabs - 1);
        slab_sum[slab] += contrib;
    }

    float best_score = 0.f;
    float best_sum = 0.f;
    int best_slab = -1;
    for (int s = 0; s < slabs; ++s) {
        float t_k = depth_min + (float(s) + 0.5f) * slab_width;
        float cue = (depth_cue_lambda > 0.f)
            ? __expf(-depth_cue_lambda * fmaxf(t_k - d_ref, 0.f))
            : 1.f;
        float score = slab_sum[s] * cue;
        if (score > best_score) {
            best_score = score;
            best_sum = slab_sum[s];
            best_slab = s;
        }
    }

    // image gets the cued score (this is the depth-attenuated MIP output);
    // depth_map gets the physical slab-center depth of the winner, +inf when
    // the pixel is empty. best_sum is kept around should you ever want the
    // un-cued intensity as a second channel -- swap it in here.
    (void)best_sum;
    image[pixel] = best_score;
    depth_map[pixel] = (best_slab >= 0)
        ? depth_min + (float(best_slab) + 0.5f) * slab_width
        : INFINITY;
}

torch::Tensor contiguous_float(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "expected CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "expected float32 tensor");
    return t.contiguous();
}

torch::Tensor contiguous_u8(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "expected CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kUInt8, "expected uint8 tensor");
    return t.contiguous();
}

}  // namespace

std::vector<torch::Tensor> transform_to_camera_space(
    const torch::Tensor& means,
    const torch::Tensor& log_scales,
    const torch::Tensor& quats,
    const torch::Tensor& camera_view_matrix
) {
    auto means_c = contiguous_float(means);
    auto scales_c = contiguous_float(log_scales);
    auto quats_c = contiguous_float(quats);
    auto view_c = contiguous_float(camera_view_matrix);
    TORCH_CHECK(means_c.dim() == 2 && means_c.size(1) == 3, "means must be [N,3]");
    TORCH_CHECK(scales_c.dim() == 2 && scales_c.size(1) == 3, "log_scales must be [N,3]");
    TORCH_CHECK(quats_c.dim() == 2 && quats_c.size(1) == 4, "quats must be [N,4] (w,x,y,z)");
    TORCH_CHECK(view_c.numel() == 16, "camera_view_matrix must be 4x4");

    auto n = means_c.size(0);
    auto means_cam = torch::empty({n, 3}, means_c.options());
    auto cov_cam = torch::empty({n, 3, 3}, means_c.options());
    auto depths = torch::empty({n}, means_c.options());
    if (n == 0) return {means_cam, cov_cam, depths};

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    transform_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means_c.data_ptr<float>(),
        scales_c.data_ptr<float>(),
        quats_c.data_ptr<float>(),
        view_c.data_ptr<float>(),
        (int)n,
        means_cam.data_ptr<float>(),
        cov_cam.data_ptr<float>(),
        depths.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {means_cam, cov_cam, depths};
}

std::vector<torch::Tensor> project_to_image_plane(
    const torch::Tensor& means_cam,
    const torch::Tensor& cov_cam,
    const torch::Tensor& projection_matrix,
    int image_h,
    int image_w,
    float eps
) {
    auto means_c = contiguous_float(means_cam);
    auto cov_c = contiguous_float(cov_cam);
    auto proj_c = contiguous_float(projection_matrix);
    TORCH_CHECK(means_c.dim() == 2 && means_c.size(1) == 3, "means_cam must be [N,3]");
    TORCH_CHECK(proj_c.numel() == 16, "projection_matrix must be 4x4");

    auto n = means_c.size(0);
    auto u_norm = torch::empty({n}, means_c.options());
    auto v_norm = torch::empty({n}, means_c.options());
    auto u_px = torch::empty({n}, means_c.options());
    auto v_px = torch::empty({n}, means_c.options());
    auto ndc_depths = torch::empty({n}, means_c.options());
    auto inv_screen = torch::empty({n, 2, 2}, means_c.options());
    auto valid_mask = torch::empty({n}, torch::TensorOptions().dtype(torch::kUInt8).device(means_c.device()));
    if (n == 0) return {u_norm, v_norm, u_px, v_px, ndc_depths, inv_screen, valid_mask};

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    project_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means_c.data_ptr<float>(),
        cov_c.data_ptr<float>(),
        proj_c.data_ptr<float>(),
        (int)n,
        image_h,
        image_w,
        eps,
        u_norm.data_ptr<float>(),
        v_norm.data_ptr<float>(),
        u_px.data_ptr<float>(),
        v_px.data_ptr<float>(),
        ndc_depths.data_ptr<float>(),
        inv_screen.data_ptr<float>(),
        valid_mask.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {u_norm, v_norm, u_px, v_px, ndc_depths, inv_screen, valid_mask};
}

// valid_mask (from project_to_image_plane) is now a required input and is
// folded into keep_mask here, so the driver cannot forget to apply it and
// singular-covariance Gaussians can never render as phantom blobs.
torch::Tensor cull_gaussians(
    const torch::Tensor& means_cam,
    const torch::Tensor& depths,
    const torch::Tensor& inten,
    const torch::Tensor& valid_mask,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    double z_min,
    double z_max,
    double opacity_threshold,
    double z_forward_sign
) {
    auto means_c = contiguous_float(means_cam);
    auto depth_c = contiguous_float(depths);
    auto inten_c = contiguous_float(inten);
    auto valid_c = contiguous_u8(valid_mask);
    auto n = means_c.size(0);
    TORCH_CHECK(depth_c.numel() == n && inten_c.numel() == n && valid_c.numel() == n,
                "depths, inten, valid_mask must all have length N");

    auto keep_mask = torch::empty({n}, torch::TensorOptions().dtype(torch::kUInt8).device(means_c.device()));
    if (n == 0) return keep_mask;

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    cull_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means_c.data_ptr<float>(),
        depth_c.data_ptr<float>(),
        inten_c.data_ptr<float>(),
        valid_c.data_ptr<uint8_t>(),
        (int)n,
        (float)x_min, (float)x_max,
        (float)y_min, (float)y_max,
        (float)z_min, (float)z_max,
        (float)opacity_threshold,
        (float)z_forward_sign,
        keep_mask.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return keep_mask;
}

torch::Tensor bin_gaussians_into_tiles(
    const torch::Tensor& u_px,
    const torch::Tensor& v_px,
    const torch::Tensor& inv_screen_covariances,
    const torch::Tensor& keep_mask,
    int image_h,
    int image_w,
    int tile_size
) {
    auto u_c = contiguous_float(u_px);
    auto v_c = contiguous_float(v_px);
    auto inv_c = contiguous_float(inv_screen_covariances);
    auto keep_c = contiguous_u8(keep_mask);

    auto n = u_c.size(0);
    int tiles_x = (image_w + tile_size - 1) / tile_size;
    int tiles_y = (image_h + tile_size - 1) / tile_size;
    auto tile_mask = torch::zeros({n, tiles_y, tiles_x}, keep_c.options());
    if (n == 0) return tile_mask;

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    bin_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        u_c.data_ptr<float>(),
        v_c.data_ptr<float>(),
        inv_c.data_ptr<float>(),
        keep_c.data_ptr<uint8_t>(),
        (int)n,
        image_h,
        image_w,
        tile_size,
        tile_mask.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return tile_mask;
}

// depths must be CAMERA-SPACE depth (the third return of
// transform_to_camera_space), not the NDC depth from projection -- slabs and
// the depth cue are both linear in this quantity.
//
// The slab range is computed over KEPT Gaussians only: one culled outlier
// (behind-camera mirror, low-intensity stray) would otherwise stretch the
// range until all real structure lands in one slab and the render silently
// degenerates back to the sum-everything regime.
//
// d_ref defaults (NaN) to the near edge of the kept depth range.
// depth_cue_lambda = 0 disables the cue entirely.
std::vector<torch::Tensor> rasterize_scene(
    const torch::Tensor& u_px,
    const torch::Tensor& v_px,
    const torch::Tensor& depths,
    const torch::Tensor& inten,
    const torch::Tensor& inv_screen_covariances,
    const torch::Tensor& keep_mask,
    const torch::Tensor& tile_mask,
    int image_h,
    int image_w,
    int tile_size,
    int num_depth_slabs,
    double depth_cue_lambda,
    double d_ref
) {
    auto u_c = contiguous_float(u_px);
    auto v_c = contiguous_float(v_px);
    auto depth_c = contiguous_float(depths);
    auto inten_c = contiguous_float(inten);
    auto inv_c = contiguous_float(inv_screen_covariances);
    auto keep_c = contiguous_u8(keep_mask);
    auto tile_c = contiguous_u8(tile_mask);

    auto n = u_c.size(0);
    auto image = torch::zeros({image_h, image_w}, u_c.options());
    auto depth_map = torch::full({image_h, image_w},
                                 std::numeric_limits<float>::infinity(),
                                 u_c.options());
    int tiles_x = (image_w + tile_size - 1) / tile_size;
    int tiles_y = (image_h + tile_size - 1) / tile_size;

    if (n > 0) {
        auto kept_depths = depth_c.masked_select(keep_c.to(torch::kBool));
        if (kept_depths.numel() > 0) {
            float depth_min = kept_depths.min().item<float>();
            float depth_max = kept_depths.max().item<float>();
            float dref = std::isnan(d_ref) ? depth_min : (float)d_ref;
            int slabs = std::max(1, std::min(num_depth_slabs, MAX_DEPTH_SLABS));

            const int threads = 256;
            const int blocks = (image_h * image_w + threads - 1) / threads;
            raster_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                u_c.data_ptr<float>(),
                v_c.data_ptr<float>(),
                depth_c.data_ptr<float>(),
                inten_c.data_ptr<float>(),
                inv_c.data_ptr<float>(),
                keep_c.data_ptr<uint8_t>(),
                tile_c.data_ptr<uint8_t>(),
                (int)n,
                image_h,
                image_w,
                tile_size,
                tiles_x,
                tiles_y,
                depth_min,
                depth_max,
                slabs,
                (float)depth_cue_lambda,
                dref,
                image.data_ptr<float>(),
                depth_map.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        // If nothing survived culling, image stays zero and depth_map stays
        // +inf -- a well-defined "empty render", not an error.
    }

    return {image, depth_map};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transform_to_camera_space", &transform_to_camera_space, "transform_to_camera_space");
    m.def("project_to_image_plane", &project_to_image_plane, "project_to_image_plane");
    m.def("cull_gaussians", &cull_gaussians, "cull_gaussians",
          py::arg("means_cam"), py::arg("depths"), py::arg("inten"), py::arg("valid_mask"),
          py::arg("x_min"), py::arg("x_max"),
          py::arg("y_min"), py::arg("y_max"),
          py::arg("z_min"), py::arg("z_max"),
          py::arg("opacity_threshold"),
          py::arg("z_forward_sign") = 0.0);
    m.def("bin_gaussians_into_tiles", &bin_gaussians_into_tiles, "bin_gaussians_into_tiles");
    m.def("rasterize_scene", &rasterize_scene, "rasterize_scene",
          py::arg("u_px"), py::arg("v_px"), py::arg("depths"), py::arg("inten"),
          py::arg("inv_screen_covariances"), py::arg("keep_mask"), py::arg("tile_mask"),
          py::arg("image_h"), py::arg("image_w"), py::arg("tile_size"),
          py::arg("num_depth_slabs") = 16,
          py::arg("depth_cue_lambda") = 0.0,
          py::arg("d_ref") = std::numeric_limits<double>::quiet_NaN());
}