import sys, argparse, torch, torch.nn.functional as F, h5py, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '/root/project/scripts')

import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()
from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset, _load_eval_kernel
from diff_gaussian_rasterization import _C as C

device = torch.device('cuda')

# smoke_data blocks are 50^3 (see configs/smoke_config.yml) -> ssim_crop must match
cfg = argparse.Namespace(
    scale_min_clamp=1e-5, mahal_max_clamp=20.0, init_scale=0.05,
    init_inten=0.1, init_scale_z_factor=1.0, n_init=5000,
    swc_path=None, chunk_n=1000, eval_samples=200_000,
    ssim_crop=50, batch=2048,
    grad_sample_weight=0.0, lambda_ssim=0.2,
)

# --- load model directly via GaussianCloud.load (handles the .pth checkpoint format) ---
ckpt_path = '/root/project/models_smoke/block_z000_y001_x006/best.pth'
aabb = AABB.unit()
gc = GaussianCloud.load(ckpt_path, aabb, device, cfg)
print(f"Loaded {gc.N} Gaussians from {ckpt_path}")

# --- pure Gaussian MIP splatting (project's own fused splat_mip kernel, no
#     rasterization/tiling/frustum-culling camera pipeline) ---
H, W = 50, 50
kernel = _load_eval_kernel()
lo, hi = gc.aabb.lo.cpu(), gc.aabb.hi.cpu()
lo_x, hi_x = float(lo[0]), float(hi[0])
lo_y, hi_y = float(lo[1]), float(hi[1])
lo_z, hi_z = float(lo[2]), float(hi[2])

def render_splat(density_scale):
    flat = kernel.splat_mip(
        gc.means.contiguous(), gc.log_s.contiguous(), gc.quats.contiguous(), gc.inten.contiguous(),
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
        H, W, 64, 0,                      # depth_samples=64, view_axis=0 -> looking down Z (xy MIP)
        float(gc.scale_min), float(gc.mahal_clamp),
        density_scale,
    )
    return flat.reshape(H, W)

# splat_mip tone-maps via mapped = 1 - exp(-density_scale * accumulated_intensity);
# the kernel's default density_scale (1e-4) is tuned for the much larger/sparser
# FAFB 64^3 blocks. Calibrate it here so this block's accumulated intensity maps
# into a visible range: probe with a tiny density_scale (tone-map ~linear there,
# so probe / 1e-8 recovers the raw accumulated intensity), then rescale so the
# brightest pixel lands at ~95% (1 - exp(-3) = 0.95).
probe = render_splat(1e-8)
positive_max = float(probe.max().item()) / 1e-8
density_scale = 3.0 / max(positive_max, 1e-6)
print(f"estimated peak accumulated intensity: {positive_max:.3f}  ->  density_scale={density_scale:.5f}")

splat_mip = render_splat(density_scale).cpu().numpy()
print(f"splat_mip range: [{splat_mip.min():.3f}, {splat_mip.max():.3f}]")

# --- fixed tile rasterizer: true fitted scales (scale_modifier=1.0), a
#     low_pass floor sized for sub-pixel splats instead of the 0.3px^2
#     default (see diff-gaussian-rasterization/cuda_rasterizer/forward.cu),
#     and num_depth_slabs=1 so MIP compositing is a plain sum instead of a
#     max-over-quantized-slabs that throws away sub-slab z structure. ---
means3D     = gc.means.contiguous()
scales      = torch.exp(gc.log_s).contiguous()
intensities = F.softplus(gc.inten).contiguous()
rotations   = gc.quats.contiguous()

z_offset = 2.5
viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
projmatrix = torch.eye(4, device=device, dtype=torch.float32)
viewmatrix[3, 2] = z_offset
projmatrix[3, 2] = z_offset

result = C.rasterize_mip_gaussians(
    means3D, intensities, scales, rotations,
    1.0,                                            # scale_modifier: true fitted size
    torch.empty(0, device=device, dtype=torch.float32),
    viewmatrix, projmatrix,
    1.0, 1.0,                                        # tan_fovx, tan_fovy
    0.02,                                             # low_pass: sized for sub-pixel splats
    H, W,
    1,                                                # num_depth_slabs=1 -> plain sum, no slab quantization
    float(means3D[:, 2].min().item()) + z_offset,
    float(means3D[:, 2].max().item()) + z_offset,
    False, False,
)
_, raster_mip, _, radii, _, _, _ = result
raster_mip = raster_mip.cpu().numpy()
print(f"raster_mip (fixed): range [{raster_mip.min():.3f}, {raster_mip.max():.3f}]  "
      f"visible {(radii > 0).sum().item()}/{means3D.shape[0]}")

# --- ground-truth MIP over the same block (z-axis max projection) ---
gt_path = '/root/project/data/smoke_data/blocks/block_z0_y1_x6.h5'
with h5py.File(gt_path, 'r') as f:
    gt_vol = f['raw'][:].astype(np.float32)
gt_vol = (gt_vol - gt_vol.min()) / (gt_vol.max() - gt_vol.min())
gt_mip = gt_vol.max(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt_mip,      cmap='gray'); axes[0].set_title("GT MIP (z-axis)");        axes[0].axis('off')
axes[1].imshow(splat_mip,   cmap='gray'); axes[1].set_title("splat_mip (analytic)");   axes[1].axis('off')
axes[2].imshow(raster_mip,  cmap='gray'); axes[2].set_title("rasterizer (fixed)");     axes[2].axis('off')
plt.tight_layout()
out_png = '/root/project/scripts/eval_scripts/render_vs_gt.png'
fig.savefig(out_png, dpi=150)
print(f"Saved {out_png}")
