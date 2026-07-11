import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import diff_gaussian_rasterization._C as C


device = "cuda"

ckpt_path = "models_smoke/block_z000_y001_x006/best.pth"   # change this
ckpt = torch.load(ckpt_path, map_location=device)

# Adjust these keys if your checkpoint uses different names
means3D = ckpt["means"].to(device).float().contiguous()          # [P, 3]
log_scales = ckpt["log_scales"].to(device).float().contiguous()  # [P, 3]
rotations = ckpt["quats"].to(device).float().contiguous()        # [P, 4]
raw_intensities = ckpt["intensities"].to(device).float()

# Convert parameters
scales = torch.exp(log_scales).contiguous()
intensities = F.softplus(raw_intensities).reshape(-1).contiguous()

# Very important: GraphDECO-style code assumes quaternion layout is wxyz.
# If your checkpoint stores xyzw, uncomment this:
# rotations = rotations[:, [3, 0, 1, 2]].contiguous()

P = means3D.shape[0]
print("P:", P)
print("means range:", means3D.min(dim=0).values, means3D.max(dim=0).values)
print("scales range:", scales.min().item(), scales.max().item())
print("intensity range:", intensities.min().item(), intensities.max().item())
print("z range:", means3D[:, 2].min().item(), means3D[:, 2].max().item())

H, W = 50, 50

# Identity view/projection first, only if your means are already normalized
# and z is positive.
viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
projmatrix = torch.eye(4, device=device, dtype=torch.float32)

tan_fovx = 1.0
tan_fovy = 1.0

scale_modifier = 0.1
cov3D_precomp = torch.empty(0, device=device, dtype=torch.float32)

num_depth_slabs = 50

# Use real camera-space z range for slab assignment
depth_min = float(means3D[:, 2].min().item())
depth_max = float(means3D[:, 2].max().item())

# Avoid invalid zero-width depth range
if depth_max <= depth_min:
    depth_max = depth_min + 1.0

prefiltered = False
debug = True

result = C.rasterize_mip_gaussians(
    means3D,
    intensities,
    scales,
    rotations,
    scale_modifier,
    cov3D_precomp,
    viewmatrix,
    projmatrix,
    tan_fovx,
    tan_fovy,
    H,
    W,
    num_depth_slabs,
    depth_min,
    depth_max,
    prefiltered,
    debug,
)

rendered, out_mip, out_depth, radii, geomBuffer, binningBuffer, imgBuffer = result

print("rendered:", rendered)
print("radii min/max:", radii.min().item(), radii.max().item())
print("visible gaussians:", (radii > 0).sum().item(), "/", P)
print("mip min/max:", out_mip.min().item(), out_mip.max().item())
print("depth finite:", torch.isfinite(out_depth).sum().item(), "/", out_depth.numel())

mip_np = out_mip.detach().cpu().numpy()
depth_np = out_depth.detach().cpu().numpy()

plt.figure(figsize=(6, 6))
plt.imshow(mip_np, cmap="gray")
plt.title("Actual Gaussian MIP")
plt.colorbar()
plt.tight_layout()
plt.savefig("actual_mip_render.png", dpi=200)
plt.close()

depth_vis = depth_np.copy()
finite = np.isfinite(depth_vis)

if finite.any():
    depth_vis[~finite] = depth_vis[finite].max()
else:
    depth_vis[:] = 0.0
# direct volume rendering for comparison, if you have the original volume block
vol = h5py.File("data/smoke_data/blocks/block_z0_y1_x6.h5", "r")

# vol_block = h5py.File("data/smoke_data/blocks/block_z0_y1_x6.h5", "r")
# print("vol_block keys:", list(vol_block.keys()))
mip_render = np.max(vol['raw'], axis=0)
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(mip_render, cmap="gray")
# plt.title("Actual Volume MIP")
# plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(mip_np, cmap="gray")
plt.title("Actual Gaussian Winning Depth")
plt.tight_layout()
plt.savefig("actual_depth_render.png", dpi=200)
plt.close()

print("saved actual_mip_render.png and actual_depth_render.png")

print("scales min/max:", scales.min().item(), scales.max().item())
print("scales mean:", scales.mean().item())
print("radii min/max after render:", radii.min().item(), radii.max().item())
print("visible:", (radii > 0).sum().item(), "/", P)

visible_radii = radii[radii > 0].float()

print("visible radii count:", visible_radii.numel())
print("radii mean:", visible_radii.mean().item())
print("radii median:", visible_radii.median().item())
print("radii p90:", torch.quantile(visible_radii, 0.90).item())
print("radii p95:", torch.quantile(visible_radii, 0.95).item())
print("radii p99:", torch.quantile(visible_radii, 0.99).item())
print("radii max:", visible_radii.max().item())