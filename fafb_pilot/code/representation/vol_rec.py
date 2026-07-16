# reconstruct the volume from the Gaussian cloud
import csv
import os
import sys, argparse, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.insert(0, '/root/project/scripts')

import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()
from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset
from _3dgs._3dgs_training import _load_volume

device = torch.device('cuda')
os.makedirs("/root/project/fafb_pilot/code/representation/figures", exist_ok=True)

# smoke_data blocks are 50^3 (see configs/smoke_config.yml) -> ssim_crop must match
cfg = argparse.Namespace(
    scale_min_clamp=1e-5, mahal_max_clamp=20.0, init_scale=0.05,
    init_inten=0.1, init_scale_z_factor=1.0, n_init=5000,
    swc_path=None, chunk_n=1000, eval_samples=200_000,
    ssim_crop=50, batch=2048,
    grad_sample_weight=0.0, lambda_ssim=0.2,
)

# --- load GT volume (ground truth) ---
# _load_volume normalises to [0,1] the same way every block was normalised at
# training time -- a raw tifffile.imread() is uint8 [0,255] and would make
# every metric below (MSE/PSNR/SSIM) meaningless against pred_vol's [0,1] range.
vol_t, _, _ = _load_volume('/root/project/data/fafb/blocks/image_z32_y31_x32.tif')
aabb    = AABB.unit()
dataset = VolumeDataset(vol_t, aabb, cfg)
D, H, W = dataset.D, dataset.H, dataset.W

# --- load model directly via GaussianCloud.load (handles the .pth checkpoint format) ---
ckpt_path = '/root/project/fafb_pilot/models/blocks_v2/b_212/best.pth'
gc = GaussianCloud.load(ckpt_path, aabb, device, cfg)
print(f"Loaded {gc.N} Gaussians from {ckpt_path}")

# --- reconstruct volume slice-by-slice ---
pred_vol = np.empty((D, H, W), dtype=np.float32)
with torch.no_grad():
    for z in range(D):
        pts = dataset._indices_to_pts(
            torch.full((H * W,), z, dtype=torch.long),
            torch.arange(H, dtype=torch.long).repeat_interleave(W),
            torch.arange(W, dtype=torch.long).tile(H),
            device,
        )
        pred = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
        pred_vol[z] = pred.cpu().numpy().reshape(H, W)
#check min max of pred_vol
print(f"Reconstructed volume shape: {pred_vol.shape}  range: [{pred_vol.min():.3f}, {pred_vol.max():.3f}]")
query_pt = torch.tensor([[0.3369, 0.7638, -0.3207]], device=device)
with torch.no_grad():
    query_val = gc.forward(query_pt, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)

    # trilinear GT lookup at the same continuous point (same convention as
    # loss_sparsity_intensity: grid_sample expects (1,1,1,N,3), align_corners=True)
    grid  = query_pt.view(1, 1, 1, 1, 3)
    vol_5d = vol_t.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    gt_val = F.grid_sample(vol_5d, grid, mode='bilinear', align_corners=True).view(-1)

print(f"Predicted intensity at {query_pt.tolist()[0]}: {query_val.item():.4f}")
print(f"Original (GT) intensity at {query_pt.tolist()[0]}: {gt_val.item():.4f}")

# --- visualise middle slice ---
mid = D // 2
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(vol_t[mid].numpy(), cmap='gray', vmin=0, vmax=1); axes[0].set_title(f"GT   Z={mid}"); axes[0].axis('off')
axes[1].imshow(pred_vol[mid],      cmap='gray', vmin=0, vmax=1); axes[1].set_title(f"Pred Z={mid}"); axes[1].axis('off')
plt.tight_layout()
out_png = f"/root/project/fafb_pilot/code/representation/figures/vol_rec_mid_slice_{ckpt_path.split('/')[-2].split('.')[0]}.png"
fig.savefig(out_png, dpi=150)
print(f"Saved {out_png}")

np.save(f"/root/project/fafb_pilot/code/representation/figures/rec_{ckpt_path.split('/')[-2].split('.')[0]}.npy", pred_vol)

# define ssim 
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    """Compute the Structural Similarity Index (SSIM) between two images."""
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim_index = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_index
# save the metrics in csv file by comparing volumes MSE PSNR SSIM Max Error Output Min Output Max
with open(f"/root/project/fafb_pilot/code/representation/figures/metrics_{ckpt_path.split('/')[-2].split('.')[0]}.csv", 'w', newline='') as csvfile:
    fieldnames = ['MSE', 'PSNR', 'SSIM', 'Max Error', 'Output Min', 'Output Max']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    mse = np.mean((pred_vol - vol_t.numpy()) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    SSIM = ssim(pred_vol, vol_t.numpy())
    max_error = np.max(np.abs(pred_vol - vol_t.numpy()))
    output_min = np.min(pred_vol)
    output_max = np.max(pred_vol)

    writer.writerow({
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': SSIM,
        'Max Error': max_error,
        'Output Min': output_min,
        'Output Max': output_max
    })

    # save pdf mid slices saggital coronal axial and compare them with gt and also add difference images
gt_vol = vol_t.detach().cpu().numpy()

pred_slices = [
    pred_vol[:, :, W // 2],   # Sagittal
    pred_vol[:, H // 2, :],   # Coronal
    pred_vol[D // 2, :, :],   # Axial
]

gt_slices = [
    gt_vol[:, :, W // 2],     # Sagittal
    gt_vol[:, H // 2, :],     # Coronal
    gt_vol[D // 2, :, :],     # Axial
]

titles = ["Sagittal", "Coronal", "Axial"]

fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for i, title in enumerate(titles):
    pred_slice = pred_slices[i]
    gt_slice = gt_slices[i]
    diff = np.abs(pred_slice - gt_slice)

    # Prediction
    axs[i, 0].imshow(
        pred_slice,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axs[i, 0].set_title(f"Pred {title}")

    # Ground truth
    axs[i, 1].imshow(
        gt_slice,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axs[i, 1].set_title(f"GT {title}")

    # Absolute difference
    im = axs[i, 2].imshow(
        diff,
        cmap="hot",
        vmin=0,
        vmax=max(float(diff.max()), 1e-8),
    )
    axs[i, 2].set_title(f"Diff {title}")

    fig.colorbar(
        im,
        ax=axs[i, 2],
        fraction=0.046,
        pad=0.04,
    )

plt.tight_layout()

block_name = ckpt_path.split("/")[-2].split(".")[0]

out_pdf = (
    f"/root/project/fafb_pilot/code/representation/figures/"
    f"vol_rec_slices_{block_name}.pdf"
)

fig.savefig(
    out_pdf,
    dpi=800,
    bbox_inches="tight",
)


plt.close(fig)

print(f"Saved {out_pdf}")

# save high quality pred_slice to fafb_pilot/code/representation/figures dpi=800 using PIL.Image.save
for i, pred_slice in enumerate(pred_slices):
    from PIL import Image

    img = Image.fromarray((pred_slice * 255).astype(np.uint8))
    img.save(
        f"/root/project/fafb_pilot/code/representation/figures/pred_{titles[i]}_{block_name}.png",
        dpi=(800, 800),
    )

    gt_img = Image.fromarray((gt_slices[i] * 255).astype(np.uint8))
    gt_img.save(
        f"/root/project/fafb_pilot/code/representation/figures/gt_{titles[i]}_{block_name}.png",
        dpi=(800, 800),
    )