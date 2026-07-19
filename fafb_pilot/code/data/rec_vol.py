"""
Reconstruct a volume from a trained GaussianCloud checkpoint and compare it
against its ground-truth EM block.

USAGE
-----
Minimal (just volume + checkpoint; output goes to the default results/data
dir, no single-point sanity check):

    /venv/r3-ml/bin/python3 fafb_pilot/code/data/rec_vol.py \\
        --volume     data/fafb/blocks/image_z32_y31_x31.tif \\
        --checkpoint fafb_pilot/models/blocks_v18/b_211/best.pth

Full (custom output dir, explicit name, single-point sanity check):

    /venv/r3-ml/bin/python3 fafb_pilot/code/data/rec_vol.py \\
        --volume      data/fafb/blocks/image_z32_y31_x32.tif \\
        --checkpoint  fafb_pilot/models/blocks_v2/b_212/best.pth \\
        --output-dir  results/data \\
        --name        b_212 \\
        --query-point 0.3369 0.7638 -0.3207

STEPS
-----
1. Load the ground-truth block            (load_ground_truth)
2. Load the trained GaussianCloud         (load_model)
3. Reconstruct the volume voxel-by-voxel  (reconstruct_volume)
4. (Optional) Spot-check one query point  (check_query_point)
5. Save a mid-slice PNG                   (save_mid_slice)
6. Save the raw volumes (.npy + .tif)     (main, save_volume_tif)
7. Compute + save an MSE/PSNR/SSIM CSV    (save_metrics_csv)
8. Save a 3-plane comparison PDF + PNGs   (save_plane_comparison)

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  vol_rec_mid_slice_<name>.png   quick GT-vs-pred mid-slice check
  rec_<name>.npy                 raw reconstructed (D,H,W) float32 volume
  rec_<name>.tif                 same reconstructed volume, as a 3D tif stack
  gt_<name>.tif                  ground-truth volume, as a 3D tif stack
                                  (float32 [0,1], same scale as rec_<name>.tif --
                                  directly diffable in any tif viewer, e.g. Fiji)
  metrics_<name>.csv             MSE, PSNR, SSIM, Max Error, Output Min/Max
  vol_rec_slices_<name>.pdf      3x3 grid: {Pred,GT,Diff} x {Sagittal,Coronal,Axial}
  pred_<Plane>_<name>.png        high-res per-plane prediction slices
  gt_<Plane>_<name>.png          high-res per-plane ground-truth slices

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path, see scripts/_3dgs/_3dgs.py)
  - --volume and --checkpoint must be the SAME block -- comparing a
    checkpoint against a different block's tif silently gives meaningless
    metrics rather than an error.
"""
import csv
import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# import libs  # sets up sys.path to scripts/ + TORCH_CUDA_ARCH_LIST
from libs import imshow_gray, save_png, save_volume_tif, compute_metrics

import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()
from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset
from _3dgs._3dgs_training import _load_volume


# ── Step 1: ground truth ─────────────────────────────────────────────────────

def load_ground_truth(volume_path):
    """Load a block tif, normalised to [0,1] the same way training does."""
    vol_t, _, _ = _load_volume(volume_path)
    return vol_t


# ── Step 2: model ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, aabb, device, cfg):
    """Load a trained GaussianCloud checkpoint (.pth or .npz)."""
    gc = GaussianCloud.load(checkpoint_path, aabb, device, cfg)
    print(f"Loaded {gc.N} Gaussians from {checkpoint_path}")
    return gc


# ── Step 3: reconstruction ──────────────────────────────────────────────────

def reconstruct_volume(gc, dataset, device, chunk_n):
    """Evaluate the field at every voxel centre, one Z-slice at a time."""
    D, H, W = dataset.D, dataset.H, dataset.W
    pred_vol = np.empty((D, H, W), dtype=np.float32)
    with torch.no_grad():
        for z in range(D):
            pts = dataset._indices_to_pts(
                torch.full((H * W,), z, dtype=torch.long),
                torch.arange(H, dtype=torch.long).repeat_interleave(W),
                torch.arange(W, dtype=torch.long).tile(H),
                device,
            )
            pred = gc.forward(pts, chunk_n=chunk_n).clamp(0.0, 1.0)
            pred_vol[z] = pred.cpu().numpy().reshape(H, W)
    print(f"Reconstructed volume shape: {pred_vol.shape}  "
          f"range: [{pred_vol.min():.3f}, {pred_vol.max():.3f}]")
    return pred_vol


# ── Step 4: optional single-point sanity check ──────────────────────────────

def check_query_point(gc, vol_t, query_point, device, chunk_n):
    """Compare predicted vs. trilinear-GT intensity at one [-1,1]^3 point."""
    query_pt = torch.tensor([query_point], device=device)
    with torch.no_grad():
        query_val = gc.forward(query_pt, chunk_n=chunk_n).clamp(0.0, 1.0)

        # Same convention as loss_sparsity_intensity: grid_sample expects
        # (1,1,1,N,3), align_corners=True.
        grid = query_pt.view(1, 1, 1, 1, 3)
        vol_5d = vol_t.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
        gt_val = F.grid_sample(vol_5d, grid, mode='bilinear', align_corners=True).view(-1)

    print(f"Predicted intensity at {query_point}: {query_val.item():.4f}")
    print(f"Original (GT) intensity at {query_point}: {gt_val.item():.4f}")


# ── Step 5: mid-slice PNG ────────────────────────────────────────────────────

def save_mid_slice(pred_vol, gt_vol, output_dir, name):
    """Quick 2-panel GT-vs-pred check at the middle Z-slice."""
    mid = pred_vol.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    imshow_gray(axes[0], gt_vol[mid], f"GT   Z={mid}")
    imshow_gray(axes[1], pred_vol[mid], f"Pred Z={mid}")
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"vol_rec_mid_slice_{name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


# ── Step 7: metrics CSV ──────────────────────────────────────────────────────

def save_metrics_csv(pred_vol, gt_vol, output_dir, name):
    metrics = compute_metrics(pred_vol, gt_vol)
    out_csv = os.path.join(output_dir, f"metrics_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved {out_csv}")


# ── Step 8: 3-plane comparison ───────────────────────────────────────────────

def save_plane_comparison(pred_vol, gt_vol, output_dir, name):
    """3x3 {Pred,GT,Diff} x {Sagittal,Coronal,Axial} PDF, plus per-slice PNGs."""
    D, H, W = gt_vol.shape
    planes = {
        "Sagittal": (pred_vol[:, :, W // 2], gt_vol[:, :, W // 2]),
        "Coronal":  (pred_vol[:, H // 2, :], gt_vol[:, H // 2, :]),
        "Axial":    (pred_vol[D // 2, :, :], gt_vol[D // 2, :, :]),
    }

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for row, (title, (pred_slice, gt_slice)) in enumerate(planes.items()):
        diff = np.abs(pred_slice - gt_slice)
        imshow_gray(axs[row, 0], pred_slice, f"Pred {title}")
        imshow_gray(axs[row, 1], gt_slice, f"GT {title}")
        im = axs[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 1e-8))
        axs[row, 2].set_title(f"Diff {title}")
        axs[row, 2].axis('off')
        fig.colorbar(im, ax=axs[row, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_pdf = os.path.join(output_dir, f"vol_rec_slices_{name}.pdf")
    fig.savefig(out_pdf, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")

    for title, (pred_slice, gt_slice) in planes.items():
        save_png(pred_slice, os.path.join(output_dir, f"pred_{title}_{name}.png"))
        save_png(gt_slice, os.path.join(output_dir, f"gt_{title}_{name}.png"))


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct a volume from a trained GaussianCloud "
                    "checkpoint and compare it against its ground-truth tif block.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--volume", required=True,
                        help="Ground-truth block tif, e.g. data/fafb/blocks/image_z32_y31_x32.tif")
    parser.add_argument("--checkpoint", required=True,
                        help="Trained GaussianCloud checkpoint (.pth or .npz), "
                             "e.g. models/blocks_v2/b_212/best.pth")
    parser.add_argument("--output-dir",
                        default="/root/project/results/data",
                        help="Directory to write figures/metrics/npy into "
                             "(project convention: generated outputs live under results/).")
    parser.add_argument("--name", default=None,
                        help="Label used in output filenames. Defaults to the "
                             "checkpoint's parent directory name (matches this "
                             "project's blocks_v*/b_XXX/best.pth convention).")
    parser.add_argument("--query-point", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Optional point in [-1,1]^3 to sanity-check predicted "
                             "vs. ground-truth intensity at. Omit to skip.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-n", type=int, default=1000,
                        help="Points per forward() batch during reconstruction.")
    parser.add_argument("--scale-min-clamp", type=float, default=1e-5)
    parser.add_argument("--mahal-max-clamp", type=float, default=20.0)
    parser.add_argument("--ssim-crop", type=int, default=None,
                        help="Unused by this script's own logic (no loss is "
                             "computed here); defaults to min(64, block depth). "
                             "Kept only for VolumeDataset/cfg compatibility.")
    # The remaining flags are likewise unused by GaussianCloud.load() /
    # this script's logic -- kept only because VolumeDataset's constructor
    # expects a fully-populated cfg Namespace.
    parser.add_argument("--n-init", type=int, default=5000)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--init-inten", type=float, default=0.1)
    parser.add_argument("--init-scale-z-factor", type=float, default=1.0)
    parser.add_argument("--swc-path", default=None)
    parser.add_argument("--eval-samples", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--grad-sample-weight", type=float, default=0.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or os.path.basename(os.path.dirname(args.checkpoint))

    # Step 1: ground truth.
    vol_t = load_ground_truth(args.volume)
    aabb = AABB.unit()
    cfg = argparse.Namespace(
        scale_min_clamp=args.scale_min_clamp,
        mahal_max_clamp=args.mahal_max_clamp,
        init_scale=args.init_scale,
        init_inten=args.init_inten,
        init_scale_z_factor=args.init_scale_z_factor,
        n_init=args.n_init,
        swc_path=args.swc_path,
        chunk_n=args.chunk_n,
        eval_samples=args.eval_samples,
        ssim_crop=args.ssim_crop if args.ssim_crop is not None else min(64, vol_t.shape[0]),
        batch=args.batch,
        grad_sample_weight=args.grad_sample_weight,
        lambda_ssim=args.lambda_ssim,
    )
    dataset = VolumeDataset(vol_t, aabb, cfg)

    # Step 2: model.
    gc = load_model(args.checkpoint, aabb, device, cfg)

    # Step 3: reconstruction.
    pred_vol = reconstruct_volume(gc, dataset, device, cfg.chunk_n)
    gt_vol = vol_t.detach().cpu().numpy()

    # Step 4: optional spot-check.
    if args.query_point is not None:
        check_query_point(gc, vol_t, args.query_point, device, cfg.chunk_n)

    # Step 5: mid-slice PNG.
    save_mid_slice(pred_vol, gt_vol, args.output_dir, name)

    # Step 6: raw volumes -- .npy (exact, for downstream Python reuse) and
    # .tif (for viewers like Fiji/ImageJ). Both pred and gt are saved as tif,
    # at the same float32 [0,1] scale, so they're directly diffable.
    out_npy = os.path.join(args.output_dir, f"rec_{name}.npy")
    np.save(out_npy, pred_vol)
    print(f"Saved {out_npy}")
    save_volume_tif(pred_vol, args.output_dir, f"rec_{name}.tif")
    save_volume_tif(gt_vol, args.output_dir, f"gt_{name}.tif")

    # Step 7: metrics CSV.
    save_metrics_csv(pred_vol, gt_vol, args.output_dir, name)

    # Step 8: 3-plane comparison PDF + per-slice PNGs.
    save_plane_comparison(pred_vol, gt_vol, args.output_dir, name)


if __name__ == "__main__":
    main()
