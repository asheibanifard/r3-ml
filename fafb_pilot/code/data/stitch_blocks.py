"""
Stitch n_per_axis^3 independently-trained per-block GaussianCloud checkpoints
into one combined volume, using HARD POSITION GATING -- each combined voxel
is reconstructed using ONLY its owning block's Gaussians, never a sum/blend
across a block boundary.

BACKGROUND
----------
The project's eda.ipynb (Section 4, "Naive Sum vs. Feathered Blend vs. Hard
Partition") and fafb_pilot/code/renderer/test.ipynb ("8-block stitch with
HARD position gating") both found that naively SUMMING two neighbouring
blocks' Gaussian fields at their shared boundary introduces a real seam
artefact. Hard gating -- evaluating each spatial region using only the one
block that owns it -- was the fix that held up through 2-, 4-, and 8-block
junctions in eda.ipynb, and through a full 128^3 (2x2x2 = 8 block) stitch in
test.ipynb. This script generalises that fix from a fixed 2x2x2 case to ANY
n_per_axis x n_per_axis x n_per_axis grid of blocks.

OWNERSHIP CONVENTION (generalises eda.ipynb's [-1,0]/[0,1])
-------------------------------------------------------------
eda.ipynb's convention for n_per_axis=2: the "low" block along an axis owns
combined coordinate range [-1, 0], the "high" block owns [0, 1]. For general
n_per_axis, block index i (0 .. n_per_axis-1) along an axis owns:

    [-1 + i*(2/n_per_axis), -1 + (i+1)*(2/n_per_axis)]

See combined_range() below -- combined_range(0, 2) == (-1.0, 0.0) and
combined_range(1, 2) == (0.0, 1.0) recover eda.ipynb's convention exactly.
Hard gating means: a combined voxel is reconstructed using ONLY the
Gaussians of the one block whose owned range contains it on every axis --
never a sum across two blocks' ranges.

WHY THIS IS SIMPLE TO IMPLEMENT (not just simple to state)
--------------------------------------------------------------
Because gating is HARD (never a blend), reconstructing the combined volume
this way is mathematically identical to: reconstruct each block
independently in its own untouched [-1,1]^3 frame (exactly like
rec_vol.py's reconstruct_volume(), one block at a time), then place each
result into its own disjoint (block_voxels)^3 slice of the combined array.
No Gaussian ever needs to be remapped into a shared coordinate frame, and no
per-point ownership test is needed at reconstruction time -- gating is
satisfied "for free" by construction, since two different blocks' voxels
never share an array slice. (Remapping every block's Gaussians into one
shared frame only matters if you additionally want to render the stitched
scene as a single flat Gaussian list, e.g. through a production renderer --
see fafb_pilot/code/renderer/test.ipynb for that separate use case.)

USAGE
-----
Minimal (2x2x2 = 8 blocks, matching test.ipynb's octant):

    /venv/r3-ml/bin/python3 fafb_pilot/code/data/stitch_blocks.py \
        --blocks-dir fafb_pilot/models/blocks_v18 \
        --n-per-axis 2

Full (the whole 4x4x4 = 64-block pilot grid, custom output naming):

    /venv/r3-ml/bin/python3 fafb_pilot/code/data/stitch_blocks.py \
        --blocks-dir  fafb_pilot/models/blocks_v18 \
        --n-per-axis  4 \
        --z0 30 --y0 30 --x0 30 \
        --output-dir  results/data \
        --name        blocks_v18_full

STEPS
-----
1. For every block index (iz,iy,ix) in [0,n)^3: load its checkpoint and its
   ground-truth tif, in that block's own native frame (load_block).
2. Reconstruct that block's own volume (reconstruct_block -- identical
   approach to rec_vol.py's reconstruct_volume).
3. Place both the block's GT and its reconstruction into their disjoint
   (block_voxels)^3 slice of the combined (n*block_voxels)^3 array (in
   stitch_blocks) -- this IS hard gating, by construction.
4. Compute MSE/PSNR/SSIM comparing the combined reconstruction to combined
   GT (libs.compute_metrics).
5. Save the combined GT + reconstruction as .tif stacks, a mid-slice
   comparison figure, and a metrics CSV.
6. ALSO remap each block's own Gaussians into the shared global frame
   (remap_gaussians_to_global -- the exact same formula as
   fafb_pilot/code/representation/stitch_block_gaussians.py and
   test.ipynb's octant remap) and concatenate them into one combined
   Gaussian list, saved as gaussian_<name>.json. This is the OTHER use case
   named in "WHY THIS IS SIMPLE" above: a single flat Gaussian list, for
   feeding a production renderer (e.g. Mip_Render_Inside_Volume.cu's
   pretrained_gaussian mode) or any other downstream tool that wants plain
   JSON rather than a torch .pth checkpoint. It is NOT needed for the
   hard-gated voxel reconstruction itself (steps 1-5 above never remap).

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  rec_<name>.tif                   combined reconstruction, float32 [0,1]
  gt_<name>.tif                    combined ground truth, float32 [0,1]
  vol_stitch_mid_slice_<name>.png  GT / stitched-pred / |diff| mid-slice check
  metrics_<name>.csv               MSE, PSNR, SSIM, Max Error, Output Min/Max
  gaussian_<name>.json             combined Gaussians (all blocks, remapped
                                    into the shared global frame): means,
                                    log_scales, quats, intensities (raw,
                                    pre-softplus -- see "inten_param")

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path, see scripts/_3dgs/_3dgs.py)
  - --blocks-dir must contain b_<iz><iy><ix>/<checkpoint-name> for every
    (iz,iy,ix) in [0,n_per_axis)^3 -- retrain_pilot_blocks_v2.sh/v18.sh's
    naming convention. Each axis index is a single digit, so n_per_axis is
    limited to 10 (this project's pilot grid only ever used 4).
  - --z0/--y0/--x0 must match the offsets used when those blocks were
    trained (see retrain_pilot_blocks_v18.sh), so each block's GT tif is the
    one it was actually trained on.
"""
import csv
import json
import math
import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fafb_pilot/code
import libs  # sets up sys.path to scripts/ + TORCH_CUDA_ARCH_LIST
from libs import imshow_gray, save_volume_tif, compute_metrics

import _3dgs._3dgs as _mod
_mod.USE_CUDA_KERNEL = True
_mod._load_3dgs_kernel()
from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset
from _3dgs._3dgs_training import _load_volume


# ── Ownership convention (generalises eda.ipynb's [-1,0]/[0,1]) ─────────────

def combined_range(idx, n_per_axis):
    """The [lo, hi] combined-frame interval block index `idx` owns along one
    axis. n_per_axis=2 recovers eda.ipynb's exact convention:
        combined_range(0, 2) == (-1.0, 0.0)   # eda.ipynb's "low" side
        combined_range(1, 2) == (0.0, 1.0)    # eda.ipynb's "high" side
    """
    step = 2.0 / n_per_axis
    return (-1.0 + idx * step, -1.0 + (idx + 1) * step)


def block_name(iz, iy, ix):
    """b_<iz><iy><ix>, matching retrain_pilot_blocks_v2.sh/v18.sh's naming.
    Each index must be a single digit (0-9)."""
    if max(iz, iy, ix) > 9:
        raise ValueError(
            "block indices must be single digits (n_per_axis <= 10) to match "
            "this project's b_<iz><iy><ix> checkpoint-directory convention."
        )
    return f"b_{iz}{iy}{ix}"


# ── Step 1: load one block, native frame (no remap needed -- see docstring) ─

def load_block(blocks_dir, data_dir, z0, y0, x0, iz, iy, ix, checkpoint_name, aabb, device, cfg):
    """Load one block's GaussianCloud + its own ground-truth tif, both in
    that block's own native [-1,1]^3 frame -- untouched, exactly as trained."""
    bname = block_name(iz, iy, ix)
    ckpt_path = os.path.join(blocks_dir, bname, checkpoint_name)
    gc = GaussianCloud.load(ckpt_path, aabb, device, cfg)

    vol_path = os.path.join(data_dir, f"image_z{z0 + iz}_y{y0 + iy}_x{x0 + ix}.tif")
    vol_t, _, _ = _load_volume(vol_path)

    return bname, gc, vol_t


# ── Step 2: reconstruct one block's own volume ──────────────────────────────

def reconstruct_block(gc, dataset, device, chunk_n):
    """Evaluate one block's own field at every voxel centre -- same approach
    as rec_vol.py's reconstruct_volume()."""
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
    return pred_vol


# ── Step 6: remap one block's own Gaussians into the shared global frame ───
# (only needed for the gaussian_<name>.json output -- NOT for the hard-gated
# voxel reconstruction in steps 1-5, which never remaps anything; see
# fafb_pilot/code/representation/stitch_block_gaussians.py, whose exact
# convention this mirrors.)

def remap_gaussians_to_global(gc, iz, iy, ix, n_per_axis):
    """Remap one block's own [-1,1]^3-local means/log_scales into the shared
    global frame it owns (combined_range(iz/iy/ix, n_per_axis) on each axis).
    Quaternions and intensities are unaffected by a uniform (isotropic)
    rescale, so they pass through unchanged."""
    extent = 1.0 / n_per_axis
    log_extent = math.log(extent)

    # means stored (x, y, z) -- matches VolumeDataset._indices_to_pts and
    # stitch_block_gaussians.py's remap_block().
    center = torch.tensor(
        [
            -1.0 + (ix + 0.5) * (2.0 / n_per_axis),
            -1.0 + (iy + 0.5) * (2.0 / n_per_axis),
            -1.0 + (iz + 0.5) * (2.0 / n_per_axis),
        ],
        dtype=gc.means.dtype, device=gc.means.device,
    )

    means_global = center.unsqueeze(0) + gc.means.detach() * extent
    log_scales_global = gc.log_s.detach() + log_extent

    return means_global.cpu(), log_scales_global.cpu(), gc.quats.detach().cpu(), gc.inten.detach().cpu()


# ── Step 3: stitch every block into the combined, hard-gated volume ────────

def stitch_blocks(blocks_dir, data_dir, n_per_axis, z0, y0, x0, checkpoint_name, device, cfg):
    """Reconstruct every block and place it into its own disjoint slice of
    the combined array -- hard gating "for free" by construction (see
    module docstring)."""
    aabb = AABB.unit()

    print(f"Stitching {n_per_axis}^3 = {n_per_axis ** 3} blocks, hard-gated "
          f"(no cross-block blending).")
    print("Per-axis ownership (generalises eda.ipynb's [-1,0]/[0,1]):")
    for idx in range(n_per_axis):
        lo, hi = combined_range(idx, n_per_axis)
        print(f"  block index {idx}: owns combined range [{lo:.3f}, {hi:.3f}]")

    gt_cube, rec_cube, block_voxels = None, None, None
    all_means, all_log_scales, all_quats, all_inten = [], [], [], []

    for iz in range(n_per_axis):
        for iy in range(n_per_axis):
            for ix in range(n_per_axis):
                bname, gc, vol_t = load_block(
                    blocks_dir, data_dir, z0, y0, x0, iz, iy, ix,
                    checkpoint_name, aabb, device, cfg,
                )
                dataset = VolumeDataset(vol_t, aabb, cfg)
                pred_vol = reconstruct_block(gc, dataset, device, cfg.chunk_n)

                if gt_cube is None:
                    block_voxels = vol_t.shape[0]
                    cube_voxels = n_per_axis * block_voxels
                    gt_cube = np.zeros((cube_voxels,) * 3, dtype=np.float32)
                    rec_cube = np.zeros((cube_voxels,) * 3, dtype=np.float32)

                sz = slice(iz * block_voxels, (iz + 1) * block_voxels)
                sy = slice(iy * block_voxels, (iy + 1) * block_voxels)
                sx = slice(ix * block_voxels, (ix + 1) * block_voxels)
                gt_vol_np = vol_t.detach().cpu().numpy()
                gt_cube[sz, sy, sx] = gt_vol_np
                rec_cube[sz, sy, sx] = pred_vol

                own_psnr = compute_metrics(pred_vol, gt_vol_np)['PSNR']
                print(f"  [{bname}] {gc.N} Gaussians, own-block vol_PSNR = {own_psnr:.2f} dB")

                # Step 6: remap this block's Gaussians into the global frame
                # for the gaussian_<name>.json output (see module docstring).
                means_g, log_scales_g, quats_g, inten_g = remap_gaussians_to_global(
                    gc, iz, iy, ix, n_per_axis
                )
                all_means.append(means_g)
                all_log_scales.append(log_scales_g)
                all_quats.append(quats_g)
                all_inten.append(inten_g)

    combined_gaussians = {
        "means": torch.cat(all_means, dim=0),
        "log_scales": torch.cat(all_log_scales, dim=0),
        "quats": torch.cat(all_quats, dim=0),
        "intensities": torch.cat(all_inten, dim=0),
        "inten_param": "softplus",
    }

    return gt_cube, rec_cube, combined_gaussians


# ── Step 5: save outputs ─────────────────────────────────────────────────────

def save_mid_slice(pred_vol, gt_vol, output_dir, name):
    """GT / stitched-pred / |diff| check at the combined cube's middle Z-slice."""
    mid = pred_vol.shape[0] // 2
    diff = np.abs(pred_vol[mid] - gt_vol[mid])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    imshow_gray(axes[0], gt_vol[mid], f"GT   Z={mid}")
    imshow_gray(axes[1], pred_vol[mid], f"Stitched Pred Z={mid}")
    im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 1e-8))
    axes[2].set_title(f"|Diff| (mean={diff.mean():.4f})")
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"vol_stitch_mid_slice_{name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_gaussians_json(combined_gaussians, output_dir, name):
    """Combined, globally-remapped Gaussians as plain JSON -- means/log_scales
    are (N,3), quats are (N,4) [w,x,y,z], intensities are (N,) raw values
    (apply softplus per 'inten_param' to get the density weight v_k, matching
    GaussianCloud.save()'s own on-disk convention)."""
    out = {
        "n_gaussians": combined_gaussians["means"].shape[0],
        "means": combined_gaussians["means"].tolist(),
        "log_scales": combined_gaussians["log_scales"].tolist(),
        "quats": combined_gaussians["quats"].tolist(),
        "intensities": combined_gaussians["intensities"].tolist(),
        "inten_param": combined_gaussians["inten_param"],
    }
    out_json = os.path.join(output_dir, f"gaussian_{name}.json")
    with open(out_json, 'w') as f:
        json.dump(out, f)
    print(f"Saved {out_json} ({out['n_gaussians']} Gaussians)")


def save_metrics_csv(pred_vol, gt_vol, output_dir, name):
    metrics = compute_metrics(pred_vol, gt_vol)
    out_csv = os.path.join(output_dir, f"metrics_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved {out_csv}")
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch n_per_axis^3 GaussianCloud checkpoints into one "
                    "combined volume using hard position gating.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blocks-dir", required=True,
                        help="Directory containing b_<iz><iy><ix>/<checkpoint-name> "
                             "subdirectories, e.g. fafb_pilot/models/blocks_v18")
    parser.add_argument("--n-per-axis", type=int, default=2,
                        help="Blocks per axis (n_per_axis^3 blocks total). "
                             "2 matches test.ipynb's 8-block octant.")
    parser.add_argument("--data-dir", default="/root/project/data/fafb/blocks",
                        help="Directory containing image_z{Z}_y{Y}_x{X}.tif ground-truth blocks.")
    parser.add_argument("--z0", type=int, default=30)
    parser.add_argument("--y0", type=int, default=30)
    parser.add_argument("--x0", type=int, default=30,
                        help="--z0/--y0/--x0: offsets added to (iz,iy,ix) to form each "
                             "block's tif filename -- must match how the blocks were "
                             "trained (see retrain_pilot_blocks_v18.sh).")
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--output-dir",
                        default="/root/project/results/data",
                        help="Directory to write the stitched tif/figure/metrics into "
                             "(project convention: generated outputs live under results/).")
    parser.add_argument("--name", default=None,
                        help="Label used in output filenames. Defaults to "
                             "<blocks-dir basename>_n<n-per-axis>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-n", type=int, default=1000,
                        help="Points per forward() batch during reconstruction.")
    parser.add_argument("--scale-min-clamp", type=float, default=1e-5)
    parser.add_argument("--mahal-max-clamp", type=float, default=20.0)
    # The remaining flags are unused by GaussianCloud.load()/this script's own
    # logic -- kept only because VolumeDataset's constructor expects a
    # fully-populated cfg Namespace (same convention as rec_vol.py).
    parser.add_argument("--ssim-crop", type=int, default=64)
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
    name = args.name or f"{os.path.basename(os.path.normpath(args.blocks_dir))}_n{args.n_per_axis}"

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
        ssim_crop=args.ssim_crop,
        batch=args.batch,
        grad_sample_weight=args.grad_sample_weight,
        lambda_ssim=args.lambda_ssim,
    )

    # Step 1-3 (+6): stitch (load + reconstruct + place, per block; also
    # remaps each block's Gaussians into the shared global frame).
    gt_cube, rec_cube, combined_gaussians = stitch_blocks(
        args.blocks_dir, args.data_dir, args.n_per_axis,
        args.z0, args.y0, args.x0, args.checkpoint_name, device, cfg,
    )
    print(f"Combined cube: {rec_cube.shape}")

    # Step 5: save outputs (raw volumes, mid-slice figure, metrics -- Step 4's
    # metrics are computed as part of both save_metrics_csv and the printed
    # summary below).
    save_volume_tif(rec_cube, args.output_dir, f"rec_{name}.tif")
    save_volume_tif(gt_cube, args.output_dir, f"gt_{name}.tif")
    save_mid_slice(rec_cube, gt_cube, args.output_dir, name)
    metrics = save_metrics_csv(rec_cube, gt_cube, args.output_dir, name)
    save_gaussians_json(combined_gaussians, args.output_dir, name)

    print(f"\nCombined stitched-volume metrics: "
          f"vol_PSNR={metrics['PSNR']:.2f} dB, SSIM={metrics['SSIM']:.4f}")


if __name__ == "__main__":
    main()
