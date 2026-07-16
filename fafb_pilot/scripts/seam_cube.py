#!/usr/bin/env python3
"""Extract a 64^3 "seam cube" straddling the boundary between two X-adjacent
trained blocks, for both the dense ground-truth volume and the stitched
Gaussian reconstruction -- the core diagnostic for Section 4.2 (Cross-Block
Stitching Artefact Characterisation): does the naive additive stitching of
Section 3.5 introduce a visible/measurable artefact right at a block seam?

Block layout matches the fafb_pilot grid: block (iz, iy, ix) (each axis in
[0, n_per_axis)) maps to raw tif file
    {blocks_dir}/image_z{base_z+iz}_y{base_y+iy}_x{base_x+ix}.tif
and trained checkpoint
    {models_dir}/b_{iz}{iy}{ix}/{checkpoint_name}
-- the same b_<iz><iy><ix> naming used by stitch_block_gaussians.py.

Given block A = (iz, iy, ix) and its +X neighbour B = (iz, iy, ix+1):
  1. Both raw tif volumes are loaded (with the same per-block [0,1]
     normalisation used at training time, see _3dgs_training._load_volume)
     and concatenated along X into one (64, 64, 128) combined GT volume.
  2. Both trained GaussianClouds are remapped into a shared frame that is 2
     blocks wide along X (and a single block wide along Y/Z, since A and B
     share the same Y/Z tile) -- the same axis_transform algebra as
     stitch_block_gaussians.py / scripts/render_scripts/sliding_window_camera.py,
     specialised to n=2 on X only.
  3. A cube_size^3 window is placed at --x-center (a voxel index into the
     128-wide combined X range; default 64 = the seam itself). With the
     default cube_size=64 this puts exactly the right half of block A and
     the left half of block B inside the cube. Both the GT sub-volume and
     the Gaussian reconstruction (filter-by-mean + dense re-evaluation on a
     fresh local grid, matching sliding_window_camera.py's
     filter_gaussians_in_block / reconstruct_local_cube) are extracted from
     this same window.
  4. vol_PSNR between the two is reported (same -10*log10(mse) definition
     used throughout the project, see scripts/_3dgs/_3dgs.py:vol_psnr), and
     GT / Prediction / |Diff| middle-slice visualisations (with the seam
     location marked) plus the raw cubes are saved.

Usage:
  /venv/r3-ml/bin/python3 fafb_pilot/scripts/seam_cube.py \
    --block-a 2 1 1 \
    --models-dir fafb_pilot/models/blocks_v2 \
    --out-dir   fafb_pilot/results/seam_cubes/b211_b212

Slide the cube away from the seam (e.g. to sanity-check a same-block
interior region has no artefact) with --x-center:
  --x-center 32   # fully inside block A, no seam
  --x-center 96   # fully inside block B, no seam
  --x-center 64   # default -- straddles the seam
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts._3dgs._3dgs import AABB, GaussianCloud  # noqa: E402


def _load_block_volume(path: Path) -> np.ndarray:
    """Same per-block [0,1] normalisation as _3dgs_training._load_volume."""
    raw = tifffile.imread(str(path)).astype(np.float32)
    vmin, vmax = float(raw.min()), float(raw.max())
    if vmax > vmin:
        raw = (raw - vmin) / (vmax - vmin)
    return raw  # (Z, Y, X)


def voxel_to_norm(idx: float, n: int) -> float:
    """index in [0, n-1] -> [-1, 1], same convention used project-wide
    (see scripts/render_scripts/sliding_window_camera.py:voxel_to_global_coord)."""
    return 2.0 * idx / (n - 1) - 1.0


def remap_block_x_pair(ckpt: dict, slot: int) -> dict:
    """Remap one block's checkpoint into the shared 2-block-wide-along-X frame.

    slot=0 -> block A occupies the left half ([-1,-0]) of the combined X range,
    slot=1 -> block B occupies the right half ([0,1]). Y and Z are left
    untouched: A and B share the same Y/Z tile, so no rescale is needed on
    those axes (equivalent to axis_transform with n=1).
    """
    extent_x = 0.5
    center_x = -1.0 + (slot + 0.5) * 1.0  # -0.5 for slot 0, +0.5 for slot 1

    means = ckpt["means"].clone()
    log_scales = ckpt["log_scales"].clone()
    means[:, 0] = center_x + means[:, 0] * extent_x
    log_scales[:, 0] = log_scales[:, 0] + math.log(extent_x)

    return {
        "means": means,
        "log_scales": log_scales,
        "quats": ckpt["quats"].clone(),
        "intensities": ckpt["intensities"].clone(),
    }


def _save_seam_slices(gt_cube: np.ndarray, recon_cube: np.ndarray,
                       out_dir: Path, seam_offset_in_cube: float) -> None:
    """Full dense voxel-grid GT vs. Prediction vs. |Diff| comparison for the
    seam cube, at all three mid-planes -- same 3x3 layout as
    _3dgs_training._visualize_middle_slices, so this is directly comparable
    to the per-block visualisations already used elsewhere (e.g. Figure 1).
    gt_cube / recon_cube are indexed (Z, Y, X); X is the axis the seam runs
    across, so the dashed seam marker is drawn on any panel whose horizontal
    axis is X (XY and XZ), and omitted on YZ (a single fixed-X cross-section).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D, H, W = gt_cube.shape  # (Z, Y, X)
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    gt_xy, pred_xy = gt_cube[mid_d], recon_cube[mid_d]          # (Y, X)
    gt_xz, pred_xz = gt_cube[:, mid_h], recon_cube[:, mid_h]    # (Z, X)
    gt_yz, pred_yz = gt_cube[:, :, mid_w], recon_cube[:, :, mid_w]  # (Z, Y)

    rows = [
        (gt_xy, pred_xy, np.abs(gt_xy - pred_xy), "XY (mid-Z)", True),
        (gt_xz, pred_xz, np.abs(gt_xz - pred_xz), "XZ (mid-Y)", True),
        (gt_yz, pred_yz, np.abs(gt_yz - pred_yz), "YZ (mid-X)", False),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), facecolor='white')
    for row, (gt, pred, diff, title, has_x_axis) in enumerate(rows):
        for col, (img, label) in enumerate(zip([gt, pred, diff], ["GT", "Prediction", "|Diff|"])):
            ax = axes[row, col]
            is_diff = (label == "|Diff|")
            ax.imshow(img, cmap='hot' if is_diff else 'gray', vmin=0, vmax=None if is_diff else 1)
            if has_x_axis and 0 <= seam_offset_in_cube <= img.shape[1]:
                ax.axvline(seam_offset_in_cube, color='cyan', linewidth=1.2, linestyle='--')
            ax.set_title(f"{title} - {label}", fontsize=10)
            ax.axis('off')

    fig.suptitle("Seam cube — dense voxel-grid GT vs. Prediction "
                  "(dashed line marks the block A|B boundary)",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / "seam_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block-a", nargs=3, type=int, required=True, metavar=("IZ", "IY", "IX"),
                     help="Grid index of the LEFT block; its +X neighbour is used as block B")
    ap.add_argument("--n-per-axis", type=int, default=4, help="Pilot grid size per axis")
    ap.add_argument("--base-zyx", nargs=3, type=int, default=[30, 30, 30],
                     help="z,y,x offset added to grid indices for the raw tif filename")
    ap.add_argument("--block-shape", type=int, default=64, help="Raw block size in voxels")
    ap.add_argument("--blocks-dir", default="data/fafb/blocks")
    ap.add_argument("--models-dir", required=True,
                     help="Directory containing b_<iz><iy><ix>/ checkpoint subdirs")
    ap.add_argument("--checkpoint-name", default="best.pth")
    ap.add_argument("--cube-size", type=int, default=64)
    ap.add_argument("--x-center", type=int, default=None,
                     help="Voxel index (0..2*block_shape-1) to centre the sliding cube on; "
                          "defaults to the seam (= block_shape), which -- with the default "
                          "cube_size=block_shape -- puts exactly the right half of block A "
                          "and the left half of block B inside the cube.")
    ap.add_argument("--chunk-n", type=int, default=1024,
                     help="Gaussians evaluated per chunk in forward() (PyTorch fallback path "
                          "chunks over Gaussians, not points -- memory scales as "
                          "cube_size^3 * chunk_n, so keep this small; matches the default "
                          "used by sliding_window_camera.py's reconstruct_local_cube)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    iz, iy, ix = args.block_a
    if ix + 1 >= args.n_per_axis:
        raise ValueError(f"block A's ix={ix} has no +X neighbour within n_per_axis={args.n_per_axis}")
    block_b = (iz, iy, ix + 1)

    bz, by, bx = args.base_zyx
    W = args.block_shape  # per-block width along X
    combined_W = 2 * W
    x_center = args.x_center if args.x_center is not None else W
    cube = args.cube_size
    half = cube / 2.0
    x_lo_idx, x_hi_idx = x_center - half, x_center + half
    if x_lo_idx < 0 or x_hi_idx > combined_W:
        raise ValueError(f"cube [{x_lo_idx}, {x_hi_idx}) falls outside the combined "
                          f"[0, {combined_W}) X range -- reduce --cube-size or move --x-center")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Ground truth: load + concatenate along X, then slice the window ----
    def tif_path(idx3):
        z, y, x = idx3
        return Path(args.blocks_dir) / f"image_z{bz + z}_y{by + y}_x{bx + x}.tif"

    vol_a = _load_block_volume(tif_path((iz, iy, ix)))
    vol_b = _load_block_volume(tif_path(block_b))
    combined_gt = np.concatenate([vol_a, vol_b], axis=2)  # (Z, Y, X): X axis doubles

    ix0 = round(x_lo_idx)
    ix1 = ix0 + cube
    gt_cube = combined_gt[:, :, ix0:ix1]
    print(f"GT seam cube: X voxels [{ix0}, {ix1}) of combined width {combined_W} "
          f"(seam at {W}) -> shape {gt_cube.shape}")

    # ---- 2) Gaussian reconstruction: remap both blocks, filter to the window ----
    aabb = AABB.unit()
    cfg = argparse.Namespace(scale_min_clamp=1e-6, mahal_max_clamp=20.0)

    def ckpt_dict(idx3):
        z, y, x = idx3
        path = Path(args.models_dir) / f"b_{z}{y}{x}" / args.checkpoint_name
        return torch.load(str(path), map_location="cpu")

    ck_a = remap_block_x_pair(ckpt_dict((iz, iy, ix)), slot=0)
    ck_b = remap_block_x_pair(ckpt_dict(block_b), slot=1)

    means = torch.cat([ck_a["means"], ck_b["means"]], dim=0).to(device)
    log_s = torch.cat([ck_a["log_scales"], ck_b["log_scales"]], dim=0).to(device)
    quats = torch.cat([ck_a["quats"], ck_b["quats"]], dim=0).to(device)
    inten = torch.cat([ck_a["intensities"], ck_b["intensities"]], dim=0).to(device)

    x_lo, x_hi = voxel_to_norm(x_lo_idx, combined_W), voxel_to_norm(x_hi_idx - 1, combined_W)
    in_box = (means[:, 0] >= x_lo) & (means[:, 0] <= x_hi)
    print(f"Gaussians in seam window: {int(in_box.sum())} of {means.shape[0]} combined "
          f"(block A: {ck_a['means'].shape[0]}, block B: {ck_b['means'].shape[0]})")

    local_gc = GaussianCloud.__new__(GaussianCloud)
    local_gc.aabb, local_gc.device = aabb, device
    local_gc.scale_min, local_gc.mahal_clamp = cfg.scale_min_clamp, cfg.mahal_max_clamp
    local_gc.means = means[in_box]
    local_gc.log_s = log_s[in_box]
    local_gc.quats = quats[in_box]
    local_gc.inten = inten[in_box]

    # Recentre/rescale the filtered Gaussians into a fresh [-1,1]^3 frame
    # spanning exactly this window (Y/Z already span the full [-1,1] since
    # the window uses the full block extent on those axes).
    center = torch.tensor([(x_lo + x_hi) / 2, 0.0, 0.0], device=device)
    scale = torch.tensor([2.0 / (x_hi - x_lo), 1.0, 1.0], device=device)
    local_gc.means = (local_gc.means - center) * scale
    local_gc.log_s = local_gc.log_s + torch.log(scale).unsqueeze(0)

    with torch.no_grad():
        zc = torch.linspace(-1, 1, cube, device=device)
        yc = torch.linspace(-1, 1, cube, device=device)
        xc = torch.linspace(-1, 1, cube, device=device)
        zz, yy, xx = torch.meshgrid(zc, yc, xc, indexing='ij')
        pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        if local_gc.means.shape[0] == 0:
            recon_cube = np.zeros((cube, cube, cube), dtype=np.float32)
        else:
            recon_cube = local_gc.forward(pts, chunk_n=args.chunk_n).reshape(
                cube, cube, cube).clamp(0, 1).cpu().numpy().astype(np.float32)

    # ---- 3) Metrics + save ----
    mse = float(np.mean((recon_cube - gt_cube) ** 2))
    vol_psnr_db = float('inf') if mse == 0 else -10.0 * math.log10(mse)
    print(f"Seam-cube vol_PSNR: {vol_psnr_db:.3f} dB  (mse={mse:.6e})")

    np.save(out_dir / "gt_cube.npy", gt_cube)
    np.save(out_dir / "recon_cube.npy", recon_cube)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "block_a": list(args.block_a), "block_b": list(block_b),
            "x_center": x_center, "cube_size": cube,
            "n_gaussians_in_window": int(in_box.sum()),
            "mse": mse, "vol_psnr": vol_psnr_db,
        }, f, indent=2)

    seam_offset_in_cube = W - ix0  # where the true A|B boundary falls inside this cube
    _save_seam_slices(gt_cube, recon_cube, out_dir, seam_offset_in_cube)
    print(f"Saved: {out_dir}/{{gt_cube.npy, recon_cube.npy, metrics.json, seam_slices.png}}")


if __name__ == "__main__":
    main()
