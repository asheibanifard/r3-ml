"""
Task 2 of fafb_pilot/code/claude.txt: stitch 2, then 4, then 8 blocks (using
stitch_blocks.py's hard-gated stitching), and for each arrangement compare
the combined reconstruction's middle Z-slice against the corresponding
ground-truth slice with PSNR/SSIM/LPIPS.

BACKGROUND
----------
stitch_blocks.py already reports WHOLE-VOLUME metrics (libs.compute_metrics)
for one fixed cubic n_per_axis^3 arrangement. This script instead runs THREE
arrangements back to back -- 2 blocks (a 1x1x2 strip along x), 4 blocks (a
1x2x2 sheet), 8 blocks (the familiar 2x2x2 octant) -- via the newly
generalised stitch_blocks.stitch_blocks_grid(nz, ny, nx, ...), and scores
each one's middle Z-slice with the standard per-image metrics
(libs.slice_metrics), to see whether reconstruction quality holds steady as
more block boundaries are introduced (the key claim behind hard gating: no
seam penalty).

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/data/stitch_quality_report.py \\
        --gaussian-json fafb_pilot/code/data/gaussians.json

STEPS
-----
1. For each of the 3 fixed grids -- (1,1,2)="2 blocks", (1,2,2)="4 blocks",
   (2,2,2)="8 blocks" -- stitch it (stitch_blocks.stitch_blocks_grid).
2. Take the combined cube's middle Z-slice (pred + GT).
3. Score that slice with PSNR/SSIM/LPIPS (libs.slice_metrics) -- as well as
   the whole-cube metrics (libs.compute_metrics), for reference.
4. Save one GT/Pred/|Diff| mid-slice figure per arrangement, one summary
   figure (metrics vs. block count), and one metrics CSV (one row per
   arrangement).

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  stitch_quality_<name>_2blocks.png   GT/Pred/|Diff| mid-slice, 1x1x2 grid
  stitch_quality_<name>_4blocks.png   GT/Pred/|Diff| mid-slice, 1x2x2 grid
  stitch_quality_<name>_8blocks.png   GT/Pred/|Diff| mid-slice, 2x2x2 grid
  stitch_quality_<name>_summary.png   mid-slice PSNR/SSIM/LPIPS vs block count
  stitch_quality_<name>.csv           one row per arrangement (both mid-slice
                                       and whole-cube metrics)
  stitch_quality_<name>.xlsx          same rows, as a spreadsheet

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path + LPIPS network)
  - --gaussian-json and --blocks-dir are mutually exclusive, same convention
    as stitch_blocks.py/rec_vol.py. --gaussian-json is strongly recommended
    here since the same export is reused across all 3 arrangements (14
    blocks loaded in total: 2+4+8) without re-parsing per block.
"""
import csv
import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fafb_pilot/code
import libs  # sets up sys.path to scripts/ + TORCH_CUDA_ARCH_LIST
from libs import imshow_gray, load_gaussians_json, slice_metrics, compute_metrics, save_metrics_excel

import stitch_blocks as sb  # reuses stitch_blocks_grid (sets up the CUDA kernel as an import side effect)

# The 3 arrangements named in the task: 2, 4, and 8 blocks. Grid dims are
# (nz, ny, nx); a plain n_per_axis^3 cube is the nz==ny==nx special case
# (see stitch_blocks.stitch_blocks_grid's own docstring).
ARRANGEMENTS = [
    ("2blocks", (1, 1, 2)),
    ("4blocks", (1, 2, 2)),
    ("8blocks", (2, 2, 2)),
]


# ── Steps 1-3: stitch one arrangement + score its mid-slice ─────────────────

def evaluate_arrangement(grid, blocks_dir, data_dir, z0, y0, x0, checkpoint_name, device, cfg,
                         gaussians_json_data):
    nz, ny, nx = grid
    gt_cube, rec_cube, _combined_gaussians = sb.stitch_blocks_grid(
        blocks_dir, data_dir, nz, ny, nx, z0, y0, x0, checkpoint_name, device, cfg,
        gaussians_json_data=gaussians_json_data,
    )
    mid = gt_cube.shape[0] // 2
    slice_m = slice_metrics(rec_cube[mid], gt_cube[mid], device=device)
    volume_m = compute_metrics(rec_cube, gt_cube)
    return gt_cube, rec_cube, slice_m, volume_m


# ── Step 4: save outputs ─────────────────────────────────────────────────────

def save_arrangement_figure(gt_cube, rec_cube, slice_m, label, grid, output_dir, name):
    mid = gt_cube.shape[0] // 2
    gt_slice, pred_slice = gt_cube[mid], rec_cube[mid]
    diff = np.abs(pred_slice - gt_slice)

    nz, ny, nx = grid
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    imshow_gray(axes[0], gt_slice, f"GT   Z={mid}  ({nz}x{ny}x{nx} grid)")
    imshow_gray(
        axes[1], pred_slice,
        f"Stitched Pred\nPSNR={slice_m['PSNR']:.2f} SSIM={slice_m['SSIM']:.4f} LPIPS={slice_m['LPIPS']:.4f}"
    )
    im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 1e-8))
    axes[2].set_title(f"|Diff| (mean={diff.mean():.4f})")
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"stitch_quality_{name}_{label}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_summary_figure(results, output_dir, name):
    """Mid-slice PSNR/SSIM/LPIPS vs. block count -- flat lines here are the
    headline result: hard gating pays no seam penalty as more block
    boundaries are introduced."""
    n_blocks = [np.prod(grid) for _label, grid, *_ in results]
    psnr = [r[2]['PSNR'] for r in results]
    ssim = [r[2]['SSIM'] for r in results]
    lpips_v = [r[2]['LPIPS'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, values, title, ylabel in zip(
        axes, [psnr, ssim, lpips_v],
        ["Mid-slice PSNR", "Mid-slice SSIM", "Mid-slice LPIPS"],
        ["dB", "", "(lower is better)"],
    ):
        ax.plot(n_blocks, values, marker='o')
        ax.set_xticks(n_blocks)
        ax.set_xlabel("Blocks stitched")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"stitch_quality_{name}_summary.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_metrics_csv(results, output_dir, name):
    fieldnames = [
        'label', 'nz', 'ny', 'nx', 'n_blocks',
        'slice_PSNR', 'slice_SSIM', 'slice_LPIPS',
        'volume_PSNR', 'volume_SSIM', 'volume_MSE',
    ]
    rows = []
    for label, grid, slice_m, volume_m in results:
        nz, ny, nx = grid
        rows.append({
            'label': label, 'nz': nz, 'ny': ny, 'nx': nx, 'n_blocks': nz * ny * nx,
            'slice_PSNR': slice_m['PSNR'], 'slice_SSIM': slice_m['SSIM'], 'slice_LPIPS': slice_m['LPIPS'],
            'volume_PSNR': volume_m['PSNR'], 'volume_SSIM': volume_m['SSIM'], 'volume_MSE': volume_m['MSE'],
        })

    out_csv = os.path.join(output_dir, f"stitch_quality_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")

    save_metrics_excel(rows, output_dir, f"stitch_quality_{name}.xlsx")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch 2, 4, and 8 blocks and compare each arrangement's "
                    "mid-slice reconstruction quality against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blocks-dir", default=None,
                        help="Directory containing b_<iz><iy><ix>/<checkpoint-name> "
                             "subdirectories. Mutually exclusive with --gaussian-json.")
    parser.add_argument("--gaussian-json", default=None,
                        help="Path to a gaussians_json.py-style export covering all "
                             "blocks used by the 3 arrangements -- alternative to "
                             "--blocks-dir/--checkpoint-name, and strongly recommended "
                             "here since it's loaded once and reused across all 3 runs.")
    parser.add_argument("--data-dir", default="/root/project/data/fafb/blocks")
    parser.add_argument("--z0", type=int, default=30)
    parser.add_argument("--y0", type=int, default=30)
    parser.add_argument("--x0", type=int, default=30)
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--output-dir", default="/root/project/fafb_pilot/results/data")
    parser.add_argument("--name", default="blocks_v18",
                        help="Label used in output filenames.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-n", type=int, default=1000)
    parser.add_argument("--scale-min-clamp", type=float, default=1e-5)
    parser.add_argument("--mahal-max-clamp", type=float, default=20.0)
    # The remaining flags are unused by GaussianCloud.load()/this script's own
    # logic -- kept only because VolumeDataset's constructor expects a
    # fully-populated cfg Namespace (same convention as stitch_blocks.py).
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

    if bool(args.blocks_dir) == bool(args.gaussian_json):
        raise SystemExit("Specify exactly one of --blocks-dir or --gaussian-json.")

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    gaussians_json_data = load_gaussians_json(args.gaussian_json) if args.gaussian_json else None

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

    results = []
    for label, grid in ARRANGEMENTS:
        print(f"\n=== {label} ({grid[0]}x{grid[1]}x{grid[2]}) ===")
        gt_cube, rec_cube, slice_m, volume_m = evaluate_arrangement(
            grid, args.blocks_dir, args.data_dir, args.z0, args.y0, args.x0,
            args.checkpoint_name, device, cfg, gaussians_json_data,
        )
        print(f"  Mid-slice:   PSNR={slice_m['PSNR']:.2f} dB  SSIM={slice_m['SSIM']:.4f}  LPIPS={slice_m['LPIPS']:.4f}")
        print(f"  Whole-cube:  PSNR={volume_m['PSNR']:.2f} dB  SSIM={volume_m['SSIM']:.4f}")

        save_arrangement_figure(gt_cube, rec_cube, slice_m, label, grid, args.output_dir, args.name)
        results.append((label, grid, slice_m, volume_m))

    save_summary_figure(results, args.output_dir, args.name)
    save_metrics_csv(results, args.output_dir, args.name)


if __name__ == "__main__":
    main()
