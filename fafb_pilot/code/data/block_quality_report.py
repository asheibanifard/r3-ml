"""
Task 1 of fafb_pilot/code/claude.txt: report reconstruction quality (PSNR,
SSIM, LPIPS) for a handful of randomly-chosen blocks, using each block's
middle Z-slice as the 2D image compared against the ground truth.

BACKGROUND
----------
rec_vol.py and stitch_blocks.py already report WHOLE-VOLUME metrics
(libs.compute_metrics -- one MSE/PSNR/SSIM number per 64^3 volume). This
script instead reports the standard per-image metrics (skimage's windowed
SSIM, LPIPS's perceptual distance) on a single representative 2D slice per
block -- a quick, visual sanity check across several blocks at once, rather
than an exhaustive per-block evaluation.

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/data/block_quality_report.py \\
        --gaussian-json fafb_pilot/code/data/gaussians.json \\
        --n-blocks 4 --seed 0

STEPS
-----
1. Randomly sample --n-blocks distinct (iz,iy,ix) block indices from the
   [0,n_per_axis)^3 pilot grid (pick_random_blocks).
2. For each: load its GaussianCloud + GT tif (stitch_blocks.load_block) and
   reconstruct its full volume (stitch_blocks.reconstruct_block).
3. Take each volume's middle Z-slice and score it against the GT slice with
   PSNR/SSIM/LPIPS (libs.slice_metrics).
4. Save one grid figure (GT | Pred | |Diff| per block, metrics in the title)
   and one metrics CSV (one row per block).

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  block_quality_<name>.png   GT / Pred / |Diff| mid-slice grid, one row per block
  block_quality_<name>.csv   block name, PSNR, SSIM, LPIPS -- one row per block
  block_quality_<name>.xlsx  same rows, as a spreadsheet

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path + LPIPS network)
  - --gaussian-json and --blocks-dir are mutually exclusive, same convention
    as stitch_blocks.py/rec_vol.py.
"""
import csv
import os
import random
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fafb_pilot/code
import libs  # sets up sys.path to scripts/ + TORCH_CUDA_ARCH_LIST
from libs import imshow_gray, load_gaussians_json, slice_metrics, save_metrics_excel

import stitch_blocks as sb  # reuses load_block/reconstruct_block/block_name
from _3dgs._3dgs import AABB, VolumeDataset


# ── Step 1: pick blocks ─────────────────────────────────────────────────────

def pick_random_blocks(n_per_axis, n_blocks, seed):
    """n_blocks distinct (iz,iy,ix) triples sampled from the [0,n_per_axis)^3
    pilot grid, without replacement."""
    rng = random.Random(seed)
    all_indices = [
        (iz, iy, ix)
        for iz in range(n_per_axis)
        for iy in range(n_per_axis)
        for ix in range(n_per_axis)
    ]
    return rng.sample(all_indices, n_blocks)


# ── Steps 2-3: reconstruct one block + score its mid-slice ──────────────────

def evaluate_block(iz, iy, ix, blocks_dir, data_dir, z0, y0, x0, checkpoint_name, device, cfg,
                   gaussians_json_data):
    """Load, reconstruct, and mid-slice-score one block."""
    aabb = AABB.unit()
    bname, gc, vol_t = sb.load_block(
        blocks_dir, data_dir, z0, y0, x0, iz, iy, ix, checkpoint_name,
        aabb, device, cfg, gaussians_json_data=gaussians_json_data,
    )
    dataset = VolumeDataset(vol_t, aabb, cfg)
    pred_vol = sb.reconstruct_block(gc, dataset, device, cfg.chunk_n)
    gt_vol = vol_t.detach().cpu().numpy()

    mid = gt_vol.shape[0] // 2
    metrics = slice_metrics(pred_vol[mid], gt_vol[mid], device=device)
    return bname, gc.N, gt_vol, pred_vol, metrics


# ── Step 4: save outputs ─────────────────────────────────────────────────────

def save_report_figure(results, output_dir, name):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (bname, n_gauss, gt_vol, pred_vol, metrics) in enumerate(results):
        mid = gt_vol.shape[0] // 2
        gt_slice, pred_slice = gt_vol[mid], pred_vol[mid]
        diff = np.abs(pred_slice - gt_slice)

        imshow_gray(axes[row, 0], gt_slice, f"{bname} GT Z={mid}")
        imshow_gray(
            axes[row, 1], pred_slice,
            f"{bname} Pred ({n_gauss} G)\n"
            f"PSNR={metrics['PSNR']:.2f} SSIM={metrics['SSIM']:.4f} LPIPS={metrics['LPIPS']:.4f}"
        )
        im = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 1e-8))
        axes[row, 2].set_title(f"|Diff| (mean={diff.mean():.4f})")
        axes[row, 2].axis('off')
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"block_quality_{name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_report_csv(results, output_dir, name):
    rows = [{'block': bname, 'n_gaussians': n_gauss, **metrics}
            for bname, n_gauss, gt_vol, pred_vol, metrics in results]

    out_csv = os.path.join(output_dir, f"block_quality_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['block', 'n_gaussians', 'PSNR', 'SSIM', 'LPIPS'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")

    save_metrics_excel(rows, output_dir, f"block_quality_{name}.xlsx")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report mid-slice PSNR/SSIM/LPIPS reconstruction quality "
                    "for several randomly-chosen blocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blocks-dir", default=None,
                        help="Directory containing b_<iz><iy><ix>/<checkpoint-name> "
                             "subdirectories. Mutually exclusive with --gaussian-json.")
    parser.add_argument("--gaussian-json", default=None,
                        help="Path to a gaussians_json.py-style export covering the "
                             "pilot grid -- alternative to --blocks-dir/--checkpoint-name.")
    parser.add_argument("--n-per-axis", type=int, default=4,
                        help="Blocks per axis in the trained pilot grid to sample from "
                             "(4 matches the full blocks_v18 pilot grid).")
    parser.add_argument("--n-blocks", type=int, default=4,
                        help="How many distinct random blocks to evaluate.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for block selection (reproducible sampling).")
    parser.add_argument("--data-dir", default="/root/project/data/fafb/blocks")
    parser.add_argument("--z0", type=int, default=30)
    parser.add_argument("--y0", type=int, default=30)
    parser.add_argument("--x0", type=int, default=30)
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--output-dir", default="/root/project/fafb_pilot/results/data")
    parser.add_argument("--name", default=None,
                        help="Label used in output filenames. Defaults to "
                             "n<n-blocks>_seed<seed>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-n", type=int, default=1000)
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

    if bool(args.blocks_dir) == bool(args.gaussian_json):
        raise SystemExit("Specify exactly one of --blocks-dir or --gaussian-json.")

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or f"n{args.n_blocks}_seed{args.seed}"

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

    # Step 1: pick blocks.
    picks = pick_random_blocks(args.n_per_axis, args.n_blocks, args.seed)
    print(f"Randomly selected blocks (seed={args.seed}): "
          f"{[sb.block_name(*p) for p in picks]}")

    # Steps 2-3: reconstruct + score each block's mid-slice.
    results = []
    for (iz, iy, ix) in picks:
        bname, n_gauss, gt_vol, pred_vol, metrics = evaluate_block(
            iz, iy, ix, args.blocks_dir, args.data_dir, args.z0, args.y0, args.x0,
            args.checkpoint_name, device, cfg, gaussians_json_data,
        )
        print(f"  [{bname}] {n_gauss} Gaussians -- "
              f"PSNR={metrics['PSNR']:.2f} dB, SSIM={metrics['SSIM']:.4f}, LPIPS={metrics['LPIPS']:.4f}")
        results.append((bname, n_gauss, gt_vol, pred_vol, metrics))

    # Step 4: save outputs.
    save_report_figure(results, args.output_dir, name)
    save_report_csv(results, args.output_dir, name)


if __name__ == "__main__":
    main()
