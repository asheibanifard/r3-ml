"""
Task 3 of fafb_pilot/code/claude.txt: slide an empty 64^3 evaluation cube
across the shared boundary/boundaries of a stitched multi-block scene, and
score it against the corresponding ground-truth voxel grid (PSNR/SSIM/LPIPS)
at each position -- for the 2-block, 4-block, and 8-block stitches in turn.

BACKGROUND
----------
stitch_quality_report.py (Task 2) already scores ONE fixed 64^3 crop per
arrangement (the combined cube's own middle Z-slice). This script instead
NAVIGATES a 64^3 window (the same size as one native block) THROUGH the
junction between blocks -- the direct test of whether hard gating (see
stitch_blocks.py's module docstring) pays any seam penalty when the
evaluation region itself straddles a block boundary, not just when it sits
safely inside one block.

SWEEP CONVENTION
----------------
For each arrangement, only the axes that actually have >1 block vary; a
single-block axis stays fixed at offset 0 (the window already exactly fills
it). The window's offset on each varying axis moves TOGETHER along the
diagonal connecting the two far corners, from offset 0 (window fully inside
the "low" block(s)) through offset 32 (window exactly straddling the shared
boundary/boundaries -- the hardest case) to offset 64 (window fully inside
the "high" block(s)):

  2 blocks (1x1x2 grid): only x varies -- a 1D sweep through one boundary.
  4 blocks (1x2x2 grid): y and x vary together -- sweeps diagonally through
                         the shared edge where all 4 blocks meet.
  8 blocks (2x2x2 grid): z, y, and x vary together -- sweeps through the
                         single point where all 8 blocks meet (test.ipynb's
                         octant centre).

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/data/sliding_cube_eval.py \\
        --gaussian-json fafb_pilot/code/data/gaussians.json

STEPS
-----
1. For each of the 3 fixed grids (2/4/8 blocks): stitch it once
   (stitch_blocks.stitch_blocks_grid), matching stitch_quality_report.py.
2. Sweep the 64^3 window across --n-steps positions along the block
   junction's diagonal (sweep_positions).
3. At each position, crop both the combined reconstruction and the combined
   GT to the window and score it: whole-window PSNR/SSIM (libs.compute_metrics)
   and mid-slice PSNR/SSIM/LPIPS (libs.slice_metrics) (evaluate_window).
4. Save one figure per arrangement (mid-slice snapshots at the start/
   boundary/end positions, plus PSNR/SSIM/LPIPS vs. sweep offset with the
   boundary marked) and one combined CSV (one row per arrangement per step).

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  sliding_cube_<name>_2blocks.png   snapshots + metrics-vs-offset, 1x1x2 grid
  sliding_cube_<name>_4blocks.png   snapshots + metrics-vs-offset, 1x2x2 grid
  sliding_cube_<name>_8blocks.png   snapshots + metrics-vs-offset, 2x2x2 grid
  sliding_cube_<name>.csv           one row per (arrangement, sweep step)
  sliding_cube_<name>.xlsx          same rows, as a spreadsheet

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path + LPIPS network)
  - --gaussian-json and --blocks-dir are mutually exclusive, same convention
    as stitch_blocks.py/rec_vol.py. --gaussian-json is strongly recommended
    since it's loaded once and reused across all 3 stitches.
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

CUBE_SIZE = 64  # matches the native per-block resolution

# Same 3 arrangements as stitch_quality_report.py (Task 2).
ARRANGEMENTS = [
    ("2blocks", (1, 1, 2)),
    ("4blocks", (1, 2, 2)),
    ("8blocks", (2, 2, 2)),
]


# ── Step 2: sweep positions ──────────────────────────────────────────────────

def sweep_positions(grid, cube_size, n_steps):
    """n_steps (oz, oy, ox) window offsets sweeping through the block
    junction's diagonal -- see module docstring's SWEEP CONVENTION. Returns
    (offsets, boundary_t): boundary_t is the sweep parameter value at which
    the window exactly straddles the shared boundary (always cube_size/2,
    since every block is cube_size voxels wide)."""
    nz, ny, nx = grid
    max_off = (max(nz, ny, nx) - 1) * cube_size  # 0 or cube_size for these 3 arrangements
    ts = np.linspace(0, max_off, n_steps)
    offsets = []
    for t in ts:
        oz = int(round(t)) if nz > 1 else 0
        oy = int(round(t)) if ny > 1 else 0
        ox = int(round(t)) if nx > 1 else 0
        offsets.append((oz, oy, ox))
    boundary_t = max_off / 2.0
    return offsets, ts, boundary_t


# ── Step 3: crop + score one window position ────────────────────────────────

def evaluate_window(gt_cube, rec_cube, offset, cube_size, device):
    oz, oy, ox = offset
    gt_win = gt_cube[oz:oz + cube_size, oy:oy + cube_size, ox:ox + cube_size]
    rec_win = rec_cube[oz:oz + cube_size, oy:oy + cube_size, ox:ox + cube_size]

    volume_m = compute_metrics(rec_win, gt_win)
    mid = cube_size // 2
    slice_m = slice_metrics(rec_win[mid], gt_win[mid], device=device)
    return gt_win, rec_win, volume_m, slice_m


# ── Step 4: save outputs ─────────────────────────────────────────────────────

def save_arrangement_figure(label, grid, ts, boundary_t, snapshots, slice_metrics_by_step,
                            volume_metrics_by_step, output_dir, name):
    """Top row: mid-slice GT|Pred snapshots at the start/boundary/end sweep
    positions. Bottom row: PSNR/SSIM/LPIPS vs. sweep offset, boundary marked."""
    nz, ny, nx = grid
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    for col, (t, gt_win, rec_win) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, col])
        mid = gt_win.shape[0] // 2
        side_by_side = np.concatenate([gt_win[mid], rec_win[mid]], axis=1)
        imshow_gray(ax, side_by_side, f"offset={t:.0f}  (GT | Pred)")

    psnr = [m['PSNR'] for m in slice_metrics_by_step]
    ssim = [m['SSIM'] for m in slice_metrics_by_step]
    lpips_v = [m['LPIPS'] for m in slice_metrics_by_step]
    vol_psnr = [m['PSNR'] for m in volume_metrics_by_step]

    ax_p = fig.add_subplot(gs[1, 0])
    ax_p.plot(ts, psnr, marker='o', label='mid-slice PSNR')
    ax_p.plot(ts, vol_psnr, marker='s', label='whole-window PSNR')
    ax_p.axvline(boundary_t, color='red', linestyle='--', alpha=0.5, label='boundary')
    ax_p.set_xlabel("Sweep offset (voxels)")
    ax_p.set_ylabel("dB")
    ax_p.set_title("PSNR vs. sweep offset")
    ax_p.legend(fontsize=8)
    ax_p.grid(alpha=0.3)

    ax_s = fig.add_subplot(gs[1, 1])
    ax_s.plot(ts, ssim, marker='o', color='tab:orange')
    ax_s.axvline(boundary_t, color='red', linestyle='--', alpha=0.5)
    ax_s.set_xlabel("Sweep offset (voxels)")
    ax_s.set_title("Mid-slice SSIM vs. sweep offset")
    ax_s.grid(alpha=0.3)

    ax_l = fig.add_subplot(gs[1, 2])
    ax_l.plot(ts, lpips_v, marker='o', color='tab:green')
    ax_l.axvline(boundary_t, color='red', linestyle='--', alpha=0.5)
    ax_l.set_xlabel("Sweep offset (voxels)")
    ax_l.set_title("Mid-slice LPIPS vs. sweep offset\n(lower is better)")
    ax_l.grid(alpha=0.3)

    fig.suptitle(f"{label} ({nz}x{ny}x{nx} grid): sliding 64^3 cube across the block junction")
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"sliding_cube_{name}_{label}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_metrics_csv(all_rows, output_dir, name):
    fieldnames = [
        'arrangement', 'nz', 'ny', 'nx', 'offset_z', 'offset_y', 'offset_x', 'sweep_t',
        'slice_PSNR', 'slice_SSIM', 'slice_LPIPS', 'volume_PSNR', 'volume_SSIM', 'volume_MSE',
    ]

    out_csv = os.path.join(output_dir, f"sliding_cube_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {out_csv}")

    save_metrics_excel(all_rows, output_dir, f"sliding_cube_{name}.xlsx")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slide a 64^3 evaluation cube across the block junction "
                    "of 2-, 4-, and 8-block stitches, scoring PSNR/SSIM/LPIPS "
                    "against ground truth at each position.",
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
    parser.add_argument("--n-steps", type=int, default=9,
                        help="Number of window positions swept per arrangement "
                             "(evenly spaced from offset 0 to 64).")
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

    all_rows = []
    for label, grid in ARRANGEMENTS:
        nz, ny, nx = grid
        print(f"\n=== {label} ({nz}x{ny}x{nx}) ===")

        # Step 1: stitch once (same as stitch_quality_report.py).
        gt_cube, rec_cube, _combined_gaussians = sb.stitch_blocks_grid(
            args.blocks_dir, args.data_dir, nz, ny, nx, args.z0, args.y0, args.x0,
            args.checkpoint_name, device, cfg, gaussians_json_data=gaussians_json_data,
        )

        # Step 2: sweep positions.
        offsets, ts, boundary_t = sweep_positions(grid, CUBE_SIZE, args.n_steps)

        # Step 3: score every position.
        slice_metrics_by_step, volume_metrics_by_step = [], []
        snapshots = []
        snapshot_ts = {ts[0], boundary_t, ts[-1]}
        for offset, t in zip(offsets, ts):
            gt_win, rec_win, volume_m, slice_m = evaluate_window(gt_cube, rec_cube, offset, CUBE_SIZE, device)
            print(f"  offset={t:6.1f}  slice PSNR={slice_m['PSNR']:.2f} dB  "
                  f"SSIM={slice_m['SSIM']:.4f}  LPIPS={slice_m['LPIPS']:.4f}  "
                  f"| whole-window PSNR={volume_m['PSNR']:.2f} dB")
            slice_metrics_by_step.append(slice_m)
            volume_metrics_by_step.append(volume_m)
            if t in snapshot_ts and len(snapshots) < 3:
                snapshots.append((t, gt_win, rec_win))

            all_rows.append({
                'arrangement': label, 'nz': nz, 'ny': ny, 'nx': nx,
                'offset_z': offset[0], 'offset_y': offset[1], 'offset_x': offset[2], 'sweep_t': t,
                'slice_PSNR': slice_m['PSNR'], 'slice_SSIM': slice_m['SSIM'], 'slice_LPIPS': slice_m['LPIPS'],
                'volume_PSNR': volume_m['PSNR'], 'volume_SSIM': volume_m['SSIM'], 'volume_MSE': volume_m['MSE'],
            })

        # Step 4 (figure): snapshots + metrics-vs-offset for this arrangement.
        save_arrangement_figure(
            label, grid, ts, boundary_t, snapshots,
            slice_metrics_by_step, volume_metrics_by_step, args.output_dir, args.name,
        )

    # Step 4 (CSV): every arrangement's every sweep step, one file.
    save_metrics_csv(all_rows, args.output_dir, args.name)


if __name__ == "__main__":
    main()
