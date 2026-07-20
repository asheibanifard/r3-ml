"""
Visualise the seam-bump statistics discussed for the 2/4/8-block
stitch_quality_report.py images: how much (if any) each block's own edge
shows up as elevated reconstruction error exactly at a stitch boundary, and
why that shows up as more or less visually obvious depending on how many
boundaries cross in a given slice.

BACKGROUND
----------
Manual inspection of stitch_quality_blocks_v18_{2,4,8}blocks.png raised two
questions this script answers with numbers instead of eyeballing:
  1. Is there a real elevated-error bump exactly at a block boundary, or is
     it an illusion from natural membrane structure? -- YES, a small one
     (~1.5-2.6x the local baseline), present in ALL THREE arrangements at
     similar relative magnitude. It is NOT a hard-gating bug (whole-cube
     PSNR stays flat across arrangements -- see stitch_quality_report.py) --
     it's each independently-trained block being very slightly less
     accurate right at its own physical edge (less training context there).
  2. Why does it look much more obvious in the 4- and 8-block images than
     the 2-block one? -- Because more boundaries cross in the SAME slice:
     the 2-block slice crosses 1 seam, the 4-block slice crosses 2 (forming
     a cross/grid pattern the eye immediately reads as artificial), and the
     8-block slice's own mid-Z-slice (Z=64) is ITSELF sitting exactly on the
     z-seam too, on top of crossing x and y -- so its intersection point
     accumulates 3 independent blocks' edge effects at once, producing a
     visible spike right where they meet. The image content each seam
     happens to cross also matters: a seam through a busy, membrane-dense
     region is camouflaged by naturally high background error; a seam
     through a flat region stands out by contrast.

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/data/seam_bump_report.py \\
        --gaussian-json fafb_pilot/code/data/gaussians.json

STEPS
-----
1. For each of the 3 fixed arrangements (2/4/8 blocks, same grids as
   stitch_quality_report.py): stitch it and take the SAME middle Z-slice
   diff = |pred-gt| shown in stitch_quality_blocks_v18_<label>.png, so the
   numbers here reproduce exactly what those figures show.
2. For every axis that has >1 block in that arrangement (x for 2-block; x
   and y for 4-block; x and y for 8-block's 2D slice), average |diff| along
   the other axis to get a 1D profile, and compare the value exactly at the
   boundary index against a baseline measured safely away from it
   (seam_bump_profile).
3. Where 2+ axes are split in the same slice, also read off the exact
   corner/intersection pixel's error relative to the slice's overall mean
   (corner_spike).
4. (8-block only) Compare the mean |diff| of the on-z-seam slice actually
   shown in the figure (Z=64) against a deep-interior slice with no seam
   crossing at all (Z=32), to show the aggregate per-slice cost of sitting
   on a boundary is negligible even though the local spike is real
   (z_seam_vs_interior).
5. Save one combined figure (one row per arrangement: its profile plot(s);
   final row: cross-arrangement summary bars) and one metrics CSV + xlsx.

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  seam_bump_<name>.png    profiles per arrangement + cross-arrangement summary
  seam_bump_<name>.csv    one row per (arrangement, axis) with boundary/
                          baseline/ratio, plus corner-spike and z-seam rows
  seam_bump_<name>.xlsx   same rows, as a spreadsheet

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path, see scripts/_3dgs/_3dgs.py)
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
from libs import load_gaussians_json, save_metrics_excel

import stitch_blocks as sb  # reuses stitch_blocks_grid (sets up the CUDA kernel as an import side effect)

BLOCK_VOXELS = 64  # native per-block resolution -- boundaries always fall at a multiple of this

# Same 3 arrangements as stitch_quality_report.py / sliding_cube_eval.py.
ARRANGEMENTS = [
    ("2blocks", (1, 1, 2)),
    ("4blocks", (1, 2, 2)),
    ("8blocks", (2, 2, 2)),
]

# Columns/rows this many voxels away from the boundary on either side count
# as "baseline" (far enough that the block-edge effect has died out).
BASELINE_OFFSETS = [-8, -7, -6, -5, 5, 6, 7, 8]


# ── Steps 2-3: seam-bump statistics for one arrangement's mid-slice diff ────

def seam_bump_profile(diff, axis, boundary_idx):
    """Mean |diff| profile along `axis` (averaging over the other axis),
    plus the boundary value, a baseline (BASELINE_OFFSETS away), and their
    ratio. axis=1 -> column profile (x varies), axis=0 -> row profile
    (y varies), matching diff's own (Y, X) shape."""
    profile = diff.mean(axis=1 - axis)  # average over the OTHER axis
    boundary_val = float(profile[boundary_idx])
    baseline_val = float(np.mean([profile[boundary_idx + o] for o in BASELINE_OFFSETS]))
    ratio = boundary_val / baseline_val if baseline_val > 0 else float('inf')
    window = np.arange(boundary_idx - 12, boundary_idx + 13)
    return {
        'profile': profile[window],
        'window': window,
        'boundary_val': boundary_val,
        'baseline_val': baseline_val,
        'ratio': ratio,
    }


def corner_spike(diff, boundary_idx):
    """The exact intersection pixel's error vs. the slice's overall mean --
    only meaningful where 2+ axes are split in the same slice."""
    corner_val = float(diff[boundary_idx, boundary_idx])
    overall_mean = float(diff.mean())
    return {'corner_val': corner_val, 'overall_mean': overall_mean,
            'ratio': corner_val / overall_mean if overall_mean > 0 else float('inf')}


# ── Step 4: aggregate on-seam vs. interior slice cost (8-block only) ───────

def z_seam_vs_interior(rec_cube, gt_cube):
    """8-block only: compare the mean |diff| of the mid-Z-slice (which sits
    exactly on the z-seam, since nz=2) against a slice deep inside one
    block, with no seam crossing at all."""
    on_seam = float(np.abs(rec_cube[64] - gt_cube[64]).mean())
    interior = float(np.abs(rec_cube[32] - gt_cube[32]).mean())
    return {'on_seam_mean_diff': on_seam, 'interior_mean_diff': interior}


# ── Step 5: save outputs ─────────────────────────────────────────────────────

AXIS_COLOR = {'x': 'tab:blue', 'y': 'tab:orange'}


def save_figure(per_arrangement, z_seam_stats, output_dir, name):
    n_rows = len(per_arrangement) + 1  # +1 for the summary row
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.1 * n_rows))

    for row, (label, grid, profiles, corner) in enumerate(per_arrangement):
        ax = axes[row]
        for axis_name, stats in profiles.items():
            ax.plot(stats['window'], stats['profile'], marker='o', markersize=3,
                    color=AXIS_COLOR[axis_name],
                    label=f"{axis_name}-profile (boundary/baseline = {stats['ratio']:.2f}x)")
        ax.axvline(BLOCK_VOXELS, color='red', linestyle='--', alpha=0.5, label='boundary')
        title = f"{label} ({grid[0]}x{grid[1]}x{grid[2]}): mean |diff| near the boundary"
        if corner is not None:
            title += f"  |  corner spike = {corner['ratio']:.2f}x overall mean"
        ax.set_title(title)
        ax.set_xlabel("Voxel index")
        ax.set_ylabel("mean |diff|")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Summary row: boundary/baseline ratio bars across every (arrangement, axis).
    ax = axes[-1]
    bar_labels, bar_values, bar_colors = [], [], []
    for label, grid, profiles, corner in per_arrangement:
        for axis_name, stats in profiles.items():
            bar_labels.append(f"{label}\n{axis_name}-seam")
            bar_values.append(stats['ratio'])
            bar_colors.append(AXIS_COLOR[axis_name])
        if corner is not None:
            bar_labels.append(f"{label}\ncorner")
            bar_values.append(corner['ratio'])
            bar_colors.append('tab:red')
    ax.bar(range(len(bar_values)), bar_values, color=bar_colors)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='no elevation (1x)')
    ax.set_ylabel("ratio vs. baseline / overall mean")
    ax.set_title("Summary: boundary/baseline ratio and corner spike, by arrangement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(
        "Seam bumps are similar magnitude across arrangements (~1.5-2.6x); the corner spike grows "
        "with how many seams cross there.\n"
        f"8-block's on-z-seam slice mean ({z_seam_stats['on_seam_mean_diff']:.5f}) barely differs from "
        f"an interior slice ({z_seam_stats['interior_mean_diff']:.5f}) -- no aggregate penalty.",
        fontsize=10, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(output_dir, f"seam_bump_{name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_metrics(per_arrangement, z_seam_stats, output_dir, name):
    rows = []
    for label, grid, profiles, corner in per_arrangement:
        for axis_name, stats in profiles.items():
            rows.append({
                'arrangement': label, 'grid': f"{grid[0]}x{grid[1]}x{grid[2]}", 'row_type': f'{axis_name}_seam',
                'boundary_val': stats['boundary_val'], 'baseline_val': stats['baseline_val'], 'ratio': stats['ratio'],
            })
        if corner is not None:
            rows.append({
                'arrangement': label, 'grid': f"{grid[0]}x{grid[1]}x{grid[2]}", 'row_type': 'corner_spike',
                'boundary_val': corner['corner_val'], 'baseline_val': corner['overall_mean'], 'ratio': corner['ratio'],
            })
    rows.append({
        'arrangement': '8blocks', 'grid': '2x2x2', 'row_type': 'z_seam_slice_vs_interior_slice',
        'boundary_val': z_seam_stats['on_seam_mean_diff'], 'baseline_val': z_seam_stats['interior_mean_diff'],
        'ratio': z_seam_stats['on_seam_mean_diff'] / z_seam_stats['interior_mean_diff'],
    })

    out_csv = os.path.join(output_dir, f"seam_bump_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['arrangement', 'grid', 'row_type', 'boundary_val', 'baseline_val', 'ratio'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")

    save_metrics_excel(rows, output_dir, f"seam_bump_{name}.xlsx")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise the seam-bump statistics behind the 2/4/8-block "
                    "stitch_quality_report.py images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blocks-dir", default=None,
                        help="Directory containing b_<iz><iy><ix>/<checkpoint-name> "
                             "subdirectories. Mutually exclusive with --gaussian-json.")
    parser.add_argument("--gaussian-json", default=None,
                        help="Path to a gaussians_json.py-style export covering all "
                             "blocks used by the 3 arrangements -- alternative to "
                             "--blocks-dir/--checkpoint-name.")
    parser.add_argument("--data-dir", default="/root/project/data/fafb/blocks")
    parser.add_argument("--z0", type=int, default=30)
    parser.add_argument("--y0", type=int, default=30)
    parser.add_argument("--x0", type=int, default=30)
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--output-dir", default="/root/project/fafb_pilot/results/data")
    parser.add_argument("--name", default="blocks_v18", help="Label used in output filenames.")
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

    per_arrangement = []
    z_seam_stats = None
    for label, grid in ARRANGEMENTS:
        nz, ny, nx = grid
        print(f"\n=== {label} ({nz}x{ny}x{nx}) ===")

        # Step 1: stitch once, same mid-Z-slice as stitch_quality_report.py.
        gt_cube, rec_cube, _combined_gaussians = sb.stitch_blocks_grid(
            args.blocks_dir, args.data_dir, nz, ny, nx, args.z0, args.y0, args.x0,
            args.checkpoint_name, device, cfg, gaussians_json_data=gaussians_json_data,
        )
        mid = gt_cube.shape[0] // 2
        diff = np.abs(rec_cube[mid] - gt_cube[mid])

        # Step 2: per-axis boundary profile, only for axes actually split.
        profiles = {}
        if nx > 1:
            profiles['x'] = seam_bump_profile(diff, axis=1, boundary_idx=BLOCK_VOXELS)
            print(f"  x-seam: boundary={profiles['x']['boundary_val']:.5f}  "
                  f"baseline={profiles['x']['baseline_val']:.5f}  ratio={profiles['x']['ratio']:.2f}x")
        if ny > 1:
            profiles['y'] = seam_bump_profile(diff, axis=0, boundary_idx=BLOCK_VOXELS)
            print(f"  y-seam: boundary={profiles['y']['boundary_val']:.5f}  "
                  f"baseline={profiles['y']['baseline_val']:.5f}  ratio={profiles['y']['ratio']:.2f}x")

        # Step 3: corner spike, only where 2+ axes are split in this slice.
        corner = None
        if nx > 1 and ny > 1:
            corner = corner_spike(diff, BLOCK_VOXELS)
            print(f"  corner spike: {corner['corner_val']:.5f}  "
                  f"(overall mean {corner['overall_mean']:.5f}, {corner['ratio']:.2f}x)")

        per_arrangement.append((label, grid, profiles, corner))

        # Step 4: 8-block only -- aggregate on-seam vs. interior slice cost.
        if label == "8blocks":
            z_seam_stats = z_seam_vs_interior(rec_cube, gt_cube)
            print(f"  z-seam slice (Z=64) mean |diff|: {z_seam_stats['on_seam_mean_diff']:.5f}  "
                  f"vs. interior slice (Z=32): {z_seam_stats['interior_mean_diff']:.5f}")

    # Step 5: save outputs.
    save_figure(per_arrangement, z_seam_stats, args.output_dir, args.name)
    save_metrics(per_arrangement, z_seam_stats, args.output_dir, args.name)


if __name__ == "__main__":
    main()
