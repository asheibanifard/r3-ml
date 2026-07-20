"""
Paper-grade version of seam_bump_report.py's single-slice, N=1 observations.
Fixes the 5 weaknesses identified when reviewing that script's numbers for
inclusion in a paper:

  1. N=1 per arrangement          -> sample --n-samples DIFFERENT placements
                                      per arrangement (different corners of
                                      the pilot grid), report mean +/- SEM.
  2. Single mid-slice only         -> average over the FULL 3D volume (every
                                      Z and every in-plane row/column), not
                                      one 2D cross-section.
  3. Single-pixel corner spike     -> average over a small neighbourhood
                                      region around the corner, not 1 voxel.
  4. No membrane-structure control -> N/A directly, but averaging over many
                                      placements (different physical content
                                      each time) washes out any one slice's
                                      coincidental membrane alignment.
  5. No isolated-block control     -> ADDED: measure the same edge-vs-centre
                                      effect on lone, non-stitched blocks (no
                                      neighbour, no stitching at all). If the
                                      isolated-block ratio matches the
                                      stitched-boundary ratio, that's direct
                                      evidence the effect is generic
                                      "independently-trained block is
                                      slightly worse at its own edge", NOT an
                                      artefact specific to stitching.

BACKGROUND
----------
seam_bump_report.py found small boundary/baseline error ratios (~1.5-2.6x)
in the 2/4/8-block stitch_quality_report.py mid-slices, and a corner-pixel
spike that grew with how many seams crossed there (0.07x for 4 blocks,
3.51x for 8 blocks). Those numbers came from exactly one slice through
exactly one fixed set of blocks each -- not defensible as a quantitative
paper claim. This script re-measures the same phenomena with proper
statistics.

METHOD
------
Every block in the --n-per-axis-available^3 pilot grid (default 4, i.e. all
64 blocks_v18 blocks) is loaded and reconstructed EXACTLY ONCE
(precompute_block_cache) -- reconstructing a block's own field never depends
on which stitched arrangement it ends up in, so this cache is reused for
every one of the (up to) 48+36+27 possible placements across all 3
arrangement types, instead of re-running the CUDA kernel per placement.

For each arrangement (2/4/8 blocks): sample_placements() picks --n-samples
distinct, non-overlapping-with-itself corners of the pilot grid (e.g. for
the 2-block 1x1x2 arrangement, a "corner" is any (iz,iy,ix) such that
(iz,iy,ix) and (iz,iy,ix+1) are both inside the pilot grid). Each sampled
placement's combined cube is assembled from the cache (assemble_from_cache
-- cheap array copies, no GPU work), then:
  - seam_bump_profile_3d() measures the boundary/baseline ratio for each
    split axis, averaged over BOTH other axes across the FULL volume (not
    one slice) -- e.g. for the 8-block cube's x-axis, every one of the
    128*128 (Z,Y) voxels at each X index contributes to that index's mean,
    not just the 128 pixels in one Z-slice.
  - corner_region_stat() measures a small neighbourhood around the
    corner/edge where 2+ axes meet (skipped for the 2-block arrangement,
    which has no corner).
Ratios are pooled across all --n-samples placements per arrangement and
reported as mean +/- SEM (standard error of the mean, i.e. std/sqrt(n) --
the relevant quantity for judging whether a mean is distinguishable from
1.0x, as opposed to std which describes spread between individual samples).

isolated_block_edge_ratios() runs the IDENTICAL boundary-vs-baseline
measurement (same BASELINE_OFFSETS, same single-voxel-at-the-edge
definition as seam_bump_profile_3d) on every INDIVIDUAL cached block's own
TWO physical faces along each of its 3 axes -- no stitching, no neighbour,
nothing else involved. Using the exact same definition (not a wider
edge-band-vs-centre-band average) makes this an apples-to-apples control:
if this ratio matches the stitched-boundary ratios above, the effect is
generic to independent training, not a stitching artefact.

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/data/seam_bump_statistics.py \\
        --gaussian-json fafb_pilot/code/data/gaussians.json \\
        --n-samples 8

STEPS
-----
1. Precompute every pilot-grid block's own reconstruction ONCE
   (precompute_block_cache).
2. Isolated-block control: edge-vs-centre ratio on every cached block, no
   stitching involved (isolated_block_edge_ratios).
3. For each arrangement (2/4/8 blocks): sample --n-samples placements
   (sample_placements), assemble each from the cache (assemble_from_cache),
   and measure full-volume boundary/baseline ratios per split axis
   (seam_bump_profile_3d) plus the corner-region ratio where applicable
   (corner_region_stat).
4. Aggregate every ratio list to mean +/- SEM (n samples).
5. Save one summary figure (bar chart, mean +/- SEM, isolated-block control
   as a reference line) and one metrics CSV + xlsx with the full per-group
   n/mean/std/SEM breakdown.

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  seam_bump_stats_<name>.png   mean +/- SEM bar chart, all groups + control
  seam_bump_stats_<name>.csv   one row per group: n, mean, std, sem
  seam_bump_stats_<name>.xlsx  same rows, as a spreadsheet

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (fused kernel path, see scripts/_3dgs/_3dgs.py)
  - --gaussian-json and --blocks-dir are mutually exclusive, same convention
    as stitch_blocks.py/rec_vol.py. --gaussian-json is strongly recommended:
    the precompute step loads every block in the pilot grid once regardless.
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
from libs import load_gaussians_json, save_metrics_excel

import stitch_blocks as sb  # reuses load_block/reconstruct_block/AABB/VolumeDataset

BLOCK_VOXELS = 64  # native per-block resolution -- boundaries always fall at a multiple of this
BASELINE_OFFSETS = [-8, -7, -6, -5, 5, 6, 7, 8]  # voxels away from a boundary counted as "baseline"
CORNER_WINDOW = 2   # +/- voxels around a corner/edge counted as the "corner region"

ARRANGEMENTS = [
    ("2blocks", (1, 1, 2)),
    ("4blocks", (1, 2, 2)),
    ("8blocks", (2, 2, 2)),
]
AXIS_NAMES = ('z', 'y', 'x')


# ── Step 1: precompute every pilot-grid block's own reconstruction once ────

def precompute_block_cache(n_per_axis, blocks_dir, data_dir, z0, y0, x0, checkpoint_name, device, cfg,
                           gaussians_json_data):
    """{(iz,iy,ix): (gt_vol, pred_vol)} for every block in the
    [0,n_per_axis)^3 pilot grid -- computed ONCE, reused for every placement
    sampled below (a block's own reconstruction doesn't depend on which
    stitched arrangement it ends up in)."""
    aabb = sb.AABB.unit()
    cache = {}
    total = n_per_axis ** 3
    print(f"Precomputing {total} block reconstructions (cached, reused across every placement sample)...")
    for iz in range(n_per_axis):
        for iy in range(n_per_axis):
            for ix in range(n_per_axis):
                bname, gc, vol_t = sb.load_block(
                    blocks_dir, data_dir, z0, y0, x0, iz, iy, ix, checkpoint_name,
                    aabb, device, cfg, gaussians_json_data=gaussians_json_data,
                )
                dataset = sb.VolumeDataset(vol_t, aabb, cfg)
                pred_vol = sb.reconstruct_block(gc, dataset, device, cfg.chunk_n)
                gt_vol = vol_t.detach().cpu().numpy()
                cache[(iz, iy, ix)] = (gt_vol, pred_vol)
    print(f"Cached {len(cache)} blocks.")
    return cache


# ── Step 2: isolated single-block edge-vs-centre control (no stitching) ────

def isolated_block_edge_ratios(cache):
    """Same measurement definition as seam_bump_profile_3d (single-voxel
    value at a boundary vs. a baseline band BASELINE_OFFSETS away) --
    applied to a lone block's own TWO physical faces (index 0 and
    BLOCK_VOXELS-1) along each of its 3 axes, independently. No stitching,
    no neighbouring block -- this is the control: using the identical
    boundary-vs-baseline definition means the ratio here is directly
    comparable to the stitched-boundary ratios below, isolating whether
    "worse near a boundary" is inherent to independent training rather than
    something stitching itself introduces."""
    lo_offsets = [o for o in BASELINE_OFFSETS if o > 0]        # baseline for the index-0 face
    hi_offsets = [-o for o in lo_offsets]                      # baseline for the index-(N-1) face
    ratios = []
    for (gt_vol, pred_vol) in cache.values():
        diff = np.abs(pred_vol - gt_vol)
        for axis in range(3):
            other_axes = tuple(a for a in range(3) if a != axis)
            profile = diff.mean(axis=other_axes)

            baseline_lo = float(np.mean([profile[o] for o in lo_offsets]))
            if baseline_lo > 0:
                ratios.append(float(profile[0]) / baseline_lo)

            baseline_hi = float(np.mean([profile[o] for o in hi_offsets]))
            if baseline_hi > 0:
                ratios.append(float(profile[-1]) / baseline_hi)
    return ratios


# ── Step 3: sample placements + assemble from cache + measure ───────────────

def sample_placements(n_per_axis_available, grid, n_samples, seed):
    """Up to n_samples distinct, non-overlapping-with-grid-bounds corners
    (iz,iy,ix) such that placing `grid` there stays inside the
    [0,n_per_axis_available)^3 pilot grid."""
    nz, ny, nx = grid
    max_cz = n_per_axis_available - nz
    max_cy = n_per_axis_available - ny
    max_cx = n_per_axis_available - nx
    all_corners = [
        (cz, cy, cx)
        for cz in range(max_cz + 1)
        for cy in range(max_cy + 1)
        for cx in range(max_cx + 1)
    ]
    rng = random.Random(seed)
    n = min(n_samples, len(all_corners))
    return rng.sample(all_corners, n)


def assemble_from_cache(cache, corner, grid):
    """Combined (gt_cube, rec_cube) for one placement, built purely from
    already-computed per-block volumes in `cache` -- no GPU work."""
    cz, cy, cx = corner
    nz, ny, nx = grid
    gt_cube = np.zeros((nz * BLOCK_VOXELS, ny * BLOCK_VOXELS, nx * BLOCK_VOXELS), dtype=np.float32)
    rec_cube = np.zeros_like(gt_cube)
    for dz in range(nz):
        for dy in range(ny):
            for dx in range(nx):
                gt_vol, pred_vol = cache[(cz + dz, cy + dy, cx + dx)]
                sz = slice(dz * BLOCK_VOXELS, (dz + 1) * BLOCK_VOXELS)
                sy = slice(dy * BLOCK_VOXELS, (dy + 1) * BLOCK_VOXELS)
                sx = slice(dx * BLOCK_VOXELS, (dx + 1) * BLOCK_VOXELS)
                gt_cube[sz, sy, sx] = gt_vol
                rec_cube[sz, sy, sx] = pred_vol
    return gt_cube, rec_cube


def seam_bump_profile_3d(diff_cube, axis, boundary_idx):
    """Boundary/baseline ratio along `axis`, averaged over the FULL volume
    (both other axes, every voxel) -- not one 2D slice."""
    other_axes = tuple(a for a in range(3) if a != axis)
    profile = diff_cube.mean(axis=other_axes)
    boundary_val = float(profile[boundary_idx])
    baseline_val = float(np.mean([profile[boundary_idx + o] for o in BASELINE_OFFSETS]))
    return boundary_val / baseline_val if baseline_val > 0 else float('inf')


def corner_region_stat(diff_cube, grid, boundary_idx):
    """Small-neighbourhood-around-the-corner ratio vs. the cube's overall
    mean. None where fewer than 2 axes are split (no corner/edge exists)."""
    split_axes = [a for a, n in enumerate(grid) if n > 1]
    if len(split_axes) < 2:
        return None
    lo, hi = boundary_idx - CORNER_WINDOW, boundary_idx + CORNER_WINDOW + 1
    slices = tuple(slice(lo, hi) if a in split_axes else slice(None) for a in range(3))
    region_mean = float(diff_cube[slices].mean())
    overall_mean = float(diff_cube.mean())
    return region_mean / overall_mean if overall_mean > 0 else float('inf')


# ── Step 4: aggregate to mean +/- SEM ────────────────────────────────────────

def summarize(ratios):
    arr = np.asarray(ratios, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    return {'n': n, 'mean': mean, 'std': std, 'sem': sem}


# ── Step 5: save outputs ─────────────────────────────────────────────────────

def save_figure(groups, output_dir, name):
    """groups: list of (label, summary_dict, color)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [g[0] for g in groups]
    means = [g[1]['mean'] for g in groups]
    sems = [g[1]['sem'] for g in groups]
    colors = [g[2] for g in groups]
    ns = [g[1]['n'] for g in groups]

    x = np.arange(len(groups))
    ax.bar(x, means, yerr=sems, capsize=4, color=colors)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.6, label='no elevation (1x)')
    for xi, mean, n in zip(x, means, ns):
        ax.text(xi, mean + 0.05, f"n={n}", ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
    ax.set_ylabel("ratio vs. baseline / centre (mean +/- SEM)")
    ax.set_title(
        "Edge/boundary error elevation: isolated (non-stitched) blocks vs. stitched arrangements\n"
        "(full-volume-averaged, multiple sampled placements per arrangement)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"seam_bump_stats_{name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_metrics(groups, output_dir, name):
    rows = [{'group': label, **summary} for label, summary, _color in groups]
    out_csv = os.path.join(output_dir, f"seam_bump_stats_{name}.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['group', 'n', 'mean', 'std', 'sem'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")
    save_metrics_excel(rows, output_dir, f"seam_bump_stats_{name}.xlsx")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-grade seam-bump statistics: multiple sampled placements, "
                    "full-volume averaging, and an isolated-block control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blocks-dir", default=None,
                        help="Directory containing b_<iz><iy><ix>/<checkpoint-name> "
                             "subdirectories. Mutually exclusive with --gaussian-json.")
    parser.add_argument("--gaussian-json", default=None,
                        help="Path to a gaussians_json.py-style export covering the "
                             "pilot grid -- alternative to --blocks-dir/--checkpoint-name.")
    parser.add_argument("--n-per-axis-available", type=int, default=4,
                        help="Blocks per axis available to sample placements from "
                             "(4 matches the full blocks_v18 pilot grid).")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Distinct placements sampled per arrangement (2/4/8-block).")
    parser.add_argument("--seed", type=int, default=0)
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

    # Step 1: precompute every pilot-grid block's own reconstruction once.
    cache = precompute_block_cache(
        args.n_per_axis_available, args.blocks_dir, args.data_dir,
        args.z0, args.y0, args.x0, args.checkpoint_name, device, cfg,
        gaussians_json_data=gaussians_json_data,
    )

    # Step 2: isolated-block control (no stitching involved at all).
    control_ratios = isolated_block_edge_ratios(cache)
    control_summary = summarize(control_ratios)
    print(f"\nIsolated-block edge-vs-centre control: n={control_summary['n']} "
          f"(blocks x axes), mean={control_summary['mean']:.3f} +/- SEM {control_summary['sem']:.3f}")

    groups = [("isolated block\n(no stitching)", control_summary, 'gray')]

    # Steps 3-4: sample placements per arrangement, measure, aggregate.
    for label, grid in ARRANGEMENTS:
        print(f"\n=== {label} ({grid[0]}x{grid[1]}x{grid[2]}) ===")
        placements = sample_placements(args.n_per_axis_available, grid, args.n_samples, args.seed)
        print(f"  sampled {len(placements)} placements: {placements}")

        axis_ratios = {AXIS_NAMES[a]: [] for a in range(3) if grid[a] > 1}
        corner_ratios = []
        for corner in placements:
            gt_cube, rec_cube = assemble_from_cache(cache, corner, grid)
            diff_cube = np.abs(rec_cube - gt_cube)
            for a in range(3):
                if grid[a] > 1:
                    axis_ratios[AXIS_NAMES[a]].append(seam_bump_profile_3d(diff_cube, a, BLOCK_VOXELS))
            corner_stat = corner_region_stat(diff_cube, grid, BLOCK_VOXELS)
            if corner_stat is not None:
                corner_ratios.append(corner_stat)

        color = {"2blocks": "tab:blue", "4blocks": "tab:orange", "8blocks": "tab:green"}[label]
        for axis_name, ratios in axis_ratios.items():
            s = summarize(ratios)
            print(f"  {axis_name}-seam: n={s['n']} mean={s['mean']:.3f} +/- SEM {s['sem']:.3f}")
            groups.append((f"{label}\n{axis_name}-seam", s, color))
        if corner_ratios:
            s = summarize(corner_ratios)
            print(f"  corner region: n={s['n']} mean={s['mean']:.3f} +/- SEM {s['sem']:.3f}")
            groups.append((f"{label}\ncorner", s, 'tab:red'))

    # Step 5: save outputs.
    save_figure(groups, args.output_dir, args.name)
    save_metrics(groups, args.output_dir, args.name)


if __name__ == "__main__":
    main()
