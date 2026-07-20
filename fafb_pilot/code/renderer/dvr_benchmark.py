"""
Rendering benchmark: ground truth vs. THREE variants of "our model", across
3 volume sizes and multiple screen resolutions.

BACKGROUND
----------
An earlier version of this script only compared GT against our BOUNDED
reconstruction (rec_size<N>.tif) rendered through the same dense_voxel
algorithm as GT -- that shows the Gaussian model's own reconstruction
fidelity, but never actually renders the live Gaussian model itself. This
version adds the two real Gaussian-rendering paths built earlier this
session, so all four ways of looking at this scene are compared side by
side, at the same resolutions, against the same ground truth:

  gt               Real EM tissue, dense_voxel DVR (ray-marching MIP
                    through a hardware 3D texture -- "ray tracing from all
                    voxels"). The reference everything else is scored against.
  ours_bounded      Our hard-gated, per-voxel-clamped Gaussian
                    reconstruction (rec_size<N>.tif), rendered through the
                    SAME dense_voxel algorithm as GT. Fast, faithful (see
                    fidelity_comparison_size*.png), but requires the
                    reconstruction to already be baked to a voxel grid.
  ours_raymarched   The LIVE Gaussian mixture itself, ray-marched
                    (pretrained_gaussian_hard_gated) -- no baking, but
                    evaluates the continuous field off the training grid,
                    which earlier measurements showed costs real fidelity.
  ours_rasterized   The LIVE Gaussian mixture, rasterized instead of
                    ray-marched (pretrained_gaussian_rasterized) -- same
                    fidelity trade-off as ours_raymarched, faster per frame.

A practical constraint shapes this script: ray-marching the live Gaussian
mixture was already measured at 1.06 FPS for the 256-block scene at just
128x128 (see the renderer performance work earlier this session). At
1024x1024 (64x more pixels) that would take on the order of tens of minutes
PER FRAME -- impractical to include in an automated sweep. So the two live
Gaussian paths are only benchmarked at GAUSSIAN_SCREEN_SIZES (128/256, both
tractable), with a smaller GAUSSIAN_BENCHMARK_FRAMES, while gt/ours_bounded
(both cheap dense-texture sampling) run the full SCREEN_SIZES sweep up to
1024x1024 at the full BENCHMARK_FRAMES. This asymmetry is itself a finding,
not just a script limitation -- see the printed summary at the end.

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/dvr_benchmark.py

STEPS
-----
1. For each volume size (64/128/256): export gt_size<N>.tif, rec_size<N>.tif
   (both to dense_voxel format) and gaussian_size<N>.pth (to pretrained_
   gaussian format) -- export_volume / export_gaussians.
2. Render GT and ours_bounded through dense_voxel at every screen size in
   SCREEN_SIZES; render ours_raymarched and ours_rasterized through their
   own representation types at every screen size in GAUSSIAN_SCREEN_SIZES
   (render_dense_voxel / render_gaussian).
3. Score every non-GT representation against GT at the matching (volume
   size, screen size): PSNR, SSIM, LPIPS (libs.slice_metrics).
4. Save one long-format CSV + xlsx (one row per representation x volume
   size x screen size) and two comparison figures (FPS; PSNR/SSIM/LPIPS),
   one line per representation per volume size.

OUTPUT FILES (written to --output-dir, named by --name)
---------------------------------------------------------
  dvr_benchmark.csv / .xlsx    one row per (representation, volume_size, screen_size)
  dvr_benchmark_fps.png        FPS vs. screen size, all 4 representations, per volume size
  dvr_benchmark_quality.png    PSNR/SSIM/LPIPS vs. screen size (non-GT representations only)

REQUIREMENTS
------------
  - Run with the project's venv: /venv/r3-ml/bin/python3
  - Needs a CUDA GPU (LPIPS network + the renderer itself)
  - Requires gt_size<N>.tif, rec_size<N>.tif, and gaussian_size<N>.pth to
    already exist (produced by stitch_blocks.py --n-per-axis {1,2,4} --name
    size{64,128,256}) for each of the 3 volume sizes -- run stitch_blocks.py
    first if any are missing; this script does not build them itself.
  - Requires Mip_Render_Inside_Volume to already be compiled (run.sh or
    bake_and_render.sh's nvcc step) in this directory.
"""
import csv
import os
import re
import subprocess
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fafb_pilot/code
import libs  # sets up sys.path to scripts/ + TORCH_CUDA_ARCH_LIST (lightweight -- no _3dgs import here)
from libs import slice_metrics, save_metrics_excel

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_EXE = os.path.join(SCRIPT_DIR, "Mip_Render_Inside_Volume")
EXPORTER = os.path.join(SCRIPT_DIR, "export_renderer_bin.py")
BIN_DIR = os.path.join(SCRIPT_DIR, "bins")
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
STITCH_DATA_DIR = "/root/project/fafb_pilot/results/data"

VOLUME_SIZES = [64, 128, 256]
N_PER_AXIS = {64: 1, 128: 2, 256: 4}

SCREEN_SIZES = [128, 256, 512, 1024]           # gt / ours_bounded (dense_voxel -- cheap)
GAUSSIAN_SCREEN_SIZES = [128, 256]             # ours_raymarched / ours_rasterized (live Gaussian -- expensive)

RAY_SAMPLES = 64
BENCHMARK_FRAMES = 200                          # dense_voxel: cheap, full precision
GAUSSIAN_BENCHMARK_FRAMES = 5                   # live Gaussian paths: kept small, see module docstring

FPS_PATTERN = re.compile(r"FPS:\s*([\d.]+)")


# ── Step 1: export volumes / Gaussians to the renderer's binary formats ────

def export_volume(tif_path, bin_path):
    """Bounded [0,1] .tif -> dense_voxel binary (export_renderer_bin.py, as
    a subprocess so this script doesn't need torch's training-side deps)."""
    subprocess.run(
        [sys.executable, EXPORTER, "dense_voxel", tif_path, bin_path, "--normalise", "none"],
        check=True, capture_output=True, text=True,
    )


def export_gaussians(pth_path, bin_path):
    """Combined Gaussian checkpoint -> pretrained_gaussian binary (same
    flags used throughout this project's renderer scripts)."""
    subprocess.run(
        [
            sys.executable, EXPORTER, "pretrained_gaussian", pth_path, bin_path,
            "--means-key", "means", "--scales-key", "log_scales",
            "--quaternions-key", "quats", "--intensity-key", "intensities",
            "--scale-activation", "exp", "--intensity-activation", "softplus",
            "--quaternion-order", "wxyz",
        ],
        check=True, capture_output=True, text=True,
    )


# ── Step 2: render, parse FPS ────────────────────────────────────────────────

def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        color = header == 'PF'
        dims = f.readline().decode().strip()
        while dims.startswith('#'):
            dims = f.readline().decode().strip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f', width * height * (3 if color else 1))
        data = np.reshape(data, (height, width, 3) if color else (height, width))
        return np.ascontiguousarray(np.flipud(data))


def _run_renderer(args):
    result = subprocess.run([CUDA_EXE] + args, check=True, capture_output=True, text=True)
    match = FPS_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError(f"Could not find FPS in renderer output:\n{result.stdout}\n{result.stderr}")
    return float(match.group(1))


def render_dense_voxel(bin_path, out_pfm, screen_size):
    """dense_voxel DVR (ray-marching MIP through a hardware 3D texture --
    "ray tracing from all voxels") at screen_size x screen_size."""
    fps = _run_renderer([
        "dense_voxel", bin_path, out_pfm,
        str(screen_size), str(screen_size),
        str(RAY_SAMPLES), str(BENCHMARK_FRAMES),
        "0", "0", "0", "90", "-1", "-1", "-1", "1", "1", "1",
    ])
    return read_pfm(out_pfm), fps


def render_gaussian(mode, bin_path, out_pfm, screen_size, n_per_axis):
    """Render the LIVE Gaussian mixture -- mode is 'pretrained_gaussian_hard_gated'
    (ray-marched) or 'pretrained_gaussian_rasterized'. Both need the trailing
    n_per_axis argument; GAUSSIAN_BENCHMARK_FRAMES keeps this tractable (see
    module docstring)."""
    fps = _run_renderer([
        mode, bin_path, out_pfm,
        str(screen_size), str(screen_size),
        str(RAY_SAMPLES), str(GAUSSIAN_BENCHMARK_FRAMES),
        "0", "0", "0", "90", "-1", "-1", "-1", "1", "1", "1",
        str(n_per_axis),
    ])
    return read_pfm(out_pfm), fps


# ── Steps 1-3: run the full experiment ──────────────────────────────────────

def run_experiment(device):
    rows = []
    for volume_size in VOLUME_SIZES:
        n_per_axis = N_PER_AXIS[volume_size]
        gt_tif = os.path.join(STITCH_DATA_DIR, f"gt_size{volume_size}.tif")
        rec_tif = os.path.join(STITCH_DATA_DIR, f"rec_size{volume_size}.tif")
        gaussian_pth = os.path.join(STITCH_DATA_DIR, f"gaussian_size{volume_size}.pth")
        for path in (gt_tif, rec_tif, gaussian_pth):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found -- run stitch_blocks.py for size {volume_size} first."
                )

        gt_bin = os.path.join(BIN_DIR, f"dvr_gt_size{volume_size}.bin")
        rec_bin = os.path.join(BIN_DIR, f"dvr_rec_size{volume_size}.bin")
        gaussian_bin = os.path.join(BIN_DIR, f"dvr_gaussian_size{volume_size}.bin")
        print(f"\n=== Exporting volume_size={volume_size} (n_per_axis={n_per_axis}) ===")
        export_volume(gt_tif, gt_bin)
        export_volume(rec_tif, rec_bin)
        export_gaussians(gaussian_pth, gaussian_bin)

        # --- gt / ours_bounded: cheap dense_voxel, full screen-size sweep ---
        for screen_size in SCREEN_SIZES:
            print(f"--- volume_size={volume_size}, screen_size={screen_size}: gt / ours_bounded ---")
            gt_pfm = os.path.join(RESULT_DIR, f"dvr_gt_v{volume_size}_s{screen_size}.pfm")
            rec_pfm = os.path.join(RESULT_DIR, f"dvr_ours_bounded_v{volume_size}_s{screen_size}.pfm")

            gt_image, gt_fps = render_dense_voxel(gt_bin, gt_pfm, screen_size)
            rows.append({'representation': 'gt', 'volume_size': volume_size, 'screen_size': screen_size,
                        'fps': gt_fps, 'PSNR': float('nan'), 'SSIM': float('nan'), 'LPIPS': float('nan')})

            rec_image, ours_fps = render_dense_voxel(rec_bin, rec_pfm, screen_size)
            m = slice_metrics(rec_image, gt_image, device=device)
            print(f"    gt FPS={gt_fps:,.0f}  ours_bounded FPS={ours_fps:,.0f}  "
                  f"PSNR={m['PSNR']:.2f} dB  SSIM={m['SSIM']:.4f}  LPIPS={m['LPIPS']:.4f}")
            rows.append({'representation': 'ours_bounded', 'volume_size': volume_size, 'screen_size': screen_size,
                        'fps': ours_fps, 'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m['LPIPS']})

            # --- ours_raymarched / ours_rasterized: only at the tractable screen sizes ---
            if screen_size in GAUSSIAN_SCREEN_SIZES:
                for label, mode in [
                    ('ours_raymarched', 'pretrained_gaussian_hard_gated'),
                    ('ours_rasterized', 'pretrained_gaussian_rasterized'),
                ]:
                    print(f"--- volume_size={volume_size}, screen_size={screen_size}: {label} ---")
                    out_pfm = os.path.join(RESULT_DIR, f"dvr_{label}_v{volume_size}_s{screen_size}.pfm")
                    image, fps = render_gaussian(mode, gaussian_bin, out_pfm, screen_size, n_per_axis)
                    m = slice_metrics(image, gt_image, device=device)
                    print(f"    {label} FPS={fps:,.1f}  "
                          f"PSNR={m['PSNR']:.2f} dB  SSIM={m['SSIM']:.4f}  LPIPS={m['LPIPS']:.4f}")
                    rows.append({'representation': label, 'volume_size': volume_size, 'screen_size': screen_size,
                                'fps': fps, 'PSNR': m['PSNR'], 'SSIM': m['SSIM'], 'LPIPS': m['LPIPS']})
    return rows


# ── Step 4: save outputs ─────────────────────────────────────────────────────

REPRESENTATIONS = ['gt', 'ours_bounded', 'ours_raymarched', 'ours_rasterized']
STYLE = {
    'gt': dict(color='black', marker='o', linestyle='-'),
    'ours_bounded': dict(color='tab:blue', marker='s', linestyle='--'),
    'ours_raymarched': dict(color='tab:orange', marker='^', linestyle=':'),
    'ours_rasterized': dict(color='tab:green', marker='d', linestyle='-.'),
}


def save_csv(rows, output_dir, name):
    out_csv = os.path.join(output_dir, f"{name}.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")
    save_metrics_excel(rows, output_dir, f"{name}.xlsx")


def save_fps_figure(rows, output_dir, name):
    fig, axes = plt.subplots(1, len(VOLUME_SIZES), figsize=(6 * len(VOLUME_SIZES), 5), sharey=True)
    for ax, volume_size in zip(axes, VOLUME_SIZES):
        for rep in REPRESENTATIONS:
            subset = sorted(
                [r for r in rows if r['volume_size'] == volume_size and r['representation'] == rep],
                key=lambda r: r['screen_size'],
            )
            if not subset:
                continue
            ax.plot([r['screen_size'] for r in subset], [r['fps'] for r in subset],
                    label=rep, **STYLE[rep])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(SCREEN_SIZES)
        ax.set_xticklabels(SCREEN_SIZES)
        ax.set_xlabel("Screen size (pixels, square)")
        ax.set_title(f"volume={volume_size}³")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel("FPS (log scale)")
    fig.suptitle("Rendering FPS by representation, screen size, and volume size")
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"{name}_fps.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def save_quality_figure(rows, output_dir, name):
    non_gt = [rep for rep in REPRESENTATIONS if rep != 'gt']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for rep in non_gt:
        for volume_size in VOLUME_SIZES:
            subset = sorted(
                [r for r in rows if r['volume_size'] == volume_size and r['representation'] == rep],
                key=lambda r: r['screen_size'],
            )
            if not subset:
                continue
            xs = [r['screen_size'] for r in subset]
            label = f"{rep}, volume={volume_size}³"
            axes[0].plot(xs, [r['PSNR'] for r in subset], marker='o', label=label)
            axes[1].plot(xs, [r['SSIM'] for r in subset], marker='o', label=label)
            axes[2].plot(xs, [r['LPIPS'] for r in subset], marker='o', label=label)

    for ax, title in zip(axes, ["PSNR (dB)", "SSIM", "LPIPS (lower is better)"]):
        ax.set_xscale('log', base=2)
        ax.set_xticks(SCREEN_SIZES)
        ax.set_xticklabels(SCREEN_SIZES)
        ax.set_xlabel("Screen size (pixels, square)")
        ax.set_title(f"vs. GT: {title}")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"{name}_quality.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rendering benchmark: GT vs. 3 variants of our model "
                    "(bounded/dense_voxel, live ray-marched, live rasterized), "
                    "across every (volume size, screen size) combination.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default=RESULT_DIR,
                        help="Directory to write the CSV/xlsx/figures into.")
    parser.add_argument("--name", default="dvr_benchmark",
                        help="Label used in output filenames.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(BIN_DIR, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    rows = run_experiment(device)

    save_csv(rows, args.output_dir, args.name)
    save_fps_figure(rows, args.output_dir, args.name)
    save_quality_figure(rows, args.output_dir, args.name)


if __name__ == "__main__":
    main()
