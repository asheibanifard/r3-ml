"""
Compares the two multi-block stitching strategies produced by
stitch_experiment.cu, across 2/4/8-block arrangements (2x1x1, 2x2x1, 2x2x2):

  - Baked-then-stitch: each block baked independently (hard-gated, no
    cross-block blending) onto its own 64^3 grid, placed into a combined
    128^3 grid, rendered with the same DVR/MIP kernel used for ground truth
    everywhere else in this project.
  - Gaussian-stitch: all blocks' Gaussians concatenated (with means shifted
    to each block's world position) and rendered directly with the
    tile-based rasterizer -- no intermediate grid, no per-block boundary.

Produces, per block count: GPU FPS for both methods (already measured by the
CUDA program via repeated GPU-event benchmarking), and PSNR/SSIM/LPIPS of
gaussian-stitch against baked-stitch (the higher-fidelity, grid-evaluated
reference) averaged over the 60-frame camera sweep -- plus a representative
frame comparison figure and summary plots across block counts.

USAGE
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/scratch_gs/compare_stitching.py
"""
import argparse
import glob
import os
import struct
from pathlib import Path

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib.pyplot as plt

# Categorical palette (validated for CVD-safety earlier this session): blue/green
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
COLOR_BAKED = "#2a78d6"
COLOR_RASTER = "#008300"


def read_frame(path):
    with open(path, "rb") as f:
        w = struct.unpack("<i", f.read(4))[0]
        h = struct.unpack("<i", f.read(4))[0]
        data = np.frombuffer(f.read(w * h * 4), dtype="<f4").reshape(h, w)
    return data.copy()


def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def lpips_2d(a, b, lpips_fn):
    at = torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
    bt = torch.from_numpy(b).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        return float(lpips_fn(at, bt).item())


def read_fps_summary(path):
    values = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            k, v = parts
            try:
                values[k] = float(v)
            except ValueError:
                values[k] = v
    return values


def main():
    ap = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    ap.add_argument("--base_dir", type=Path, default=script_dir)
    ap.add_argument("--block_counts", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--out_dir", type=Path, default=script_dir / "results_stitching")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lpips_fn = lpips.LPIPS(net="alex").eval()

    rows = []
    for n in args.block_counts:
        frames_dir = args.base_dir / f"frames_stitch{n}"
        fps = read_fps_summary(frames_dir / "fps_summary.txt")

        baked_paths = sorted(glob.glob(str(frames_dir / "baked_*.bin")))
        psnr_vals, ssim_vals, lpips_vals = [], [], []
        for bp in baked_paths:
            tag = os.path.basename(bp).split("_", 1)[1]
            rp = frames_dir / f"raster_{tag}"
            if not rp.exists():
                continue
            baked = read_frame(bp)
            raster = read_frame(str(rp))
            psnr_vals.append(psnr(baked, raster))
            ssim_vals.append(ssim_fn(baked, raster, data_range=1.0))
            lpips_vals.append(lpips_2d(baked, raster, lpips_fn))

        row = {
            "n_blocks": n,
            "n_gaussians": int(fps["n_gaussians_total"]),
            "baked_fps": fps["baked_stitch_fps"],
            "raster_fps": fps["gaussian_stitch_fps"],
            "bake_time_s": fps["bake_time_seconds"],
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "lpips": float(np.mean(lpips_vals)),
        }
        rows.append(row)
        print(f"n_blocks={n:2d}  n_gaussians={row['n_gaussians']:5d}  "
              f"baked_fps={row['baked_fps']:8.1f}  raster_fps={row['raster_fps']:7.1f}  "
              f"PSNR={row['psnr']:5.2f}  SSIM={row['ssim']:.4f}  LPIPS={row['lpips']:.4f}")

    # ---- CSV ----
    csv_path = args.out_dir / "stitching_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("n_blocks,n_gaussians,baked_fps,raster_fps,bake_time_s,psnr_db,ssim,lpips\n")
        for r in rows:
            f.write(f"{r['n_blocks']},{r['n_gaussians']},{r['baked_fps']:.3f},{r['raster_fps']:.3f},"
                    f"{r['bake_time_s']:.4f},{r['psnr']:.4f},{r['ssim']:.4f},{r['lpips']:.4f}\n")
    print(f"Saved {csv_path}")

    # ---- FPS vs block count ----
    ns = [r["n_blocks"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.plot(ns, [r["baked_fps"] for r in rows], "s-", color=COLOR_BAKED, linewidth=2, markersize=9, label="Baked-then-stitch")
    ax.plot(ns, [r["raster_fps"] for r in rows], "^-", color=COLOR_RASTER, linewidth=2, markersize=9, label="Gaussian-stitch")
    ax.set_xticks(ns)
    ax.set_xlabel("Number of stitched blocks", color=PRIMARY_INK)
    ax.set_ylabel("GPU FPS", color=PRIMARY_INK)
    ax.set_title("Stitching throughput vs. block count", color=PRIMARY_INK, fontweight="bold")
    ax.set_yscale("log")
    ax.tick_params(colors=MUTED_INK)
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    fig.savefig(args.out_dir / "fps_vs_block_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_dir / 'fps_vs_block_count.png'}")

    # ---- Quality vs block count ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    axes[0].plot(ns, [r["psnr"] for r in rows], "o-", color=COLOR_RASTER, linewidth=2, markersize=9)
    axes[0].set_title("PSNR (gaussian-stitch vs. baked-stitch)"); axes[0].set_xlabel("Number of blocks"); axes[0].set_ylabel("dB")
    axes[1].plot(ns, [r["ssim"] for r in rows], "o-", color=COLOR_RASTER, linewidth=2, markersize=9)
    axes[1].set_title("SSIM"); axes[1].set_xlabel("Number of blocks")
    axes[2].plot(ns, [r["lpips"] for r in rows], "o-", color=COLOR_RASTER, linewidth=2, markersize=9)
    axes[2].set_title("LPIPS (lower better)"); axes[2].set_xlabel("Number of blocks")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set_xticks(ns)
        ax.tick_params(colors=MUTED_INK)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
    plt.tight_layout()
    fig.savefig(args.out_dir / "quality_vs_block_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_dir / 'quality_vs_block_count.png'}")

    # ---- Summary table (image) ----
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.5 * len(rows)))
    ax.axis("off")
    header = ["Blocks", "Gaussians", "Baked FPS", "Raster FPS", "PSNR (dB)", "SSIM", "LPIPS"]
    table_data = [header]
    for r in rows:
        table_data.append([
            str(r["n_blocks"]), str(r["n_gaussians"]),
            f"{r['baked_fps']:.1f}", f"{r['raster_fps']:.1f}",
            f"{r['psnr']:.2f}", f"{r['ssim']:.4f}", f"{r['lpips']:.4f}",
        ])
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for j in range(len(header)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")
    plt.tight_layout()
    fig.savefig(args.out_dir / "stitching_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_dir / 'stitching_table.png'}")

    # ---- Representative frame comparison, one per block count ----
    fig, axes = plt.subplots(len(args.block_counts), 3, figsize=(15, 5 * len(args.block_counts)))
    if len(args.block_counts) == 1:
        axes = axes[None, :]
    for i, n in enumerate(args.block_counts):
        frames_dir = args.base_dir / f"frames_stitch{n}"
        baked = read_frame(str(frames_dir / "baked_0000.bin"))
        raster = read_frame(str(frames_dir / "raster_0000.bin"))
        diff = np.abs(baked - raster)
        im0 = axes[i, 0].imshow(baked, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{n} blocks: Baked-then-stitch")
        im1 = axes[i, 1].imshow(raster, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"{n} blocks: Gaussian-stitch")
        im2 = axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 1e-8))
        p = psnr(baked, raster)
        axes[i, 2].set_title(f"|Diff| (PSNR={p:.2f} dB)")
        for ax, im in [(axes[i, 0], im0), (axes[i, 1], im1), (axes[i, 2], im2)]:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(args.out_dir / "frame_comparison_all_blocks.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_dir / 'frame_comparison_all_blocks.png'}")

    print(f"\nAll outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
