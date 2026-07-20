"""
Full comparison across 3 rendering kernels -- GT DVR, Baked+DVR, and the
tile-based Gaussian rasterizer -- across 5 screen sizes (64/128/256/512/1024),
at a fixed 64^3 voxel grid (screen size only affects rendering, not training,
so a single trained/baked model is reused for every screen size).

Reads the per-screen-size output directories written by
gaussian_splat_scratch (frames_s64/, frames_s128/, ... each with
gt_*.bin/baked_*.bin/rec_*.bin frames + fps_summary.txt) plus the one-time
volume_gt.bin/volume_baked.bin pair, and produces:

  - vol_psnr / vol_ssim / vol_lpips: baked reconstructed volume vs GT volume
    (3D, computed once; SSIM/LPIPS are slice-averaged over Z, matching this
    project's established whole-volume-metric convention -- LPIPS has no
    native 3D form, so slice-averaging is the standard way to extend a 2D
    perceptual metric to a volume)
  - screen_psnr / screen_ssim / screen_lpips: per screen size, comparing
    Baked+DVR and the rasterizer against GT DVR, averaged over all 60 frames
  - FPS for all 3 kernels, per screen size

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/scratch_gs/compare_sweep.py
"""
import argparse
import glob
import os
import struct

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib.pyplot as plt


def read_frame(path):
    with open(path, "rb") as f:
        w = struct.unpack("<i", f.read(4))[0]
        h = struct.unpack("<i", f.read(4))[0]
        data = np.frombuffer(f.read(w * h * 4), dtype="<f4").reshape(h, w)
    return data.copy()


def read_volume(path):
    with open(path, "rb") as f:
        g = struct.unpack("<i", f.read(4))[0]
        data = np.frombuffer(f.read(g * g * g * 4), dtype="<f4").reshape(g, g, g)
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


def volume_ssim_slice_avg(a, b):
    return float(np.mean([ssim_fn(a[z], b[z], data_range=1.0) for z in range(a.shape[0])]))


def volume_lpips_slice_avg(a, b, lpips_fn):
    return float(np.mean([lpips_2d(a[z], b[z], lpips_fn) for z in range(a.shape[0])]))


def screen_metrics(frames_dir, prefix_ref, prefix_test, lpips_fn):
    ref_paths = sorted(glob.glob(os.path.join(frames_dir, f"{prefix_ref}_*.bin")))
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for pr in ref_paths:
        frame_tag = os.path.basename(pr).split("_", 1)[1]
        pt = os.path.join(frames_dir, f"{prefix_test}_{frame_tag}")
        if not os.path.exists(pt):
            continue
        ref = read_frame(pr)
        test = read_frame(pt)
        psnr_vals.append(psnr(ref, test))
        ssim_vals.append(ssim_fn(ref, test, data_range=1.0))
        lpips_vals.append(lpips_2d(ref, test, lpips_fn))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), float(np.mean(lpips_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--screen_sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.base_dir, "results_sweep")
    os.makedirs(out_dir, exist_ok=True)

    lpips_fn = lpips.LPIPS(net="alex").eval()

    # ---- Volume-level metrics (once -- doesn't depend on screen size) ----
    gt_vol = read_volume(os.path.join(args.base_dir, "volume_gt.bin"))
    baked_vol = read_volume(os.path.join(args.base_dir, "volume_baked.bin"))
    vol_psnr = psnr(gt_vol, baked_vol)
    vol_ssim = volume_ssim_slice_avg(gt_vol, baked_vol)
    vol_lpips = volume_lpips_slice_avg(gt_vol, baked_vol, lpips_fn)
    print(f"Volume metrics (baked vs GT, 64^3): PSNR={vol_psnr:.2f} dB  SSIM={vol_ssim:.4f}  LPIPS={vol_lpips:.4f}")

    # ---- Per-screen-size metrics ----
    rows = []
    for size in args.screen_sizes:
        frames_dir = os.path.join(args.base_dir, f"frames_s{size}")
        fps = {}
        with open(os.path.join(frames_dir, "fps_summary.txt")) as f:
            for line in f:
                k, v = line.split()
                fps[k] = float(v)
        baked_psnr, baked_ssim, baked_lpips = screen_metrics(frames_dir, "gt", "baked", lpips_fn)
        rast_psnr, rast_ssim, rast_lpips = screen_metrics(frames_dir, "gt", "rec", lpips_fn)
        row = {
            "screen_size": size,
            "dvr_fps": fps["dvr_fps"], "baked_fps": fps["baked_fps"], "rasterizer_fps": fps["rasterizer_fps"],
            "dvr_gpu_fps": fps.get("dvr_gpu_fps", float("nan")),
            "baked_gpu_fps": fps.get("baked_gpu_fps", float("nan")),
            "rasterizer_gpu_fps": fps.get("rasterizer_gpu_fps", float("nan")),
            "baked_psnr": baked_psnr, "baked_ssim": baked_ssim, "baked_lpips": baked_lpips,
            "rast_psnr": rast_psnr, "rast_ssim": rast_ssim, "rast_lpips": rast_lpips,
        }
        rows.append(row)
        print(f"size={size:5d}  wall[dvr={fps['dvr_fps']:8.1f} baked={fps['baked_fps']:8.1f} rast={fps['rasterizer_fps']:7.1f}]  "
              f"gpu[dvr={row['dvr_gpu_fps']:8.1f} baked={row['baked_gpu_fps']:8.1f} rast={row['rasterizer_gpu_fps']:7.1f}]  "
              f"baked[PSNR={baked_psnr:5.2f} SSIM={baked_ssim:.4f} LPIPS={baked_lpips:.4f}]  "
              f"rast[PSNR={rast_psnr:5.2f} SSIM={rast_ssim:.4f} LPIPS={rast_lpips:.4f}]")

    # ---- CSV ----
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    with open(csv_path, "w") as f:
        f.write("screen_size,dvr_fps,baked_fps,rasterizer_fps,dvr_gpu_fps,baked_gpu_fps,rasterizer_gpu_fps,"
                "vol_psnr,vol_ssim,vol_lpips,"
                "baked_psnr,baked_ssim,baked_lpips,rast_psnr,rast_ssim,rast_lpips\n")
        for r in rows:
            f.write(f"{r['screen_size']},{r['dvr_fps']:.3f},{r['baked_fps']:.3f},{r['rasterizer_fps']:.3f},"
                    f"{r['dvr_gpu_fps']:.3f},{r['baked_gpu_fps']:.3f},{r['rasterizer_gpu_fps']:.3f},"
                    f"{vol_psnr:.4f},{vol_ssim:.4f},{vol_lpips:.4f},"
                    f"{r['baked_psnr']:.4f},{r['baked_ssim']:.4f},{r['baked_lpips']:.4f},"
                    f"{r['rast_psnr']:.4f},{r['rast_ssim']:.4f},{r['rast_lpips']:.4f}\n")
    print(f"Saved {csv_path}")

    # ---- FPS vs screen size plot: wall-clock vs GPU-only side by side ----
    # (GPU-only, via CUDA events, is the number that reflects actual rendering
    # cost -- wall-clock also includes CPU-side kernel-launch dispatch and the
    # cudaDeviceSynchronize round-trip. Plotted together since for this
    # workload they turn out to closely track each other at every size, which
    # is itself informative: it means the FPS trend is real GPU behavior, not
    # a host-dispatch-overhead artifact.)
    sizes = [r["screen_size"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sizes, [r["dvr_fps"] for r in rows], "o-", label="GT DVR")
    axes[0].plot(sizes, [r["baked_fps"] for r in rows], "s-", label="Baked+DVR")
    axes[0].plot(sizes, [r["rasterizer_fps"] for r in rows], "^-", label="Gaussian rasterizer")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xticks(sizes); axes[0].set_xticklabels([str(s) for s in sizes])
    axes[0].set_xlabel("Screen size (pixels, square)"); axes[0].set_ylabel("FPS")
    axes[0].set_title("Wall-clock FPS"); axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(sizes, [r["dvr_gpu_fps"] for r in rows], "o-", label="GT DVR")
    axes[1].plot(sizes, [r["baked_gpu_fps"] for r in rows], "s-", label="Baked+DVR")
    axes[1].plot(sizes, [r["rasterizer_gpu_fps"] for r in rows], "^-", label="Gaussian rasterizer")
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xticks(sizes); axes[1].set_xticklabels([str(s) for s in sizes])
    axes[1].set_xlabel("Screen size (pixels, square)"); axes[1].set_ylabel("FPS")
    axes[1].set_title("GPU-only FPS (CUDA events)"); axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("FPS vs screen size (64³ volume, 800 Gaussians)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fps_vs_screen_size.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'fps_vs_screen_size.png')}")

    # ---- Quality vs screen size plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(sizes, [r["baked_psnr"] for r in rows], "s-", label="Baked+DVR")
    axes[0].plot(sizes, [r["rast_psnr"] for r in rows], "^-", label="Rasterizer")
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(sizes); axes[0].set_xticklabels([str(s) for s in sizes])
    axes[0].set_title("Screen PSNR vs GT DVR"); axes[0].set_xlabel("Screen size"); axes[0].set_ylabel("dB"); axes[0].legend()

    axes[1].plot(sizes, [r["baked_ssim"] for r in rows], "s-", label="Baked+DVR")
    axes[1].plot(sizes, [r["rast_ssim"] for r in rows], "^-", label="Rasterizer")
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(sizes); axes[1].set_xticklabels([str(s) for s in sizes])
    axes[1].set_title("Screen SSIM vs GT DVR"); axes[1].set_xlabel("Screen size"); axes[1].legend()

    axes[2].plot(sizes, [r["baked_lpips"] for r in rows], "s-", label="Baked+DVR")
    axes[2].plot(sizes, [r["rast_lpips"] for r in rows], "^-", label="Rasterizer")
    axes[2].set_xscale("log", base=2); axes[2].set_xticks(sizes); axes[2].set_xticklabels([str(s) for s in sizes])
    axes[2].set_title("Screen LPIPS vs GT DVR (lower better)"); axes[2].set_xlabel("Screen size"); axes[2].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "quality_vs_screen_size.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'quality_vs_screen_size.png')}")

    # ---- Combined sweep table (image) ----
    fig, ax = plt.subplots(figsize=(20, 1.2 + 0.5 * len(rows)))
    ax.axis("off")
    header = ["Screen", "DVR FPS\n(wall)", "Baked FPS\n(wall)", "Raster FPS\n(wall)",
              "DVR FPS\n(GPU)", "Baked FPS\n(GPU)", "Raster FPS\n(GPU)",
              "Baked PSNR", "Baked SSIM", "Baked LPIPS", "Raster PSNR", "Raster SSIM", "Raster LPIPS"]
    table_data = [header]
    for r in rows:
        table_data.append([
            f"{r['screen_size']}", f"{r['dvr_fps']:.1f}", f"{r['baked_fps']:.1f}", f"{r['rasterizer_fps']:.1f}",
            f"{r['dvr_gpu_fps']:.1f}", f"{r['baked_gpu_fps']:.1f}", f"{r['rasterizer_gpu_fps']:.1f}",
            f"{r['baked_psnr']:.2f}", f"{r['baked_ssim']:.4f}", f"{r['baked_lpips']:.4f}",
            f"{r['rast_psnr']:.2f}", f"{r['rast_ssim']:.4f}", f"{r['rast_lpips']:.4f}",
        ])
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for j in range(len(header)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'sweep_table.png')}")

    # ---- Volume metrics, its own small table ----
    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.axis("off")
    vol_table = [["vol_PSNR (dB)", "vol_SSIM", "vol_LPIPS"],
                 [f"{vol_psnr:.2f}", f"{vol_ssim:.4f}", f"{vol_lpips:.4f}"]]
    t2 = ax.table(cellText=vol_table, loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(11)
    t2.scale(1, 1.8)
    for j in range(3):
        t2[(0, j)].set_facecolor("#4472C4")
        t2[(0, j)].set_text_props(color="white", weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "volume_metrics_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'volume_metrics_table.png')}")

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
