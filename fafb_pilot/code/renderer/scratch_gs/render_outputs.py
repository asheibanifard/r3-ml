"""
Companion script for gaussian_splat_scratch.cu.

Reads the per-frame GT (vanilla DVR on the voxel grid) and Reconstruction
(from-scratch Gaussian rasterizer) frames written by the CUDA program, and
produces:

  gt.mp4                    GT frame sequence
  reco.mp4                  Reconstruction frame sequence
  diff.mp4                  |GT - Reco| frame sequence
  gt_rec_diff_plots/*.png   GT | Reco | |Diff| plot for a handful of frames
  metrics_over_frames.png   PSNR / SSIM / LPIPS per frame
  metrics_table.png         Summary table (LPIPS, PSNR, SSIM, FPS)
  metrics_table.csv         Same table, machine-readable

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/scratch_gs/render_outputs.py \\
        --frames_dir fafb_pilot/code/renderer/scratch_gs/frames \\
        --out_dir    fafb_pilot/code/renderer/scratch_gs/results
"""
import argparse
import glob
import os
import struct
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
import torch
import lpips


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


def write_mp4(frames, path, fps=24):
    """frames: list of HxW float arrays in [0,1] -> grayscale mp4 via ffmpeg."""
    h, w = frames[0].shape
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray8",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for f in frames:
        img8 = np.clip(f, 0, 1) * 255.0
        proc.stdin.write(img8.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames"))
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--fps", type=int, default=24, help="playback fps for the output mp4s")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gt_paths = sorted(glob.glob(os.path.join(args.frames_dir, "gt_*.bin")))
    rec_paths = sorted(glob.glob(os.path.join(args.frames_dir, "rec_*.bin")))
    assert len(gt_paths) == len(rec_paths) and len(gt_paths) > 0, "No matching gt_/rec_ frame pairs found"
    n_frames = len(gt_paths)
    print(f"Found {n_frames} GT/Reconstruction frame pairs in {args.frames_dir}")

    gt_frames = [read_frame(p) for p in gt_paths]
    rec_frames = [read_frame(p) for p in rec_paths]
    diff_frames = [np.abs(g - r) for g, r in zip(gt_frames, rec_frames)]

    # ---- mp4 assembly ----
    write_mp4(gt_frames, os.path.join(args.out_dir, "gt.mp4"), fps=args.fps)
    write_mp4(rec_frames, os.path.join(args.out_dir, "reco.mp4"), fps=args.fps)
    diff_max = max(float(d.max()) for d in diff_frames) or 1.0
    write_mp4([d / diff_max for d in diff_frames], os.path.join(args.out_dir, "diff.mp4"), fps=args.fps)

    # ---- per-frame metrics (PSNR, SSIM, LPIPS) ----
    lpips_fn = lpips.LPIPS(net="alex").eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for g, r in zip(gt_frames, rec_frames):
        psnr_vals.append(psnr(g, r))
        ssim_vals.append(ssim_fn(g, r, data_range=1.0))
        g_t = torch.from_numpy(g).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
        r_t = torch.from_numpy(r).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lpips_vals.append(float(lpips_fn(g_t, r_t).item()))

    psnr_vals = np.array(psnr_vals)
    ssim_vals = np.array(ssim_vals)
    lpips_vals = np.array(lpips_vals)

    # ---- GT/Reco/Diff plots for a handful of representative frames ----
    plot_dir = os.path.join(args.out_dir, "gt_rec_diff_plots")
    os.makedirs(plot_dir, exist_ok=True)
    sample_idx = sorted(set([0, n_frames // 4, n_frames // 2, (3 * n_frames) // 4, n_frames - 1]))
    for i in sample_idx:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(gt_frames[i], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT (vanilla DVR, voxel grid)")
        axes[0].set_xlabel("pixel x"); axes[0].set_ylabel("pixel y")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(rec_frames[i], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Reconstruction (Gaussian rasterizer)")
        axes[1].set_xlabel("pixel x"); axes[1].set_ylabel("pixel y")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(diff_frames[i], cmap="hot", vmin=0, vmax=max(float(diff_frames[i].max()), 1e-8))
        axes[2].set_title(f"|Diff| (PSNR={psnr_vals[i]:.2f} dB, SSIM={ssim_vals[i]:.4f})")
        axes[2].set_xlabel("pixel x"); axes[2].set_ylabel("pixel y")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = os.path.join(plot_dir, f"frame_{i:04d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    # ---- metrics-over-frames plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(psnr_vals); axes[0].set_title("PSNR per frame"); axes[0].set_xlabel("frame"); axes[0].set_ylabel("dB")
    axes[1].plot(ssim_vals); axes[1].set_title("SSIM per frame"); axes[1].set_xlabel("frame")
    axes[2].plot(lpips_vals); axes[2].set_title("LPIPS per frame (lower better)"); axes[2].set_xlabel("frame")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "metrics_over_frames.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.join(args.out_dir, 'metrics_over_frames.png')}")

    # ---- FPS summary (written by the CUDA program) ----
    fps_info = {}
    fps_summary_path = os.path.join(args.frames_dir, "fps_summary.txt")
    if os.path.exists(fps_summary_path):
        with open(fps_summary_path) as f:
            for line in f:
                k, v = line.split()
                fps_info[k] = float(v)

    dvr_fps = fps_info.get("dvr_fps", float("nan"))
    rast_fps = fps_info.get("rasterizer_fps", float("nan"))

    # ---- metrics table (LPIPS, PSNR, SSIM, FPS) ----
    rows = [
        ("gt (vanilla DVR baseline)", dvr_fps, float("nan"), float("nan"), float("nan")),
        ("ours (Gaussian rasterizer)", rast_fps, float(psnr_vals.mean()), float(ssim_vals.mean()), float(lpips_vals.mean())),
    ]
    csv_path = os.path.join(args.out_dir, "metrics_table.csv")
    with open(csv_path, "w") as f:
        f.write("representation,fps,psnr_db,ssim,lpips\n")
        for name, fps, p, s, l in rows:
            f.write(f"{name},{fps:.3f},{p:.4f},{s:.4f},{l:.4f}\n")
    print(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    table_data = [["Representation", "FPS", "PSNR (dB)", "SSIM", "LPIPS"]]
    for name, fps, p, s, l in rows:
        table_data.append([
            name,
            f"{fps:.2f}",
            "-" if np.isnan(p) else f"{p:.2f}",
            "-" if np.isnan(s) else f"{s:.4f}",
            "-" if np.isnan(l) else f"{l:.4f}",
        ])
    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=[0.42, 0.145, 0.145, 0.145, 0.145])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for j in range(5):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")
    for i in range(1, len(table_data)):
        table[(i, 0)].set_text_props(ha="left")
    plt.tight_layout()
    table_path = os.path.join(args.out_dir, "metrics_table.png")
    fig.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {table_path}")

    print("\n=== Summary ===")
    print(f"  n_gaussians   = {int(fps_info.get('n_gaussians', -1))}")
    print(f"  n_frames      = {n_frames}")
    print(f"  DVR baseline  : {dvr_fps:.2f} FPS")
    print(f"  Ours (raster) : {rast_fps:.2f} FPS   PSNR={psnr_vals.mean():.2f} dB  SSIM={ssim_vals.mean():.4f}  LPIPS={lpips_vals.mean():.4f}")


if __name__ == "__main__":
    main()
