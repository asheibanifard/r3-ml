"""
Convert every GT / ours_bounded .pfm from dvr_benchmark.py to .png, and plot
GT | Pred (bounded) | |Diff| for each (volume_size, screen_size) combination,
with pixel-coordinate axes left ON (not hidden).

USAGE
-----
    /venv/r3-ml/bin/python3 fafb_pilot/code/renderer/plot_gt_pred_diff.py

STEPS
-----
1. Find every dvr_gt_v<V>_s<S>.pfm / dvr_ours_bounded_v<V>_s<S>.pfm pair
   already on disk (produced by dvr_benchmark.py).
2. Convert each individual .pfm to its own .png (reuses pfm_to_png.py's
   convention: [0,1] display range).
3. For each (volume_size, screen_size) pair, plot GT | Pred | |Diff| side by
   side, with matplotlib's default pixel-coordinate axes shown (no axis('off')).

OUTPUT FILES (written to fafb_pilot/code/renderer/results/)
---------------------------------------------------------
  dvr_gt_v<V>_s<S>.png            GT render, standalone
  dvr_ours_bounded_v<V>_s<S>.png  Ours (bounded) render, standalone
  gt_pred_diff_v<V>_s<S>.png      GT | Pred | |Diff|, axes on
"""
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PAIR_PATTERN = re.compile(r"dvr_gt_v(\d+)_s(\d+)\.pfm$")


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


def save_png(image, path):
    """Standalone [0,1] grayscale PNG, matching pfm_to_png.py's own display
    convention, but written directly here to avoid a subprocess per file."""
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_comparison_figure(gt, pred, volume_size, screen_size, out_path):
    diff = np.abs(pred - gt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"GT (volume={volume_size}³, screen={screen_size})")
    axes[0].set_xlabel("pixel x")
    axes[0].set_ylabel("pixel y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Pred (ours_bounded)")
    axes[1].set_xlabel("pixel x")
    axes[1].set_ylabel("pixel y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=max(float(diff.max()), 1e-8))
    axes[2].set_title(f"|Diff| (mean={diff.mean():.4f}, max={diff.max():.4f})")
    axes[2].set_xlabel("pixel x")
    axes[2].set_ylabel("pixel y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    gt_paths = sorted(glob.glob(os.path.join(RESULT_DIR, "dvr_gt_v*_s*.pfm")))
    if not gt_paths:
        raise FileNotFoundError(f"No dvr_gt_v*_s*.pfm files found in {RESULT_DIR}")

    for gt_path in gt_paths:
        match = PAIR_PATTERN.search(os.path.basename(gt_path))
        if not match:
            continue
        volume_size, screen_size = int(match.group(1)), int(match.group(2))

        pred_path = os.path.join(RESULT_DIR, f"dvr_ours_bounded_v{volume_size}_s{screen_size}.pfm")
        if not os.path.exists(pred_path):
            print(f"Skipping volume={volume_size}, screen={screen_size}: no matching ours_bounded .pfm")
            continue

        gt = read_pfm(gt_path)
        pred = read_pfm(pred_path)

        # Step 2: standalone PNGs for each raw .pfm.
        save_png(gt, os.path.join(RESULT_DIR, f"dvr_gt_v{volume_size}_s{screen_size}.png"))
        save_png(pred, os.path.join(RESULT_DIR, f"dvr_ours_bounded_v{volume_size}_s{screen_size}.png"))

        # Step 3: GT | Pred | |Diff|, axes on.
        out_path = os.path.join(RESULT_DIR, f"gt_pred_diff_v{volume_size}_s{screen_size}.png")
        save_comparison_figure(gt, pred, volume_size, screen_size, out_path)


if __name__ == "__main__":
    main()
