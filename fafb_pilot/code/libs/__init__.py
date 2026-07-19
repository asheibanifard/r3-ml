"""
Shared import/environment setup for fafb_pilot/code/ scripts that need the
project's core _3dgs training/model module (scripts/_3dgs/_3dgs.py).

Usage (at the top of a script under fafb_pilot/code/<subdir>/):

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fafb_pilot/code
    import libs  # noqa: F401  -- sets up sys.path + env vars as a side effect

    import _3dgs._3dgs as _mod
    _mod.USE_CUDA_KERNEL = True
    _mod._load_3dgs_kernel()
    from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset

This does NOT copy or move scripts/_3dgs/ -- it is the single canonical
source used by the whole training pipeline (batch retrains, CUDA kernel
source discovery via __file__-relative paths inside _3dgs.py, the
retrain_pilot_blocks_v2.sh/v18.sh drivers which invoke it as a direct script
path, CLAUDE.md documentation, etc). Copying it here would create a second
copy that silently drifts out of sync with the original -- this package
only makes the existing module reachable from fafb_pilot/code/ scripts
without each one repeating the same sys.path-insert + env-var boilerplate.
"""
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# RTX 5000 Ada / Ada Lovelace (compute_89) -- matches this project's own
# nvcc builds (see e.g. fafb_pilot/code/renderer/run.sh). Set before any
# _3dgs CUDA kernel JIT-compiles, or torch.utils.cpp_extension warns and
# compiles for every architecture torch can detect on this machine.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


# ── Shared volume/plotting/metrics helpers ──────────────────────────────────
# Used by both fafb_pilot/code/data/rec_vol.py and stitch_blocks.py, so they
# live here instead of being copy-pasted across scripts.

def imshow_gray(ax, image, title):
    """Grayscale [0,1] image on a matplotlib axis, with a title, axis off."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')


def save_png(image, path):
    """Save a [0,1] float array as an 8-bit grayscale PNG."""
    Image.fromarray((image * 255).astype(np.uint8)).save(path, dpi=(800, 800))


def save_volume_tif(volume, output_dir, filename):
    """Save a (D,H,W) float32 [0,1] volume as a 3D tif stack (viewable in
    Fiji/ImageJ, directly diffable against a matching pair saved the same
    way)."""
    out_tif = os.path.join(output_dir, filename)
    tifffile.imwrite(out_tif, volume.astype(np.float32))
    print(f"Saved {out_tif}")


def whole_volume_ssim(img1, img2, C1=0.01 ** 2, C2=0.03 ** 2):
    """Whole-volume Structural Similarity Index between two arrays."""
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))


def compute_metrics(pred_vol, gt_vol):
    """MSE, PSNR, SSIM (whole-volume), max error, and output min/max."""
    mse = float(np.mean((pred_vol - gt_vol) ** 2))
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': whole_volume_ssim(pred_vol, gt_vol),
        'Max Error': float(np.max(np.abs(pred_vol - gt_vol))),
        'Output Min': float(np.min(pred_vol)),
        'Output Max': float(np.max(pred_vol)),
    }
