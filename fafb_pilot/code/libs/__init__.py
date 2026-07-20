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
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
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


def load_gaussians_json(json_path):
    """Load a gaussians_json.py-style export: one JSON file holding EVERY
    block's checkpoint, keyed by block name (e.g. "b_211"), each value the
    same dict a .pth checkpoint holds (means/log_scales/quats/intensities/
    inten_param) with tensors converted to plain lists. This file can be
    hundreds of MB (e.g. all 8 blocks_v18 pilot blocks -> ~470 MB), so load
    it ONCE and reuse the returned dict rather than re-parsing per block."""
    print(f"Loading {json_path} (this can take a while for large exports)...")
    with open(json_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} blocks: {sorted(data.keys())}")
    return data


def gaussian_cloud_from_entry(entry, aabb, device, cfg):
    """Build a GaussianCloud from one already-loaded gaussians_json.py entry
    (a dict of plain lists), bypassing GaussianCloud.load()'s disk read but
    mirroring its tensor-building exactly -- see scripts/_3dgs/_3dgs.py's
    GaussianCloud.load(). Assumes the current 'softplus' intensity
    convention (true for anything saved by GaussianCloud.save(), which is
    the only source gaussians_json.py reads from)."""
    from _3dgs._3dgs import GaussianCloud

    obj = GaussianCloud.__new__(GaussianCloud)
    obj.aabb        = aabb
    obj.device      = device
    obj.scale_min   = cfg.scale_min_clamp
    obj.mahal_clamp = cfg.mahal_max_clamp

    def _t(key):
        return torch.tensor(entry[key], dtype=torch.float32, device=device).requires_grad_(True)

    obj.means = _t('means')
    obj.log_s = _t('log_scales')
    obj.quats = _t('quats')
    obj.inten = _t('intensities')
    obj.reset_grad_acc()
    return obj


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


# ── Slice-level (2D image) metrics: PSNR / SSIM / LPIPS ─────────────────────
# Distinct from compute_metrics()/whole_volume_ssim() above, which are global
# whole-VOLUME formulas. These operate on single 2D slices using the standard
# per-image definitions (skimage's windowed SSIM, LPIPS's perceptual network)
# -- used by block_quality_report.py, stitch_quality_report.py and
# sliding_cube_eval.py to score mid-slices / sliding-window crops.

_LPIPS_FN = {}


def get_lpips_fn(device):
    """Lazily construct (and cache per-device) an LPIPS(net='alex') model.
    'alex' is the fastest of LPIPS's three backbones and the one the
    PerceptualSimilarity paper reports as best-correlated with human judgment
    for this kind of low-level distortion comparison."""
    key = str(device)
    if key not in _LPIPS_FN:
        import lpips
        _LPIPS_FN[key] = lpips.LPIPS(net='alex').to(device).eval()
    return _LPIPS_FN[key]


def lpips_distance(pred_2d, gt_2d, device):
    """LPIPS perceptual distance between two [0,1] grayscale 2D arrays.
    LPIPS expects 3-channel images in roughly [-1,1], so each slice is
    replicated across channels and rescaled."""
    fn = get_lpips_fn(device)

    def _prep(img):
        t = torch.tensor(img, dtype=torch.float32, device=device)
        t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # (1,3,H,W)
        return t * 2.0 - 1.0

    with torch.no_grad():
        d = fn(_prep(pred_2d), _prep(gt_2d))
    return float(d.item())


def slice_metrics(pred_2d, gt_2d, device=None):
    """PSNR + SSIM (skimage, windowed -- the standard per-image definitions)
    between two [0,1] 2D slices, plus LPIPS if a device is given (LPIPS
    needs a small CNN forward pass, so it's opt-in)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    metrics = {
        'PSNR': float(peak_signal_noise_ratio(gt_2d, pred_2d, data_range=1.0)),
        'SSIM': float(structural_similarity(gt_2d, pred_2d, data_range=1.0)),
    }
    if device is not None:
        metrics['LPIPS'] = lpips_distance(pred_2d, gt_2d, device)
    return metrics


def save_metrics_excel(rows, output_dir, filename):
    """Save a list of metric-row dicts (one dict per row -- the same rows
    written to a metrics CSV elsewhere) as a single-sheet .xlsx file, for
    reviewers who want a spreadsheet rather than a CSV."""
    import pandas as pd

    out_xlsx = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    print(f"Saved {out_xlsx}")
