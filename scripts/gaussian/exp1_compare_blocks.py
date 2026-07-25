# scripts/exp1_compare_blocks.py
"""
Experiment 1: compare the trained gaussian_volume/_3dgs.py (GaussianCloud)
models across all four processed_data/ block types (simple, typical,
complex, edge_rich).

For each block, reconstructs the full 64x64x64 volume from best.pth and
computes PSNR, 3D SSIM, and LPIPS (mean over axial slices) against the
ground-truth volume, plus a Gaussian-count-based compression ratio against
the raw uint8 source file. It also reconstructs every epoch_NNNN.pth
snapshot to build per-epoch SSIM/LPIPS training curves (loss/PSNR come
directly from log.json).

Outputs (outputs/exp1/):
    summary.json                    cross-block metrics table
    comparison_bar_chart.pdf         PSNR/SSIM/LPIPS/compression bar chart
    <block>_gt_recon_diff.pdf        3x3 axial/coronal/sagittal x
                                     reconstruction/ground-truth/diff grid
    <block>_training_curves.pdf      loss/psnr/ssim/lpips vs epoch

Usage:
    /venv/r3-ml/bin/python3 scripts/exp1_compare_blocks.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import lpips
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_VOLUME_DIR = PROJECT_ROOT / "gaussian_volume"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# _3dgs.py imports "from _3dgs_training import ..." as a top-level module,
# so its own directory must be on sys.path (matches how it's normally run
# directly: `python gaussian_volume/_3dgs.py`, which Python does automatically).
if str(GAUSSIAN_VOLUME_DIR) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_VOLUME_DIR))

import _3dgs  # noqa: E402  (gaussian_volume/_3dgs.py — GaussianCloud pipeline)

# Trusted, locally-produced checkpoints — silence torch.load's
# weights_only=False advisory (repeated ~250 times across all reconstructions).
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

_3dgs.USE_CUDA_KERNEL = True

BLOCKS = ("simple", "typical", "complex", "edge_rich")

# GaussianCloud checkpoint layout: means(3) + log_scales(3) + quats(4) + inten(1).
FLOATS_PER_GAUSSIAN = 11
BYTES_PER_GAUSSIAN = FLOATS_PER_GAUSSIAN * 4  # float32

OUT_DIR = PROJECT_ROOT / "outputs" / "exp1"
DPI = 800


def _gaussian_window_3d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Separable 3D Gaussian blur kernel, shaped for a depthwise conv3d."""

    coordinates = (
        torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    )
    gaussian_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    window_3d = (
        gaussian_1d[:, None, None]
        * gaussian_1d[None, :, None]
        * gaussian_1d[None, None, :]
    )

    return window_3d.view(1, 1, window_size, window_size, window_size)


def ssim_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    sigma: float,
    data_range: float,
    eps: float,
) -> torch.Tensor:
    """Differentiable-style 3D SSIM for a pair of [D,H,W] volumes."""

    pred = prediction.unsqueeze(0).unsqueeze(0)
    gt = target.unsqueeze(0).unsqueeze(0)

    minimum_dimension = min(pred.shape[-3:])
    actual_window_size = min(window_size, minimum_dimension)
    if actual_window_size % 2 == 0:
        actual_window_size -= 1
    actual_window_size = max(actual_window_size, 1)

    window = _gaussian_window_3d(actual_window_size, sigma, pred.device, pred.dtype)
    padding = actual_window_size // 2

    conv = lambda x: torch.nn.functional.conv3d(x, window, padding=padding)

    mu_pred = conv(pred)
    mu_gt = conv(gt)
    mu_pred_sq = mu_pred.square()
    mu_gt_sq = mu_gt.square()
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred_sq = (conv(pred * pred) - mu_pred_sq).clamp_min(0.0)
    sigma_gt_sq = (conv(gt * gt) - mu_gt_sq).clamp_min(0.0)
    sigma_pred_gt = conv(pred * gt) - mu_pred_gt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_pred_gt + c1) * (2.0 * sigma_pred_gt + c2)
    denominator = (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)

    return (numerator / denominator.clamp_min(eps)).mean()


def resolve_block_paths(block_name: str) -> tuple[Path, Path]:
    """
    Find a block's processed_data input .tif and its outputs/<block>/<stem>
    run directory (the layout produced by train_all_blocks_3dgs.sh /
    _3dgs.py's --out auto-derivation).
    """

    data_dir = PROJECT_ROOT / "processed_data" / block_name
    tif_files = sorted(data_dir.glob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No .tif found in {data_dir}")

    volume_path = tif_files[0]
    run_dir = PROJECT_ROOT / "outputs" / block_name / volume_path.stem

    return volume_path, run_dir


def load_ground_truth(volume_path: Path, device: torch.device) -> torch.Tensor:
    """Load and min-max normalize a volume exactly as _3dgs_training.py does."""

    raw = tifffile.imread(str(volume_path)).astype(np.float32)
    vmin, vmax = float(raw.min()), float(raw.max())

    if vmax > vmin:
        raw = (raw - vmin) / (vmax - vmin)

    return torch.from_numpy(raw).to(device)


def dense_grid_points(
    shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build every voxel centre of a [D,H,W] volume in AABB.unit() [-1,1]^3
    coordinates, ordered [x,y,z] to match GaussianCloud.forward's convention.
    """

    depth, height, width = shape

    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, depth, device=device),
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )

    return torch.stack(
        [xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)],
        dim=-1,
    )


@torch.no_grad()
def reconstruct_volume(
    checkpoint_path: Path,
    cfg: argparse.Namespace,
    aabb: "_3dgs.AABB",
    device: torch.device,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, int]:
    """Load a GaussianCloud checkpoint and reconstruct the full dense volume."""

    gc = _3dgs.GaussianCloud.load(checkpoint_path, aabb, device, cfg)
    pts = dense_grid_points(shape, device)

    prediction = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)

    return prediction.reshape(shape), gc.N


@torch.no_grad()
def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    mse = torch.mean((prediction - target) ** 2).item()

    if mse <= eps:
        return 99.0

    return float(-10.0 * np.log10(mse))


def to_lpips_input(slice_2d: torch.Tensor) -> torch.Tensor:
    """[H,W] in [0,1] -> [1,3,H,W] in [-1,1], the range/shape lpips expects."""

    image = slice_2d.clamp(0.0, 1.0) * 2.0 - 1.0

    return image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)


@torch.no_grad()
def lpips_over_axial_slices(
    lpips_model: lpips.LPIPS,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Mean LPIPS over every axial (Z) slice. LPIPS is inherently a 2D metric;
    averaging over the full stack gives a volume-level score.
    """

    values = [
        lpips_model(
            to_lpips_input(prediction[z]),
            to_lpips_input(target[z]),
        ).item()
        for z in range(prediction.shape[0])
    ]

    return float(np.mean(values))


@torch.no_grad()
def lpips_single_slice(
    lpips_model: lpips.LPIPS,
    prediction_slice: torch.Tensor,
    target_slice: torch.Tensor,
) -> float:
    return float(
        lpips_model(
            to_lpips_input(prediction_slice),
            to_lpips_input(target_slice),
        ).item()
    )


def save_gt_recon_diff_pdf(
    prediction: torch.Tensor,
    target: torch.Tensor,
    path: Path,
) -> None:
    """
    3 (Axial/Coronal/Sagittal) x 3 (Reconstruction/Ground truth/Diff) mid-slice
    comparison grid, matching scripts/train.py's checkpoint visualization.
    """

    prediction_array = prediction.detach().cpu().numpy()
    target_array = target.detach().cpu().numpy()
    diff_array = prediction_array - target_array

    depth, height, width = prediction_array.shape

    orientations = (
        ("Axial", (depth // 2, slice(None), slice(None))),
        ("Coronal", (slice(None), height // 2, slice(None))),
        ("Sagittal", (slice(None), slice(None), width // 2)),
    )

    figure, axes = plt.subplots(3, 3, figsize=(9, 9))

    for row, (orientation_name, index) in enumerate(orientations):
        prediction_slice = prediction_array[index]
        target_slice = target_array[index]
        diff_slice = diff_array[index]

        diff_limit = max(float(np.abs(diff_slice).max()), 1e-6)

        panels = (
            (f"{orientation_name} — Reconstruction", prediction_slice, "gray", 0.0, 1.0),
            (f"{orientation_name} — Ground truth", target_slice, "gray", 0.0, 1.0),
            (f"{orientation_name} — Diff", diff_slice, "bwr", -diff_limit, diff_limit),
        )

        for column, (title, image, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row, column]
            artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=8)
            # axis.axis("on")
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(path, dpi=DPI, format="pdf")
    plt.close(figure)


def save_training_curves_pdf(
    history: list[dict],
    epoch_ssim: dict[int, float],
    epoch_lpips: dict[int, float],
    path: Path,
) -> None:
    epochs = [entry["epoch"] for entry in history]
    loss = [entry["loss"] for entry in history]
    psnr = [entry["psnr"] for entry in history]

    ssim_epochs = sorted(epoch_ssim)
    ssim_values = [epoch_ssim[epoch] for epoch in ssim_epochs]

    lpips_epochs = sorted(epoch_lpips)
    lpips_values = [epoch_lpips[epoch] for epoch in lpips_epochs]

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(epochs, loss, color="tab:blue")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")

    axes[0, 1].plot(epochs, psnr, color="tab:green")
    axes[0, 1].set_title("PSNR (sampled)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("PSNR (dB)")

    axes[1, 0].plot(ssim_epochs, ssim_values, marker="o", ms=3, color="tab:orange")
    axes[1, 0].set_title("SSIM (full volume, from epoch checkpoints)")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("SSIM")

    axes[1, 1].plot(lpips_epochs, lpips_values, marker="o", ms=3, color="tab:red")
    axes[1, 1].set_title("LPIPS (axial mid-slice, from epoch checkpoints)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("LPIPS (lower is better)")

    figure.tight_layout()
    figure.savefig(path, dpi=DPI, format="pdf")
    plt.close(figure)


def save_comparison_bar_chart(summary: dict[str, dict], path: Path) -> None:
    names = list(summary)

    metrics = (
        ("psnr", "PSNR (dB, higher is better)"),
        ("ssim", "SSIM (higher is better)"),
        ("lpips", "LPIPS (lower is better)"),
        ("compression_ratio", "Compression ratio (x, vs raw uint8)"),
    )

    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for axis, (key, title) in zip(axes, metrics):
        values = [summary[name][key] for name in names]
        axis.bar(names, values, color="tab:blue")
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    figure.savefig(path, dpi=DPI, format="pdf")
    plt.close(figure)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aabb = _3dgs.AABB.unit().to(device)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    summary: dict[str, dict] = {}

    for block_name in BLOCKS:
        volume_path, run_dir = resolve_block_paths(block_name)
        print(f"[{block_name}] volume={volume_path}  run_dir={run_dir}")

        with (run_dir / "config.json").open() as config_file:
            cfg = argparse.Namespace(**json.load(config_file))

        target = load_ground_truth(volume_path, device)
        shape = tuple(int(size) for size in target.shape)

        # ---- Final best.pth comparison ----------------------------------

        prediction, n_gaussians = reconstruct_volume(
            run_dir / "best.pth", cfg, aabb, device, shape
        )

        psnr_value = compute_psnr(prediction, target)
        ssim_value = float(
            ssim_3d(
                prediction,
                target,
                window_size=11,
                sigma=1.5,
                data_range=1.0,
                eps=1e-8,
            )
        )
        lpips_value = lpips_over_axial_slices(lpips_model, prediction, target)

        original_bytes = int(np.prod(shape))  # raw uint8 source, 1 byte/voxel
        model_bytes = n_gaussians * BYTES_PER_GAUSSIAN
        compression_ratio = original_bytes / model_bytes

        summary[block_name] = {
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
            "n_gaussians": n_gaussians,
            "original_bytes": original_bytes,
            "model_bytes": model_bytes,
            "compression_ratio": compression_ratio,
        }

        save_gt_recon_diff_pdf(
            prediction, target, OUT_DIR / f"{block_name}_gt_recon_diff.pdf"
        )

        # ---- Training curves ---------------------------------------------

        with (run_dir / "log.json").open() as log_file:
            history = json.load(log_file)

        epoch_ssim: dict[int, float] = {}
        epoch_lpips: dict[int, float] = {}
        mid_z = shape[0] // 2

        for checkpoint_path in sorted(run_dir.glob("epoch_*.pth")):
            epoch_num = int(checkpoint_path.stem.split("_")[1])

            checkpoint_prediction, _ = reconstruct_volume(
                checkpoint_path, cfg, aabb, device, shape
            )

            epoch_ssim[epoch_num] = float(
                ssim_3d(
                    checkpoint_prediction,
                    target,
                    window_size=11,
                    sigma=1.5,
                    data_range=1.0,
                    eps=1e-8,
                )
            )
            epoch_lpips[epoch_num] = lpips_single_slice(
                lpips_model,
                checkpoint_prediction[mid_z],
                target[mid_z],
            )

        save_training_curves_pdf(
            history,
            epoch_ssim,
            epoch_lpips,
            OUT_DIR / f"{block_name}_training_curves.pdf",
        )

        print(
            f"[{block_name}] psnr={psnr_value:.2f} dB  ssim={ssim_value:.4f}  "
            f"lpips={lpips_value:.4f}  N={n_gaussians}  "
            f"compression={compression_ratio:.1f}x"
        )

    with (OUT_DIR / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    save_comparison_bar_chart(summary, OUT_DIR / "comparison_bar_chart.pdf")

    print("\nSummary (outputs/exp1/summary.json):")
    for name, metrics in summary.items():
        print(
            f"  {name:>10}: PSNR={metrics['psnr']:6.2f} dB  "
            f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}  "
            f"N={metrics['n_gaussians']:>6}  "
            f"compression={metrics['compression_ratio']:.1f}x"
        )


if __name__ == "__main__":
    main()
