# voxel/training.py

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .checkpoint import save_siren_checkpoint
from .lr_schedule import cosine_lr_schedule
from .model import SirenVoxelField
from .rendering import compute_volume_psnr

# Every checkpoint's companion visualization is rendered at this resolution
# (matches gaussian_volume/representation/training.py's convention, so both pipelines'
# checkpoint directories are visually comparable at a glance).
CHECKPOINT_FIGURE_DPI = 800


def save_reconstruction_pdf(
    prediction: Tensor,
    target: Tensor,
    path: str | Path,
    dpi: int,
) -> None:
    """
    Render axial/coronal/sagittal mid-slices comparing the checkpoint's
    reconstruction against the ground-truth target, plus their difference.

    Called once per saved checkpoint, at the same path with a `.pdf` suffix,
    so every `.pth` has a matching 3 (orientation) x 3 (reconstruction /
    ground truth / diff) grid of that point in training.
    """
    prediction_array = prediction.detach().to(device="cpu", dtype=torch.float32).numpy()
    target_array = target.detach().to(device="cpu", dtype=torch.float32).numpy()

    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction and target must have the same shape, "
            f"got {prediction_array.shape} and {target_array.shape}."
        )

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
            axis.axis("off")

            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(path, dpi=dpi, format="pdf")
    plt.close(figure)


def _save_checkpoint_with_pdf(
    path: Path,
    parameters: Tensor,
    reconstructed: Tensor,
    ground_truth_volume: Tensor,
) -> None:
    save_siren_checkpoint(path, parameters)
    save_reconstruction_pdf(
        reconstructed, ground_truth_volume, path.with_suffix(".pdf"), dpi=CHECKPOINT_FIGURE_DPI
    )


def train_impl(
    field: SirenVoxelField,
    ground_truth_volume: Tensor,
    *,
    iterations: int,
    learning_rate: float,
    out_dir: str | Path,
    checkpoint_every: int,
    lr_min_ratio: float = 1.0,
    lr_warmup_steps: int = 0,
    lr_warmup_init_factor: float = 0.1,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Train a SirenVoxelField to reconstruct ground_truth_volume.

    Mirrors the training loop in the original main.cu: a pre-training
    init.pth snapshot, a periodic <iteration>.pth snapshot + best.pth
    (highest full-grid volume PSNR seen so far) every checkpoint_every
    iterations (always including iteration 1 and the final iteration), and
    a last.pth snapshot at the end. Every .pth also gets a matching .pdf
    (axial/coronal/sagittal reconstruction-vs-ground-truth-vs-diff grid),
    mirroring gaussian_volume/representation/training.py's checkpoint visualizations.

    learning_rate is a base LR shaped by cosine_lr_schedule (linear warmup
    over lr_warmup_steps, then cosine annealing to lr_min_ratio *
    learning_rate) — lr_min_ratio=1.0 (the default) makes this a constant
    learning_rate throughout, matching the pre-schedule behavior.

    Returns a summary dict with the final loss, the best volume PSNR
    reached, and the final learning rate used.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}.")

    if checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be positive, got {checkpoint_every}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_checkpoint_with_pdf(
        out_dir / "init.pth", field.parameters, field.reconstruct(), ground_truth_volume
    )

    log("\nTraining SIREN continuous volume function")
    log(
        f"Grid: {field.config.grid_size}^3, hidden width: {field.config.hidden_size}, "
        f"batch: {field.config.train_batch}"
    )

    best_volume_psnr = float("-inf")
    last_loss = float("nan")

    # PSNR climbs on nearly every logged step early in training, so
    # rendering best.pdf on every single improvement would make that
    # CPU-bound render the training loop's bottleneck. best.pth itself
    # (cheap) still saves on every improvement; the PDF is throttled to at
    # most once per checkpoint_every steps, same cadence as the periodic
    # snapshot.
    last_best_pdf_iteration = -checkpoint_every
    current_lr = learning_rate

    for iteration in range(1, iterations + 1):
        current_lr = cosine_lr_schedule(
            iteration,
            iterations=iterations,
            base_lr=learning_rate,
            min_ratio=lr_min_ratio,
            warmup_steps=lr_warmup_steps,
            warmup_init_factor=lr_warmup_init_factor,
        )
        last_loss = field.training_step(ground_truth_volume, iteration, current_lr)

        is_checkpoint_step = (
            iteration == 1 or iteration % checkpoint_every == 0 or iteration == iterations
        )
        if not is_checkpoint_step:
            continue

        reconstructed = field.reconstruct()
        volume_psnr = compute_volume_psnr(
            ground_truth_volume,
            reconstructed,
            hidden_size=field.config.hidden_size,
            hidden_layers=field.config.hidden_layers,
        )

        log(
            f"iteration {iteration:5d}   minibatch voxel MSE {last_loss:.8f}   "
            f"volume PSNR {volume_psnr:.3f} dB   lr {current_lr:.2e}"
        )

        _save_checkpoint_with_pdf(
            out_dir / f"{iteration:06d}.pth", field.parameters, reconstructed, ground_truth_volume
        )

        if volume_psnr > best_volume_psnr:
            best_volume_psnr = volume_psnr

            if iteration - last_best_pdf_iteration >= checkpoint_every:
                _save_checkpoint_with_pdf(
                    out_dir / "best.pth", field.parameters, reconstructed, ground_truth_volume
                )
                last_best_pdf_iteration = iteration
            else:
                save_siren_checkpoint(out_dir / "best.pth", field.parameters)

    _save_checkpoint_with_pdf(
        out_dir / "last.pth", field.parameters, field.reconstruct(), ground_truth_volume
    )

    return {
        "final_loss": last_loss,
        "best_volume_psnr": best_volume_psnr,
        "final_learning_rate": current_lr,
    }
