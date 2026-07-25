# gaussian_volume/training.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .checkpoint import save_gaussian_checkpoint
from .config import config_section
from .densification import DensificationConfig, densify_gaussians, should_densify
from .losses import reconstruction_loss
from .model import GaussianVolumeModel
from .pruning import PruningConfig, prune_gaussians, should_prune

# Every checkpoint's companion visualization is rendered at this resolution.
CHECKPOINT_FIGURE_DPI = 800


def create_optimizer(
    model: GaussianVolumeModel,
    *,
    mean_lr: float,
    scale_lr: float,
    rotation_lr: float,
    confidence_lr: float,
    feature_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Create Adam with separate learning rates for each Gaussian parameter type.
    """
    parameter_groups = model.parameter_groups(
        mean_lr=mean_lr,
        scale_lr=scale_lr,
        rotation_lr=rotation_lr,
        confidence_lr=confidence_lr,
        feature_lr=feature_lr,
    )

    return torch.optim.Adam(
        parameter_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_densification_config(config_data: dict[str, Any]) -> DensificationConfig:
    """
    Build a DensificationConfig from the [densification] config table,
    falling back to the pilot's tuned defaults for any field left unset.
    """
    section = config_section(config_data, "densification")

    return DensificationConfig(
        enabled=bool(section.get("enabled", True)),
        from_step=int(section.get("from_step", 500)),
        until_step=int(section.get("until_step", 20_000)),
        interval=int(section.get("interval", 100)),
        gradient_threshold=float(section.get("gradient_threshold", 2e-4)),
        population_exponent=float(section.get("population_exponent", 0.8)),
        clone_scale_threshold=float(section.get("clone_scale_threshold", 2.0)),
        split_scale_threshold=float(section.get("split_scale_threshold", 2.0)),
        max_new_gaussians=int(section.get("max_new_gaussians", 1000)),
        max_gaussians=int(section.get("max_gaussians", 50_000)),
        clone_jitter_fraction=float(section.get("clone_jitter_fraction", 0.25)),
        clone_confidence_fraction=float(section.get("clone_confidence_fraction", 0.5)),
        split_children=int(section.get("split_children", 2)),
        split_offset_fraction=float(section.get("split_offset_fraction", 0.5)),
        split_scale_factor=float(section.get("split_scale_factor", 0.7)),
        minimum_confidence=float(section.get("minimum_confidence", 1e-4)),
    )


def build_pruning_config(config_data: dict[str, Any]) -> PruningConfig:
    """
    Build a PruningConfig from the [pruning] config table, falling back to
    the pilot's tuned defaults for any field left unset.
    """
    section = config_section(config_data, "pruning")

    maximum_scale = section.get("maximum_scale")

    return PruningConfig(
        enabled=bool(section.get("enabled", True)),
        from_step=int(section.get("from_step", 500)),
        until_step=int(section.get("until_step", 20_000)),
        interval=int(section.get("interval", 100)),
        minimum_confidence=float(section.get("minimum_confidence", 0.01)),
        minimum_feature=float(section.get("minimum_feature", 0.0)),
        minimum_scale=float(section.get("minimum_scale", 1e-3)),
        maximum_scale=(None if maximum_scale is None else float(maximum_scale)),
        remove_outside_volume=bool(section.get("remove_outside_volume", True)),
        outside_margin=float(section.get("outside_margin", 0.0)),
        minimum_gaussians=int(section.get("minimum_gaussians", 64)),
        confidence_score_weight=float(section.get("confidence_score_weight", 1.0)),
        feature_score_weight=float(section.get("feature_score_weight", 1.0)),
    )


@torch.no_grad()
def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    eps: float,
) -> torch.Tensor:
    """
    Compute volume-level PSNR.
    """
    mse = torch.mean((pred - target) ** 2)

    return 20.0 * torch.log10(
        torch.tensor(data_range, device=pred.device, dtype=pred.dtype)
    ) - 10.0 * torch.log10(mse.clamp_min(eps))


def periodic_checkpoint_path(output_dir: Path, step: int) -> Path:
    """
    Periodic-checkpoint path named by iteration, e.g. `002500.pth`.

    One file per save (not overwritten), distinct from `best.pth` (highest
    PSNR so far) and `last.pth` (final state).
    """
    return output_dir / f"{step:06d}.pth"


def save_reconstruction_pdf(
    prediction: torch.Tensor,
    target: torch.Tensor,
    path: Path,
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


def rebuild_model(
    *,
    old_model: GaussianVolumeModel,
    means: torch.Tensor,
    scales: torch.Tensor,
    quaternions: torch.Tensor,
    confidences: torch.Tensor,
    features: torch.Tensor,
) -> GaussianVolumeModel:
    return GaussianVolumeModel(
        means=means,
        scales=scales,
        quaternions=quaternions,
        confidences=confidences,
        features=features,
        minimum_scale=old_model.minimum_scale,
        minimum_confidence=old_model.minimum_confidence,
        constrain_features=old_model.constrain_features,
        feature_min=old_model.feature_min,
        feature_max=old_model.feature_max,
    ).to(old_model.device)


def train_model(
    *,
    model: GaussianVolumeModel,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
    data_range: float,
    log,
    iterations: int,
    cutoff_sigma: float,
    epsilon: float,
    log_every: int,
    checkpoint_every: int,
    output_dir: Path,
    input_path: Path,
    dataset: str,
    mean_lr: float,
    scale_lr: float,
    rotation_lr: float,
    confidence_lr: float,
    feature_lr: float,
    weight_decay: float,
    densification_config: DensificationConfig,
    pruning_config: PruningConfig,
    mean_clamp_margin: float,
) -> tuple[GaussianVolumeModel, torch.optim.Optimizer, list[dict[str, float]]]:
    """
    Fit the Gaussian representation to the target voxel grid.

    The target is fixed. The trainable quantities are:

        means
        scales
        quaternions
        confidences
        features

    Densification and pruning replace `model` (and its optimizer) with a
    freshly constructed instance whenever the Gaussian population changes,
    so the current model/optimizer are returned alongside the history
    rather than mutated in place on the caller's original objects.
    """
    model.train()

    volume_shape = tuple(int(size) for size in target.shape)

    history: list[dict[str, float]] = []

    best_psnr = float("-inf")

    # PSNR climbs on nearly every logged step early in training, so
    # rendering best.pdf (a 3x3 grid at CHECKPOINT_FIGURE_DPI) on every
    # single improvement would make that CPU-bound render the training
    # loop's bottleneck. best.pth itself (cheap) still saves on every
    # improvement; the PDF is throttled to at most once per
    # checkpoint_every steps, same cadence as the periodic snapshot.
    last_best_pdf_step = -checkpoint_every

    # Densify/prune fire on their own interval (a training hyperparameter,
    # independent of log_every). Rather than printing a line every time one
    # fires, tally activity here and report a summary alongside the regular
    # metrics log, at log_every cadence.
    cloned_since_log = 0
    split_since_log = 0
    densify_events_since_log = 0
    pruned_since_log = 0
    prune_events_since_log = 0

    # Training speed, reported alongside each metrics log.
    last_log_step = 0
    last_log_time = time.perf_counter()

    initial_gaussian_count = model.num_gaussians

    for step in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------------------
        # Reconstruct the full voxel grid from the current Gaussians.
        # --------------------------------------------------------------

        prediction = model(shape=volume_shape, cutoff_sigma=cutoff_sigma, epsilon=epsilon)

        # --------------------------------------------------------------
        # Compute:
        #
        #   loss = 0.7 * L1 + 0.3 * (1 - SSIM)
        # --------------------------------------------------------------

        loss, metrics = reconstruction_loss(prediction, target, data_range=data_range)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

        # --------------------------------------------------------------
        # Backpropagation:
        #
        # loss
        #   ↓
        # reconstructed voxels
        #   ↓
        # CUDA normalized Gaussian backward
        #   ↓
        # precision / mean / confidence / feature gradients
        #   ↓
        # scale and quaternion gradients through math3d.py
        # --------------------------------------------------------------

        loss.backward()
        population_changed = False

        if should_densify(step, densification_config):
            densified = densify_gaussians(
                model,
                initial_count=initial_gaussian_count,
                config=densification_config,
                volume_shape=volume_shape,
            )

            if densified.new_count > 0:
                model = rebuild_model(
                    old_model=model,
                    means=densified.means,
                    scales=densified.scales,
                    quaternions=densified.quaternions,
                    confidences=densified.confidences,
                    features=densified.features,
                )

                population_changed = True

                cloned_since_log += densified.cloned_count
                split_since_log += densified.split_parent_count
                densify_events_since_log += 1

        if should_prune(step, pruning_config):
            pruned = prune_gaussians(model, config=pruning_config, volume_shape=volume_shape)

            if pruned.removed_count > 0:
                model = rebuild_model(
                    old_model=model,
                    means=pruned.means,
                    scales=pruned.scales,
                    quaternions=pruned.quaternions,
                    confidences=pruned.confidences,
                    features=pruned.features,
                )

                population_changed = True

                pruned_since_log += pruned.removed_count
                prune_events_since_log += 1

        if population_changed:
            optimizer = create_optimizer(
                model,
                mean_lr=mean_lr,
                scale_lr=scale_lr,
                rotation_lr=rotation_lr,
                confidence_lr=confidence_lr,
                feature_lr=feature_lr,
                weight_decay=weight_decay,
            )

        # Optional gradient clipping protects against unstable early steps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        if not population_changed:
            optimizer.step()

        # Keep centres inside the target volume.
        model.clamp_means_to_volume(volume_shape, margin=mean_clamp_margin)

        # --------------------------------------------------------------
        # Validate physical parameters.
        # --------------------------------------------------------------

        with torch.no_grad():
            if not torch.isfinite(model.means).all():
                raise RuntimeError(f"Gaussian means became invalid at step {step}.")

            if not torch.isfinite(model.scales).all():
                raise RuntimeError(f"Gaussian scales became invalid at step {step}.")

            if not torch.isfinite(model.confidences).all():
                raise RuntimeError(f"Gaussian confidences became invalid at step {step}.")

            if not torch.isfinite(model.features).all():
                raise RuntimeError(f"Gaussian features became invalid at step {step}.")

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------

        should_log = step == 1 or step % log_every == 0 or step == iterations

        if should_log:
            with torch.no_grad():
                psnr = compute_psnr(prediction, target, data_range=data_range, eps=1e-12)
                current_psnr = float(psnr.item())

                # Update the running best *before* logging, so the printed
                # line always shows the best vol_psnr seen so far (the
                # exact full-grid metric — matches the training
                # distribution, unlike a noisier continuous-sample psnr)
                # rather than this step's possibly-lower raw value.
                is_new_best = current_psnr > best_psnr
                if is_new_best:
                    best_psnr = current_psnr

                record = {
                    "step": float(step),
                    "loss": float(loss.detach().item()),
                    "l1": float(metrics["l1"].item()),
                    "ssim": float(metrics["ssim"].item()),
                    "ssim_loss": float(metrics["ssim_loss"].item()),
                    "psnr": current_psnr,
                    "best_psnr": best_psnr,
                    "scale_mean": float(model.scales.mean().item()),
                    "scale_min": float(model.scales.min().item()),
                    "scale_max": float(model.scales.max().item()),
                    "confidence_mean": float(model.confidences.mean().item()),
                    "feature_mean": float(model.features.mean().item()),
                }

                history.append(record)

                log(
                    f"step={step:06d} "
                    f"loss={record['loss']:.6f} "
                    f"l1={record['l1']:.6f} "
                    f"ssim={record['ssim']:.6f} "
                    f"best_vol_psnr={record['best_psnr']:.3f} dB "
                    f"scale={record['scale_mean']:.4f} "
                    f"confidence={record['confidence_mean']:.4f}"
                )

                # ------------------------------------------------------
                # Densify/prune activity since the last log (see the
                # counters reset below) — a summary instead of a line
                # per event.
                # ------------------------------------------------------

                if densify_events_since_log > 0 or prune_events_since_log > 0:
                    log(
                        f"  population: "
                        f"{densify_events_since_log} densify events "
                        f"(+{cloned_since_log} cloned, "
                        f"{split_since_log} split) "
                        f"{prune_events_since_log} prune events "
                        f"(-{pruned_since_log} removed) "
                        f"N={model.num_gaussians}"
                    )

                cloned_since_log = 0
                split_since_log = 0
                densify_events_since_log = 0
                pruned_since_log = 0
                prune_events_since_log = 0

                # ------------------------------------------------------
                # Training speed since the last log.
                # ------------------------------------------------------

                now = time.perf_counter()
                elapsed = now - last_log_time
                steps_elapsed = step - last_log_step
                iters_per_second = steps_elapsed / elapsed if elapsed > 0 else float("inf")

                log(f"  speed: {iters_per_second:.2f} it/s ({steps_elapsed} steps in {elapsed:.2f}s)")

                last_log_step = step
                last_log_time = now

                # ----------------------------------------------------------
                # Best checkpoint (highest full-grid PSNR seen so far)
                # ----------------------------------------------------------

                if is_new_best:
                    best_path = output_dir / "best.pth"

                    save_gaussian_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        volume_shape=volume_shape,
                        normalization=metadata.get("normalization"),
                        extra={
                            "input_path": str(input_path),
                            "dataset": dataset,
                            "cutoff_sigma": cutoff_sigma,
                            "epsilon": epsilon,
                            "psnr": best_psnr,
                        },
                    )

                    if step - last_best_pdf_step >= checkpoint_every:
                        save_reconstruction_pdf(
                            prediction,
                            target,
                            best_path.with_suffix(".pdf"),
                            dpi=CHECKPOINT_FIGURE_DPI,
                        )
                        last_best_pdf_step = step

        # --------------------------------------------------------------
        # Rolling periodic checkpoint
        # --------------------------------------------------------------

        should_checkpoint = (
            checkpoint_every > 0 and step % checkpoint_every == 0 and step < iterations
        )

        if should_checkpoint:
            periodic_path = periodic_checkpoint_path(output_dir, step)

            save_gaussian_checkpoint(
                periodic_path,
                model=model,
                optimizer=optimizer,
                step=step,
                volume_shape=volume_shape,
                normalization=metadata.get("normalization"),
                extra={
                    "input_path": str(input_path),
                    "dataset": dataset,
                    "cutoff_sigma": cutoff_sigma,
                    "epsilon": epsilon,
                },
            )

            save_reconstruction_pdf(
                prediction,
                target,
                periodic_path.with_suffix(".pdf"),
                dpi=CHECKPOINT_FIGURE_DPI,
            )

            log(f"Saved periodic checkpoint: {periodic_path}")

    return model, optimizer, history
