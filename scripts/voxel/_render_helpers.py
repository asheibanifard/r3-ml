# scripts/voxel/_render_helpers.py
"""Shared rendering/benchmark/evaluation helpers for reconstruct.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from voxel import render_direct_volume, render_voxel_rasterizer


def benchmark_renderer(name: str, launch, warmup_frames: int, benchmark_frames: int) -> float:
    for _ in range(warmup_frames):
        launch()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_frames):
        launch()
    stop_event.record()
    torch.cuda.synchronize()

    milliseconds_per_frame = start_event.elapsed_time(stop_event) / benchmark_frames
    fps = 1000.0 / milliseconds_per_frame
    print(f"{name:<28} {milliseconds_per_frame:9.4f} ms/frame   {fps:10.2f} FPS")
    return fps


def render_pair(volume_gt, volume_pred, camera, config):
    gt_image = render_direct_volume(
        volume_gt, camera,
        image_width=config.image_width, image_height=config.image_height,
        dvr_steps=config.dvr_steps, density_scale=config.density_scale,
        hidden_size=config.hidden_size, hidden_layers=config.hidden_layers,
    )
    pred_image = render_voxel_rasterizer(
        volume_pred, camera,
        image_width=config.image_width, image_height=config.image_height,
        density_scale=config.density_scale,
        hidden_size=config.hidden_size, hidden_layers=config.hidden_layers,
    )
    return gt_image, pred_image


def compute_lpips_image(
    gt_image: Tensor, pred_image: Tensor, *, network: str, device: torch.device
) -> float:
    """
    LPIPS between two single-channel [H,W] rendered images in [0,1].

    LPIPS runs a pretrained image network, so it needs 3-channel RGB in
    [-1,1] — grayscale is replicated across channels (same convention as
    evaluate.py's compute_lpips for non-volume inputs).
    """
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS requires the `lpips` package. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    def to_tensor(image: Tensor) -> Tensor:
        rgb = image.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        return (rgb * 2.0 - 1.0).to(device)

    model = lpips.LPIPS(net=network).to(device)
    with torch.no_grad():
        value = model(to_tensor(gt_image), to_tensor(pred_image)).item()

    return float(value)


def save_rendered_pdf(
    path,
    gt_image: np.ndarray,
    pred_image: np.ndarray,
    diff_image: np.ndarray,
    *,
    metrics: dict,
) -> None:
    """
    Save a GT | prediction | difference rendered comparison as a single PDF,
    annotated with PSNR/SSIM/LPIPS and both renderers' FPS.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    panels = (
        (axes[0], gt_image, "GT: direct volume rendering"),
        (axes[1], pred_image, "Prediction: voxel rasterization"),
        (axes[2], diff_image, "Absolute difference"),
    )

    for axis, image, title in panels:
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(
        f"PSNR {metrics['psnr']:.3f} dB   SSIM {metrics['ssim']:.4f}   "
        f"LPIPS {metrics['lpips']:.4f}   "
        f"GT FPS {metrics['gt_fps']:.1f}   Pred FPS {metrics['pred_fps']:.1f}",
        y=1.04,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
