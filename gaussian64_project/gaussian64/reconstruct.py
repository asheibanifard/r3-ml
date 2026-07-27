from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gaussian64.config import load_config
from gaussian64.io import save_json, save_mips, save_tiff
from gaussian64.jit import reconstruct_jit, reconstruct_torch
from gaussian64.losses import ssim3d
from gaussian64.model import GaussianVolume
from gaussian64.synthetic import make_synthetic_volume


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("jit", "torch"), default=None)
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    backend = args.backend or config["reconstruction"]["backend"]
    if backend == "jit" and not torch.cuda.is_available():
        raise RuntimeError("The JIT backend requires CUDA; pass --backend torch for CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    shape = tuple(checkpoint["shape"])
    model = GaussianVolume.from_checkpoint(checkpoint, device).eval()

    if backend == "jit":
        reconstruction = reconstruct_jit(model, shape, args.verbose_build)
    else:
        reconstruction = reconstruct_torch(
            model,
            shape,
            voxel_chunk=config["reconstruction"]["voxel_chunk"],
            gaussian_chunk=config["reconstruction"]["gaussian_chunk"],
        )

    target = make_synthetic_volume(shape, device)
    error = (target - reconstruction).abs()
    mse = torch.mean((target - reconstruction) ** 2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    metric_ssim = ssim3d(
        reconstruction[None, None],
        target[None, None],
        config["training"]["ssim_window_size"],
        config["training"]["ssim_sigma"],
    )

    output = Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_tiff(output / "target.tif", target)
    save_tiff(output / "reconstruction.tif", reconstruction)
    save_tiff(output / "error.tif", error)
    save_mips(output, "target", target)
    save_mips(output, "reconstruction", reconstruction)
    metrics = {
        "backend": backend,
        "gaussians": int(model.means.shape[0]),
        "mse": float(mse),
        "psnr_db": float(psnr),
        "ssim3d": float(metric_ssim),
        "maximum_absolute_error": float(error.max()),
    }
    save_json(output / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
