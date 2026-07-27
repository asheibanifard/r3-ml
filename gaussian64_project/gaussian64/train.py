from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gaussian64.config import load_config
from gaussian64.io import save_tiff
from gaussian64.losses import ssim3d
from gaussian64.model import GaussianVolume
from gaussian64.synthetic import make_synthetic_volume


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def initialize_from_target(
    target: torch.Tensor,
    n: int,
    uniform_fraction: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    d, h, w = target.shape
    flat = target.flatten()
    weighted_count = int(round(n * (1.0 - uniform_fraction)))
    uniform_count = n - weighted_count
    probability = (flat + 1e-4) / (flat + 1e-4).sum()
    weighted = torch.multinomial(probability, weighted_count, replacement=True)
    uniform = torch.randint(0, flat.numel(), (uniform_count,), device=target.device)
    indices = torch.cat((weighted, uniform))

    z = indices // (h * w)
    y = (indices // w) % h
    x = indices % w
    means = torch.stack((x, y, z), dim=-1).float() + 0.5
    means = means + 0.20 * torch.randn_like(means)
    features = target[z, y, x]
    return means, features


def patch_coordinates(
    origin: tuple[int, int, int],
    size: int,
    device: torch.device,
) -> torch.Tensor:
    z0, y0, x0 = origin
    z, y, x = torch.meshgrid(
        torch.arange(z0, z0 + size, device=device, dtype=torch.float32) + 0.5,
        torch.arange(y0, y0 + size, device=device, dtype=torch.float32) + 0.5,
        torch.arange(x0, x0 + size, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Training configuration expects a CUDA-capable PyTorch installation.")

    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    volume_config = config["volume"]
    shape = (
        volume_config["depth"],
        volume_config["height"],
        volume_config["width"],
    )
    output = Path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    target = make_synthetic_volume(shape, device)
    save_tiff(output / "target.tif", target)

    model_config = config["model"]
    means, features = initialize_from_target(target, model_config["n_gaussians"])
    model = GaussianVolume(
        means=means,
        features=features,
        initial_scale=model_config["initial_scale"],
        initial_confidence=model_config["initial_confidence"],
        minimum_scale=model_config["minimum_scale"],
        maximum_scale=model_config["maximum_scale"],
        epsilon=model_config["epsilon"],
        cutoff_sigma=model_config["cutoff_sigma"],
    ).to(device)

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": optimizer_config["mean_lr"]},
            {"params": [model.log_scales], "lr": optimizer_config["scale_lr"]},
            {"params": [model.quaternions], "lr": optimizer_config["rotation_lr"]},
            {"params": [model.confidence_logits], "lr": optimizer_config["confidence_lr"]},
            {"params": [model.feature_logits], "lr": optimizer_config["feature_lr"]},
        ]
    )

    training = config["training"]
    patch = training["patch_size"]
    if patch > min(shape):
        raise ValueError("patch_size cannot exceed a volume dimension")

    for step in range(1, training["iterations"] + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        total_l1 = torch.zeros((), device=device)
        total_ssim = torch.zeros((), device=device)

        for _ in range(training["patches_per_step"]):
            z0 = random.randint(0, shape[0] - patch)
            y0 = random.randint(0, shape[1] - patch)
            x0 = random.randint(0, shape[2] - patch)
            coordinates = patch_coordinates((z0, y0, x0), patch, device)
            prediction = model.evaluate(coordinates).reshape(1, 1, patch, patch, patch)
            reference = target[
                z0 : z0 + patch,
                y0 : y0 + patch,
                x0 : x0 + patch,
            ][None, None]
            l1 = F.l1_loss(prediction, reference)
            ssim = ssim3d(
                prediction,
                reference,
                training["ssim_window_size"],
                training["ssim_sigma"],
            )
            loss = training["l1_weight"] * l1 + training["ssim_weight"] * (1.0 - ssim)
            total_loss = total_loss + loss / training["patches_per_step"]
            total_l1 = total_l1 + l1.detach() / training["patches_per_step"]
            total_ssim = total_ssim + ssim.detach() / training["patches_per_step"]

        total_loss.backward()
        optimizer.step()
        model.clamp_parameters(shape)

        if step == 1 or step % training["log_every"] == 0:
            print(
                f"step={step:05d} loss={total_loss.item():.6f} "
                f"l1={total_l1.item():.6f} ssim={total_ssim.item():.5f}"
            )

        if step % training["checkpoint_every"] == 0:
            torch.save(model.checkpoint(shape), output / f"model_step_{step:05d}.pt")

    torch.save(model.checkpoint(shape), output / "model.pt")
    print(f"Saved {output / 'model.pt'}")


if __name__ == "__main__":
    main()
