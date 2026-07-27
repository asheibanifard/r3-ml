from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def gaussian_window_3d(size: int, sigma: float, device: torch.device) -> Tensor:
    axis = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel = kernel / kernel.sum()
    window = (
        kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    )
    return window[None, None]


def ssim3d(prediction: Tensor, target: Tensor, window_size: int, sigma: float) -> Tensor:
    window = gaussian_window_3d(window_size, sigma, prediction.device)
    padding = window_size // 2
    mu_x = F.conv3d(prediction, window, padding=padding)
    mu_y = F.conv3d(target, window, padding=padding)
    sigma_x = F.conv3d(prediction * prediction, window, padding=padding) - mu_x.square()
    sigma_y = F.conv3d(target * target, window, padding=padding) - mu_y.square()
    sigma_xy = F.conv3d(prediction * target, window, padding=padding) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()
