from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def quaternion_to_rotation(quaternion: Tensor) -> Tensor:
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


class GaussianVolume(nn.Module):
    def __init__(
        self,
        means: Tensor,
        features: Tensor,
        initial_scale: float,
        initial_confidence: float,
        minimum_scale: float,
        maximum_scale: float,
        epsilon: float,
        cutoff_sigma: float,
    ) -> None:
        super().__init__()
        n = means.shape[0]
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(
            torch.full((n, 3), math.log(initial_scale), device=means.device)
        )
        identity = torch.zeros((n, 4), device=means.device)
        identity[:, 0] = 1.0
        self.quaternions = nn.Parameter(identity)
        self.confidence_logits = nn.Parameter(
            torch.full((n,), inverse_softplus(initial_confidence), device=means.device)
        )
        feature_probability = features.clamp(1e-4, 1.0 - 1e-4)
        self.feature_logits = nn.Parameter(torch.logit(feature_probability))
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.epsilon = float(epsilon)
        self.cutoff_sigma = float(cutoff_sigma)

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp().clamp(self.minimum_scale, self.maximum_scale)

    @property
    def confidences(self) -> Tensor:
        return F.softplus(self.confidence_logits).clamp_min(1e-8)

    @property
    def features(self) -> Tensor:
        return torch.sigmoid(self.feature_logits)

    def precision6(self) -> Tensor:
        rotation = quaternion_to_rotation(self.quaternions)
        inv_variance = self.scales.reciprocal().square()
        precision = rotation @ torch.diag_embed(inv_variance) @ rotation.transpose(-1, -2)
        return torch.stack(
            (
                precision[:, 0, 0], precision[:, 0, 1], precision[:, 0, 2],
                precision[:, 1, 1], precision[:, 1, 2], precision[:, 2, 2],
            ),
            dim=-1,
        )

    def evaluate(
        self,
        coordinates: Tensor,
        gaussian_chunk: int = 512,
    ) -> Tensor:
        """Autograd-capable normalized Gaussian evaluation at [M,3] xyz points."""
        numerator = torch.zeros(coordinates.shape[0], device=coordinates.device)
        denominator = torch.zeros_like(numerator)
        precision = self.precision6()
        cutoff2 = self.cutoff_sigma ** 2

        for start in range(0, self.means.shape[0], gaussian_chunk):
            stop = min(start + gaussian_chunk, self.means.shape[0])
            delta = coordinates[:, None, :] - self.means[None, start:stop, :]
            p = precision[start:stop]
            dx, dy, dz = delta.unbind(-1)
            mahal = (
                p[None, :, 0] * dx * dx
                + 2 * p[None, :, 1] * dx * dy
                + 2 * p[None, :, 2] * dx * dz
                + p[None, :, 3] * dy * dy
                + 2 * p[None, :, 4] * dy * dz
                + p[None, :, 5] * dz * dz
            )
            weight = torch.exp(-0.5 * mahal) * (mahal < cutoff2)
            weight = weight * self.confidences[None, start:stop]
            numerator = numerator + (weight * self.features[None, start:stop]).sum(dim=1)
            denominator = denominator + weight.sum(dim=1)

        return numerator / (denominator + self.epsilon)

    def clamp_parameters(self, shape: tuple[int, int, int]) -> None:
        d, h, w = shape
        with torch.no_grad():
            self.means[:, 0].clamp_(0.0, float(w))
            self.means[:, 1].clamp_(0.0, float(h))
            self.means[:, 2].clamp_(0.0, float(d))
            self.log_scales.clamp_(
                math.log(self.minimum_scale), math.log(self.maximum_scale)
            )

    def checkpoint(self, shape: tuple[int, int, int]) -> dict:
        return {
            "state_dict": self.state_dict(),
            "shape": shape,
            "minimum_scale": self.minimum_scale,
            "maximum_scale": self.maximum_scale,
            "epsilon": self.epsilon,
            "cutoff_sigma": self.cutoff_sigma,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: torch.device) -> "GaussianVolume":
        state = checkpoint["state_dict"]
        means = state["means"].to(device)
        features = torch.sigmoid(state["feature_logits"].to(device))
        model = cls(
            means=means,
            features=features,
            initial_scale=1.0,
            initial_confidence=1.0,
            minimum_scale=checkpoint["minimum_scale"],
            maximum_scale=checkpoint["maximum_scale"],
            epsilon=checkpoint["epsilon"],
            cutoff_sigma=checkpoint["cutoff_sigma"],
        ).to(device)
        model.load_state_dict({key: value.to(device) for key, value in state.items()})
        return model
