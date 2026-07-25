# gaussian_volume/checkpoint.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from .model import GaussianVolumeModel, inverse_softplus, _inverse_sigmoid_affine


def save_gaussian_checkpoint(
    path: str,
    *,
    model: GaussianVolumeModel,
    optimizer: Optional[Optimizer],
    step: int,
    volume_shape: Sequence[int],
    normalization: dict,
    extra: Optional[dict],
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    gaussians = {
        name: tensor.cpu()
        for name, tensor in model.export_gaussians().items()
    }

    payload = {
        "step": step,
        "volume_shape": tuple(volume_shape),
        "normalization": normalization,
        "gaussians": gaussians,
        "model_config": {
            "minimum_scale": model.minimum_scale,
            "minimum_confidence": model.minimum_confidence,
            "constrain_features": model.constrain_features,
            "feature_min": model.feature_min,
            "feature_max": model.feature_max,
        },
    }

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    if extra:
        payload["extra"] = extra

    torch.save(payload, file_path)


def load_gaussian_checkpoint(
    path: str,
    *,
    model: GaussianVolumeModel,
    optimizer: Optional[Optimizer],
    map_location: torch.device,
    strict: bool,
) -> dict:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    payload = torch.load(file_path, map_location=map_location)
    gaussians = payload["gaussians"]

    with torch.no_grad():
        _load_matching_parameter(model.means, gaussians["means"], strict=strict)
        _load_matching_parameter(
            model.raw_scales,
            inverse_softplus(
                gaussians["scales"].to(map_location), model.minimum_scale
            ),
            strict=strict,
        )
        _load_matching_parameter(model.quats, gaussians["quaternions"], strict=strict)
        _load_matching_parameter(
            model.raw_confidences,
            inverse_softplus(
                gaussians["confidences"].to(map_location), model.minimum_confidence
            ),
            strict=strict,
        )

        if model.constrain_features:
            raw_features = _inverse_sigmoid_affine(
                gaussians["features"].to(map_location),
                model.feature_min,
                model.feature_max,
            )
        else:
            raw_features = gaussians["features"]

        _load_matching_parameter(model.raw_features, raw_features, strict=strict)

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload


def _load_matching_parameter(parameter: Tensor, value: Tensor, *, strict: bool) -> None:
    value = value.to(device=parameter.device, dtype=parameter.dtype)

    if parameter.shape != value.shape:
        if strict:
            raise ValueError(
                f"Checkpoint parameter shape {tuple(value.shape)} does not match "
                f"model parameter shape {tuple(parameter.shape)}."
            )
        parameter.resize_(value.shape)

    parameter.copy_(value)
