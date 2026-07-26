# voxel/lr_schedule.py

from __future__ import annotations

import math


def cosine_lr_schedule(
    iteration: int,
    *,
    iterations: int,
    base_lr: float,
    min_ratio: float,
    warmup_steps: int,
    warmup_init_factor: float,
) -> float:
    """
    Linear warmup over the first warmup_steps iterations (from
    warmup_init_factor * base_lr up to base_lr), then cosine annealing from
    base_lr down to min_ratio * base_lr over the remaining iterations —
    same shape as gaussian_volume's [scheduler] convention
    (configs/config.toml's minimum_lr_ratio/warmup_steps), applied to the
    single flat learning rate this CUDA training loop uses (no
    per-parameter-group optimizer to schedule separately).

    min_ratio=1.0 makes this a no-op (constant base_lr throughout) —
    configs/siren.yml's default before the schedule is intentionally
    enabled.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}.")

    if base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}.")

    if not (0.0 < min_ratio <= 1.0):
        raise ValueError(f"min_ratio must be in (0, 1], got {min_ratio}.")

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    if warmup_steps >= iterations:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be less than iterations ({iterations})."
        )

    if not (0.0 < warmup_init_factor <= 1.0):
        raise ValueError(
            f"warmup_init_factor must be in (0, 1], got {warmup_init_factor}."
        )

    if warmup_steps > 0 and iteration <= warmup_steps:
        return base_lr * (
            warmup_init_factor + (1.0 - warmup_init_factor) * iteration / warmup_steps
        )

    min_lr = min_ratio * base_lr
    progress = (iteration - warmup_steps) / (iterations - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
