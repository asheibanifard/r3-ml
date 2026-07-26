# voxel/config.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoxelFieldConfig:
    """
    Every tunable constant from the original main.cu, now runtime-configurable.

    hidden_size/hidden_layers are compile-time for the CUDA extension (see
    extension/loader.py) — changing either triggers a JIT recompile the
    first time that combination is used. Everything else (grid_size, image
    dimensions, train_batch, dvr_steps, density_scale, omegas) is a plain
    kernel argument and never triggers a recompile.

    All fields are required — no numerical defaults, matching this
    project's convention that every value must be supplied explicitly by
    the caller (CLI or config file).
    """

    grid_size: int
    image_width: int
    image_height: int
    hidden_size: int
    hidden_layers: int
    train_batch: int
    dvr_steps: int
    density_scale: float
    seed: int
    first_omega_0: float
    hidden_omega_0: float

    def __post_init__(self) -> None:
        for name in (
            "grid_size", "image_width", "image_height",
            "hidden_size", "hidden_layers", "train_batch", "dvr_steps",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.density_scale <= 0:
            raise ValueError(
                f"density_scale must be positive, got {self.density_scale}."
            )

        if self.first_omega_0 <= 0:
            raise ValueError(
                f"first_omega_0 must be positive, got {self.first_omega_0}."
            )

        if self.hidden_omega_0 <= 0:
            raise ValueError(
                f"hidden_omega_0 must be positive, got {self.hidden_omega_0}."
            )
