# voxel/model.py

from __future__ import annotations

import torch
from torch import Tensor

from .config import VoxelFieldConfig
from .extension.loader import load_voxel_extension


class SirenVoxelField:
    """
    A SIREN continuous field F_theta(x,y,z) fit to a dense voxel volume,
    trained by the hand-written CUDA kernels in extension/siren.cu (no
    torch.autograd involved — forward, analytic backward, and the Adam
    update all run as raw CUDA kernels).
    """

    def __init__(self, config: VoxelFieldConfig, *, device: torch.device) -> None:
        self.config = config
        self.device = device
        self._extension = load_voxel_extension(
            config.hidden_size, config.hidden_layers
        )

        expected = self._extension.siren_param_count()
        self.parameters = self._extension.initialize_siren(
            config.seed, device, config.hidden_omega_0
        )

        if self.parameters.numel() != expected:
            raise RuntimeError(
                f"Compiled extension reports {expected} parameters but "
                f"initialize_siren returned {self.parameters.numel()}."
            )

        self.first_moment = torch.zeros_like(self.parameters)
        self.second_moment = torch.zeros_like(self.parameters)

    @property
    def num_parameters(self) -> int:
        return self.parameters.numel()

    def training_step(self, target_volume: Tensor, iteration: int, learning_rate: float) -> float:
        """
        Run one fused forward + analytic backward pass over a random batch
        of `config.train_batch` voxels, then an Adam update. Returns the
        minibatch MSE loss.
        """
        gradients, batch_loss = self._extension.siren_training_step(
            target_volume,
            self.parameters,
            self.config.grid_size,
            self.config.train_batch,
            iteration,
            self.config.first_omega_0,
            self.config.hidden_omega_0,
        )
        self._extension.adam_update(
            self.parameters,
            gradients,
            self.first_moment,
            self.second_moment,
            iteration,
            learning_rate,
        )
        return batch_loss.item()

    def reconstruct(self, grid_size: int | None = None) -> Tensor:
        """
        Query the SIREN field at every voxel centre of a `grid_size^3` grid
        (defaults to `config.grid_size`).
        """
        return self._extension.reconstruct_voxel_grid(
            self.parameters,
            grid_size or self.config.grid_size,
            self.config.first_omega_0,
            self.config.hidden_omega_0,
        )

    def load_parameters(self, parameters: Tensor) -> None:
        if parameters.numel() != self.num_parameters:
            raise ValueError(
                f"Expected {self.num_parameters} parameters, got {parameters.numel()}."
            )
        self.parameters = parameters.to(device=self.device, dtype=torch.float32).contiguous()
        self.first_moment = torch.zeros_like(self.parameters)
        self.second_moment = torch.zeros_like(self.parameters)
