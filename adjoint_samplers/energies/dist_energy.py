# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.distributions import Distribution

from adjoint_samplers.energies.base_energy import BaseEnergy


class DistEnergy(BaseEnergy):
    """An energy function given a distribution"""

    def __init__(self, dist: Distribution, device: str = "cpu") -> None:
        super().__init__(name=dist.name, dim=dist.dim)
        self.dist = dist
        self.dist.to(device)
        self.device = device

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        return -self.dist.log_prob(x)

    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        # return analytic score if implemented,
        # otherwise fall back to default autograd
        if hasattr(self.dist, "score"):
            return -self.dist.score(x)
        else:
            return super().grad_E(x)
