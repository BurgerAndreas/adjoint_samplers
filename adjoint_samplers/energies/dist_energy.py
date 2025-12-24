# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.distributions import Distribution

from adjoint_samplers.energies.base_energy import BaseEnergy


class DistEnergy(BaseEnergy):
    """An energy function given a distribution"""

    def __init__(
        self, dist: Distribution, device: str = "cpu", gad: bool = False, **kwargs
    ) -> None:
        super().__init__(name=dist.name, dim=dist.dim, gad=gad, **kwargs)
        self.dist = dist
        self.dist.to(device)
        self.device = device

    def eval(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return -self.dist.log_prob(x) * beta

    def grad_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        # return analytic score if implemented,
        # otherwise fall back to default autograd
        if hasattr(self.dist, "score"):
            return -self.dist.score(x) * beta
        else:
            return super().grad_E(x, beta=beta)
