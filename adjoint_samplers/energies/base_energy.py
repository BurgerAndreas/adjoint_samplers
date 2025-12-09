# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict
import torch


class BaseEnergy:
    def __init__(self, name, dim):
        super().__init__()
        self.name = name
        self.dim = dim

    # E(x)
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # ∇E(x)
    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)

            E = self.eval(x)
            grad_E = torch.autograd.grad(E.sum(), x, create_graph=False)[0]

        return grad_E

    # score := - ∇E(x)
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return -self.grad_E(x)

    def __call__(self, x: torch.Tensor) -> Dict:
        assert x.ndim == 2 and x.shape[-1] == self.dim

        # forces: ∇E = - ∇ log p, as p(x) = C exp(-E(x))
        output_dict = {}
        output_dict["forces"] = self.grad_E(x)
        return output_dict
