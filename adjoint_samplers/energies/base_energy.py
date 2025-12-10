# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict
import torch


class BaseEnergy:
    def __init__(self, name, dim, gad=False):
        super().__init__()
        self.name = name
        self.dim = dim
        self.gad = gad

    # E(x)
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # ∇E(x) - internal autograd implementation
    def _grad_E(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)

            E = self.eval(x)
            grad_E = torch.autograd.grad(E.sum(), x, create_graph=False)[0]

        return grad_E

    # ∇E(x) - public method (can be overridden by subclasses)
    def grad_E(self, x: torch.Tensor) -> torch.Tensor:
        if self.gad:
            return self._grad_E_gad(x)
        return self._grad_E(x)

    # ∇²E(x) - Hessian matrix - internal autograd implementation
    def _hessian_E(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian matrix H_ij = ∂²E/∂x_i ∂x_j using autograd.

        Returns
        -------
        hessian : torch.Tensor
            Hessian matrix of shape (batch, dim, dim)
        """
        batch_size = x.shape[0]
        dim = x.shape[1]
        hessian = torch.zeros(batch_size, dim, dim, device=x.device, dtype=x.dtype)

        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            E = self.eval(x)
            grad_E = torch.autograd.grad(E.sum(), x, create_graph=True)[0]

            # Compute Hessian by taking gradient of each component of grad_E
            for i in range(dim):
                grad_i = grad_E[:, i]
                hessian_i = torch.autograd.grad(
                    grad_i.sum(), x, create_graph=False, retain_graph=True
                )[0]
                hessian[:, i, :] = hessian_i

        return hessian

    # ∇²E(x) - Hessian matrix - public method (can be overridden by subclasses)
    def hessian_E(self, x: torch.Tensor) -> torch.Tensor:
        return self._hessian_E(x)

    # GAD vector field: ẋ = -∇V(x) + 2⟨∇V, v₁(x)⟩v₁(x)
    def _grad_E_gad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute GAD vector field instead of standard gradient.

        GAD formula: -∇V(x) + 2⟨∇V, v₁(x)⟩v₁(x)
        where v₁(x) is the eigenvector of the Hessian with smallest eigenvalue.
        """
        batch_size = x.shape[0]

        # Compute standard gradient ∇V
        grad_V = self._grad_E(x)

        # Compute Hessian H
        H = self.hessian_E(x)

        # Compute GAD for each sample in batch
        gad_vectors = []
        for i in range(batch_size):
            # Eigendecomposition of Hessian for sample i
            eigenvals, eigenvecs = torch.linalg.eigh(H[i])

            # v₁ is the eigenvector with smallest eigenvalue (first column)
            v1 = eigenvecs[:, 0]

            # ∇V for this sample
            grad_V_i = grad_V[i]

            # Compute ⟨∇V, v₁⟩
            inner_product = torch.dot(grad_V_i, v1)

            # GAD: -∇V + 2⟨∇V, v₁⟩v₁
            gad_i = -grad_V_i + 2 * inner_product * v1
            gad_vectors.append(gad_i)

        return torch.stack(gad_vectors, dim=0)

    # score := - ∇E(x)
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return -self.grad_E(x)

    def __call__(self, x: torch.Tensor) -> Dict:
        assert x.ndim == 2 and x.shape[-1] == self.dim

        # forces: ∇E = - ∇ log p, as p(x) = C exp(-E(x))
        output_dict = {}
        output_dict["forces"] = self.grad_E(x)
        return output_dict
