# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict
import torch


class BaseEnergy:
    def __init__(
        self,
        name,
        dim,
        gad=False,
        norm_gad_outside_ts=False,
        norm_gad_outside_ts_2=False,
        newton_raphson=False,
        newton_raphson_2=False,
        newton_raphson_then_norm_outside_ts=False,
        newton_raphson_outside_ts=False,
    ):
        super().__init__()
        self.name = name
        self.dim = dim
        self.gad = gad
        self.norm_gad_outside_ts = norm_gad_outside_ts
        self.norm_gad_outside_ts_2 = norm_gad_outside_ts_2
        self.newton_raphson = newton_raphson
        self.newton_raphson_2 = newton_raphson_2
        self.newton_raphson_then_norm_outside_ts = newton_raphson_then_norm_outside_ts
        self.newton_raphson_outside_ts = newton_raphson_outside_ts

    def check_norm_strategy(self) -> bool:
        # only at max one can be true at the same time
        _all = [
            self.norm_gad_outside_ts,
            self.norm_gad_outside_ts_2,
            self.newton_raphson,
            self.newton_raphson_2,
            self.newton_raphson_then_norm_outside_ts,
            self.newton_raphson_outside_ts,
        ]
        return sum(_all) <= 1

    # E(x)
    def eval(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        raise NotImplementedError()

    # ∇E(x) - internal autograd implementation
    def _grad_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)

            E = self.eval(x, beta=beta)
            grad_E = torch.autograd.grad(E.sum(), x, create_graph=False)[0]

        return grad_E

    # ∇E(x) - public method (can be overridden by subclasses)
    def grad_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        if self.gad:
            return self._grad_E_gad(x, beta=beta)
        return self._grad_E(x, beta=beta)

    # ∇²E(x) - Hessian matrix - internal autograd implementation
    def _hessian_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
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
            E = self.eval(x, beta=beta)
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
    def hessian_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self._hessian_E(x, beta=beta)

    # GAD vector field: ẋ = -∇V(x) + 2⟨∇V, v1(x)⟩v1(x)
    def _grad_E_gad(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Compute GAD vector field instead of standard gradient.

        GAD formula: -∇V(x) + 2⟨∇V, v1(x)⟩v1(x)
        where v1(x) is the eigenvector of the Hessian with smallest eigenvalue.
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

            # v1 is the eigenvector with smallest eigenvalue (first column)
            v1 = eigenvecs[:, 0]

            # ∇V for this sample
            grad_V_i = grad_V[i]

            # Compute ⟨∇V, v1⟩
            inner_product = torch.dot(grad_V_i, v1)

            # GAD: -∇V + 2⟨∇V, v1⟩v1
            gad_i = -grad_V_i + 2 * inner_product * v1

            # Optionally norm GAD to unity if eigenvalue product > 0
            if self.norm_gad_outside_ts:
                eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
                gad_mag = torch.linalg.norm(gad_i, dim=-1, keepdim=True)
                # Avoid division by zero if you want gradients
                # gad_mag += 1e-10
                # Only norm where eigenvalue product > 0 and norm < 1
                # (make small vectors larger, never smaller)
                min_norm = 1.0  # minimum norm we want
                mask = (eigval_product > 0) & (gad_mag.squeeze(-1) < min_norm)
                gad_i = torch.where(
                    mask.unsqueeze(-1),
                    gad_i / gad_mag,  # norm to unity
                    gad_i,  # Leave unchanged
                )

            if self.norm_gad_outside_ts_2:
                eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
                gad_mag = torch.linalg.norm(gad_i, dim=-1, keepdim=True)
                # Avoid division by zero if you want gradients
                # gad_mag += 1e-10
                mask = eigval_product > 0
                min_norm = 1.0  # minimum norm we want
                # only magnify small vectors, never shrink large ones
                clip_coef = torch.clamp(min_norm / gad_mag, min=1.0)
                # Only norm where eigenvalue product > 0
                gad_i = torch.where(
                    mask.unsqueeze(-1),
                    gad_i * clip_coef,  # Rescale
                    gad_i,  # Leave unchanged
                )

            if self.newton_raphson:
                # compute |H|^-1 * gad
                L = eigenvals  # B,2
                V = eigenvecs  # B,2,2
                # gad: B,2
                # Project Force onto Eigenbasis (change of coordinates)
                # Coefficients c_i = v_i dot F # B,2
                coeffs = (V.transpose(-1, -2) @ gad_i.unsqueeze(-1)).squeeze(-1)

                # Scale coefficients by 1/|lambda|
                L_abs = torch.abs(L)
                mask = L_abs > 1e-8

                scaled_coeffs = torch.zeros_like(coeffs)
                scaled_coeffs[mask] = coeffs[mask] / L_abs[mask]

                # Project back to spatial coordinates
                # step = sum(scaled_c_i * v_i)
                gad_i = (V @ scaled_coeffs.unsqueeze(-1)).squeeze(-1)
                
            if self.newton_raphson_outside_ts:
                # compute |H|^-1 * gad
                L = eigenvals  # B,2
                V = eigenvecs  # B,2,2
                # gad: B,2
                # Project Force onto Eigenbasis (change of coordinates)
                # Coefficients c_i = v_i dot F # B,2
                coeffs = (V.transpose(-1, -2) @ gad_i.unsqueeze(-1)).squeeze(-1)

                # Scale coefficients by 1/|lambda|
                L_abs = torch.abs(L)
                mask = L_abs > 1e-8

                scaled_coeffs = torch.zeros_like(coeffs)
                scaled_coeffs[mask] = coeffs[mask] / L_abs[mask]

                # Project back to spatial coordinates
                # step = sum(scaled_c_i * v_i)
                gad_scaled = (V @ scaled_coeffs.unsqueeze(-1)).squeeze(-1)
                
                eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
                mask = (eigval_product > 0)
                gad_i = torch.where(
                    mask.unsqueeze(-1),
                    gad_scaled, 
                    gad_i,  # Leave unchanged
                )

            if self.newton_raphson_2:
                abs_eigvals = torch.abs(eigenvals)

                # To avoid division by zero, only invert non-zero eigenvalues
                mask = abs_eigvals > 1e-8
                inv_abs_eigvals = torch.zeros_like(abs_eigvals)
                inv_abs_eigvals[mask] = 1.0 / abs_eigvals[mask]

                # Reconstruct the matrix: V @ diag(1/|L|) @ V^T (batched)
                # Scale columns of V by inv_abs_eigvals, then batched matmul with V^T.
                V = eigenvecs  # (B, 2, 2)
                H_abs_inv = (V * inv_abs_eigvals.unsqueeze(-2)) @ V.transpose(-1, -2)

                gad_i = (H_abs_inv @ gad_i.unsqueeze(-1)).squeeze(-1)

            if self.newton_raphson_then_norm_outside_ts:
                # compute |H|^-1 * gad
                L = eigenvals  # B,2
                V = eigenvecs  # B,2,2
                # gad: B,2
                # Project Force onto Eigenbasis (change of coordinates)
                # Coefficients c_i = v_i dot F # B,2
                coeffs = (V.transpose(-1, -2) @ gad_i.unsqueeze(-1)).squeeze(-1)

                # Scale coefficients by 1/|lambda|
                L_abs = torch.abs(L)
                mask = L_abs > 1e-8

                scaled_coeffs = torch.zeros_like(coeffs)
                scaled_coeffs[mask] = coeffs[mask] / L_abs[mask]

                # Project back to spatial coordinates
                # step = sum(scaled_c_i * v_i)
                gad_i = (V @ scaled_coeffs.unsqueeze(-1)).squeeze(-1)

                # norm gad to unity if eigenvalue product > 0
                eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
                gad_mag = torch.linalg.norm(gad_i, dim=-1, keepdim=True)
                # Avoid division by zero if you want gradients
                # gad_mag += 1e-10
                mask = eigval_product > 0
                min_norm = 1.0  # minimum norm we want

                # # only magnify small vectors, never shrink large ones
                # clip_coef = torch.clamp(min_norm / gad_mag, min=1.0)
                # # Only norm where eigenvalue product > 0
                # gad = torch.where(
                #     mask.unsqueeze(-1),
                #     gad * clip_coef,  # Rescale
                #     gad,  # Leave unchanged
                # )

                mask = (eigval_product > 0) & (gad_mag.squeeze(-1) < min_norm)
                gad_i = torch.where(
                    mask.unsqueeze(-1),
                    gad_i / gad_mag,  # norm to unity
                    gad_i,  # Leave unchanged
                )
                
            # GAD is a "force", we need to return the "gradient"
            gad_i = -gad_i * beta
            gad_vectors.append(gad_i)

        return torch.stack(gad_vectors, dim=0)

    # score := - ∇E(x)
    def score(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return -self.grad_E(x, beta=beta)

    def __call__(self, x: torch.Tensor, beta: float = 1.0) -> Dict:
        assert x.ndim == 2 and x.shape[-1] == self.dim

        # forces: ∇E = - ∇ log p, as p(x) = C exp(-E(x))
        output_dict = {}
        output_dict["forces"] = self.grad_E(x, beta=beta)
        return output_dict
