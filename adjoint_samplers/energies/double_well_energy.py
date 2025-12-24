# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from bgflow import Energy as bgflowEnergy
from adjoint_samplers.energies.base_energy import BaseEnergy


# from https://github.com/noegroup/bgflow/blob/main/bgflow/distribution/energy/multi_double_well_potential.py
class MultiDoubleWellPotential(bgflowEnergy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._spatial_dim = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._spatial_dim)
        dists = dists - self._offset

        energies = self._a * dists**4 + self._b * dists**2 + self._c
        return energies.sum(-1, keepdim=True)


class DoubleWellEnergy(BaseEnergy):
    def __init__(
        self,
        dim,
        n_particles,
        device="cpu",
        gad=False,
        **kwargs,
    ):
        # Set the name and dim
        super().__init__(f"dw{n_particles}", dim, gad=gad, **kwargs)

        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles

        self.device = device

        self.multi_double_well = MultiDoubleWellPotential(
            dim=dim,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

    def eval(self, samples: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self.multi_double_well._energy(samples).squeeze(-1) * beta

    def _grad_E(self, samples: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Internal method to compute standard gradients analytically."""
        return self.gradient_analytic(samples) * beta

    def grad_E(self, samples: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Compute gradients for batch of samples.

        If GAD is enabled, computes GAD vector field instead of standard gradient.
        Otherwise uses analytical gradient implementation.
        """
        if self.gad:
            return self._grad_E_gad(samples, beta=beta)
        return self._grad_E(samples, beta=beta)

    def hessian_E(self, samples: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Override to use analytical Hessian implementation."""
        return self.hessian_analytic(samples) * beta

    def gradient_analytic(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient analytically.

        For double well energy: E = sum_{i<j} [a*(d_ij - offset)^4 + b*(d_ij - offset)^2 + c]
        where d_ij = ||x_i - x_j||

        Gradient with respect to x_i:
        ∂E/∂x_i = sum_{j≠i} [4*a*(d_ij - offset)^3 + 2*b*(d_ij - offset)] * (x_i - x_j) / d_ij
        """
        batch_size = samples.shape[0]
        x = samples.reshape(batch_size, self.n_particles, self.n_spatial_dim)

        # Get parameters
        a = self.multi_double_well._a
        b = self.multi_double_well._b
        offset = self.multi_double_well._offset

        # Compute pairwise differences: diff[i,j] = x_i - x_j
        # Shape: (batch, n_particles, n_particles, spatial_dim)
        diff = x[:, :, None, :] - x[:, None, :, :]

        # Compute distances: d_ij = ||x_i - x_j||
        # Shape: (batch, n_particles, n_particles)
        dists = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)

        # Compute d_shifted = d_ij - offset
        d_shifted = dists - offset

        # Compute ∂E/∂d = 4*a*(d - offset)^3 + 2*b*(d - offset)
        # Shape: (batch, n_particles, n_particles)
        dE_dd = 4 * a * d_shifted**3 + 2 * b * d_shifted

        # Compute gradient contribution per pair: (∂E/∂d / d_ij) * (x_i - x_j)
        # Shape: (batch, n_particles, n_particles, spatial_dim)
        grad_contrib = (dE_dd / dists)[:, :, :, None] * diff

        # Sum contributions for each particle (sum over all pairs involving particle i)
        # Shape: (batch, n_particles, spatial_dim)
        grad = grad_contrib.sum(dim=2)

        # Reshape back to (batch, dim)
        return grad.reshape(batch_size, self.dim)

    def hessian_analytic(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute Hessian matrix analytically.

        For double well energy: E = sum_{i<j} [a*(d_ij - offset)^4 + b*(d_ij - offset)^2 + c]
        where d_ij = ||x_i - x_j||

        Hessian H_kl = ∂²E/∂x_k ∂x_l is computed by summing contributions from all pairs.
        For each pair (i,j), the Hessian has contributions to blocks:
        - H[i,i], H[i,j], H[j,i], H[j,j]
        """
        batch_size = samples.shape[0]
        x = samples.reshape(batch_size, self.n_particles, self.n_spatial_dim)

        # Get parameters
        a = self.multi_double_well._a
        b = self.multi_double_well._b
        offset = self.multi_double_well._offset

        # Initialize Hessian: (batch, dim, dim)
        hessian = torch.zeros(
            batch_size, self.dim, self.dim, device=samples.device, dtype=samples.dtype
        )

        # Compute pairwise differences: diff[i,j] = x_i - x_j
        # Shape: (batch, n_particles, n_particles, spatial_dim)
        diff = x[:, :, None, :] - x[:, None, :, :]

        # Compute distances: d_ij = ||x_i - x_j||
        # Shape: (batch, n_particles, n_particles)
        dists = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)

        # Compute d_shifted = d_ij - offset
        d_shifted = dists - offset

        # Compute f(d) = 4*a*(d - offset)^3 + 2*b*(d - offset)
        # Shape: (batch, n_particles, n_particles)
        f_d = 4 * a * d_shifted**3 + 2 * b * d_shifted

        # Compute f'(d) = 12*a*(d - offset)^2 + 2*b
        # Shape: (batch, n_particles, n_particles)
        f_prime_d = 12 * a * d_shifted**2 + 2 * b

        # For each pair (i,j), compute Hessian contributions
        # We only need to consider i < j to avoid double counting
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                # Get pair-specific values
                # Shape: (batch, spatial_dim)
                r = diff[:, i, j, :]  # x_i - x_j
                d = dists[:, i, j]  # ||x_i - x_j||
                d_shifted_pair = d_shifted[:, i, j]
                f_d_pair = f_d[:, i, j]
                f_prime_d_pair = f_prime_d[:, i, j]

                # Normalized direction vector: r_hat = r / d
                # Shape: (batch, spatial_dim)
                r_hat = r / (d[:, None] + 1e-8)

                # Outer product: r_hat ⊗ r_hat
                # Shape: (batch, spatial_dim, spatial_dim)
                r_hat_outer = r_hat[:, :, None] * r_hat[:, None, :]

                # Identity matrix in spatial_dim
                # Shape: (batch, spatial_dim, spatial_dim)
                I = torch.eye(
                    self.n_spatial_dim, device=samples.device, dtype=samples.dtype
                )[None, :, :].expand(batch_size, -1, -1)

                # Hessian block for pair (i,j)
                # H_block = f'(d) * (r_hat ⊗ r_hat) + f(d) * (I - r_hat ⊗ r_hat) / d
                # Shape: (batch, spatial_dim, spatial_dim)
                H_block = f_prime_d_pair[:, None, None] * r_hat_outer + (
                    f_d_pair[:, None, None] / (d[:, None, None] + 1e-8)
                ) * (I - r_hat_outer)

                # Add contributions to Hessian blocks
                # H[i,i] += H_block
                # H[j,j] += H_block
                # H[i,j] -= H_block
                # H[j,i] -= H_block

                i_start = i * self.n_spatial_dim
                i_end = (i + 1) * self.n_spatial_dim
                j_start = j * self.n_spatial_dim
                j_end = (j + 1) * self.n_spatial_dim

                hessian[:, i_start:i_end, i_start:i_end] += H_block
                hessian[:, j_start:j_end, j_start:j_end] += H_block
                hessian[:, i_start:i_end, j_start:j_end] -= H_block
                hessian[:, j_start:j_end, i_start:i_end] -= H_block

        return hessian

    def to(self, device):
        self.multi_double_well.to(device)


# modified from https://github.com/noegroup/bgflow/blob/main/bgflow/utils/geometry.py
def compute_distances(x, n_particles, spatial_dim, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in spatial_dim.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, spatial_dim)
    # distances = torch.cdist(x, x)
    diff = (
        x[:, :, None, :] - x[:, None, :, :]
    )  # Shape: (batch, n_particles, n_particles, spatial_dim)
    distances = torch.sqrt(
        torch.sum(diff**2, axis=-1) + 1e-8
    )  # Shape: (batch, n_particles, n_particles)
    if remove_duplicates:
        distances = distances[
            :, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1
        ]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances


if __name__ == "__main__":
    import time

    # Gradients: Analytic is ~166x faster (0.094 ms vs 15.640 ms)
    # Hessians: Analytic is ~3.84x faster (0.829 ms vs 3.179 ms)

    # Test that gradient_analytic matches grad_E
    torch.manual_seed(42)

    n_particles = 4
    spatial_dim = 2
    dim = n_particles * spatial_dim
    batch_size = 10

    energy = DoubleWellEnergy(dim=dim, n_particles=n_particles, device="cpu")

    # Generate random test samples
    samples = torch.randn(batch_size, dim)

    # Test gradients
    print("Testing gradients...")
    t0 = time.perf_counter()
    grad_autograd = energy._grad_E(samples)
    t1 = time.perf_counter()
    grad_time_autograd = t1 - t0

    t0 = time.perf_counter()
    grad_analytic = energy.gradient_analytic(samples)
    t1 = time.perf_counter()
    grad_time_analytic = t1 - t0

    atol = 1e-5
    rtol = 1e-5
    is_close_grad = torch.allclose(grad_autograd, grad_analytic, atol=atol, rtol=rtol)

    max_diff_grad = (grad_autograd - grad_analytic).abs().max().item()
    mean_diff_grad = (grad_autograd - grad_analytic).abs().mean().item()

    if is_close_grad:
        print(f"✓ Gradient test passed! gradient_analytic matches grad_E")
    else:
        print(f"✗ Gradient test failed! gradient_analytic does not match grad_E")
    print(f"  Max difference: {max_diff_grad:.2e}")
    print(f"  Mean difference: {mean_diff_grad:.2e}")
    print(
        f"  Timing - autograd: {grad_time_autograd * 1000:.3f} ms, analytic: {grad_time_analytic * 1000:.3f} ms"
    )
    print(f"  Speedup: {grad_time_autograd / grad_time_analytic:.2f}x")

    # Test Hessians
    print("\nTesting Hessians...")
    t0 = time.perf_counter()
    hessian_autograd = energy._hessian_E(samples)
    t1 = time.perf_counter()
    hess_time_autograd = t1 - t0

    t0 = time.perf_counter()
    hessian_analytic = energy.hessian_analytic(samples)
    t1 = time.perf_counter()
    hess_time_analytic = t1 - t0

    atol_hess = 1e-4
    rtol_hess = 1e-4
    is_close_hess = torch.allclose(
        hessian_autograd, hessian_analytic, atol=atol_hess, rtol=rtol_hess
    )

    max_diff_hess = (hessian_autograd - hessian_analytic).abs().max().item()
    mean_diff_hess = (hessian_autograd - hessian_analytic).abs().mean().item()

    # Check symmetry (Hessian should be symmetric)
    sym_diff = (
        (hessian_analytic - hessian_analytic.transpose(-2, -1)).abs().max().item()
    )

    if is_close_hess:
        print(f"✓ Hessian test passed! hessian_analytic matches hessian_E")
    else:
        print(f"✗ Hessian test failed! hessian_analytic does not match hessian_E")
    print(f"  Max difference: {max_diff_hess:.2e}")
    print(f"  Mean difference: {mean_diff_hess:.2e}")
    print(f"  Symmetry error: {sym_diff:.2e}")
    print(
        f"  Timing - autograd: {hess_time_autograd * 1000:.3f} ms, analytic: {hess_time_analytic * 1000:.3f} ms"
    )
    print(f"  Speedup: {hess_time_autograd / hess_time_analytic:.2f}x")
