# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from bgflow import Energy as bgflowEnergy
from bgflow.utils import distance_vectors, distances_from_vectors
from adjoint_samplers.energies.base_energy import BaseEnergy


def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


# from https://github.com/jarridrb/DEM/blob/main/dem/energies/lennardjones_energy.py
class LennardJonesPotential(bgflowEnergy):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

    def _energy(self, x):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = (
            lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor
        )

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(
                *batch_shape
            )
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def _log_prob(self, x):
        return -self._energy(x)


class LennardJonesEnergy(BaseEnergy):
    def __init__(
        self,
        dim,
        n_particles,
        oscillator=True,
        device="cpu",
    ):
        # Set the name and dim
        super().__init__(f"lj{n_particles}", dim)

        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles

        self.device = device

        self.lennard_jones = LennardJonesPotential(
            dim=dim,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            oscillator=oscillator,
            oscillator_scale=1.0,
            two_event_dims=False,
            energy_factor=1.0,
        )

    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        return -self.lennard_jones._log_prob(samples).squeeze(-1)

    # def grad_E(self, samples: torch.Tensor) -> torch.Tensor:
    #     """Override to use analytical gradient implementation."""
    #     return self.gradient_analytic(samples)

    # def hessian_E(self, samples: torch.Tensor) -> torch.Tensor:
    #     """Override to use analytical Hessian implementation."""
    #     return self.hessian_analytic(samples)

    def gradient_analytic(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient analytically.

        For Lennard-Jones energy: E = sum_{i<j} [eps * ((rm/r_ij)^12 - 2*(rm/r_ij)^6)] * energy_factor
        plus optional oscillator term: E_osc = 0.5 * scale * sum_i ||x_i - x_mean||^2

        Gradient with respect to x_i:
        ∂E/∂x_i = sum_{j≠i} [eps * 12*rm^6/r_ij^7 * (1 - rm^6/r_ij^6)] * energy_factor * (x_i - x_j) / r_ij
        plus oscillator term: scale * (x_i - x_mean)
        """
        batch_size = samples.shape[0]
        x = samples.reshape(batch_size, self.n_particles, self.n_spatial_dim)

        # Get parameters
        eps = self.lennard_jones._eps
        rm = self.lennard_jones._rm
        energy_factor = self.lennard_jones._energy_factor
        oscillator = self.lennard_jones.oscillator
        oscillator_scale = self.lennard_jones._oscillator_scale

        # Compute pairwise differences: diff[i,j] = x_i - x_j
        # Shape: (batch, n_particles, n_particles, spatial_dim)
        diff = x[:, :, None, :] - x[:, None, :, :]

        # Compute distances: r_ij = ||x_i - x_j||
        # Shape: (batch, n_particles, n_particles)
        dists = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)

        # Create mask to exclude diagonal (i == j) where distance is zero
        mask = torch.eye(self.n_particles, device=samples.device, dtype=torch.bool)[
            None, :, :
        ].expand(batch_size, -1, -1)

        # Compute rm/r for each pair
        # Shape: (batch, n_particles, n_particles)
        rm_over_r = rm / (dists + 1e-8)

        # Compute ∂E_pair/∂r = eps * 12*rm^6/r^7 * (1 - rm^6/r^6) * energy_factor
        # Shape: (batch, n_particles, n_particles)
        dE_dr = (
            eps
            * 12
            * (rm**6)
            / ((dists + 1e-8) ** 7)
            * (1 - rm_over_r**6)
            * energy_factor
        )

        # Zero out diagonal contributions (i == j)
        dE_dr[mask] = 0.0

        # Compute gradient contribution per pair: (∂E/∂r / r_ij) * (x_i - x_j)
        # Use dists (not dists_masked) for the division to match the actual distances
        # Shape: (batch, n_particles, n_particles, spatial_dim)
        grad_contrib = (dE_dr / (dists + 1e-8))[:, :, :, None] * diff

        # Sum contributions for each particle (sum over all pairs involving particle i)
        # The energy computation uses distances_from_vectors which counts each physical pair twice.
        # However, when we compute the gradient using the full distance matrix and sum over j≠i,
        # we only count each pair once. Since the energy counts each pair twice, we need to
        # multiply by 2 to match.
        # Shape: (batch, n_particles, spatial_dim)
        grad = 2.0 * grad_contrib.sum(dim=2)

        # Add oscillator term if enabled
        if oscillator:
            # Oscillator gradient: ∂E_osc/∂x_i = scale * (x_i - x_mean)
            x_mean = x.mean(dim=1, keepdim=True)  # Shape: (batch, 1, spatial_dim)
            grad_osc = oscillator_scale * (x - x_mean)
            grad = grad + grad_osc

        # Reshape back to (batch, dim)
        return grad.reshape(batch_size, self.dim)

    def hessian_analytic(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute Hessian matrix analytically.

        For Lennard-Jones energy: E = sum_{i<j} [eps * ((rm/r_ij)^12 - 2*(rm/r_ij)^6)] * energy_factor
        plus optional oscillator term: E_osc = 0.5 * scale * sum_i ||x_i - x_mean||^2

        Hessian H_kl = ∂²E/∂x_k ∂x_l is computed by summing contributions from all pairs.
        """
        batch_size = samples.shape[0]
        x = samples.reshape(batch_size, self.n_particles, self.n_spatial_dim)

        # Get parameters
        eps = self.lennard_jones._eps
        rm = self.lennard_jones._rm
        energy_factor = self.lennard_jones._energy_factor
        oscillator = self.lennard_jones.oscillator
        oscillator_scale = self.lennard_jones._oscillator_scale

        # Initialize Hessian: (batch, dim, dim)
        hessian = torch.zeros(
            batch_size, self.dim, self.dim, device=samples.device, dtype=samples.dtype
        )

        # Compute pairwise differences: diff[i,j] = x_i - x_j
        # Shape: (batch, n_particles, n_particles, spatial_dim)
        diff = x[:, :, None, :] - x[:, None, :, :]

        # Compute distances: r_ij = ||x_i - x_j||
        # Shape: (batch, n_particles, n_particles)
        dists = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)

        # Create mask to exclude diagonal (i == j) where distance is zero
        mask = torch.eye(self.n_particles, device=samples.device, dtype=torch.bool)[
            None, :, :
        ].expand(batch_size, -1, -1)
        dists_masked = dists.clone()
        dists_masked[mask] = 1.0  # Set diagonal to 1.0 to avoid division issues

        # Compute rm/r for each pair
        # Shape: (batch, n_particles, n_particles)
        rm_over_r = rm / (dists_masked + 1e-8)

        # Compute f(r) = ∂E/∂r = eps * 12*rm^6/r^7 * (1 - rm^6/r^6) * energy_factor
        # Shape: (batch, n_particles, n_particles)
        f_r = (
            2.0
            * eps
            * 12
            * (rm**6)
            / ((dists_masked + 1e-8) ** 7)
            * (1 - rm_over_r**6)
            * energy_factor
        )

        # Compute f'(r) = ∂²E/∂r²
        # f'(r) = eps * 12*rm^6/r^8 * (13*(rm/r)^6 - 7) * energy_factor
        # Shape: (batch, n_particles, n_particles)
        f_prime_r = (
            2.0
            * eps
            * 12
            * (rm**6)
            / ((dists_masked + 1e-8) ** 8)
            * (13 * rm_over_r**6 - 7)
            * energy_factor
        )

        # For each pair (i,j), compute Hessian contributions
        # We only need to consider i < j to avoid double counting
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                # Get pair-specific values
                # Shape: (batch, spatial_dim)
                r = diff[:, i, j, :]  # x_i - x_j
                d = dists[:, i, j]  # ||x_i - x_j||
                f_r_pair = f_r[:, i, j]
                f_prime_r_pair = f_prime_r[:, i, j]

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
                # H_block = f''(r) * (r_hat ⊗ r_hat) + f'(r) * (I - r_hat ⊗ r_hat) / r
                # where f'(r) = ∂E/∂r and f''(r) = ∂²E/∂r²
                # Shape: (batch, spatial_dim, spatial_dim)
                H_block = f_prime_r_pair[:, None, None] * r_hat_outer + (
                    f_r_pair[:, None, None] / (d[:, None, None] + 1e-8)
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

        # Add oscillator term if enabled
        if oscillator:
            # Oscillator Hessian: H_ij = scale * (δ_ij * (1 - 1/N) - 1/N)
            # For each particle i, j:
            for i in range(self.n_particles):
                for j in range(self.n_particles):
                    i_start = i * self.n_spatial_dim
                    i_end = (i + 1) * self.n_spatial_dim
                    j_start = j * self.n_spatial_dim
                    j_end = (j + 1) * self.n_spatial_dim

                    if i == j:
                        # Diagonal: scale * (1 - 1/N) * I
                        hessian[:, i_start:i_end, j_start:j_end] += (
                            oscillator_scale
                            * (1 - 1.0 / self.n_particles)
                            * torch.eye(
                                self.n_spatial_dim,
                                device=samples.device,
                                dtype=samples.dtype,
                            )[None, :, :]
                        )
                    else:
                        # Off-diagonal: -scale / N * I
                        hessian[:, i_start:i_end, j_start:j_end] -= (
                            oscillator_scale
                            / self.n_particles
                            * torch.eye(
                                self.n_spatial_dim,
                                device=samples.device,
                                dtype=samples.dtype,
                            )[None, :, :]
                        )

        return hessian

    def to(self, device):
        self.lennard_jones.to(device)


if __name__ == "__main__":
    import time

    # Test that gradient_analytic and hessian_analytic match autograd versions
    torch.manual_seed(42)

    for n_particles in [2, 4, 7]:
        for oscillator in [False, True]:
            print(
                f"\n\n",
                "=" * 3,
                f"Testing with n_particles={n_particles} and oscillator={oscillator}...",
            )
            spatial_dim = 3
            dim = n_particles * spatial_dim
            batch_size = 10

            energy = LennardJonesEnergy(
                dim=dim, n_particles=n_particles, oscillator=oscillator, device="cpu"
            )

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
            is_close_grad = torch.allclose(
                grad_autograd, grad_analytic, atol=atol, rtol=rtol
            )

            max_diff_grad = (grad_autograd - grad_analytic).abs().max().item()
            mean_diff_grad = (grad_autograd - grad_analytic).abs().mean().item()

            if is_close_grad:
                print(f"✓ Gradient test passed! gradient_analytic matches _grad_E")
            else:
                print(
                    f"✗ Gradient test failed! gradient_analytic does not match _grad_E"
                )
                # print(f"  Autograd: \n{grad_autograd}")
                # print(f"  Analytic: \n{grad_analytic}")
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
                (hessian_analytic - hessian_analytic.transpose(-2, -1))
                .abs()
                .max()
                .item()
            )

            if is_close_hess:
                print(f"✓ Hessian test passed! hessian_analytic matches _hessian_E")
            else:
                print(
                    f"✗ Hessian test failed! hessian_analytic does not match _hessian_E"
                )
                # print(f"  Autograd: \n{hessian_autograd}")
                # print(f"  Analytic: \n{hessian_analytic}")
            print(f"  Max difference: {max_diff_hess:.2e}")
            print(f"  Mean difference: {mean_diff_hess:.2e}")
            print(f"  Symmetry error: {sym_diff:.2e}")
            print(
                f"  Timing - autograd: {hess_time_autograd * 1000:.3f} ms, analytic: {hess_time_analytic * 1000:.3f} ms"
            )
            print(f"  Speedup: {hess_time_autograd / hess_time_analytic:.2f}x")
