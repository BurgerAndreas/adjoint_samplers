# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adjoint_samplers.utils.dist_utils import (
    Gauss,
    Delta,
    CenteredParticlesGauss,
    CenteredParticlesHarmonic,
)
from adjoint_samplers.utils.eval_utils import interatomic_dist


def plot_samples_and_distances():
    """Sample from all source distributions and plot samples + distance histograms."""
    device = "cpu"
    num_samples = 1000
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Default parameters matching dw4 problem
    dim = 8
    n_particles = 4
    spatial_dim = 2
    scale = 1.0

    # Create distributions
    distributions = [
        ("Gauss", Gauss(dim=dim, scale=scale, device=device)),
        ("Delta", Delta(dim=dim, device=device)),
        (
            "meanfree",
            CenteredParticlesGauss(
                n_particles=n_particles,
                spatial_dim=spatial_dim,
                scale=scale,
                device=device,
            ),
        ),
        (
            "harmonic",
            CenteredParticlesHarmonic(
                n_particles=n_particles,
                spatial_dim=spatial_dim,
                scale=scale,
                device=device,
            ),
        ),
    ]

    # Create figure with subplots: 4 rows (one per distribution), 2 columns (samples + distances)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Source Distribution Samples and Distance Histograms", fontsize=16)

    for row_idx, (name, dist) in enumerate(distributions):
        print(f"Sampling from {name}...")
        samples = dist.sample([num_samples])
        samples_np = samples.detach().cpu().numpy()

        ax_samples = axes[row_idx, 0]
        ax_distances = axes[row_idx, 1]

        # Plot samples
        if name == "Gauss":
            if dim == 1:
                ax_samples.hist(samples_np.flatten(), bins=50, density=True, alpha=0.7)
                ax_samples.set_xlabel("Value")
                ax_samples.set_ylabel("Density")
                ax_samples.set_title(f"{name} Distribution (1D histogram)")
            else:
                # Plot 2D projection (first 2 dimensions)
                ax_samples.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=1)
                ax_samples.set_xlabel("Dimension 0")
                ax_samples.set_ylabel("Dimension 1")
                ax_samples.set_title(f"{name} Distribution (2D projection)")
                ax_samples.grid(True, alpha=0.3)

        elif name == "Delta":
            # Delta is deterministic - all samples are the same
            unique_values = np.unique(samples_np)
            if len(unique_values) == 1:
                ax_samples.axvline(
                    unique_values[0],
                    color="r",
                    linewidth=2,
                    label=f"Value: {unique_values[0]:.3f}",
                )
                ax_samples.set_xlim(unique_values[0] - 1, unique_values[0] + 1)
                ax_samples.set_ylim(0, 1.1)
                ax_samples.legend()
                ax_samples.set_xlabel("Value")
                ax_samples.set_title(f"{name} Distribution (deterministic)")
            else:
                # Fallback: histogram
                ax_samples.hist(samples_np.flatten(), bins=50, density=True, alpha=0.7)
                ax_samples.set_xlabel("Value")
                ax_samples.set_ylabel("Density")
                ax_samples.set_title(f"{name} Distribution")

        else:
            # Particle-based distributions (meanfree, harmonic)
            # Reshape to (num_samples, n_particles, spatial_dim)
            samples_reshaped = samples_np.reshape(num_samples, n_particles, spatial_dim)

            # Plot particle positions (show a subset of samples)
            n_show = min(100, num_samples)
            for i in range(n_show):
                pos = samples_reshaped[i]
                ax_samples.scatter(pos[:, 0], pos[:, 1], alpha=0.1, s=10, c="blue")

            # Also plot mean positions
            mean_pos = samples_reshaped.mean(axis=0)
            ax_samples.scatter(
                mean_pos[:, 0],
                mean_pos[:, 1],
                s=100,
                c="red",
                marker="x",
                linewidths=3,
                label="Mean",
            )
            ax_samples.set_xlabel("X")
            ax_samples.set_ylabel("Y")
            ax_samples.set_title(
                f"{name} Distribution (particle positions, {n_show} samples)"
            )
            ax_samples.legend()
            ax_samples.grid(True, alpha=0.3)
            ax_samples.set_aspect("equal")

        # Plot distance histograms
        if name in ["meanfree", "harmonic"]:
            # Compute interatomic distances
            distances = (
                interatomic_dist(samples, n_particles, spatial_dim)
                .detach()
                .cpu()
                .numpy()
            )
            distances_flat = distances.reshape(-1)

            ax_distances.hist(distances_flat, bins=50, density=True, alpha=0.7)
            ax_distances.set_xlabel("Interatomic Distance")
            ax_distances.set_ylabel("Density")
            ax_distances.set_title(f"{name} Interatomic Distance Distribution")
            ax_distances.grid(True, alpha=0.3)
        else:
            # For Gauss and Delta, show N/A
            ax_distances.text(
                0.5,
                0.5,
                "N/A\n(Not a particle system)",
                ha="center",
                va="center",
                transform=ax_distances.transAxes,
                fontsize=12,
            )
            ax_distances.set_title(f"{name} Interatomic Distance Distribution")
            ax_distances.set_xticks([])
            ax_distances.set_yticks([])

    plt.tight_layout()

    # Save figure
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "source_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path.resolve()}")

    plt.close()


if __name__ == "__main__":
    plot_samples_and_distances()
