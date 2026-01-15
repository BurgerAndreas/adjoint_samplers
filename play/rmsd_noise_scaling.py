#!/usr/bin/env python3
"""
Script to compute and plot how RMSD scales with noise added to individual particles.
Tests 3, 4, and 5 particles in 3D structures where all particles are 2.5 units apart,
then adds increasing Gaussian noise and computes RMSD between original and noisy systems.
Averages over 10 systems per noise level.
"""

from tkinter import Frame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation
import warnings
from pathlib import Path

def compute_rmsd(coords1, coords2):
    """
    Compute RMSD between two sets of coordinates after optimal alignment.

    Args:
        coords1, coords2: numpy arrays of shape (N, 3) where N is number of atoms

    Returns:
        float: RMSD value
    """
    # Center both structures
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)

    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Find optimal rotation using Kabsch algorithm
    # Compute covariance matrix
    H = np.dot(coords1_centered.T, coords2_centered)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Check for reflection
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))

    # Optimal rotation matrix
    R = np.dot(Vt.T, np.dot(np.diag([1, 1, d]), U.T))

    # Apply rotation to coords2
    coords2_aligned = np.dot(coords2_centered, R)

    # Compute RMSD
    diff = coords1_centered - coords2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd

def create_reference_structure(n_particles):
    """
    Create reference structure with n_particles where all particles are approximately 2.5 units apart.

    Args:
        n_particles: number of particles (3, 4, 5, 8, or 16)

    Returns:
        numpy array: coordinates of shape (n_particles, 3)
    """
    if n_particles == 3:
        # Equilateral triangle
        coords = np.zeros((3, 3))
        coords[0] = [0, 0, 0]
        coords[1] = [2.5, 0, 0]
        coords[2] = [1.25, 2.5 * np.sqrt(3) / 2, 0]

    elif n_particles == 4:
        # Regular tetrahedron
        # Using standard tetrahedron coordinates
        coords = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)

        # Scale to have edge length 2.5
        # Current edge length between [1,1,1] and [1,-1,-1] is sqrt((0)^2 + (2)^2 + (2)^2) = sqrt(8) ≈ 2.828
        # So scale factor = 2.5 / 2.828 ≈ 0.883
        scale_factor = 2.5 / np.sqrt(8)
        coords *= scale_factor

    elif n_particles == 5:
        # Trigonal bipyramid - note: cannot have all pairwise distances equal for 5 particles in 3D
        coords = np.zeros((5, 3))

        # Axial particles along z-axis at distance that makes axial-equatorial = 2.5
        h = 2.5 / np.sqrt(3)  # Height from center to axial particles
        coords[0] = [0, 0, h]   # Top axial
        coords[1] = [0, 0, -h]  # Bottom axial

        # Equatorial particles in xy plane at distance that makes equatorial-equatorial = 2.5
        r = 2.5 / np.sqrt(3)  # Distance from center to equatorial particles
        for i in range(3):
            angle = 2 * np.pi * i / 3
            coords[i+2] = [r * np.cos(angle), r * np.sin(angle), 0]

    elif n_particles == 8:
        # Cube structure
        # Place particles at corners of a cube with edge length 2.5
        coords = np.zeros((8, 3))
        corners = [
            [0, 0, 0], [2.5, 0, 0], [2.5, 2.5, 0], [0, 2.5, 0],
            [0, 0, 2.5], [2.5, 0, 2.5], [2.5, 2.5, 2.5], [0, 2.5, 2.5]
        ]
        coords = np.array(corners, dtype=float)

    elif n_particles == 16:
        # 4x4 grid structure in 3D space
        # Create a 4x4x4 grid but only use 16 points with approximately 2.5 spacing
        coords = np.zeros((16, 3))
        spacing = 2.5
        idx = 0
        for i in range(4):
            for j in range(4):
                x = i * spacing
                y = j * spacing
                z = ((i + j) % 2) * spacing  # Alternate z levels for better distribution
                coords[idx] = [x, y, z]
                idx += 1

    else:
        raise ValueError(f"Unsupported number of particles: {n_particles}. Use 3, 4, 5, 8, or 16.")

    return coords

def add_noise_to_structure(coords, noise_level):
    """
    Add Gaussian noise to each particle's coordinates.

    Args:
        coords: numpy array of shape (N, 3)
        noise_level: standard deviation of Gaussian noise

    Returns:
        numpy array: noisy coordinates
    """
    noise = np.random.normal(0, noise_level, coords.shape)
    return coords + noise

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Set up seaborn poster theme with 0.7 font scaling
    sns.set_theme(style="white", context="poster", font_scale=0.7)
    colours = sns.color_palette("deep", 5)

    # Particle counts to test
    particle_counts = [3, 4, 5, 8, 16]

    # Noise levels to test
    noise_levels = np.linspace(0.0, 2.0, 21)  # 0.0 to 2.0 in 0.1 increments

    # Number of systems per noise level
    n_systems = 100

    # Plot setup
    plt.figure(figsize=(12, 8))

    for i_particle, n_particles in enumerate(particle_counts):
        colour = colours[i_particle]
        print(f"\n{'='*50}")
        print(f"Computing RMSD for {n_particles} particles")

        # Create reference structure
        ref_coords = create_reference_structure(n_particles)
        print(f"Reference structure created with {len(ref_coords)} particles")
        print("Particle positions:")
        for i, pos in enumerate(ref_coords):
            print(f"  Particle {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Verify all pairwise distances are 2.5
        distances = []
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                dist = np.linalg.norm(ref_coords[i] - ref_coords[j])
                distances.append(dist)
        print(f"All pairwise distances: {distances}")
        print(f"Target distance: 2.5, Max deviation: {max(abs(d-2.5) for d in distances):.6f}")

        # Store results for this particle count
        avg_rmsds = []
        std_rmsds = []

        print("\nComputing RMSD for different noise levels...")

        for noise_level in noise_levels:
            rmsds = []

            for _ in range(n_systems):
                # Create noisy version
                noisy_coords = add_noise_to_structure(ref_coords, noise_level)

                # Compute RMSD
                rmsd = compute_rmsd(ref_coords, noisy_coords)
                rmsds.append(rmsd)

            # Average over systems
            avg_rmsd = np.mean(rmsds)
            std_rmsd = np.std(rmsds)

            avg_rmsds.append(avg_rmsd)
            std_rmsds.append(std_rmsd)

            print(f"Noise level {noise_level:.1f}: Avg RMSD = {avg_rmsd:.3f} ± {std_rmsd:.3f}")

        # Add dodge offset for error bars
        dodge_offset = (i_particle - 1) * 0.02  # -0.02, 0.0, +0.02 for 3, 4, 5 particles
        x_dodged = noise_levels + dodge_offset

        # Plot results for this particle count
        plt.errorbar(x_dodged, avg_rmsds, yerr=std_rmsds,
                     fmt='o-', capsize=3, markersize=4, linewidth=2, 
                     color=colour,
                     label=f'{n_particles} particles')

    # Finalize plot
    plt.xlabel('Noise Level (Standard Deviation)')
    plt.ylabel('RMSD')
    plt.title('RMSD Scaling with Gaussian Noise')
    plt.legend(frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.01)

    # Save plot
    fname = Path("plots") / "rmsd_noise_scaling_3d.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"\nPlot saved as {fname.resolve()}")

if __name__ == "__main__":
    main()
