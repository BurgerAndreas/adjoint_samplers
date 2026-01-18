import io
import os
import sys
import math
import multiprocessing as mp
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import PIL
import PIL.ImageDraw
import pymol
from pymol import cmd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import wandb
import hdbscan
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

from adjoint_samplers.utils.align_unordered_mols import (
    REORDER_HUNGARIAN,
    rmsd_unordered_from_numpy,
)
from adjoint_samplers.energies.scine_energy import (
    element_type_to_symbol,
)
from adjoint_samplers.utils.ase_utils import (
    samples_to_ase_atoms,
    _get_atom_types_from_energy,
)
from ase import Atoms
from dscribe.descriptors import MBTR
import warnings

warnings.filterwarnings("ignore", message=".*using precomputed metric.*")
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # https://stackoverflow.com/a/61756899
    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )

def _rmsd_row_block(args):
    """
    Compute a block of the upper-triangular RMSD matrix for rows [start_idx, end_idx).
    Returns (start_idx, end_idx, block_matrix).
    """
    start_idx, end_idx, coords, atoms = args
    n = coords.shape[0]
    block = np.zeros((end_idx - start_idx, n))
    for local_i, i in enumerate(range(start_idx, end_idx)):
        for j in range(i + 1, n):
            rmsd_val = rmsd_unordered_from_numpy(
                atoms,
                coords[i],
                atoms,
                coords[j],
                reorder=True,
                reorder_method_str=REORDER_HUNGARIAN,
            )
            block[local_i, j] = rmsd_val
    return start_idx, end_idx, block

def compute_rmsd_matrix(samples, energy, cfg, eval_dir, eval_dict, tag="rmsd"):
    """Compute pairwise RMSD matrix for samples."""
    coords = (
        samples.detach().cpu().numpy().reshape(samples.shape[0], energy.n_particles, -1)
    )
    if coords.shape[2] == 2:
        pad = np.zeros((coords.shape[0], energy.n_particles, 1))
        coords = np.concatenate([coords, pad], axis=2)
    atoms = np.ones(energy.n_particles, dtype=int)
    n = coords.shape[0]
    rmsd_matrix = np.zeros((n, n))

    num_workers = min(mp.cpu_count(), getattr(cfg, "rmsd_num_workers", 10))
    block_size = max(1, math.ceil(n / num_workers))
    tasks = []
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        tasks.append((start, end, coords, atoms))
    with mp.Pool(processes=num_workers) as pool:
        for start_idx, end_idx, block in tqdm(
            pool.imap_unordered(_rmsd_row_block, tasks),
            total=len(tasks),
            desc="Computing RMSD matrix (parallel)",
        ):
            rmsd_matrix[start_idx:end_idx] += block
    rmsd_matrix = rmsd_matrix + rmsd_matrix.T

    # # Plot RMSD matrix as heatmap
    # fig, ax = plt.subplots(figsize=(10, 8))
    # sns.heatmap(
    #     rmsd_matrix,
    #     cmap="viridis",
    #     square=True,
    #     cbar_kws={"label": "RMSD"},
    #     ax=ax,
    # )
    # ax.set_title(f"Pairwise RMSD Matrix ({tag})")
    # ax.set_xlabel("Sample Index")
    # ax.set_ylabel("Sample Index")
    # plt.tight_layout()
    # fig.canvas.draw()
    # rmsd_heatmap_img = fig2img(fig)
    # fname = eval_dir / f"rmsd_heatmap_{tag}.png"
    # rmsd_heatmap_img.save(fname)
    # print(f"Saved RMSD heatmap ({tag}) to\n {fname.resolve()}")
    # plt.close(fig)
    # eval_dict[f"rmsd_heatmap_{tag}"] = wandb.Image(rmsd_heatmap_img)

    return rmsd_matrix


def cluster_density_peaks(distance_matrix, samples, energy, cfg, eval_dir, eval_dict, tag="density_peaks", beta: float = 1.0):
    """
    Density Peaks clustering (Rodriguez & Laio, 2014) on precomputed distance matrix.
    
    This method identifies cluster centers as points with high local density and 
    large distance from points of higher density. It's particularly effective for 
    finding modes in molecular conformational space.
    
    Args:
        distance_matrix: Precomputed pairwise distance matrix (n_samples x n_samples)
        samples: Sample coordinates
        energy: Energy function
        cfg: Configuration object with density_peaks parameters
        eval_dir: Directory for saving outputs
        eval_dict: Dictionary for logging metrics
        tag: Tag for naming outputs
        beta: Temperature parameter for energy evaluation
    
    Returns:
        Tuple of (cluster_centers, labels)
    """
    print(f"[Density Peaks {tag}] Starting clustering...")
    
    n = distance_matrix.shape[0]
    
    # Get parameters from config with defaults
    dc = getattr(cfg.density_peaks, "dc", None)
    if dc is None:
        # Auto-select dc as 2% of max distance (Rodriguez & Laio recommendation)
        dc = np.percentile(distance_matrix[np.triu_indices(n, k=1)], 2)
        print(f"[Density Peaks {tag}] Auto-selected dc={dc:.4f}")
    
    # Compute local density using Gaussian kernel
    print(f"[Density Peaks {tag}] Computing local densities...")
    rho = np.zeros(n)
    for i in range(n):
        rho[i] = np.sum(np.exp(-(distance_matrix[i, :] / dc) ** 2)) - 1  # exclude self
    
    # Compute minimum distance to higher density point
    print(f"[Density Peaks {tag}] Computing delta values...")
    delta = np.zeros(n)
    nearest_higher = np.zeros(n, dtype=int)
    
    # Sort by density (descending)
    rho_sorted_idx = np.argsort(-rho)
    
    for i, idx in enumerate(rho_sorted_idx):
        if i == 0:
            # Point with highest density
            delta[idx] = np.max(distance_matrix[idx, :])
            nearest_higher[idx] = -1
        else:
            # Find minimum distance to points with higher density
            higher_density_points = rho_sorted_idx[:i]
            distances_to_higher = distance_matrix[idx, higher_density_points]
            min_idx = np.argmin(distances_to_higher)
            delta[idx] = distances_to_higher[min_idx]
            nearest_higher[idx] = higher_density_points[min_idx]
    
    # Compute decision graph metric (gamma = rho * delta)
    gamma = rho * delta
    
    # # Plot decision graph
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # # Decision graph: rho vs delta
    # scatter = axes[0].scatter(rho, delta, c=gamma, cmap='viridis', alpha=0.6, s=20)
    # axes[0].set_xlabel('Local Density (ρ)')
    # axes[0].set_ylabel('Distance to Higher Density (δ)')
    # axes[0].set_title(f'Decision Graph ({tag})')
    # axes[0].grid(True, alpha=0.3)
    # plt.colorbar(scatter, ax=axes[0], label='γ = ρ × δ')
    
    # # Gamma values sorted
    # gamma_sorted = np.sort(gamma)[::-1]
    # axes[1].plot(gamma_sorted, 'o-', markersize=4)
    # axes[1].set_xlabel('Rank')
    # axes[1].set_ylabel('γ = ρ × δ')
    # axes[1].set_title(f'Sorted γ Values ({tag})')
    # axes[1].grid(True, alpha=0.3)
    # axes[1].set_yscale('log')
    
    # plt.tight_layout()
    # fig.canvas.draw()
    # decision_graph_img = fig2img(fig)
    # fname = eval_dir / f"density_peaks_decision_graph_{tag}.png"
    # decision_graph_img.save(fname)
    # print(f"Saved decision graph ({tag}) to\n {fname.resolve()}")
    # plt.close(fig)
    # eval_dict[f"density_peaks_decision_graph_{tag}"] = wandb.Image(decision_graph_img)
    
    # Select cluster centers
    # Method 1: Use threshold on gamma
    gamma_threshold = getattr(cfg.density_peaks, "gamma_threshold", None)
    if gamma_threshold is None:
        # Auto-select: points with gamma > mean + 2*std
        gamma_threshold = np.mean(gamma) + 2 * np.std(gamma)
        print(f"[Density Peaks {tag}] Auto-selected gamma_threshold={gamma_threshold:.4f}")
    
    cluster_centers = np.where(gamma > gamma_threshold)[0]
    
    # Alternative: select top k centers
    max_centers = getattr(cfg.density_peaks, "max_centers", 9)
    if len(cluster_centers) > max_centers:
        top_k_indices = np.argsort(-gamma)[:max_centers]
        cluster_centers = top_k_indices
        print(f"[Density Peaks {tag}] Limited to top {max_centers} centers")
    
    if len(cluster_centers) == 0:
        print(f"[Density Peaks {tag}] No cluster centers found, using top center")
        cluster_centers = np.array([np.argmax(gamma)])
    
    print(f"[Density Peaks {tag}] Found {len(cluster_centers)} cluster centers")
    
    # Assign remaining points to clusters
    labels = -np.ones(n, dtype=int)
    for i, center_idx in enumerate(cluster_centers):
        labels[center_idx] = i
    
    # Assign points in order of decreasing density
    for idx in rho_sorted_idx:
        if labels[idx] == -1:
            # Assign to same cluster as nearest higher density point
            higher_idx = nearest_higher[idx]
            if higher_idx >= 0:
                labels[idx] = labels[higher_idx]
    
    # Count cluster sizes
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique_labels, counts)}
    num_clusters = len(unique_labels)
    
    eval_dict[f"density_peaks_{tag}_num_clusters"] = num_clusters
    print(f"[Density Peaks {tag}] clusters={num_clusters}, noise={np.sum(labels == -1)}")
    print(f"[Density Peaks {tag}] cluster sizes: {label_counts}")
    
    # # Render cluster centers
    # render_medoids_and_grid(
    #     cluster_centers.tolist(),
    #     samples,
    #     energy,
    #     eval_dir,
    #     tag,
    #     eval_dict,
    #     draw_label_cutoff=getattr(cfg, "draw_label_cutoff", 3.8),
    #     draw_label_max_pairs=getattr(cfg, "draw_label_max_pairs", 10),
    #     beta=beta,
    # )
    
    return cluster_centers, labels

def select_medoids_from_labels(labels, distance_matrix, max_medoids=9):
    """Pick up to max_medoids medoids per cluster, ordered by cluster size."""
    uniq, counts = np.unique(labels, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(uniq, counts)}
    medoid_indices = []
    sorted_labels = [
        lab
        for lab, _ in sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
        if lab != -1
    ]
    for lab in sorted_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        pdists = distance_matrix[np.ix_(idxs, idxs)]
        medoid_local = np.argmin(pdists.sum(axis=1))
        medoid_indices.append(int(idxs[medoid_local]))
        if len(medoid_indices) >= max_medoids:
            break
    return medoid_indices, label_counts

def cluster_rmsd_hdbscan(samples, energy, cfg, eval_dir, eval_dict, rmsd_matrix=None, tag="rmsd", beta: float = 1.0):
    """HDBSCAN on unordered-aligned RMSD matrix; render medoids."""
    if rmsd_matrix is None:
        rmsd_matrix = compute_rmsd_matrix(samples, energy, cfg, eval_dir, eval_dict, tag)

    hdbscan_clusterer_rmsd = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=getattr(cfg.hdbscan, "min_samples", None),
        cluster_selection_epsilon=getattr(cfg.hdbscan, "cluster_selection_epsilon", 0.0),
        metric="precomputed",
    )
    labels_rmsd = hdbscan_clusterer_rmsd.fit_predict(rmsd_matrix)
    medoid_indices_rmsd, label_counts_rmsd = select_medoids_from_labels(
        labels_rmsd, rmsd_matrix
    )
    num_clusters_rmsd = len([k for k in label_counts_rmsd.keys() if k != -1])
    eval_dict[f"hdbscan_{tag}_num_clusters"] = num_clusters_rmsd
    print(
        f"[HDBSCAN {tag}] clusters={num_clusters_rmsd}, noise={label_counts_rmsd.get(-1, 0)}"
    )
    # render_medoids_and_grid(
    #     medoid_indices_rmsd,
    #     samples,
    #     energy,
    #     eval_dir,
    #     tag,
    #     eval_dict,
    #     draw_label_cutoff=getattr(cfg, "draw_label_cutoff", 3.8),
    #     draw_label_max_pairs=getattr(cfg, "draw_label_max_pairs", 10),
    #     beta=beta,
    # )
    return medoid_indices_rmsd, labels_rmsd


def cluster_rmsd_density_peaks(samples, energy, cfg, eval_dir, eval_dict, tag="density_peaks", beta: float = 1.0, rmsd_matrix=None):
    """Density Peaks clustering on unordered-aligned RMSD matrix; render cluster centers."""
    if rmsd_matrix is None:
        rmsd_matrix = compute_rmsd_matrix(samples, energy, cfg, eval_dir, eval_dict, tag)
    
    cluster_centers, labels = cluster_density_peaks(
        rmsd_matrix,
        samples,
        energy,
        cfg,
        eval_dir,
        eval_dict,
        tag=tag,
        beta=beta,
    )
    
    return cluster_centers, labels


def cluster_mbtr(samples, energy, cfg, eval_dir, eval_dict, tag="mbtr", beta: float = 1.0):
    """HDBSCAN on MBTR (Many-Body Tensor Representation) descriptors; render medoids."""
    # Check if atom types are available (required for MBTR)
    atom_types = _get_atom_types_from_energy(energy)
    if atom_types is None:
        print(
            f"[HDBSCAN {tag}] Error: Atom types not available, skipping MBTR clustering"
        )
        return []

    print(
        f"[HDBSCAN {tag}] Computing MBTR descriptors for {samples.shape[0]} samples..."
    )

    # Convert samples to ASE Atoms objects
    atoms_list = samples_to_ase_atoms(
        samples, atom_types, energy.n_particles, energy.n_spatial_dim
    )

    # Get unique species for MBTR
    unique_species = sorted(set(atom_types))

    # Initialize MBTR descriptor
    mbtr = MBTR(
        species=unique_species,
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 100},
            "weighting": {"function": "unity"},
        },
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "sigma": 0.1, "n": 100},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
        },
        k3={
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "sigma": 0.1, "n": 100},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
        },
        periodic=False,
        flatten=True,
        sparse=False,
    )

    # Compute MBTR descriptors for all samples
    print(f"[HDBSCAN {tag}] Computing MBTR descriptors...")
    mbtr_features = mbtr.create(atoms_list)

    # Compute pairwise distance matrix using MBTR features
    print(f"[HDBSCAN {tag}] Computing pairwise distance matrix from MBTR features...")
    mbtr_distance_matrix = pairwise_distances(mbtr_features)

    # # Plot MBTR distance matrix as heatmap
    # fig, ax = plt.subplots(figsize=(10, 8))
    # sns.heatmap(
    #     mbtr_distance_matrix,
    #     cmap="viridis",
    #     square=True,
    #     cbar_kws={"label": "MBTR Distance"},
    #     ax=ax,
    # )
    # ax.set_title(f"Pairwise MBTR Distance Matrix ({tag})")
    # ax.set_xlabel("Sample Index")
    # ax.set_ylabel("Sample Index")
    # plt.tight_layout()
    # fig.canvas.draw()
    # mbtr_heatmap_img = fig2img(fig)
    # fname = eval_dir / f"mbtr_heatmap_{tag}.png"
    # mbtr_heatmap_img.save(fname)
    # print(f"Saved MBTR heatmap ({tag}) to\n {fname.resolve()}")
    # plt.close(fig)
    # eval_dict[f"mbtr_heatmap_{tag}"] = wandb.Image(mbtr_heatmap_img)

    # Perform HDBSCAN clustering
    hdbscan_clusterer_mbtr = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        metric="precomputed",
    )
    labels_mbtr = hdbscan_clusterer_mbtr.fit_predict(mbtr_distance_matrix)
    medoid_indices_mbtr, label_counts_mbtr = select_medoids_from_labels(
        labels_mbtr, mbtr_distance_matrix
    )
    num_clusters_mbtr = len([k for k in label_counts_mbtr.keys() if k != -1])
    eval_dict[f"hdbscan_{tag}_num_clusters"] = num_clusters_mbtr
    print(
        f"[HDBSCAN {tag}] clusters={num_clusters_mbtr}, noise={label_counts_mbtr.get(-1, 0)}"
    )
    # render_medoids_and_grid(
    #     medoid_indices_mbtr,
    #     samples,
    #     energy,
    #     eval_dir,
    #     tag,
    #     eval_dict,
    #     draw_label_cutoff=getattr(cfg, "draw_label_cutoff", 3.8),
    #     draw_label_max_pairs=getattr(cfg, "draw_label_max_pairs", 10),
    #     beta=beta,
    # )
    return medoid_indices_mbtr

def plot_2d_projection_rmsd(
    samples,
    energy,
    eval_dir,
    eval_dict,
    cfg,
    tag="projection",
    cluster_labels=None,
    n_samples_max=5000,
    rmsd_matrix=None,
):
    """Plot 2D projections of samples using t-SNE and UMAP.
    
    Args:
        samples: Sample positions
        energy: Energy function
        eval_dir: Directory to save plots
        eval_dict: Dictionary to store wandb images
        tag: Tag for naming files
        cluster_labels: Optional cluster labels for coloring
        n_samples_max: Maximum number of samples to use
        rmsd_matrix: Optional precomputed RMSD matrix. If provided, uses this
                     instead of computing interatomic distance features.
    """
    # Subsample if needed
    n_samples = samples.shape[0]
    if n_samples > n_samples_max:
        print(
            f"Subsampling from {n_samples} to {n_samples_max} samples for projection..."
        )
        indices = np.random.choice(n_samples, n_samples_max, replace=False)
        samples = samples[indices]
        if cluster_labels is not None:
            cluster_labels = cluster_labels[indices]
        if rmsd_matrix is not None:
            rmsd_matrix = rmsd_matrix[np.ix_(indices, indices)]
    
    # Use RMSD matrix if provided, otherwise compute distance features
    if rmsd_matrix is None:
        rmsd_matrix = compute_rmsd_matrix(samples, energy, cfg, eval_dir, eval_dict, tag="rmsd")
    # t-SNE and UMAP support precomputed distance matrices
    metric = "precomputed"
    features = rmsd_matrix
    n_features = rmsd_matrix.shape[0]
        
    # Check if we have enough features for 2D projection
    if n_features < 2:
        print(
            f"Skipping 2D projection: only {n_features} feature(s) available "
            f"(need at least 2 for t-SNE/UMAP projection)"
        )
        return None, None

    # Apply t-SNE
    print("Computing t-SNE projection...")
    # t-SNE with precomputed metric requires init="random" instead of default "pca"
    tsne = TSNE(n_components=2, random_state=42, metric=metric, init="random")
    tsne_coords = tsne.fit_transform(features)

    # Apply UMAP
    print("Computing UMAP projection...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, metric=metric)
    umap_coords = umap_reducer.fit_transform(features)

    # Create t-SNE plot
    fig_tsne = plt.figure(figsize=(8, 6))
    ax_tsne = fig_tsne.add_subplot(111)
    if cluster_labels is not None:
        scatter_tsne = ax_tsne.scatter(
            tsne_coords[:, 0],
            tsne_coords[:, 1],
            c=cluster_labels,
            cmap="tab10",
            alpha=0.6,
            s=10,
        )
        plt.colorbar(scatter_tsne, ax=ax_tsne, label="Cluster")
    else:
        ax_tsne.scatter(
            tsne_coords[:, 0],
            tsne_coords[:, 1],
            alpha=0.6,
            s=10,
        )
    ax_tsne.set_title(f"t-SNE Projection ({tag})")
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    ax_tsne.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_tsne.canvas.draw()
    tsne_img = fig2img(fig_tsne)
    fname_tsne = eval_dir / f"projection_2d_tsne_{tag}.png"
    tsne_img.save(fname_tsne)
    print(f"Saved t-SNE projection ({tag}) to\n {fname_tsne.resolve()}")
    plt.close(fig_tsne)

    # Create UMAP plot
    fig_umap = plt.figure(figsize=(8, 6))
    ax_umap = fig_umap.add_subplot(111)
    if cluster_labels is not None:
        scatter_umap = ax_umap.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=cluster_labels,
            cmap="tab10",
            alpha=0.6,
            s=10,
        )
        plt.colorbar(scatter_umap, ax=ax_umap, label="Cluster")
    else:
        ax_umap.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            alpha=0.6,
            s=10,
        )
    ax_umap.set_title(f"UMAP Projection ({tag})")
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_umap.canvas.draw()
    umap_img = fig2img(fig_umap)
    fname_umap = eval_dir / f"projection_2d_umap_{tag}.png"
    umap_img.save(fname_umap)
    print(f"Saved UMAP projection ({tag}) to\n {fname_umap.resolve()}")
    plt.close(fig_umap)

    # Log to wandb separately
    eval_dict[f"projection_2d_tsne_{tag}"] = wandb.Image(tsne_img)
    eval_dict[f"projection_2d_umap_{tag}"] = wandb.Image(umap_img)

    return tsne_coords, umap_coords

