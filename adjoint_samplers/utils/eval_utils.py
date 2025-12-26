# Copyright (c) Meta Platforms, Inc. and affiliates.

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

from adjoint_samplers.utils.frequency_analysis import analyze_frequencies_torch
from adjoint_samplers.utils.align_unordered_mols import (
    REORDER_HUNGARIAN,
    rmsd_unordered_from_numpy,
)
from adjoint_samplers.energies.scine_energy import (
    element_type_to_symbol,
)
from ase import Atoms
from dscribe.descriptors import MBTR

# Initialize PyMOL once at module level
_pymol_initialized = False


class _Silence:
    """Redirect Python and OS-level stdout/stderr to /dev/null."""

    def __enter__(self):
        self._null = open(os.devnull, "w")
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        self._old_fd_out = os.dup(1)
        self._old_fd_err = os.dup(2)
        os.dup2(self._null.fileno(), 1)
        os.dup2(self._null.fileno(), 2)
        sys.stdout = self._null
        sys.stderr = self._null

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._old_fd_out, 1)
        os.dup2(self._old_fd_err, 2)
        sys.stdout = self._old_out
        sys.stderr = self._old_err
        os.close(self._old_fd_out)
        os.close(self._old_fd_err)
        self._null.close()


def _init_pymol():
    global _pymol_initialized
    if not _pymol_initialized:
        with _Silence():
            pymol.finish_launching(["pymol", "-c", "-q"])
        _pymol_initialized = True
    # Silence command echoing (e.g., "PyMOL>viewport ...") on stdout.
    cmd.feedback("disable", "all", "actions")
    cmd.feedback("disable", "all", "results")
    cmd.feedback("disable", "all", "warnings")


def _apply_pastel_colors():
    """Apply pastel colors from seaborn's deep palette to elements."""
    palette = sns.color_palette("deep")
    # Map elements to palette colors (C, O, N, H, S, P, F, Cl, Br, I, etc.)
    # PyMOL expects RGB values as a list [R, G, B] with values 0-1
    element_colors = {
        "C": list(palette[2]),  # green
        "O": list(palette[3]),  # red
        "N": list(palette[0]),  # blue
        "H": list(palette[7]),  # gray
        "S": list(palette[7]),  # gray
        "P": list(palette[1]),  # orange
        "F": list(palette[4]),  # purple
        "Cl": list(palette[2]),  # green
        "Br": list(palette[5]),  # brown
        "I": list(palette[6]),  # pink
    }

    for elem, rgb in element_colors.items():
        cmd.set_color(f"elem_{elem}", rgb)
        cmd.color(f"elem_{elem}", f"elem {elem}")


def _add_distance_labels(obj_name="mol", cutoff=3.8, max_pairs=10):
    """Draw distance labels between nearby atom pairs to avoid clutter.
    cutoff: distance in Angstroms to draw bond labels for.
    max_pairs: maximum number of pairs to draw labels for.
    """
    model = cmd.get_model(obj_name)
    atoms = model.atom
    n = len(atoms)
    if n == 0:
        return

    pairs = []
    for i in range(n):
        ai = atoms[i]
        for j in range(i + 1, n):
            aj = atoms[j]
            dx = ai.coord[0] - aj.coord[0]
            dy = ai.coord[1] - aj.coord[1]
            dz = ai.coord[2] - aj.coord[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist <= cutoff:
                pairs.append((dist, i + 1, j + 1))
    pairs.sort(key=lambda t: t[0])
    for k, (_, i_idx, j_idx) in enumerate(pairs[:max_pairs]):
        dist_name = f"dist_{i_idx}_{j_idx}"
        cmd.distance(
            dist_name, f"{obj_name} and index {i_idx}", f"{obj_name} and index {j_idx}"
        )
        cmd.show("labels", dist_name)
        cmd.set("label_size", 14, dist_name)
        # 12 different fonts from 5-16pt
        cmd.set("label_font_id", 7, dist_name)
        cmd.set("label_color", "black", dist_name)
        cmd.set("label_outline_color", "white", dist_name)
        cmd.set("dash_width", 1, dist_name)
        cmd.set("dash_color", "black", dist_name)


def get_fig_axes(ncol, nrow=1, ax_length_in=2.0):
    figsize = (ncol * ax_length_in, nrow * ax_length_in)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, ncol)
    return fig, axes


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    # https://stackoverflow.com/a/61756899
    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


# from https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


# from https://github.com/jarridrb/DEM/blob/main/dem/models/components/distribution_distances.py
def ot(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    x1 = x1[col_ind]
    return x1


# from https://github.com/jarridrb/DEM/blob/main/dem/models/components/distribution_distances.py
def dist_point_clouds(x0, x1):
    M = []
    for i in range(len(x0)):
        reordered = []
        for j in range(len(x1)):
            x1_reordered = ot(x0[i], x1[j])
            reordered.append(x1_reordered)
        reordered = torch.stack(reordered)
        R, t = torch.vmap(find_rigid_alignment)(
            x0[i][None].repeat(len(x1), 1, 1), reordered
        )
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    M = torch.stack(M).squeeze()
    return M


# from https://github.com/jarridrb/DEM/blob/main/dem/utils/data_utils.py
def interatomic_dist(x, n_particles, n_spatial_dim):
    B, D = x.shape
    assert D == n_particles * n_spatial_dim, (
        f"x={x.shape} != n_particles={n_particles} * n_spatial_dim={n_spatial_dim}"
    )

    x = x.view(B, n_particles, n_spatial_dim)

    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,
    ]
    return torch.linalg.norm(distances, dim=-1)


def interatomic_dist_by_type(x, atom_types, n_particles, n_spatial_dim):
    """
    Compute pairwise distances grouped by atom type pairs.
    Returns feature vectors where distances are sorted within each atom type pair group.

    Args:
        x: Tensor of shape (B, n_particles * n_spatial_dim) with coordinates
        atom_types: List of atom type symbols (e.g., ["C", "H", "C", "H"])
        n_particles: Number of particles
        n_spatial_dim: Spatial dimension (2 or 3)

    Returns:
        Tensor of shape (B, total_distances) with distances grouped and sorted by atom type pairs
    """
    B, D = x.shape
    assert D == n_particles * n_spatial_dim, (
        f"x={x.shape} != n_particles={n_particles} * n_spatial_dim={n_spatial_dim}"
    )
    assert len(atom_types) == n_particles, (
        f"len(atom_types)={len(atom_types)} != n_particles={n_particles}"
    )

    x = x.view(B, n_particles, n_spatial_dim)

    # Compute pairwise distances (same as interatomic_dist)
    distances = x[:, None, :, :] - x[:, :, None, :]
    triu_mask = torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1
    distances = distances[:, triu_mask]
    distances = torch.linalg.norm(distances, dim=-1)  # (B, n_pairs)

    # Create mapping from distance index to atom type pair
    # triu_mask gives us pairs (i, j) where i < j
    pair_indices = torch.nonzero(triu_mask, as_tuple=False)  # (n_pairs, 2)
    atom_type_pairs = []
    for idx in range(pair_indices.shape[0]):
        i, j = pair_indices[idx, 0].item(), pair_indices[idx, 1].item()
        type_i, type_j = atom_types[i], atom_types[j]
        # Use lexicographic ordering for consistency: (min, max)
        pair = tuple(sorted([type_i, type_j]))
        atom_type_pairs.append(pair)

    # Group distances by atom type pair
    unique_pairs = sorted(set(atom_type_pairs))  # Consistent ordering
    grouped_features = []

    for pair in unique_pairs:
        # Get indices where this pair occurs
        pair_mask = torch.tensor(
            [atom_type_pairs[idx] == pair for idx in range(len(atom_type_pairs))],
            dtype=torch.bool,
        )
        pair_distances = distances[:, pair_mask]  # (B, n_pairs_of_this_type)
        # Sort within each group
        pair_distances_sorted, _ = torch.sort(pair_distances, dim=1)
        grouped_features.append(pair_distances_sorted)

    # Concatenate all groups
    features = torch.cat(grouped_features, dim=1)  # (B, total_distances)
    return features


def build_xyz_from_positions(positions, atom_type="C", atom_types=None, center=True):
    """
    Build XYZ format string from positions.

    Args:
        positions: torch.Tensor or np.ndarray of shape (N, 3)
        atom_type: Single atom type string (default "C") - used if atom_types is None
        atom_types: List of atom type strings, one per atom (overrides atom_type if provided)
        center: Whether to center the positions
    Returns:
        XYZ format string
    """
    pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else positions
    if center:
        pos = pos - pos.mean(axis=0, keepdims=True)
    lines = [f"{pos.shape[0]}", "generated"]

    # Use atom_types if provided, otherwise use single atom_type for all atoms
    if atom_types is not None:
        if len(atom_types) != pos.shape[0]:
            raise ValueError(
                f"Number of atom_types ({len(atom_types)}) must match number of positions ({pos.shape[0]})"
            )
        for i, p in enumerate(pos):
            lines.append(f"{atom_types[i]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    else:
        for p in pos:
            lines.append(f"{atom_type} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines)


def set_pymol_settings(cutoff=3.8, max_pairs=10, flat=False):
    # Try both sticks and spheres to ensure visibility
    cmd.show("sticks")
    cmd.show("spheres", "all")
    cmd.set("stick_radius", 0.1)
    cmd.set("stick_quality", 20)
    cmd.set("sphere_scale", 0.1)
    # cmd.set("orthoscopic", 1) # parallel projection, no depth foreshortening
    cmd.set("ray_shadows", 0)  # disable ray-tracing shadows
    cmd.set("ray_shadow", 0)
    # cmd.set("light_count", 1) # 0 = no highlights
    if flat:
        # no highlights
        cmd.set("shininess", 0)
        cmd.set("specular", 0.0)
        # 2 = flat
        cmd.set("ambient", 2)  # ambient light = brightness
    # cmd.set("reflect", 0) # dark
    # cmd.set("direct", 0) # black
    cmd.set("ray_opaque_background", 1)
    _apply_pastel_colors()
    _add_distance_labels("mol", cutoff, max_pairs)
    cmd.zoom("mol", buffer=0.2)


def align_to_standard_frame(coords):
    """
    Center at the center of mass and rotate so the first principal axis maps to +z
    and the second principal axis maps to +x.
    """
    if coords.shape[1] == 2:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    center = coords.mean(axis=0, keepdims=True)
    centered = coords - center
    cov = centered.T @ centered
    U, _, _ = np.linalg.svd(cov)
    target_axes = np.eye(3)
    rotation = target_axes @ U.T
    aligned = centered @ rotation.T
    return aligned


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


def render_xyz_to_png(
    xyz_str,
    width=300,
    height=300,
    draw_label_cutoff=3.8,
    draw_label_max_pairs=10,
    flat=False,
):
    _init_pymol()

    # Clean up any existing molecules
    cmd.delete("all")

    with NamedTemporaryFile(delete=False, suffix=".xyz", mode="w") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(xyz_str)

    cmd.load(str(tmp_path), "mol")
    with _Silence():
        cmd.viewport(width, height)
    cmd.bg_color("white")
    cmd.hide("all")
    set_pymol_settings(
        cutoff=draw_label_cutoff, max_pairs=draw_label_max_pairs, flat=flat
    )
    cmd.refresh()

    with NamedTemporaryFile(delete=False, suffix=".png") as png_tmp:
        png_path = Path(png_tmp.name)

    # cmd.ray(width, height)
    cmd.png(str(png_path), width=width, height=height, dpi=300)

    with open(png_path, "rb") as f:
        png_bytes = f.read()

    png_path.unlink(missing_ok=True)
    tmp_path.unlink(missing_ok=True)
    cmd.delete("all")

    return png_bytes


def render_xyz_grid(
    xyz_strings,
    ncols=3,
    width=900,
    height=900,
    draw_label_cutoff=3.8,
    draw_label_max_pairs=10,
    flat=False,
):
    """Render up to ncols*ncols XYZ strings into a grid PNG bytes."""
    n = len(xyz_strings)
    n = min(n, ncols * ncols)
    cell_w = width // ncols
    cell_h = height // ncols

    _init_pymol()

    imgs = []

    for i in range(n):
        with NamedTemporaryFile(delete=False, suffix=".xyz", mode="w") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(xyz_strings[i])

        mol_name = "mol"
        cmd.load(str(tmp_path), mol_name)
        with _Silence():
            cmd.viewport(cell_w, cell_h)
        cmd.bg_color("white")
        cmd.hide("all")
        set_pymol_settings(
            cutoff=draw_label_cutoff, max_pairs=draw_label_max_pairs, flat=flat
        )
        cmd.refresh()

        with NamedTemporaryFile(delete=False, suffix=".png") as png_tmp:
            png_path = Path(png_tmp.name)

        cmd.ray(cell_w, cell_h)
        cmd.png(str(png_path), width=cell_w, height=cell_h, dpi=300)

        with open(png_path, "rb") as f:
            png_bytes = f.read()

        img = PIL.Image.open(io.BytesIO(png_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        imgs.append(img)

        cmd.delete("all")
        tmp_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)

    grid = PIL.Image.new("RGB", (cell_w * ncols, cell_h * ncols), color=(255, 255, 255))
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, ncols)
        grid.paste(img, (c * cell_w, r * cell_h))

    # Draw thin separators between tiles for clarity (after pasting).
    draw = PIL.ImageDraw.Draw(grid)
    line_color = (200, 200, 200)
    for c in range(1, ncols):
        x = c * cell_w
        draw.line([(x, 0), (x, cell_h * ncols)], fill=line_color, width=1)
    for r in range(1, ncols):
        y = r * cell_h
        draw.line([(0, y), (cell_w * ncols, y)], fill=line_color, width=1)

    return grid


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


def _get_atom_types_from_energy(energy):
    """Get list of atom type symbols from energy object if available."""
    if hasattr(energy, "elements"):
        # ScineEnergy has elements as list of ElementType
        return [element_type_to_symbol(elem) for elem in energy.elements]
    return None


def samples_to_ase_atoms(samples, atom_types, n_particles, n_spatial_dim):
    """
    Convert torch tensor samples to list of ASE Atoms objects.

    Args:
        samples: Tensor of shape (B, n_particles * n_spatial_dim) with coordinates
        atom_types: List of atom type symbols (e.g., ["C", "H", "C", "H"])
        n_particles: Number of particles
        n_spatial_dim: Spatial dimension (2 or 3)

    Returns:
        List of ASE Atoms objects
    """
    B, D = samples.shape
    assert D == n_particles * n_spatial_dim, (
        f"samples={samples.shape} != n_particles={n_particles} * n_spatial_dim={n_spatial_dim}"
    )
    assert len(atom_types) == n_particles, (
        f"len(atom_types)={len(atom_types)} != n_particles={n_particles}"
    )

    # Reshape to (B, n_particles, n_spatial_dim)
    coords = samples.view(B, n_particles, n_spatial_dim)

    # Convert to numpy
    coords_np = coords.detach().cpu().numpy()

    # Handle 2D case by padding with zeros
    if n_spatial_dim == 2:
        coords_np = np.concatenate([coords_np, np.zeros((B, n_particles, 1))], axis=2)

    # Create ASE Atoms objects for each sample
    atoms_list = []
    for i in range(B):
        atoms = Atoms(symbols=atom_types, positions=coords_np[i])
        atoms_list.append(atoms)

    return atoms_list


def render_medoids_and_grid(
    medoid_indices,
    samples,
    energy,
    eval_dir,
    tag,
    eval_dict,
    draw_label_cutoff=3.8,
    draw_label_max_pairs=10,
    flat=False,
):
    """Render per-medoid PNGs and a grid for the provided indices."""
    if len(medoid_indices) == 0:
        print(f"No clusters found for {tag}")
        return
    print(f"Rendering {len(medoid_indices)} medoid representatives for {tag}...")
    medoid_xyz = []
    medoid_energies = []
    medoid_grad_norms = []
    medoid_labels = []
    atom_types = _get_atom_types_from_energy(energy)
    for i, idx in enumerate(medoid_indices):
        sample = samples[idx : idx + 1]
        energy_val = energy.eval(sample).detach()
        medoid_energies.append(float(energy_val.item()))
        grad = energy.grad_E(sample)
        grad_norm = grad.view(grad.shape[0], -1).norm(dim=1)
        medoid_grad_norms.append(float(grad_norm.item()))
        label = f"medoid_{tag}_{i}"
        medoid_labels.append(label)
        # eval_dict[f"{label}_energy"] = medoid_energies[-1]
        # eval_dict[f"{label}_grad_norm"] = medoid_grad_norms[-1]
        pos = (
            samples[idx]
            .detach()
            .reshape(energy.n_particles, energy.n_spatial_dim)
            .cpu()
            .numpy()
        )
        if pos.shape[1] == 2:
            pos = np.concatenate([pos, np.zeros((pos.shape[0], 1))], axis=1)
        pos = align_to_standard_frame(pos)
        xyz = build_xyz_from_positions(
            pos, atom_type="C", atom_types=atom_types, center=False
        )
        medoid_xyz.append(xyz)
    for i, xyz_str in enumerate(medoid_xyz[:3]):
        png_bytes = render_xyz_to_png(
            xyz_str,
            width=600,
            height=600,
            draw_label_cutoff=draw_label_cutoff,
            draw_label_max_pairs=draw_label_max_pairs,
        )
        medoid_img = PIL.Image.open(io.BytesIO(png_bytes))
        if medoid_img.mode != "RGB":
            medoid_img = medoid_img.convert("RGB")
        fname = eval_dir / f"medoid_{tag}_{i}.png"
        medoid_img.save(fname)
        print(f"Saved medoid {i} ({tag}) to\n {fname.resolve()}")
        eval_dict[f"medoid_{tag}_{i}"] = wandb.Image(medoid_img)
    if len(medoid_xyz) > 0:
        medoid_grid_img = render_xyz_grid(
            medoid_xyz,
            ncols=3,
            width=900,
            height=900,
            draw_label_cutoff=draw_label_cutoff,
            draw_label_max_pairs=draw_label_max_pairs,
            flat=flat,
        )
        fname = eval_dir / f"hdbscan_medoid_grid_{tag}.png"
        medoid_grid_img.save(fname)
        print(f"Saved medoid grid ({tag}) to\n {fname.resolve()}")
        eval_dict[f"hdbscan_medoid_grid_{tag}"] = wandb.Image(medoid_grid_img)
    if len(medoid_energies) > 0:
        eval_dict[f"{tag}_medoid_energies"] = wandb.Histogram(medoid_energies)
        eval_dict[f"{tag}_medoid_grad_norms"] = wandb.Histogram(medoid_grad_norms)
        order = np.argsort(medoid_energies)
        energies_sorted = [medoid_energies[k] for k in order]
        # only get the idx (last character) of the label
        labels_sorted = [medoid_labels[k][-1] for k in order]
        sns.reset_defaults()
        sns.set_theme(context="poster", palette="deep", font_scale=0.6)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=labels_sorted, y=energies_sorted, ax=ax)
        ax.set_title(f"Medoid energies ({tag})")
        ax.set_xlabel("Medoid")
        ax.set_ylabel("Energy")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout(pad=0.1)
        fig.canvas.draw()
        bar_img = fig2img(fig)
        fname = eval_dir / f"medoid_{tag}_energy_bar.png"
        bar_img.save(fname)
        print(f"Saved medoid energy bar plot ({tag}) to\n {fname.resolve()}")
        eval_dict[f"medoid_{tag}_energy_bar"] = wandb.Image(bar_img)
        plt.close("all")
        sns.reset_defaults()
        plt.rcdefaults()


def run_frequency_analysis(medoid_indices, samples, energy, eval_dict, tag, beta=1.0):
    """Compute minima/TS/other counts on medoids and log with tag."""
    if len(medoid_indices) == 0:
        return
    if energy.n_spatial_dim not in (2, 3):
        print(
            f"Warning: energy.n_spatial_dim is {energy.n_spatial_dim}, skipping frequency analysis for {tag}"
        )
        return
    freq_minima = 0
    freq_ts = 0
    freq_other = 0
    freq_samples = 0
    for idx in medoid_indices:
        atoms = ["x"] * energy.n_particles
        hess = energy.hessian_E(samples[idx : idx + 1], beta=beta).detach()[0]
        if energy.n_spatial_dim == 3:
            freq = analyze_frequencies_torch(
                hessian=hess,
                cart_coords=samples[idx],
                atomsymbols=atoms,
                # ev_thresh=-1e-6,
            )
            neg_num = int(freq["neg_num"])
        else:
            h_flat = hess.reshape(samples[idx].numel(), samples[idx].numel())
            h_flat = (h_flat + h_flat.T) / 2.0
            eigvals = torch.linalg.eigvalsh(h_flat)
            neg_num = int((eigvals < -1e-6).sum().item())
        freq_samples += 1
        if neg_num == 0:
            freq_minima += 1
        elif neg_num == 1:
            freq_ts += 1
        else:
            freq_other += 1
    if freq_samples > 0:
        prefix = f"{tag}_"
        total = float(freq_samples)
        freq_minima_ratio = freq_minima / total
        freq_ts_ratio = freq_ts / total
        freq_other_ratio = freq_other / total
        eval_dict[f"{prefix}freq_minima"] = freq_minima_ratio
        eval_dict[f"{prefix}freq_transition_states"] = freq_ts_ratio
        eval_dict[f"{prefix}freq_other"] = freq_other_ratio
        eval_dict[f"{prefix}freq_total_samples"] = freq_samples


def plot_energy_distance_hist(
    samples,
    energy,
    epoch,
    eval_dir,
    eval_dict,
    beta=1.0,
    energy_min=None,
    energy_max=None,
    dist_min=None,
    dist_max=None,
    vert_lines=None,
):
    """Plot energy and interatomic distance histograms and log to eval_dict."""
    print("Plotting energy and distance histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Energy histogram
    energy_values = energy.eval(samples, beta=beta).detach().cpu().numpy()
    axes[0].hist(energy_values, bins=50, density=True)
    axes[0].set_xlabel("Energy")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Energy Distribution (epoch {epoch})")
    axes[0].grid(True)
    if energy_min is not None or energy_max is not None:
        emin = energy_min if energy_min is not None else axes[0].get_xlim()[0]
        emax = energy_max if energy_max is not None else axes[0].get_xlim()[1]
        axes[0].set_xlim(emin, emax)

    # Interatomic distance histogram (if applicable)
    if hasattr(energy, "n_particles") and hasattr(energy, "n_spatial_dim"):
        distances_full = interatomic_dist(
            samples, energy.n_particles, energy.n_spatial_dim
        ).detach()
        distances = distances_full.cpu().numpy().reshape(-1)
        axes[1].hist(distances, bins=50, density=True)
        axes[1].set_xlabel("Interatomic Distance")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"Interatomic Distance Distribution (epoch {epoch})")
        axes[1].grid(True)
        if dist_min is not None or dist_max is not None:
            dmin = dist_min if dist_min is not None else axes[1].get_xlim()[0]
            dmax = dist_max if dist_max is not None else axes[1].get_xlim()[1]
            axes[1].set_xlim(dmin, dmax)
        if vert_lines is not None:
            for x in vert_lines:
                axes[1].axvline(x=x, color="black", linestyle="--", linewidth=1)
    else:
        axes[1].text(
            0.5,
            0.5,
            "N/A\n(Not a particle system)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Interatomic Distance Distribution")
        distances_full = None

    plt.tight_layout()
    fig.canvas.draw()  # ensure matplotlib renders before conversion
    energy_dist_hist_img = fig2img(fig)
    fname = eval_dir / "energy_dist_hist.png"
    energy_dist_hist_img.save(fname)
    print(f"Saved energy dist hist to\n {fname.resolve()}")
    plt.close(fig)

    # also log the histogram image to wandb
    eval_dict["energy_dist_hist"] = wandb.Image(energy_dist_hist_img)
    return distances_full


def plot_2d_projection(
    samples,
    energy,
    eval_dir,
    eval_dict,
    tag="projection",
    cluster_labels=None,
    n_samples_max=5000,
):
    """Plot 2D projections of samples using t-SNE and UMAP on sorted pairwise distances."""
    # Check if atom types are available for type-grouped clustering
    atom_types = _get_atom_types_from_energy(energy)

    if atom_types is not None:
        # Use type-grouped distances when atom types are available
        dist_features_sorted = (
            interatomic_dist_by_type(
                samples, atom_types, energy.n_particles, energy.n_spatial_dim
            )
            .detach()
            .cpu()
            .numpy()
        )
    else:
        # Fall back to original sorted distances behavior
        distances_full = interatomic_dist(
            samples, energy.n_particles, energy.n_spatial_dim
        ).detach()
        dist_features = (
            distances_full.cpu().numpy().reshape(distances_full.shape[0], -1)
        )
        dist_features_sorted = np.sort(dist_features, axis=1)

    # Subsample if needed
    n_samples = dist_features_sorted.shape[0]
    n_features = dist_features_sorted.shape[1]
    if n_samples > n_samples_max:
        print(
            f"Subsampling from {n_samples} to {n_samples_max} samples for projection..."
        )
        indices = np.random.choice(n_samples, n_samples_max, replace=False)
        dist_features_sorted = dist_features_sorted[indices]
        if cluster_labels is not None:
            cluster_labels = cluster_labels[indices]

    # Check if we have enough features for 2D projection
    if n_features < 2:
        print(
            f"Skipping 2D projection: only {n_features} feature(s) available "
            f"(need at least 2 for t-SNE/UMAP projection)"
        )
        return None, None

    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(dist_features_sorted)

    # Apply UMAP
    print("Computing UMAP projection...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = umap_reducer.fit_transform(dist_features_sorted)

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


def cluster_intradist(
    distances_full, samples, energy, cfg, eval_dir, eval_dict, tag="intradist"
):
    """HDBSCAN on intra-distance features (sorted or type-grouped); render medoids."""
    # Check if atom types are available for type-grouped clustering
    atom_types = _get_atom_types_from_energy(energy)

    if atom_types is not None:
        # Use type-grouped distances when atom types are available
        dist_features = (
            interatomic_dist_by_type(
                samples, atom_types, energy.n_particles, energy.n_spatial_dim
            )
            .detach()
            .cpu()
            .numpy()
        )
        print(
            f"[HDBSCAN {tag}] Using type-grouped distances (atom types: {set(atom_types)})"
        )
    else:
        # Fall back to original sorted distances behavior
        dist_features = (
            distances_full.cpu().numpy().reshape(distances_full.shape[0], -1)
        )
        dist_features = np.sort(dist_features, axis=1)
        print(f"[HDBSCAN {tag}] Using sorted distances (no atom type information)")

    dist_matrix_sorted = pairwise_distances(dist_features)
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.hdbscan.min_samples)
    labels = hdbscan_clusterer.fit_predict(dist_features)
    medoid_indices, label_counts = select_medoids_from_labels(
        labels, dist_matrix_sorted
    )
    num_clusters = len([k for k in label_counts.keys() if k != -1])
    eval_dict[f"hdbscan_{tag}_num_clusters"] = num_clusters
    print(f"[HDBSCAN {tag}] clusters={num_clusters}, noise={label_counts.get(-1, 0)}")
    render_medoids_and_grid(
        medoid_indices,
        samples,
        energy,
        eval_dir,
        tag,
        eval_dict,
        draw_label_cutoff=getattr(cfg, "draw_label_cutoff", 3.8),
        draw_label_max_pairs=getattr(cfg, "draw_label_max_pairs", 10),
        flat=getattr(cfg, "render_medoids_flat", False),
    )
    return medoid_indices, labels


def cluster_rmsd(samples, energy, cfg, eval_dir, eval_dict, tag="rmsd"):
    """HDBSCAN on unordered-aligned RMSD matrix; render medoids."""
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

    # Plot RMSD matrix as heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        rmsd_matrix,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "RMSD"},
        ax=ax,
    )
    ax.set_title(f"Pairwise RMSD Matrix ({tag})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    plt.tight_layout()
    fig.canvas.draw()
    rmsd_heatmap_img = fig2img(fig)
    fname = eval_dir / f"rmsd_heatmap_{tag}.png"
    rmsd_heatmap_img.save(fname)
    print(f"Saved RMSD heatmap ({tag}) to\n {fname.resolve()}")
    plt.close(fig)
    eval_dict[f"rmsd_heatmap_{tag}"] = wandb.Image(rmsd_heatmap_img)

    hdbscan_clusterer_rmsd = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_samples,
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
    render_medoids_and_grid(
        medoid_indices_rmsd,
        samples,
        energy,
        eval_dir,
        tag,
        eval_dict,
        draw_label_cutoff=getattr(cfg, "draw_label_cutoff", 3.8),
        draw_label_max_pairs=getattr(cfg, "draw_label_max_pairs", 10),
    )
    return medoid_indices_rmsd, labels_rmsd


def cluster_mbtr(samples, energy, cfg, eval_dir, eval_dict, tag="mbtr"):
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

    # Plot MBTR distance matrix as heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        mbtr_distance_matrix,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "MBTR Distance"},
        ax=ax,
    )
    ax.set_title(f"Pairwise MBTR Distance Matrix ({tag})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    plt.tight_layout()
    fig.canvas.draw()
    mbtr_heatmap_img = fig2img(fig)
    fname = eval_dir / f"mbtr_heatmap_{tag}.png"
    mbtr_heatmap_img.save(fname)
    print(f"Saved MBTR heatmap ({tag}) to\n {fname.resolve()}")
    plt.close(fig)
    eval_dict[f"mbtr_heatmap_{tag}"] = wandb.Image(mbtr_heatmap_img)

    # Perform HDBSCAN clustering
    hdbscan_clusterer_mbtr = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_samples,
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
    render_medoids_and_grid(
        medoid_indices_mbtr,
        samples,
        energy,
        eval_dir,
        tag,
        eval_dict,
        draw_label_cutoff=getattr(cfg, "draw_label_cutoff", 3.8),
        draw_label_max_pairs=getattr(cfg, "draw_label_max_pairs", 10),
    )
    return medoid_indices_mbtr
