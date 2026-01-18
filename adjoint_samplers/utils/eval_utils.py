# Copyright (c) Meta Platforms, Inc. and affiliates.

import io
import os
import sys
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
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import umap

from adjoint_samplers.utils.frequency_analysis import analyze_frequencies_torch
from adjoint_samplers.utils.frequency_analysis_2d import analyze_frequencies_2d_torch
from adjoint_samplers.energies.scine_energy import (
    element_type_to_symbol,
)
from adjoint_samplers.utils.ase_utils import (
    samples_to_ase_atoms,
    _get_atom_types_from_energy,
)

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
    beta: float = 1.0,
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
        energy_val = energy.eval(sample, beta=beta).detach()
        medoid_energies.append(float(energy_val.item()))
        grad = energy.grad_E(sample, beta=beta)
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
            # In 2D, remove redundant rigid-body modes (2 translations + 1 rotation)
            # before counting negative eigenvalues.
            freq = analyze_frequencies_2d_torch(
                hessian=hess,
                cart_coords=samples[idx],
                ev_thresh=-1e-6,
            )
            neg_num = int(freq["neg_num"])
        freq_samples += 1
        if neg_num == 0:
            freq_minima += 1
        elif neg_num == 1:
            freq_ts += 1
        else:
            freq_other += 1
    if freq_samples > 0:
        prefix = f"{tag}/"
        total = float(freq_samples)
        freq_minima_ratio = freq_minima / total
        freq_ts_ratio = freq_ts / total
        freq_other_ratio = freq_other / total
        # Log absolute counts
        eval_dict[f"{prefix}num_minima"] = freq_minima
        eval_dict[f"{prefix}num_transition_states"] = freq_ts
        eval_dict[f"{prefix}num_other"] = freq_other
        eval_dict[f"{prefix}num_clusters"] = freq_samples
        # Log ratios (for backwards compatibility and convenience)
        eval_dict[f"{prefix}freq_minima"] = freq_minima_ratio
        eval_dict[f"{prefix}freq_transition_states"] = freq_ts_ratio
        eval_dict[f"{prefix}freq_other"] = freq_other_ratio


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


def compute_rmsd_to_gt(sample, gt_positions):
    """
    Compute RMSD between a sample and a ground truth geometry after optimal alignment.

    Args:
        sample: Tensor of shape (n_atoms, 3)
        gt_positions: Tensor of shape (n_atoms, 3)

    Returns:
        RMSD value (float)
    """
    # Align sample to ground truth using Kabsch algorithm
    R, t = find_rigid_alignment(sample, gt_positions)
    aligned = (R @ sample.T).T + t
    rmsd = torch.sqrt(((aligned - gt_positions) ** 2).sum(dim=1).mean())
    return rmsd.item()


def evaluate_gt_matching(
    samples,
    gt_geometries,
    energy,
    eval_dict,
    tag="gt",
):
    """
    Evaluate how well generated samples match ground truth geometries.

    For each ground truth geometry, find the closest sample (by RMSD) and report:
    - Min RMSD to each GT geometry
    - Coverage: fraction of GT geometries matched within threshold

    Args:
        samples: Tensor of shape (n_samples, dim) - generated samples
        gt_geometries: List of dicts with position tensors
        energy: Energy object with n_particles and n_spatial_dim
        eval_dict: Dict to store evaluation metrics
        tag: Prefix for metric names
    """
    if len(gt_geometries) == 0:
        print(f"No ground truth geometries for {tag}")
        return

    n_particles = energy.n_particles
    n_spatial_dim = energy.n_spatial_dim

    # Reshape samples to (n_samples, n_atoms, 3)
    samples_reshaped = samples.view(-1, n_particles, n_spatial_dim)

    # Extract positions from gt_geometries
    gt_positions_list = []
    for gt in gt_geometries:
        # Get any position key from the geometry dict
        for key in gt.keys():
            if "positions" in key:
                pos = gt[key]
                if pos.dim() == 2 and pos.shape[0] == n_particles:
                    gt_positions_list.append(pos.to(samples.device).float())

    if len(gt_positions_list) == 0:
        print(f"No valid position tensors found in ground truth for {tag}")
        return

    print(f"Evaluating {len(samples_reshaped)} samples against {len(gt_positions_list)} GT geometries ({tag})...")

    # For each GT geometry, find the min RMSD to any sample
    min_rmsds = []
    matched_indices = []

    for gt_idx, gt_pos in enumerate(gt_positions_list):
        best_rmsd = float("inf")
        best_sample_idx = -1

        for sample_idx in range(len(samples_reshaped)):
            sample = samples_reshaped[sample_idx]
            rmsd = compute_rmsd_to_gt(sample, gt_pos)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_sample_idx = sample_idx

        min_rmsds.append(best_rmsd)
        matched_indices.append(best_sample_idx)

    min_rmsds = np.array(min_rmsds)

    # Compute coverage at different thresholds
    thresholds = [0.1, 0.25, 0.5, 1.0]
    for thresh in thresholds:
        coverage = (min_rmsds < thresh).mean()
        eval_dict[f"{tag}/coverage_rmsd_{thresh}"] = coverage

    # Log statistics
    eval_dict[f"{tag}/min_rmsd_mean"] = float(min_rmsds.mean())
    eval_dict[f"{tag}/min_rmsd_std"] = float(min_rmsds.std())
    eval_dict[f"{tag}/min_rmsd_min"] = float(min_rmsds.min())
    eval_dict[f"{tag}/min_rmsd_max"] = float(min_rmsds.max())
    eval_dict[f"{tag}/num_gt"] = len(gt_positions_list)

    # Log histogram
    eval_dict[f"{tag}/min_rmsd_hist"] = wandb.Histogram(min_rmsds)

    print(f"  {tag}: mean_rmsd={min_rmsds.mean():.4f}, coverage@0.5={eval_dict[f'{tag}/coverage_rmsd_0.5']:.2%}")


