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
    """Draw distance labels between nearby atom pairs to avoid clutter."""
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
    assert D == n_particles * n_spatial_dim

    x = x.view(B, n_particles, n_spatial_dim)

    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,
    ]
    return torch.linalg.norm(distances, dim=-1)


def build_xyz_from_positions(positions, atom_type="C", center=True):
    """
    positions: torch.Tensor or np.ndarray of shape (N, 3)
    returns XYZ string
    """
    pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else positions
    if center:
        pos = pos - pos.mean(axis=0, keepdims=True)
    lines = [f"{pos.shape[0]}", "generated"]
    for p in pos:
        lines.append(f"{atom_type} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines)


def set_pymol_settings():
    # Try both sticks and spheres to ensure visibility
    cmd.show("sticks")
    cmd.show("spheres", "all")
    cmd.set("stick_radius", 0.1)
    cmd.set("stick_quality", 20)
    cmd.set("sphere_scale", 0.1)
    # cmd.set("orthoscopic", 1) # parallel projection, no depth foreshortening
    # cmd.set("ray_shadows", 0) # disable ray-tracing shadows
    # cmd.set("ray_shadow", 0)
    # cmd.set("light_count", 0)
    cmd.set("shininess", 0)
    cmd.set("specular", 0.0)  # no highlights
    cmd.set("ambient", 2)  # ambient light = brightness
    # cmd.set("reflect", 0)
    # cmd.set("direct", 0)
    cmd.set("ray_opaque_background", 1)
    _apply_pastel_colors()
    _add_distance_labels("mol")
    cmd.zoom("mol", buffer=0.2)


def render_xyz_to_png(xyz_str, width=300, height=300):
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
    set_pymol_settings()
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


def render_xyz_grid(xyz_strings, ncols=3, width=900, height=900):
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
        set_pymol_settings()
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
