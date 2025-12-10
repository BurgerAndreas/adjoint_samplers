# Copyright (c) Meta Platforms, Inc. and affiliates.

import asyncio
import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import PIL
import py3Dmol
import torch
from matplotlib import pyplot as plt
from pyppeteer import launch
from scipy.optimize import linear_sum_assignment


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


def render_xyz_to_png(xyz_str, width=300, height=300):
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}})
    view.zoomTo()

    html = view.write_html()

    async def _html_to_png_bytes():
        with NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(html.encode("utf-8"))

        browser = await launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = await browser.newPage()
        await page.setViewport({"width": width, "height": height})
        await page.goto(tmp_path.as_uri(), waitUntil="networkidle0")
        await page.waitFor(100)
        png_bytes = await page.screenshot(fullPage=False)
        await browser.close()
        tmp_path.unlink(missing_ok=True)
        return png_bytes

    return asyncio.run(_html_to_png_bytes())


def render_xyz_grid(xyz_strings, ncols=3, width=900, height=900):
    """Render up to ncols*ncols XYZ strings into a grid PNG bytes."""
    n = len(xyz_strings)
    n = min(n, ncols * ncols)
    cell_w = width // ncols
    cell_h = height // ncols

    imgs = []
    for i in range(n):
        png_bytes = render_xyz_to_png(xyz_strings[i], width=cell_w, height=cell_h)
        imgs.append(PIL.Image.open(io.BytesIO(png_bytes)))

    grid = PIL.Image.new("RGB", (cell_w * ncols, cell_h * ncols), color=(255, 255, 255))
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, ncols)
        grid.paste(img, (c * cell_w, r * cell_h))

    return grid
