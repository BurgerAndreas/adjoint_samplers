# Adapted from the 3D Eckart projection utilities in adjoint_samplers.utils.frequency_analysis
#
# For 2D point clouds (N particles in R^2), the energy is invariant under:
# - 2 translations (x/y)
# - 1 rotation (about the out-of-plane axis)
#
# When counting negative eigenvalues of a Hessian, these rigid-body modes are
# redundant "zero modes" and should be projected out.

from __future__ import annotations

import torch


def _to_torch_double(array_like, device=None) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def get_trans_rot_vectors_2d_torch(
    cart_coords: torch.Tensor,
    rot_thresh: float = 1e-10,
) -> torch.Tensor:
    """Return orthonormal row vectors spanning the rigid-body subspace in 2D.

    Parameters
    ----------
    cart_coords:
        Coordinates of shape (2*N,) or (N, 2)
    rot_thresh:
        If the rotation vector norm is below this threshold, it is dropped.

    Returns
    -------
    tr_vecs:
        Orthonormal row vectors of shape (k, 2*N), with k in {2, 3}.
        k=2 if rotation is dropped (degenerate geometry), else k=3.
    """
    cc = _to_torch_double(cart_coords)
    if cc.ndim == 1:
        coords2d = cc.reshape(-1, 2)
    else:
        coords2d = cc.reshape(-1, 2)

    n = coords2d.shape[0]
    device = coords2d.device

    # Translation vectors (unit, uniform weights)
    # tx = (1,0, 1,0, ..., 1,0) / sqrt(N)
    # ty = (0,1, 0,1, ..., 0,1) / sqrt(N)
    tx = torch.zeros(2 * n, dtype=torch.float64, device=device)
    ty = torch.zeros(2 * n, dtype=torch.float64, device=device)
    tx[0::2] = 1.0
    ty[1::2] = 1.0
    tx = tx / torch.linalg.norm(tx)
    ty = ty / torch.linalg.norm(ty)

    # Rotation vector about COM: delta r = (-y, x) for an infinitesimal rotation
    com = coords2d.mean(dim=0, keepdim=True)
    centered = coords2d - com
    rot = torch.empty_like(centered)
    rot[:, 0] = -centered[:, 1]
    rot[:, 1] = centered[:, 0]
    rot = rot.reshape(-1)  # (2N,)

    tr_list = [tx, ty]
    if torch.linalg.norm(rot) > rot_thresh:
        rot = rot / torch.linalg.norm(rot)
        tr_list.append(rot)

    tr = torch.stack(tr_list, dim=0)  # (k, 2N)

    # Orthonormalize the row space (k, 2N) by QR on the transpose (2N, k)
    Q, _ = torch.linalg.qr(tr.T)  # (2N, k)
    return Q.T  # (k, 2N)


def get_trans_rot_projector_2d_torch(
    cart_coords: torch.Tensor,
    full: bool = False,
    rot_thresh: float = 1e-10,
) -> torch.Tensor:
    """Return either a full projector matrix or a basis for the complement.

    If full=False (recommended): returns P of shape ((2N-k), 2N) such that
    projecting a Hessian as P @ H @ P.T removes rigid modes.

    If full=True: returns the dense (2N,2N) projector onto the complement.
    """
    tr_vecs = get_trans_rot_vectors_2d_torch(cart_coords, rot_thresh=rot_thresh)  # (k,2N)
    if full:
        n = tr_vecs.size(1)
        P = torch.eye(n, dtype=tr_vecs.dtype, device=tr_vecs.device)
        for v in tr_vecs:
            P = P - torch.outer(v, v)
        return P

    U, S, _ = torch.linalg.svd(tr_vecs.T, full_matrices=True)  # U: (2N, 2N)
    return U[:, S.numel() :].T  # ((2N-k), 2N)


def project_hessian_remove_rigid_2d_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    rot_thresh: float = 1e-10,
) -> torch.Tensor:
    """Project (2N,2N) Hessian into vibrational subspace (2N-k, 2N-k)."""
    H = _to_torch_double(hessian, device=hessian.device)
    H = 0.5 * (H + H.T)
    P = get_trans_rot_projector_2d_torch(
        cart_coords=cart_coords,
        full=False,
        rot_thresh=rot_thresh,
    )
    H_proj = P @ H @ P.T
    return 0.5 * (H_proj + H_proj.T)


def analyze_frequencies_2d_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    ev_thresh: float = -1e-6,
    rot_thresh: float = 1e-10,
) -> dict:
    """Compute eigenvalues after removing rigid-body modes in 2D."""
    cc = _to_torch_double(cart_coords, device=hessian.device)
    coords2d = cc.reshape(-1, 2)
    n = coords2d.shape[0]

    H = _to_torch_double(hessian, device=hessian.device).reshape(2 * n, 2 * n)
    H_proj = project_hessian_remove_rigid_2d_torch(
        hessian=H,
        cart_coords=coords2d,
        rot_thresh=rot_thresh,
    )
    eigvals = torch.linalg.eigvalsh(H_proj)
    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = int(neg_inds.sum().item())
    return {
        "eigvals": eigvals,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "n_particles": int(n),
        "ndof": int(2 * n),
        "ndof_vib": int(H_proj.shape[0]),
    }


