import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def _split_xy(xy: torch.Tensor):
    if xy.shape[-1] != 2:
        raise ValueError("xy must have last dimension = 2 (x, y).")
    return xy[..., 0], xy[..., 1]


def get_potential(xy: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """
    V_4(x,y) = ((x-1/2)^2 - 1)^2 + ((y-1/2)^2 - 1)^2 + beta * (x-1/2)*(y-1/2)
    Four minima near (0.5±1, 0.5±1) when |beta| is small.
    Returns shape (...,)
    """
    x, y = _split_xy(xy)
    u = x - 0.5
    v = y - 0.5
    return (u * u - 1.0) ** 2 + (v * v - 1.0) ** 2 + beta * u * v


def get_gradient(xy: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """
    ∇V_4 = ( 4u(u^2-1) + beta*v ,  4v(v^2-1) + beta*u )
    Returns shape (..., 2)
    """
    x, y = _split_xy(xy)
    u = x - 0.5
    v = y - 0.5
    gx = 4.0 * u * (u * u - 1.0) + beta * v
    gy = 4.0 * v * (v * v - 1.0) + beta * u
    return torch.stack((gx, gy), dim=-1)


def get_hessian(xy: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """
    H_4 = [[12u^2 - 4,   beta],
           [beta,        12v^2 - 4]]
    Returns shape (..., 2, 2)
    """
    x, y = _split_xy(xy)
    u = x - 0.5
    v = y - 0.5

    hxx = 12.0 * (u * u) - 4.0
    hyy = 12.0 * (v * v) - 4.0
    hxy = torch.as_tensor(beta, dtype=xy.dtype, device=xy.device)

    H00 = hxx
    H11 = hyy
    H01 = H10 = hxy.expand_as(hxx)

    return torch.stack(
        (
            torch.stack((H00, H01), dim=-1),
            torch.stack((H10, H11), dim=-1),
        ),
        dim=-2,
    )


class FWEnergy(torch.nn.Module):
    """Four well energy.
    Keep beta=0 for symmetric wells; small |beta| couples the axes while preserving 4 minima
    """

    def __init__(
        self,
        tau: float = 1.0,
        device: str = "cpu",
        beta: float = 0.0,
    ):
        super().__init__()

        self.device = device
        self.tau = tau
        self.beta = beta

    def __call__(self, batch: torch.Tensor | dict):
        if not isinstance(batch, torch.Tensor):
            pos = batch["pos"]
        else:
            pos = batch
        energy = get_potential(pos, self.beta)
        forces = -get_gradient(pos, self.beta)
        hessian = get_hessian(pos, self.beta)

        return {
            "energy": energy,
            "forces": forces,
            "energy_grad": -forces / self.tau,
            "hessian": hessian,
        }


class FWEnergyGAD(FWEnergy):
    """Four well energy with GAD."""

    def __init__(
        self,
        norm_gad_outside_ts: bool = False,
        norm_gad_outside_ts_2: bool = False,
        newton_raphson: bool = False,
        newton_raphson_2: bool = False,
        newton_raphson_then_norm_outside_ts: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_gad_outside_ts = norm_gad_outside_ts
        self.norm_gad_outside_ts_2 = norm_gad_outside_ts_2
        self.newton_raphson = newton_raphson
        self.newton_raphson_2 = newton_raphson_2
        self.newton_raphson_then_norm_outside_ts = newton_raphson_then_norm_outside_ts

    def __call__(self, batch):
        """batch: B, 2"""
        assert (
            sum(
                [
                    self.norm_gad_outside_ts,
                    self.norm_gad_outside_ts_2,
                    self.newton_raphson,
                    self.newton_raphson_2,
                    self.newton_raphson_then_norm_outside_ts,
                ]
            )
            <= 1
        ), "Only one can be True."
        batch = batch.reshape(-1, 2)
        outs = super().__call__(batch)

        eigenvals, eigenvecs = torch.linalg.eigh(outs["hessian"])
        smallest_eigvec = eigenvecs[..., :, 0]
        gad = (
            -outs["forces"]
            + 2
            * (outs["forces"] * smallest_eigvec).sum(-1, keepdim=True)
            * smallest_eigvec
        )

        # Optionally norm GAD to unity if eigenvalue product > 0
        if self.norm_gad_outside_ts:
            eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
            gad_mag = torch.linalg.norm(gad, dim=-1, keepdim=True)
            # Avoid division by zero if you want gradients
            # gad_mag += 1e-10
            # Only norm where eigenvalue product > 0 and norm < 1
            # (make small vectors larger, never smaller)
            min_norm = 1.0  # minimum norm we want
            mask = (eigval_product > 0) & (gad_mag.squeeze(-1) < min_norm)
            gad = torch.where(
                mask.unsqueeze(-1),
                gad / gad_mag,  # norm to unity
                gad,  # Leave unchanged
            )

        if self.norm_gad_outside_ts_2:
            eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
            gad_mag = torch.linalg.norm(gad, dim=-1, keepdim=True)
            # Avoid division by zero if you want gradients
            # gad_mag += 1e-10
            mask = eigval_product > 0
            min_norm = 1.0  # minimum norm we want
            # only magnify small vectors, never shrink large ones
            clip_coef = torch.clamp(min_norm / gad_mag, min=1.0)
            # Only norm where eigenvalue product > 0
            gad = torch.where(
                mask.unsqueeze(-1),
                gad * clip_coef,  # Rescale
                gad,  # Leave unchanged
            )

        if self.newton_raphson:
            # compute |H|^-1 * gad
            L = eigenvals  # B,2
            V = eigenvecs  # B,2,2
            # gad: B,2
            # Project Force onto Eigenbasis (change of coordinates)
            # Coefficients c_i = v_i dot F # B,2
            coeffs = (V.transpose(-1, -2) @ gad.unsqueeze(-1)).squeeze(-1)

            # Scale coefficients by 1/|lambda|
            L_abs = torch.abs(L)
            mask = L_abs > 1e-8

            scaled_coeffs = torch.zeros_like(coeffs)
            scaled_coeffs[mask] = coeffs[mask] / L_abs[mask]

            # Project back to spatial coordinates
            # step = sum(scaled_c_i * v_i)
            gad = (V @ scaled_coeffs.unsqueeze(-1)).squeeze(-1)

        if self.newton_raphson_2:
            abs_eigvals = torch.abs(eigenvals)

            # To avoid division by zero, only invert non-zero eigenvalues
            mask = abs_eigvals > 1e-8
            inv_abs_eigvals = torch.zeros_like(abs_eigvals)
            inv_abs_eigvals[mask] = 1.0 / abs_eigvals[mask]

            # Reconstruct the matrix: V @ diag(1/|L|) @ V^T (batched)
            # Scale columns of V by inv_abs_eigvals, then batched matmul with V^T.
            V = eigenvecs  # (B, 2, 2)
            H_abs_inv = (V * inv_abs_eigvals.unsqueeze(-2)) @ V.transpose(-1, -2)

            gad = (H_abs_inv @ gad.unsqueeze(-1)).squeeze(-1)

        if self.newton_raphson_then_norm_outside_ts:
            # compute |H|^-1 * gad
            L = eigenvals  # B,2
            V = eigenvecs  # B,2,2
            # gad: B,2
            # Project Force onto Eigenbasis (change of coordinates)
            # Coefficients c_i = v_i dot F # B,2
            coeffs = (V.transpose(-1, -2) @ gad.unsqueeze(-1)).squeeze(-1)

            # Scale coefficients by 1/|lambda|
            L_abs = torch.abs(L)
            mask = L_abs > 1e-8

            scaled_coeffs = torch.zeros_like(coeffs)
            scaled_coeffs[mask] = coeffs[mask] / L_abs[mask]

            # Project back to spatial coordinates
            # step = sum(scaled_c_i * v_i)
            gad = (V @ scaled_coeffs.unsqueeze(-1)).squeeze(-1)

            # norm gad to unity if eigenvalue product > 0
            eigval_product = eigenvals[..., 0] * eigenvals[..., 1]
            gad_mag = torch.linalg.norm(gad, dim=-1, keepdim=True)
            # Avoid division by zero if you want gradients
            # gad_mag += 1e-10
            mask = eigval_product > 0
            min_norm = 1.0  # minimum norm we want

            # # only magnify small vectors, never shrink large ones
            # clip_coef = torch.clamp(min_norm / gad_mag, min=1.0)
            # # Only norm where eigenvalue product > 0
            # gad = torch.where(
            #     mask.unsqueeze(-1),
            #     gad * clip_coef,  # Rescale
            #     gad,  # Leave unchanged
            # )

            mask = (eigval_product > 0) & (gad_mag.squeeze(-1) < min_norm)
            gad = torch.where(
                mask.unsqueeze(-1),
                gad / gad_mag,  # norm to unity
                gad,  # Leave unchanged
            )

        return {
            **outs,
            "energy_grad": gad / self.tau,
            "eigvals1": eigenvals[..., 0],
            "eigvals2": eigenvals[..., 1],
        }

class FWEnergyPRFO(FWEnergy):
    """Four well energy with P-RFO (Partitioned Rational Function Optimization)."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __call__(self, batch):
        """batch: B, 2"""
        batch = batch.reshape(-1, 2)
        outs = super().__call__(batch)

        eigenvals, eigenvecs = torch.linalg.eigh(outs["hessian"])
        V = eigenvecs  # (..., 2, 2)
        L = eigenvals  # (..., 2)

        # Gradient (since forces = -∇E)
        g = -outs["forces"]  # (..., 2)

        # Project gradient into Hessian eigenbasis: g_coords = V^T g
        g_coords = (V.transpose(-1, -2) @ g.unsqueeze(-1)).squeeze(-1)  # (..., 2)

        # Mode-wise augmented 2x2 eigenvalues:
        # r± = 0.5 * (λ ± sqrt(λ^2 + 4 g_i^2))
        rad = torch.sqrt(L * L + 4.0 * (g_coords * g_coords))
        r_plus = 0.5 * (L + rad)
        r_minus = 0.5 * (L - rad)

        # Partitioning: transition mode (smallest eigenvalue, index 0) uses r_plus,
        # remaining mode(s) use r_minus.
        r_choice = r_minus.clone()
        r_choice[..., 0] = r_plus[..., 0]

        denom = L - r_choice  # (..., 2)
        # Stabilize rare denom≈0 cases (e.g. g≈0 and λ>0 in the r_plus branch)
        eps = 1e-8
        denom_safe = denom + eps * torch.where(denom >= 0, 1.0, -1.0)

        # Step in eigenbasis: d_i = - g_i / (λ_i - r_choice)
        d_coords = -g_coords / denom_safe  # (..., 2)

        # Transform back to Cartesian coordinates: d = V d_coords
        d = (V @ d_coords.unsqueeze(-1)).squeeze(-1)  # (..., 2)

        # Convention in this file: `plot_gad_vector_field` uses `-out["energy_grad"]`
        # as the displayed flow/force field. Returning (-d)/tau makes that equal to
        # the PRFO step direction (up to 1/tau scaling).
        prfo_energy_grad = (-d) / self.tau

        return {
            **outs,
            "energy_grad": prfo_energy_grad,
            "eigvals1": eigenvals[..., 0],
            "eigvals2": eigenvals[..., 1],
        }


@torch.no_grad()
def plot_energy(
    energy_function,
    xlim=(-1.5, 2.5),
    ylim=(-1.5, 2.5),
    num_points: int = 81,
):
    # grid
    x_vals = torch.linspace(xlim[0], xlim[1], num_points)
    y_vals = torch.linspace(ylim[0], ylim[1], num_points)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")
    XY = torch.stack((X, Y), dim=-1)

    # fields
    original_shape = XY.shape[:-1]
    XY_flat = XY.reshape(-1, 2)
    out = energy_function(XY_flat)
    V = out["energy"].reshape(original_shape)

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    Vn = V.detach().cpu().numpy()

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # energy (potential)
    cs = ax.contourf(Xn, Yn, Vn, levels=40, cmap="viridis")
    ax.set_title("Energy V(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cs, ax=ax)
    plt.tight_layout(pad=0.01)

    return fig, ax


@torch.no_grad()
def plot_gad_vector_field(
    energy_function,
    xlim=(-0.8, 1.8),
    ylim=(-0.8, 1.8),
    num_points: int = 81,
    max_score_norm: float = 100.0,
):
    # grid
    x_vals = torch.linspace(xlim[0], xlim[1], num_points)
    y_vals = torch.linspace(ylim[0], ylim[1], num_points)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")
    XY = torch.stack((X, Y), dim=-1)

    # fields
    original_shape = XY.shape[:-1]
    XY_flat = XY.reshape(-1, 2)
    out = energy_function(XY_flat)
    GAD = -out["energy_grad"].reshape(*original_shape, 2)  # -1 to get a force field

    # Clamp GAD vectors
    gad_norms = torch.sqrt((GAD**2).sum(-1, keepdim=True))
    clip_coefficient = torch.clamp(max_score_norm / (gad_norms + 1e-6), max=1)
    GAD = GAD * clip_coefficient

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # GAD vector field (quiver)
    step = max(1, num_points // 20)
    GADx = GAD[..., 0][::step, ::step].detach().cpu().numpy()
    GADy = GAD[..., 1][::step, ::step].detach().cpu().numpy()
    Xq = X[::step, ::step].detach().cpu().numpy()
    Yq = Y[::step, ::step].detach().cpu().numpy()
    mag = np.hypot(GADx, GADy)
    # norm to unit vectors so all arrows have the same size
    GADx_norm = np.divide(GADx, mag, out=np.zeros_like(GADx), where=mag > 0)
    GADy_norm = np.divide(GADy, mag, out=np.zeros_like(GADy), where=mag > 0)
    Q = ax.quiver(
        Xq,
        Yq,
        GADx_norm,
        GADy_norm,
        mag,
        angles="xy",
        scale_units="xy",
        scale=10.0,
        cmap="viridis",
    )
    Q.set_clim(0.0, 5.1)
    fig.colorbar(Q, ax=ax, label="|GAD|")
    ax.set_title("GAD Vector Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.01)

    return fig, ax


@torch.no_grad()
def plot_gad_vector_field_mag(
    energy_function,
    xlim=(-0.8, 1.8),
    ylim=(-0.8, 1.8),
    num_points: int = 81,
    max_score_norm: float = 100.0,
):
    # grid
    x_vals = torch.linspace(xlim[0], xlim[1], num_points)
    y_vals = torch.linspace(ylim[0], ylim[1], num_points)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")
    XY = torch.stack((X, Y), dim=-1)

    # fields
    original_shape = XY.shape[:-1]
    XY_flat = XY.reshape(-1, 2)
    out = energy_function(XY_flat)
    GAD = -out["energy_grad"].reshape(*original_shape, 2)  # -1 to get a force field

    # Clamp GAD vectors
    gad_norms = torch.sqrt((GAD**2).sum(-1, keepdim=True))
    clip_coefficient = torch.clamp(max_score_norm / (gad_norms + 1e-6), max=1)
    GAD = GAD * clip_coefficient

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # GAD vector field (quiver) with mag lengths
    step = max(1, num_points // 20)
    GADx = GAD[..., 0][::step, ::step].detach().cpu().numpy()
    GADy = GAD[..., 1][::step, ::step].detach().cpu().numpy()
    Xq = X[::step, ::step].detach().cpu().numpy()
    Yq = Y[::step, ::step].detach().cpu().numpy()
    mag = np.hypot(GADx, GADy)
    # Use actual vector components (not normd) so lengths are mag
    Q = ax.quiver(
        Xq,
        Yq,
        GADx,
        GADy,
        mag,
        angles="xy",
        scale_units="xy",
        scale=None,  # Auto-scale to fit
        cmap="viridis",
    )
    Q.set_clim(0.0, 5.1)
    fig.colorbar(Q, ax=ax, label="|GAD|")
    ax.set_title("GAD Vector Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.01)

    return fig, ax


@torch.no_grad()
def plot_eigenvalue_product(
    energy_function,
    xlim=(-0.8, 1.8),
    ylim=(-0.8, 1.8),
    num_points: int = 81,
):
    # grid
    x_vals = torch.linspace(xlim[0], xlim[1], num_points)
    y_vals = torch.linspace(ylim[0], ylim[1], num_points)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")
    XY = torch.stack((X, Y), dim=-1)

    # fields
    original_shape = XY.shape[:-1]
    XY_flat = XY.reshape(-1, 2)
    out = energy_function(XY_flat)
    eigvals1 = out["eigvals1"].reshape(original_shape)
    eigvals2 = out["eigvals2"].reshape(original_shape)
    eigval_product = eigvals1 * eigvals2

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    eigval_product_n = eigval_product.detach().cpu().numpy()

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Find symmetric range for divergent colormap centered at 0
    vmax = np.abs(eigval_product_n).max()
    vmin = -vmax

    # eigenvalue product with divergent colormap
    cs = ax.contourf(
        Xn, Yn, eigval_product_n, levels=40, cmap="RdBu_r", vmin=vmin, vmax=vmax
    )

    # Outline the region where product is negative (contour at 0)
    ax.contour(Xn, Yn, eigval_product_n, levels=[0], colors="black", linewidths=2)

    ax.set_title("Product of Two Smallest Eigenvalues (λ₁ × λ₂)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("λ₁ × λ₂")
    plt.tight_layout(pad=0.01)

    return fig, ax


if __name__ == "__main__":
    # Create output directory
    plot_dir = "plots/gad"
    os.makedirs(plot_dir, exist_ok=True)

    # Create GAD energy function
    energy_model = FWEnergyGAD(
        tau=1.0,
        device="cpu",
        beta=0.0,
    )

    # Plot energy
    fig_energy, ax_energy = plot_energy(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
    )
    plt.savefig(os.path.join(plot_dir, "energy.png"), dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_dir}/energy.png")
    plt.close()

    # Plot GAD vector field (normd lengths)
    fig_gad, ax_gad = plot_gad_vector_field(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    plt.savefig(
        os.path.join(plot_dir, "gad_vector_field.png"), dpi=150, bbox_inches="tight"
    )
    print(f"Saved plot to {plot_dir}/gad_vector_field.png")
    plt.close()

    # Plot GAD vector field (mag lengths)
    fig_gad_prop, ax_gad_prop = plot_gad_vector_field_mag(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    plt.savefig(
        os.path.join(plot_dir, "gad_vector_field_mag.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"Saved plot to {plot_dir}/gad_vector_field_mag.png")
    plt.close()

    # Plot eigenvalue product
    fig_eig, ax_eig = plot_eigenvalue_product(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
    )
    plt.savefig(
        os.path.join(plot_dir, "eigenvalue_product.png"), dpi=150, bbox_inches="tight"
    )
    print(f"Saved plot to {plot_dir}/eigenvalue_product.png")
    plt.close()

    # rescale GAD to unity if eigenvalue product > 0
    energy_model.norm_gad_outside_ts = True
    fig_gad_rescaled, ax_gad_rescaled = plot_gad_vector_field_mag(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    energy_model.norm_gad_outside_ts = False
    fname = os.path.join(plot_dir, "gad_vector_field_mag_rescaled.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    plt.close()

    # rescale GAD to unity if eigenvalue product > 0
    energy_model.norm_gad_outside_ts_2 = True
    fig_gad_rescaled, ax_gad_rescaled = plot_gad_vector_field_mag(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    energy_model.norm_gad_outside_ts_2 = False
    fname = os.path.join(plot_dir, "gad_vector_field_mag_rescaled2.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    plt.close()

    # newton raphson
    energy_model.newton_raphson = True
    fig_gad_rescaled, ax_gad_rescaled = plot_gad_vector_field_mag(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    energy_model.newton_raphson = False
    fname = os.path.join(plot_dir, "gad_vector_field_mag_newtonraphson.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    plt.close()

    # newton raphson 2
    energy_model.newton_raphson_2 = True
    fig_gad_rescaled, ax_gad_rescaled = plot_gad_vector_field_mag(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    energy_model.newton_raphson_2 = False
    fname = os.path.join(plot_dir, "gad_vector_field_mag_newtonraphson2.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    plt.close()

    energy_model.newton_raphson_then_norm_outside_ts = True
    fig_gad_rescaled, ax_gad_rescaled = plot_gad_vector_field_mag(
        energy_function=energy_model,
        xlim=(-0.8, 1.8),
        ylim=(-0.8, 1.8),
        num_points=201,
        max_score_norm=100.0,
    )
    energy_model.newton_raphson_then_norm_outside_ts = False
    fname = os.path.join(
        plot_dir, "gad_vector_field_mag_newtonraphson_then_norm_outside_ts.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    plt.close()
