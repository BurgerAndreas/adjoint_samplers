from ase import Atoms


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