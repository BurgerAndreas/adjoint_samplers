# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Union
import re
import numpy as np
import torch
import scine_utilities

from adjoint_samplers.energies.base_energy import BaseEnergy
from adjoint_samplers.energies.scine_hessian_batch import compute_batch


def _element_symbol_to_type(symbol: str) -> scine_utilities.ElementType:
    """Convert element symbol to scine_utilities.ElementType.

    Args:
        symbol: Element symbol (e.g., "C", "H", "N", "O")

    Returns:
        scine_utilities.ElementType corresponding to the symbol
    """
    symbol_upper = symbol.upper()
    element_map = {
        "H": scine_utilities.ElementType.H,
        "He": scine_utilities.ElementType.He,
        "Li": scine_utilities.ElementType.Li,
        "Be": scine_utilities.ElementType.Be,
        "B": scine_utilities.ElementType.B,
        "C": scine_utilities.ElementType.C,
        "N": scine_utilities.ElementType.N,
        "O": scine_utilities.ElementType.O,
        "F": scine_utilities.ElementType.F,
        "Ne": scine_utilities.ElementType.Ne,
        "Na": scine_utilities.ElementType.Na,
        "Mg": scine_utilities.ElementType.Mg,
        "Al": scine_utilities.ElementType.Al,
        "Si": scine_utilities.ElementType.Si,
        "P": scine_utilities.ElementType.P,
        "S": scine_utilities.ElementType.S,
        "Cl": scine_utilities.ElementType.Cl,
        "Ar": scine_utilities.ElementType.Ar,
        "K": scine_utilities.ElementType.K,
        "Ca": scine_utilities.ElementType.Ca,
    }
    if symbol_upper not in element_map:
        raise ValueError(f"Unsupported element symbol: {symbol}")
    return element_map[symbol_upper]


def _atomic_number_to_type(atomic_number: int) -> scine_utilities.ElementType:
    """Convert atomic number to scine_utilities.ElementType.

    Args:
        atomic_number: Atomic number (1-20 supported)

    Returns:
        scine_utilities.ElementType corresponding to the atomic number
    """
    number_map = {
        1: scine_utilities.ElementType.H,
        2: scine_utilities.ElementType.He,
        3: scine_utilities.ElementType.Li,
        4: scine_utilities.ElementType.Be,
        5: scine_utilities.ElementType.B,
        6: scine_utilities.ElementType.C,
        7: scine_utilities.ElementType.N,
        8: scine_utilities.ElementType.O,
        9: scine_utilities.ElementType.F,
        10: scine_utilities.ElementType.Ne,
        11: scine_utilities.ElementType.Na,
        12: scine_utilities.ElementType.Mg,
        13: scine_utilities.ElementType.Al,
        14: scine_utilities.ElementType.Si,
        15: scine_utilities.ElementType.P,
        16: scine_utilities.ElementType.S,
        17: scine_utilities.ElementType.Cl,
        18: scine_utilities.ElementType.Ar,
        19: scine_utilities.ElementType.K,
        20: scine_utilities.ElementType.Ca,
    }
    if atomic_number not in number_map:
        raise ValueError(f"Unsupported atomic number: {atomic_number}")
    return number_map[atomic_number]


def _parse_molecule_formula(molecule: str) -> List[str]:
    """Parse molecule formula string into list of element symbols.

    Parses formulas like "c3h4", "h2o", "ch4" into lists like
    ["C", "C", "C", "H", "H", "H", "H"], ["H", "H", "O"], ["C", "H", "H", "H", "H"].

    Args:
        molecule: Molecule formula string (e.g., "c3h4", "h2o", "ch4")

    Returns:
        List of element symbols (capitalized)
    """
    # Pattern to match: element symbol (1-2 letters, case-insensitive) followed by optional number
    # Examples: "c3" -> ("c", "3"), "H4" -> ("H", "4"), "He" -> ("He", "")
    # Handle both single-letter (C, H, N, O) and two-letter (He, Li, Be, etc.) elements
    pattern = r"([A-Za-z][a-z]?)(\d*)"

    elements = []

    matches = re.findall(pattern, molecule)

    if not matches:
        raise ValueError(f"Could not parse molecule formula: {molecule}")

    for element_symbol, count_str in matches:
        count = int(count_str) if count_str else 1
        # Capitalize element symbol properly
        # Single letter: uppercase (C, H, N, O)
        # Two letters: first uppercase, second lowercase (He, Li, Be, etc.)
        if len(element_symbol) == 1:
            element_symbol = element_symbol.upper()
        else:
            element_symbol = element_symbol[0].upper() + element_symbol[1:].lower()

        elements.extend([element_symbol] * count)

    return elements


def _convert_elements(
    elements: Union[List[scine_utilities.ElementType], List[str], List[int], str],
) -> List[scine_utilities.ElementType]:
    """Convert elements to list of scine_utilities.ElementType.

    Args:
        elements: Can be:
            - List of ElementType
            - List of element symbols (strings)
            - List of atomic numbers (ints)
            - Molecule formula string (e.g., "c3h4", "h2o")

    Returns:
        List of scine_utilities.ElementType
    """
    # Handle molecule formula string
    if isinstance(elements, str):
        element_symbols = _parse_molecule_formula(elements)
        return [_element_symbol_to_type(sym) for sym in element_symbols]

    if not elements:
        raise ValueError("Elements list cannot be empty")

    # Check if already ElementType
    if isinstance(elements[0], scine_utilities.ElementType):
        return elements

    # Check if strings (element symbols)
    if isinstance(elements[0], str):
        return [_element_symbol_to_type(sym) for sym in elements]

    # Check if integers (atomic numbers)
    if isinstance(elements[0], int):
        return [_atomic_number_to_type(z) for z in elements]

    raise ValueError(
        "Elements must be a list of ElementType, element symbols (strings), "
        "atomic numbers (ints), or a molecule formula string (e.g., 'c3h4')"
    )


class ScineEnergy(BaseEnergy):
    """Energy class for molecular systems using SCINE quantum chemistry calculations.

    This class is molecule-agnostic - it accepts atomic structure (elements or molecule name) at
    initialization and processes batches of positions. All molecule-specific
    information comes from the elements/molecule provided at initialization.

    Args:
        molecule: Molecule formula string (e.g., "c3h4", "h2o") or list of elements.
                 If None, elements must be provided.
        elements: (Deprecated, use molecule instead) List of ElementType, element symbols,
                 or atomic numbers. If molecule is provided, this is ignored.
        functional: SCINE functional name (default "DFTB0")
        n_jobs: Number of parallel jobs for SCINE computation (default 10)
        device: Device for torch tensors (default "cpu")
        gad: Gradient ascent descent flag (default False)
    """

    def __init__(
        self,
        molecule: Union[str, None] = None,
        elements: Union[
            List[scine_utilities.ElementType], List[str], List[int], None
        ] = None,
        functional: str = "DFTB0",
        n_jobs: int = 10,
        device: str = "cpu",
        gad: bool = False,
    ):
        # Prefer molecule over elements if both provided
        if molecule is not None:
            elements_input = molecule
        elif elements is not None:
            elements_input = elements
        else:
            raise ValueError("Either 'molecule' or 'elements' must be provided")

        # Convert elements to ElementType list
        self.elements = _convert_elements(elements_input)
        self.functional = functional
        self.n_jobs = n_jobs
        self.device = device

        # Compute dimensions
        self.n_particles = len(self.elements)
        self.n_spatial_dim = 3
        dim = self.n_particles * self.n_spatial_dim

        # Set name based on functional
        super().__init__(name=f"scine_{functional}", dim=dim, gad=gad)

    def eval(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Compute energy for batch of molecular geometries.

        Args:
            x: Tensor of shape [batch_size, dim] where dim = n_particles * 3
               Contains flattened atomic positions in Angstrom
            beta: Temperature scaling factor

        Returns:
            Tensor of shape [batch_size] with energies in eV, scaled by beta
        """
        batch_size = x.shape[0]

        # Reshape to [batch_size, n_particles, 3]
        positions = x.reshape(batch_size, self.n_particles, 3)

        # Convert to numpy (SCINE runs on CPU)
        positions_np = positions.detach().cpu().numpy()

        # Create batch of (elements, positions) tuples
        geometries = [(self.elements, positions_np[i]) for i in range(batch_size)]

        # Compute batch using SCINE
        results = compute_batch(
            geometries=geometries,
            functional=self.functional,
            n_jobs=self.n_jobs,
            verbose=0,
        )

        # Extract energies and handle failures
        energies = []
        for result in results:
            if result["success"]:
                energies.append(result["energy_ev"])
            else:
                # Return NaN for failed calculations
                energies.append(float("nan"))

        # Convert to torch tensor and apply beta scaling
        energies_tensor = (
            torch.tensor(energies, dtype=x.dtype, device=self.device) * beta
        )

        return energies_tensor

    def _grad_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Internal method to compute standard gradients from SCINE.

        Args:
            x: Tensor of shape [batch_size, dim] where dim = n_particles * 3
               Contains flattened atomic positions in Angstrom
            beta: Temperature scaling factor

        Returns:
            Tensor of shape [batch_size, dim] with gradients in Hartree/Bohr,
            scaled by beta
        """
        batch_size = x.shape[0]

        # Reshape to [batch_size, n_particles, 3]
        positions = x.reshape(batch_size, self.n_particles, 3)

        # Convert to numpy (SCINE runs on CPU)
        positions_np = positions.detach().cpu().numpy()

        # Create batch of (elements, positions) tuples
        geometries = [(self.elements, positions_np[i]) for i in range(batch_size)]

        # Compute batch using SCINE
        results = compute_batch(
            geometries=geometries,
            functional=self.functional,
            n_jobs=self.n_jobs,
            verbose=0,
        )

        # Extract gradients and handle failures
        gradients_list = []
        for result in results:
            if result["success"]:
                # Gradients are in Hartree/Bohr, shape (n_particles, 3)
                grad = result["gradients"]
                # Flatten to (n_particles * 3,)
                gradients_list.append(grad.flatten())
            else:
                # Return NaN for failed calculations
                gradients_list.append(np.full(self.dim, float("nan")))

        # Convert to torch tensor and apply beta scaling
        gradients_array = np.array(gradients_list)
        gradients_tensor = (
            torch.tensor(gradients_array, dtype=x.dtype, device=self.device) * beta
        )

        return gradients_tensor

    def grad_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Compute gradients for batch of molecular geometries.

        If GAD is enabled, computes GAD vector field instead of standard gradient.

        Args:
            x: Tensor of shape [batch_size, dim] where dim = n_particles * 3
               Contains flattened atomic positions in Angstrom
            beta: Temperature scaling factor

        Returns:
            Tensor of shape [batch_size, dim] with gradients in Hartree/Bohr,
            scaled by beta. If GAD is enabled, returns GAD vector field.
        """
        if self.gad:
            return self._grad_E_gad(x, beta=beta)
        return self._grad_E(x, beta=beta)

    def hessian_E(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Compute Hessian matrices for batch of molecular geometries.

        Args:
            x: Tensor of shape [batch_size, dim] where dim = n_particles * 3
               Contains flattened atomic positions in Angstrom
            beta: Temperature scaling factor

        Returns:
            Tensor of shape [batch_size, dim, dim] with Hessians in eV/Å²,
            scaled by beta
        """
        batch_size = x.shape[0]

        # Reshape to [batch_size, n_particles, 3]
        positions = x.reshape(batch_size, self.n_particles, 3)

        # Convert to numpy (SCINE runs on CPU)
        positions_np = positions.detach().cpu().numpy()

        # Create batch of (elements, positions) tuples
        geometries = [(self.elements, positions_np[i]) for i in range(batch_size)]

        # Compute batch using SCINE
        results = compute_batch(
            geometries=geometries,
            functional=self.functional,
            n_jobs=self.n_jobs,
            verbose=0,
        )

        # Extract Hessians and handle failures
        hessians_list = []
        for result in results:
            if result["success"]:
                # Hessians are in eV/Å², shape (n_particles * 3, n_particles * 3)
                hess = result["hessian_ev_ang2"]
                hessians_list.append(hess)
            else:
                # Return NaN for failed calculations
                hessians_list.append(np.full((self.dim, self.dim), float("nan")))

        # Convert to torch tensor and apply beta scaling
        hessians_array = np.array(hessians_list)
        hessians_tensor = (
            torch.tensor(hessians_array, dtype=x.dtype, device=self.device) * beta
        )

        return hessians_tensor
