from ase.calculators.calculator import Calculator, all_changes
from ase import units
import torch
import numpy as np
from adjoint_samplers.energies.scine_energy import ScineEnergy

class ScineCalculator(Calculator):
    """
    ASE Calculator wrapper for the ScineEnergy class.

    This calculator bridges the gap between ASE (which operates on single Atoms objects
    in eV and Angstroms) and ScineEnergy (which operates on batches of tensors and
    outputs gradients in Hartree/Bohr).

    Args:
        functional (str): SCINE functional name (default "DFTB0").
        n_jobs (int): Number of parallel jobs for SCINE computation (default 1).
        device (str): Device for torch tensors (default "cpu").
        **kwargs: Additional arguments passed to the base ASE Calculator.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, functional: str = "DFTB0", n_jobs: int = 1, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.functional = functional
        self.n_jobs = n_jobs
        self.device = device
        
        # Internal storage
        self._scine_engine = None
        self._last_symbols = None

    def _ensure_engine(self, atoms):
        """
        Ensures the ScineEnergy engine is initialized and matches the current atoms.
        ScineEnergy is topology-fixed (requires element list at init), so we
        recreate it if the element types or count change.
        """
        current_symbols = atoms.get_chemical_symbols()
        
        # Initialize if missing or if elements have changed
        if self._scine_engine is None or current_symbols != self._last_symbols:
            self._scine_engine = ScineEnergy(
                elements=current_symbols,
                functional=self.functional,
                n_jobs=self.n_jobs,
                device=self.device
            )
            self._last_symbols = current_symbols

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Performs the calculation using ScineEnergy.
        """
        super().calculate(atoms, properties, system_changes)
        self._ensure_engine(atoms)

        # 1. Prepare Positions
        # ScineEnergy expects a flat tensor: [batch_size, n_atoms * 3]
        # ASE positions are [n_atoms, 3] in Angstroms
        positions_arr = atoms.get_positions()
        n_atoms = len(atoms)
        
        # Create tensor on device
        positions_tensor = torch.tensor(
            positions_arr.flatten(), 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0) # Add batch dimension -> [1, N*3]

        # 2. Compute Energy
        # eval returns [batch_size] tensor in eV
        energy_tensor = self._scine_engine.eval(positions_tensor)
        energy_val = energy_tensor.item()
        
        # Handle convergence failures (ScineEnergy returns NaN on failure)
        if np.isnan(energy_val):
            raise RuntimeError("SCINE calculation failed to converge (Energy is NaN).")
            
        self.results['energy'] = energy_val

        # 3. Compute Forces
        if 'forces' in properties:
            # grad_E returns [batch_size, N*3] tensor in Hartree/Bohr
            grad_tensor = self._scine_engine.grad_E(positions_tensor)
            
            # Convert to numpy and reshape to [N, 3]
            grad_arr = grad_tensor.detach().cpu().numpy().reshape(n_atoms, 3)
            
            # Unit Conversion:
            # SCINE gradients are dE/dx in Hartree/Bohr.
            # ASE forces are -dE/dx in eV/Angstrom.
            # Factor = (Hartree -> eV) / (Bohr -> Angstrom)
            hartree_to_ev = units.Hartree
            bohr_to_ang = units.Bohr
            conversion_factor = hartree_to_ev / bohr_to_ang

            # Force = -1 * Gradient * Conversion
            forces = -grad_arr * conversion_factor
            
            # Handle convergence failures
            if np.isnan(forces).any():
                raise RuntimeError("SCINE calculation failed to converge (Forces contain NaN).")

            self.results['forces'] = forces
    
    def get_hessian(self, atoms=None):
        # Update atoms if needed
        if atoms is not None:
             self.calculate(atoms, properties=['energy'])
             
        # Convert ASE positions (Angstrom) to Tensor
        positions_arr = self.atoms.get_positions()
        positions_tensor = torch.tensor(
            positions_arr.flatten(), 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0)

        # Get Hessian from SCINE (returns [1, N*3, N*3])
        hess_tensor = self._scine_engine.hessian_E(positions_tensor)
        
        # Convert to Numpy [N*3, N*3]
        hess_ev_ang2 = hess_tensor.detach().cpu().numpy()[0]
        
        return hess_ev_ang2

if __name__ == "__main__":
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.io import write

    # 1. Create the calculator
    calc = ScineCalculator(functional="DFTB0", n_jobs=1, device="cpu")

    # 2. Attach to an ASE Atoms object
    water = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    water.calc = calc

    # 3. Calculate potential energy and forces
    print(f"Energy: {water.get_potential_energy()} eV")
    print(f"Forces:\n{water.get_forces()} eV/Å")
    
    
    #########################################################
    
    # Approximate initial coordinates (Angstrom)
    # Structure: CH3-CH(OH)-CH3
    symbols = "C3H8O"
    
    # Rough coordinates for 2-Propanol
    positions = [
        [ 0.000,  0.000,  0.000],  # C (central)
        [ 1.500,  0.000,  0.000],  # C (methyl 1)
        [-0.800,  1.300,  0.000],  # C (methyl 2)
        [ 0.000, -0.800,  1.200],  # O (hydroxyl)
        [ 1.800, -0.500,  0.900],  # H (on C1)
        [ 1.800, -0.500, -0.900],  # H (on C1)
        [ 1.900,  1.000,  0.000],  # H (on C1)
        [-0.500,  2.000,  0.800],  # H (on C2)
        [-0.500,  2.000, -0.800],  # H (on C2)
        [-1.900,  1.200,  0.000],  # H (on C2)
        [-0.400, -0.500, -0.900],  # H (on central C)
        [ 0.800, -0.600,  1.800],  # H (on O)
    ]
    
    mol = Atoms(symbols=symbols, positions=positions)


    # trajectory='opt.traj' saves steps so you can visualize them later
    opt = BFGS(mol, trajectory='isopropanol_opt.traj')

    print("\n--- Starting Relaxation ---")
    
    # Run Optimization
    # fmax=0.05 eV/Ang is a standard convergence criterion
    opt.run(fmax=0.05)

    print("\n--- Relaxation Complete ---")
    
    # Get Final Results
    final_e = mol.get_potential_energy()
    final_pos = mol.get_positions()
    
    print(f"Final Potential Energy: {final_e:.4f} eV")
    print("Final Coordinates:")
    for atom in mol:
        print(f"  {atom.symbol}: {atom.position}")

    # Optional: Save final structure to XYZ
    # write("isopropanol_optimized.xyz", mol)