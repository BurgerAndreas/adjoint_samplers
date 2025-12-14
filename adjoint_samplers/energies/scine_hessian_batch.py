import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scine_sparrow
import scine_utilities
from joblib import Parallel, delayed
from sympy.printing.pretty.pretty_symbology import B
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# uv pip install scine-utilities scine-sparrow joblib tqdm

# Backup threading environment variables before modifying them
_THREADING_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]
_THREADING_ENV_BACKUP = {var: os.environ.get(var) for var in _THREADING_ENV_VARS}


@contextmanager
def suppress_output():
    """Context manager to suppress stdout at file descriptor level, keeping stderr for exceptions."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)  # Save stdout file descriptor
    os.dup2(devnull_fd, 1)  # Redirect stdout to /dev/null
    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)  # Restore stdout
        os.close(devnull_fd)
        os.close(old_stdout_fd)


def compute_single_geometry(
    geometry_idx: int,
    elements: List[scine_utilities.ElementType],
    positions_angstrom: np.ndarray,
    functional: str,
    compute_hessian: bool = False,
) -> Dict[str, Any]:
    """
    Worker function to compute energy, gradients, and optionally Hessian for a single geometry.

    Each worker process initializes its own SCINE module manager since
    SCINE uses singletons per process.

    Args:
        geometry_idx: Index of geometry in the batch
        elements: List of ElementType for each atom
        positions_angstrom: Atomic positions in Angstrom, shape (N_atoms, 3)
        functional: Calculator functional name (e.g., "PM6", "AM1", "RM1", "MNDO", "DFTB0")
        compute_hessian: Whether to compute the Hessian matrix (default False)

    Returns:
        Dictionary with results:
        - geometry_idx: Index of geometry
        - success: Boolean indicating if calculation succeeded
        - energy_ev: Energy in eV (if successful)
        - gradients: Gradients array in Hartree/Bohr (if successful)
        - hessian_ev_ang2: Hessian matrix in eV/Å² (if successful and compute_hessian=True)
        - error: Error message (if failed)
    """
    # Force single-threaded execution per process to avoid oversubscription
    # This prevents BLAS/LAPACK libraries from spawning multiple threads
    # when we're already parallelizing at the process level with joblib
    # Backup current values and set to single-threaded
    original_env = {}
    for var in _THREADING_ENV_VARS:
        original_env[var] = os.environ.get(var)
        os.environ[var] = "1"

    # Initialize module manager in this worker process
    manager = scine_utilities.core.ModuleManager.get_instance()
    sparrow_module = Path(scine_sparrow.__file__).parent / "sparrow.module.so"
    manager.load(os.fspath(sparrow_module))

    # Get calculator
    calculator = manager.get("calculator", functional)
    if calculator is None:
        # Restore original environment variables before returning
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        return {
            "geometry_idx": geometry_idx,
            "success": False,
            "error": f"Calculator {functional} not found",
        }

    # Convert positions from Angstrom to Bohr
    positions_bohr = positions_angstrom * scine_utilities.BOHR_PER_ANGSTROM
    structure = scine_utilities.AtomCollection(elements, positions_bohr)

    # Assign structure and set required properties
    calculator.structure = structure
    properties = [
        scine_utilities.Property.Energy,
        scine_utilities.Property.Gradients,
    ]
    if compute_hessian:
        properties.append(scine_utilities.Property.Hessian)
    calculator.set_required_properties(properties)

    # Suppress SCINE output
    scine_utilities.core.Log.silent()

    # Calculate with output suppression and error handling
    with suppress_output():
        results = calculator.calculate()

    # Extract results
    energy = results.energy  # Hartree
    gradients = results.gradients  # Hartree/Bohr

    # Convert units
    hartree_to_ev = 27.211386245988
    bohr_to_ang = 0.529177210903

    energy_ev = energy * hartree_to_ev

    # Restore original environment variables
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)  # Remove if it wasn't set originally
        else:
            os.environ[var] = value

    result_dict = {
        "geometry_idx": geometry_idx,
        "success": True,
        "energy_ev": energy_ev,
        "gradients": gradients,
    }

    # Conditionally compute and add Hessian
    if compute_hessian:
        hessian = results.hessian  # Hartree/Bohr^2
        # Hessian conversion: Energy / Distance^2
        hessian_ev_ang2 = hessian * (hartree_to_ev / (bohr_to_ang**2))
        result_dict["hessian_ev_ang2"] = hessian_ev_ang2

    return result_dict


def compute_batch(
    geometries: List[Tuple[List[scine_utilities.ElementType], np.ndarray]],
    functional: str,
    n_jobs: int = -1,
    verbose: int = 0,
    compute_hessian: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute energy, gradients, and optionally Hessians for a batch of geometries in parallel.

    Args:
        geometries: List of (elements, positions_angstrom) tuples
            - elements: List of ElementType for each atom
            - positions_angstrom: Atomic positions in Angstrom, shape (N_atoms, 3)
        functional: Calculator functional name (e.g., "PM6", "AM1", "RM1", "MNDO", "DFTB0")
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
        verbose: Verbosity level for joblib (0=silent, 1=progress, 10=debug)
        compute_hessian: Whether to compute the Hessian matrix (default False)

    Returns:
        List of result dictionaries (one per geometry), ordered by geometry_idx
    """
    # Create tasks
    tasks = [
        delayed(compute_single_geometry)(
            idx, elements, positions, functional, compute_hessian
        )
        for idx, (elements, positions) in enumerate(geometries)
    ]

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(tasks)

    # Sort by geometry_idx to ensure consistent ordering
    results.sort(key=lambda x: x["geometry_idx"])

    return results


if __name__ == "__main__":
    # Example: Batch of cyclopropene geometries
    # C3H4 (cyclopropene structure)

    # Base structure
    base_elements = [
        scine_utilities.ElementType.C,
        scine_utilities.ElementType.C,
        scine_utilities.ElementType.C,  # 3 carbons
        scine_utilities.ElementType.H,
        scine_utilities.ElementType.H,
        scine_utilities.ElementType.H,
        scine_utilities.ElementType.H,  # 4 hydrogens
    ]

    # Create a batch of geometries (example: slightly perturbed positions)
    np.random.seed(42)  # For reproducibility

    base_positions = np.array(
        [
            # C1 (double bond carbon, at origin)
            [0.0, 0.0, 0.0],
            # C2 (one vertex of triangle)
            [1.51, 0.0, 0.0],
            # C3 (other vertex of triangle)
            [0.755, 1.31, 0.0],
            # H1, H2 on C1
            [-0.89, 0.0, 0.0],
            [0.0, -0.89, 0.0],
            # H3 on C2
            [2.40, 0.0, 0.0],
            # H4 on C3
            [0.755, 2.20, 0.0],
        ]
    )

    n_jobs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    functional = "DFTB0"
    n_reps = 10

    # Collect all timings in a list
    timing_data = []

    for n_jobs in n_jobs_list:
        for rep in range(n_reps):
            # Generate slightly perturbed geometries
            n_geometries = 128
            geometries = []
            for i in range(n_geometries):
                # Add small random perturbation (0.01 Å)
                perturbation = np.random.normal(0, 0.01, base_positions.shape)
                perturbed_positions = base_positions + perturbation
                geometries.append((base_elements, perturbed_positions))

            # Compute batch
            print(
                f"n_jobs={n_jobs}, rep={rep + 1}/{n_reps}: Computing Hessians for {len(geometries)} geometries using {functional}..."
            )

            start_time = time.time()
            results = compute_batch(
                geometries, functional, n_jobs=n_jobs, verbose=0, compute_hessian=True
            )
            end_time = time.time()

            elapsed_time = end_time - start_time

            # Print summary
            print(f"Completed in {elapsed_time:.4f} seconds")
            print(
                f"Average time per geometry: {elapsed_time / len(geometries):.4f} seconds"
            )

            # Statistics
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            print(f"Success: {len(successful)}/{len(results)}")
            if failed:
                print(f"Failed: {len(failed)}/{len(results)}")
                for fail in failed:
                    print(
                        f"  Geometry {fail['geometry_idx']}: {fail.get('error', 'Unknown error')}"
                    )

            if successful:
                energies = [r["energy_ev"] for r in successful]

                # Store timing data
                timing_data.append(
                    {
                        "n_jobs": n_jobs,
                        "rep": rep,
                        "time": elapsed_time,
                    }
                )
            else:
                print("Warning: All calculations failed, skipping this data point")
            print()

    # Create DataFrame
    df = pd.DataFrame(timing_data)

    # Calculate statistics per n_jobs
    df_stats = df.groupby("n_jobs")["time"].agg(["mean", "min", "max"]).reset_index()

    print("\nTiming statistics:")
    print(df_stats.round(3))

    # Plot
    sns.set_theme(context="poster", style="whitegrid", palette="deep", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot average line
    ax.plot(
        df_stats["n_jobs"], df_stats["mean"], marker="o", label="Average", linewidth=2
    )

    # Fill area between min and max
    ax.fill_between(
        df_stats["n_jobs"],
        df_stats["min"],
        df_stats["max"],
        alpha=0.3,
        label="Min-Max range",
    )

    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Hessian parallel timing ({functional})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor="none")

    plt.tight_layout(pad=0.01)

    fname = f"scine_hessian_batch_timings_{functional}.png"
    fig.savefig(fname, dpi=150)
    print(f"\nSaved: {fname}")
