#!/usr/bin/env python3
"""
Relax Transition1x isopropanol (C3H8O) reactants and products using BFGS with SCINE energy.

Reads:
  - data/tx_isopropanol_C3H8O_triplets.lmdb (LMDB of torch tensor dicts, one per reaction)

Writes:
  - data/t1x_isopropanol_C3H80_minima_scine_bfgs.lmdb (LMDB with relaxed structures)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS

from adjoint_samplers.optimization.scine_ase_calc import ScineCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-lmdb",
        type=str,
        default="data/t1x_isopropanol_C3H8O_triplets.lmdb",
    )
    parser.add_argument(
        "--out-lmdb",
        type=str,
        default="data/t1x_isopropanol_C3H80_minima_scine_bfgs.lmdb",
    )
    parser.add_argument("--functional", type=str, default="DFTB0")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    in_lmdb = Path(args.in_lmdb)
    out_lmdb = Path(args.out_lmdb)
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)

    # Create SCINE calculator
    calc = ScineCalculator(
        functional=args.functional,
        n_jobs=args.n_jobs,
        device=args.device
    )

    # Open input and output LMDB
    with lmdb.open(str(in_lmdb), readonly=True, subdir=False) as env_in, \
         lmdb.open(str(out_lmdb), map_size=int(500 * 1024 * 1024), subdir=False) as env_out:
        
        txn_in = env_in.begin()
        txn_out = env_out.begin(write=True)
        
        cursor = txn_in.cursor()
        for idx, (key, value) in enumerate(cursor):
            item = pickle.loads(value)
            rxn_id = item["rxn_id"]
            
            print(f"\n[{idx}] Processing {rxn_id}")
            
            # Extract atomic numbers (same for all states)
            atomic_numbers = item["atomic_numbers"].numpy()
            symbols = []
            z_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
            for z in atomic_numbers:
                symbols.append(z_to_symbol[int(z)])
            # Process reactant
            print(f"  Relaxing reactant...")
            pos_r = item["reactant_positions"].numpy()
            atoms_r = Atoms(symbols=symbols, positions=pos_r)
            atoms_r.calc = calc
            
            opt_r = BFGS(atoms_r, logfile=None)
            converged_r = opt_r.run(fmax=args.fmax, steps=args.max_steps)
            steps_r = opt_r.get_number_of_steps()
            print(f"    Reactant converged: {converged_r}, steps: {steps_r}")
            relaxed_pos_r = atoms_r.get_positions()
            relaxed_energy_r = atoms_r.get_potential_energy()
            relaxed_forces_r = atoms_r.get_forces()
            
            # Process product
            print(f"  Relaxing product...")
            pos_p = item["product_positions"].numpy()
            atoms_p = Atoms(symbols=symbols, positions=pos_p)
            atoms_p.calc = calc
            
            opt_p = BFGS(atoms_p, logfile=None)
            converged_p = opt_p.run(fmax=args.fmax, steps=args.max_steps)
            steps_p = opt_p.get_number_of_steps()
            print(f"    Product converged: {converged_p}, steps: {steps_p}")
            relaxed_pos_p = atoms_p.get_positions()
            relaxed_energy_p = atoms_p.get_potential_energy()
            relaxed_forces_p = atoms_p.get_forces()
            # Store relaxed structures
            relaxed_item = {
                "rxn_id": rxn_id,
                "atomic_numbers": item["atomic_numbers"],
                # Original data
                "original_reactant_positions": item["reactant_positions"],
                "original_reactant_energy": item["reactant_energy"],
                "original_product_positions": item["product_positions"],
                "original_product_energy": item["product_energy"],
                # Relaxed reactant
                "relaxed_reactant_positions": torch.tensor(relaxed_pos_r, dtype=torch.float64),
                "relaxed_reactant_energy": torch.tensor(relaxed_energy_r, dtype=torch.float64),
                "relaxed_reactant_forces": torch.tensor(relaxed_forces_r, dtype=torch.float64),
                # Relaxed product
                "relaxed_product_positions": torch.tensor(relaxed_pos_p, dtype=torch.float64),
                "relaxed_product_energy": torch.tensor(relaxed_energy_p, dtype=torch.float64),
                "relaxed_product_forces": torch.tensor(relaxed_forces_p, dtype=torch.float64),
            }
            
            txn_out.put(key, pickle.dumps(relaxed_item))
            
            # Commit every 10 reactions
            if (idx + 1) % 10 == 0:
                txn_out.commit()
                txn_out = env_out.begin(write=True)
                print(f"  Committed {idx + 1} reactions")
        
        txn_out.commit()
        print(f"\nSaved relaxed structures to {out_lmdb}")

    # Check for duplicates
    print("\n" + "=" * 60)
    print("Checking for duplicate minima...")
    print("=" * 60)

    all_positions = []
    all_labels = []

    with lmdb.open(str(out_lmdb), readonly=True, subdir=False) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                item = pickle.loads(value)
                rxn_id = item["rxn_id"]
                all_positions.append(item["relaxed_reactant_positions"].numpy())
                all_labels.append(f"{rxn_id}_reactant")
                all_positions.append(item["relaxed_product_positions"].numpy())
                all_labels.append(f"{rxn_id}_product")

    n_structures = len(all_positions)
    print(f"Total structures: {n_structures}")

    # Compute RMSD matrix using Kabsch alignment
    def compute_rmsd(pos1, pos2):
        """Compute RMSD after optimal alignment."""
        # Center both structures
        pos1_c = pos1 - pos1.mean(axis=0)
        pos2_c = pos2 - pos2.mean(axis=0)
        # Kabsch algorithm
        H = pos1_c.T @ pos2_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        pos1_aligned = pos1_c @ R
        rmsd = np.sqrt(((pos1_aligned - pos2_c) ** 2).sum(axis=1).mean())
        return rmsd

    # Find duplicates
    rmsd_threshold = 0.1  # Angstroms
    duplicates = []
    unique_indices = []
    is_duplicate = [False] * n_structures

    for i in range(n_structures):
        if is_duplicate[i]:
            continue
        unique_indices.append(i)
        for j in range(i + 1, n_structures):
            if is_duplicate[j]:
                continue
            rmsd = compute_rmsd(all_positions[i], all_positions[j])
            if rmsd < rmsd_threshold:
                duplicates.append((i, j, rmsd))
                is_duplicate[j] = True

    print(f"\nUnique minima: {len(unique_indices)}")
    print(f"Duplicate pairs (RMSD < {rmsd_threshold} Å): {len(duplicates)}")

    if duplicates:
        print("\nDuplicate pairs:")
        for i, j, rmsd in duplicates:
            print(f"  {all_labels[i]} <-> {all_labels[j]}: RMSD = {rmsd:.4f} Å")

    print(f"\nUnique structures:")
    for idx in unique_indices:
        print(f"  {all_labels[idx]}")


if __name__ == "__main__":
    main()