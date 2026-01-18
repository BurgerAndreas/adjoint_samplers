#!/usr/bin/env python3
"""
Refine Transition1x isopropanol (C3H8O) transition states using Sella with SCINE energy.

Reads:
  - data/t1x_isopropanol_C3H8O_triplets.lmdb (LMDB of torch tensor dicts, one per reaction)

Writes:
  - data/t1x_isopropanol_C3H80_ts_scine_sella.lmdb (LMDB with refined TS structures)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from sella import Sella

from adjoint_samplers.optimization.scine_ase_calc import ScineCalculator
from adjoint_samplers.utils.frequency_analysis import analyze_frequencies_torch


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
        default="data/t1x_isopropanol_C3H80_ts_scine_sella.lmdb",
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
        valid_count = 0
        skipped_count = 0
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
            
            # Extract positions
            pos_ts = item["transition_state_positions"].numpy()
            pos_r = item["reactant_positions"].numpy()
            pos_p = item["product_positions"].numpy()

            # Starting geometries to try: original TS, halfway TS-R, halfway R-P
            starting_geometries = [
                ("original_ts", pos_ts),
                ("halfway_ts_reactant", 0.5 * (pos_ts + pos_r)),
                ("halfway_reactant_product", 0.5 * (pos_r + pos_p)),
            ]

            # Sella parameter configurations
            sella_configs = [
                {"delta0": 0.1, "internal": True},
                {"delta0": 0.05, "internal": True},
                {"delta0": 0.1, "internal": False},
            ]

            found_valid_ts = False
            refined_pos_ts = None
            refined_energy_ts = None
            refined_forces_ts = None
            hessian_ts = None

            for geom_name, start_pos in starting_geometries:
                if found_valid_ts:
                    break

                print(f"  Trying starting geometry: {geom_name}")

                # First check if starting geometry is already a valid TS
                atoms_ts = Atoms(symbols=symbols, positions=start_pos.copy())
                atoms_ts.calc = calc

                hessian_ts = calc.get_hessian(atoms_ts)
                hess_torch = torch.tensor(hessian_ts, dtype=torch.float64)
                cart_coords = torch.tensor(start_pos, dtype=torch.float64).flatten()
                freq_result = analyze_frequencies_torch(
                    hessian=hess_torch,
                    cart_coords=cart_coords,
                    atomsymbols=symbols,
                )
                neg_num = int(freq_result["neg_num"])

                if neg_num == 1:
                    # Check if forces are already converged
                    forces = atoms_ts.get_forces()
                    fmax = np.max(np.abs(forces))
                    if fmax < args.fmax:
                        print(f"    ✓ Already a valid TS (1 neg eigenvalue, fmax={fmax:.4f})")
                        refined_pos_ts = start_pos
                        refined_energy_ts = atoms_ts.get_potential_energy()
                        refined_forces_ts = forces
                        found_valid_ts = True
                        break

                print(f"    Structure has {neg_num} negative eigenvalues, optimizing...")

                for config in sella_configs:
                    atoms_ts = Atoms(symbols=symbols, positions=start_pos.copy())
                    atoms_ts.calc = calc

                    print(f"      Config: delta0={config['delta0']}, internal={config['internal']}")

                    opt_ts = Sella(
                        atoms_ts,
                        trajectory=None,
                        order=1,
                        internal=config["internal"],
                        delta0=config["delta0"],
                    )
                    converged = opt_ts.run(fmax=args.fmax, steps=args.max_steps)
                    refined_pos_ts = atoms_ts.get_positions()
                    refined_energy_ts = atoms_ts.get_potential_energy()
                    refined_forces_ts = atoms_ts.get_forces()

                    num_steps = opt_ts.nsteps
                    print(f"      Steps: {num_steps}, converged: {converged}")

                    # Verify TS
                    hessian_ts = calc.get_hessian(atoms_ts)
                    hess_torch = torch.tensor(hessian_ts, dtype=torch.float64)
                    cart_coords = torch.tensor(refined_pos_ts, dtype=torch.float64).flatten()
                    freq_result = analyze_frequencies_torch(
                        hessian=hess_torch,
                        cart_coords=cart_coords,
                        atomsymbols=symbols,
                    )
                    neg_num = int(freq_result["neg_num"])

                    if neg_num == 1:
                        print(f"  ✓ Found valid TS from {geom_name}")
                        found_valid_ts = True
                        break
                    else:
                        print(f"      Still {neg_num} negative eigenvalues")

            if not found_valid_ts:
                print(f"  ✗ Could not find valid TS, skipping")
                skipped_count += 1
                continue
            
            # Store refined transition state
            refined_item = {
                "rxn_id": rxn_id,
                "atomic_numbers": item["atomic_numbers"],
                # Original TS data
                "original_ts_positions": item["transition_state_positions"],
                "original_ts_energy": item["transition_state_energy"],
                "original_ts_forces": item["transition_state_forces"],
                # Refined TS
                "refined_ts_positions": torch.tensor(refined_pos_ts, dtype=torch.float64),
                "refined_ts_energy": torch.tensor(refined_energy_ts, dtype=torch.float64),
                "refined_ts_forces": torch.tensor(refined_forces_ts, dtype=torch.float64),
            }
            
            if hessian_ts is not None:
                refined_item["refined_ts_hessian"] = torch.tensor(hessian_ts, dtype=torch.float64)
            
            txn_out.put(key, pickle.dumps(refined_item))
            valid_count += 1

            # Commit every 10 reactions
            if (idx + 1) % 10 == 0:
                txn_out.commit()
                txn_out = env_out.begin(write=True)
                print(f"  Committed {idx + 1} reactions")
        
        txn_out.commit()
        print(f"\nSaved {valid_count} valid transition states to {out_lmdb}")
        print(f"Skipped {skipped_count} entries (could not refine to valid TS)")


if __name__ == "__main__":
    main()