#!/usr/bin/env python3
"""
Export Transition1x isopropanol (C3H8O) reaction triplets (R, TS, P) and plot.

Writes:
  - data/transition1x_isopropanol_C3H8O_triplets.lmdb (LMDB of torch tensor dicts, one per reaction)
  - plots/transition1x_isopropanol_C3H8O/rxnXXXX_triplet.png (one per reaction)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import PIL.Image
from matplotlib import pyplot as plt
import torch
import lmdb
import pickle

from adjoint_samplers.utils.eval_utils import render_xyz_to_png

REFERENCE_ENERGIES = {
    1: -13.62222753701504,  # H
    6: -1029.4130839658328,  # C
    7: -1484.8710358098756,  # N
    8: -2041.8396277138045,  # O
    9: -2712.8213146878606,  # F
}


def get_molecular_reference_energy(atomic_numbers: np.ndarray) -> float:
    return float(sum(REFERENCE_ENERGIES[int(z)] for z in atomic_numbers))


def build_xyz_from_positions_and_atomic_numbers(
    positions: np.ndarray, atomic_numbers: np.ndarray, center: bool = True
) -> str:
    pos = np.asarray(positions, dtype=float)
    if center:
        pos = pos - pos.mean(axis=0, keepdims=True)

    atomic_numbers = np.asarray(atomic_numbers)
    # Minimal mapping for Transition1x (H,C,N,O,F). Extend if needed.
    z_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    symbols = [z_to_symbol[int(z)] for z in atomic_numbers]

    lines = [f"{pos.shape[0]}", "generated"]
    for sym, p in zip(symbols, pos):
        lines.append(f"{sym} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines)


def extract_entry(rxn_grp, formula: str, rxn_id: str, state: str):
    grp = rxn_grp[state]
    atomic_numbers = np.asarray(grp["atomic_numbers"], dtype=int)
    positions = np.asarray(grp["positions"][0], dtype=float)
    forces = np.asarray(grp["wB97x_6-31G(d).forces"][0], dtype=float)
    energy = float(np.asarray(grp["wB97x_6-31G(d).energy"][0]))
    ref_e = get_molecular_reference_energy(atomic_numbers)
    atomization_energy = energy - ref_e
    return atomic_numbers, positions, forces, energy, atomization_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-path",
        type=str,
        default="/ssd/Code/Datastore/Transition1x/data/transition1x.h5",
    )
    parser.add_argument("--datasplit", type=str, default="data")
    parser.add_argument("--formula", type=str, default="C3H8O")
    parser.add_argument(
        "--out-lmdb",
        type=str,
        default="data/tx_isopropanol_C3H8O_triplets.lmdb",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/t1x_isopropanol_C3H8O",
    )
    parser.add_argument("--width", type=int, default=450)
    parser.add_argument("--height", type=int, default=450)
    args = parser.parse_args()

    h5_path = args.h5_path
    datasplit = args.datasplit
    formula = args.formula
    out_lmdb = Path(args.out_lmdb)
    plot_dir = Path(args.plot_dir)

    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Extract and write LMDB
    with h5py.File(h5_path, "r") as f, lmdb.open(
        str(out_lmdb), map_size=int(200 * 1024 * 1024), subdir=False
    ) as env:
        formula_grp = f[datasplit][formula]
        rxn_ids = sorted(formula_grp.keys())

        txn = env.begin(write=True)
        for idx, rxn_id in enumerate(rxn_ids):
            rxn_grp = formula_grp[rxn_id]

            z_r, pos_r, frc_r, e_r, ae_r = extract_entry(rxn_grp, formula, rxn_id, "reactant")
            z_p, pos_p, frc_p, e_p, ae_p = extract_entry(rxn_grp, formula, rxn_id, "product")
            z_ts, pos_ts, frc_ts, e_ts, ae_ts = extract_entry(
                rxn_grp, formula, rxn_id, "transition_state"
            )

            if not np.array_equal(z_r, z_p) or not np.array_equal(z_r, z_ts):
                raise ValueError(f"atomic_numbers mismatch across states in {rxn_id}")

            item = {
                "rxn_id": rxn_id,
                "atomic_numbers": torch.tensor(z_r, dtype=torch.int64),
                "reactant_positions": torch.tensor(pos_r, dtype=torch.float64),
                "reactant_forces": torch.tensor(frc_r, dtype=torch.float64),
                "reactant_energy": torch.tensor(e_r, dtype=torch.float64),
                "reactant_atomization_energy": torch.tensor(ae_r, dtype=torch.float64),
                "product_positions": torch.tensor(pos_p, dtype=torch.float64),
                "product_forces": torch.tensor(frc_p, dtype=torch.float64),
                "product_energy": torch.tensor(e_p, dtype=torch.float64),
                "product_atomization_energy": torch.tensor(ae_p, dtype=torch.float64),
                "transition_state_positions": torch.tensor(pos_ts, dtype=torch.float64),
                "transition_state_forces": torch.tensor(frc_ts, dtype=torch.float64),
                "transition_state_energy": torch.tensor(e_ts, dtype=torch.float64),
                "transition_state_atomization_energy": torch.tensor(ae_ts, dtype=torch.float64),
            }
            key = rxn_id.encode("utf-8")
            txn.put(key, pickle.dumps(item))

            # Optionally commit every 100
            if (idx + 1) % 100 == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        print(f"Saved LMDB with {len(rxn_ids)} items to {out_lmdb}")

    # Plot one 1x3 figure per reaction (R | TS | P)
    # Re-extract all for plotting
    with h5py.File(h5_path, "r") as f:
        formula_grp = f[datasplit][formula]
        rxn_ids = sorted(formula_grp.keys())

        for i, rxn_id in enumerate(rxn_ids):
            rxn_grp = formula_grp[rxn_id]
            z_r, pos_r, _, _, _ = extract_entry(rxn_grp, formula, rxn_id, "reactant")
            z_p, pos_p, _, _, _ = extract_entry(rxn_grp, formula, rxn_id, "product")
            z_ts, pos_ts, _, _, _ = extract_entry(rxn_grp, formula, rxn_id, "transition_state")

            if not np.array_equal(z_r, z_p) or not np.array_equal(z_r, z_ts):
                raise ValueError(f"atomic_numbers mismatch across states in {rxn_id}")

            xyz_r = build_xyz_from_positions_and_atomic_numbers(pos_r, z_r)
            xyz_ts = build_xyz_from_positions_and_atomic_numbers(pos_ts, z_r)
            xyz_p = build_xyz_from_positions_and_atomic_numbers(pos_p, z_r)

            img_r = PIL.Image.open(
                io.BytesIO(render_xyz_to_png(xyz_r, width=args.width, height=args.height))
            ).convert("RGB")
            img_ts = PIL.Image.open(
                io.BytesIO(render_xyz_to_png(xyz_ts, width=args.width, height=args.height))
            ).convert("RGB")
            img_p = PIL.Image.open(
                io.BytesIO(render_xyz_to_png(xyz_p, width=args.width, height=args.height))
            ).convert("RGB")

            fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
            for ax, img, title in zip(
                axes,
                [img_r, img_ts, img_p],
                ["Reactant", "Transition state", "Product"],
            ):
                ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")

            fig.suptitle(rxn_id, fontsize=12)
            fig.tight_layout()
            out_path = plot_dir / f"{rxn_id}_triplet.png"
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()

