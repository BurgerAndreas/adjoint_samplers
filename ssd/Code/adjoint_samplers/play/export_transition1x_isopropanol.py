#!/usr/bin/env python3
"""
Export Transition1x isopropanol (C3H8O) reaction triplets (R, TS, P) and plot.

Writes:
  - data/transition1x_isopropanol_C3H8O_triplets.npz
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
        "--out-npz",
        type=str,
        default="data/transition1x_isopropanol_C3H8O_triplets.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots/transition1x_isopropanol_C3H8O",
    )
    parser.add_argument("--width", type=int, default=450)
    parser.add_argument("--height", type=int, default=450)
    args = parser.parse_args()

    h5_path = args.h5_path
    datasplit = args.datasplit
    formula = args.formula
    out_npz = Path(args.out_npz)
    out_dir = Path(args.out_dir)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        formula_grp = f[datasplit][formula]
        rxn_ids = sorted(formula_grp.keys())

        # Extract one entry per state for each rxn.
        atomic_numbers = []
        r_positions = []
        r_forces = []
        r_energy = []
        r_atom_e = []

        p_positions = []
        p_forces = []
        p_energy = []
        p_atom_e = []

        ts_positions = []
        ts_forces = []
        ts_energy = []
        ts_atom_e = []

        for rxn_id in rxn_ids:
            rxn_grp = formula_grp[rxn_id]

            z_r, pos_r, frc_r, e_r, ae_r = extract_entry(rxn_grp, formula, rxn_id, "reactant")
            z_p, pos_p, frc_p, e_p, ae_p = extract_entry(rxn_grp, formula, rxn_id, "product")
            z_ts, pos_ts, frc_ts, e_ts, ae_ts = extract_entry(
                rxn_grp, formula, rxn_id, "transition_state"
            )

            if not np.array_equal(z_r, z_p) or not np.array_equal(z_r, z_ts):
                raise ValueError(f"atomic_numbers mismatch across states in {rxn_id}")

            atomic_numbers.append(z_r)
            r_positions.append(pos_r)
            r_forces.append(frc_r)
            r_energy.append(e_r)
            r_atom_e.append(ae_r)

            p_positions.append(pos_p)
            p_forces.append(frc_p)
            p_energy.append(e_p)
            p_atom_e.append(ae_p)

            ts_positions.append(pos_ts)
            ts_forces.append(frc_ts)
            ts_energy.append(e_ts)
            ts_atom_e.append(ae_ts)

    rxn_ids_arr = np.asarray(rxn_ids, dtype=str)
    atomic_numbers_arr = np.asarray(atomic_numbers, dtype=int)

    np.savez_compressed(
        out_npz,
        rxn_ids=rxn_ids_arr,
        atomic_numbers=atomic_numbers_arr,
        reactant_positions=np.asarray(r_positions, dtype=float),
        reactant_forces=np.asarray(r_forces, dtype=float),
        reactant_energy=np.asarray(r_energy, dtype=float),
        reactant_atomization_energy=np.asarray(r_atom_e, dtype=float),
        product_positions=np.asarray(p_positions, dtype=float),
        product_forces=np.asarray(p_forces, dtype=float),
        product_energy=np.asarray(p_energy, dtype=float),
        product_atomization_energy=np.asarray(p_atom_e, dtype=float),
        transition_state_positions=np.asarray(ts_positions, dtype=float),
        transition_state_forces=np.asarray(ts_forces, dtype=float),
        transition_state_energy=np.asarray(ts_energy, dtype=float),
        transition_state_atomization_energy=np.asarray(ts_atom_e, dtype=float),
    )

    # Plot one 1x3 figure per reaction (R | TS | P)
    for i, rxn_id in enumerate(rxn_ids_arr.tolist()):
        z = atomic_numbers_arr[i]
        xyz_r = build_xyz_from_positions_and_atomic_numbers(r_positions[i], z)
        xyz_ts = build_xyz_from_positions_and_atomic_numbers(ts_positions[i], z)
        xyz_p = build_xyz_from_positions_and_atomic_numbers(p_positions[i], z)

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
        out_path = out_dir / f"{rxn_id}_triplet.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()


