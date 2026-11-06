"""
Oracle-label online-selected selected_data.xyz directly (a-SKMD Eq. 8 / UHAL already select; no FPS).

Usage:
    python label_selected_data.py --fname <dir> --oracle_model <path> \
        [--prev_samples <xyz>] [--npz_path <npz>]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from ase.io import read, write
import matplotlib.pyplot as plt
import numpy as np

from systems.alanine_dipeptide.skmd_run.skmd_functions import phi_psi_scatter, load_pes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fname", type=str, required=True,
                   help="Directory containing selected_data.xyz (run_skmd_mace.py's --fname).")
    p.add_argument("--oracle_model", type=str, required=True)
    p.add_argument("--prev_samples", type=str, default=None,
                   help="File containing all previously drawn samples (prior AL iterations).")
    p.add_argument("--npz_path", type=str, default=None,
                   help="PES .npz for the selected-samples Ramachandran plot. Skipped if unset.")
    return p.parse_args()


def main():
    args = parse_args()
    from mace.calculators import MACECalculator

    selected = read(f"{args.fname}/selected_data.xyz", index=":")

    prev_atoms_list = None
    if args.prev_samples is not None:
        prev_atoms_list = read(args.prev_samples, index=":")

    calc = MACECalculator(model_paths=args.oracle_model, device="cuda", default_dtype="float64")

    for atoms in selected:
        symbols = atoms.get_chemical_symbols()
        atoms.set_chemical_symbols(["H" if s == "X" else s for s in symbols])
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(np.max(forces))
        atoms.info["mace_energy"] = energy
        atoms.arrays["mace_forces"] = forces.copy()

    write(f"{args.fname}/diverse_subset.xyz", selected)
    print(f"Labeled {len(selected)} a-SKMD online-selected configs -> "
          f"{args.fname}/diverse_subset.xyz")

    if prev_atoms_list is not None:
        write(f"{args.fname}/new_and_previous_samples.xyz", prev_atoms_list + selected)

    if args.npz_path is not None:
        PHI, PSI, Erel, phi_atoms, psi_atoms = load_pes(args.npz_path)
        fig, ax = plt.subplots()
        cf = ax.contourf(PHI, PSI, Erel, levels=50)
        fig.colorbar(cf, ax=ax, label="Energy above min (eV)")

        phism, psism = phi_psi_scatter(selected, phi_atoms, psi_atoms)
        ax.scatter(phism, psism, color="blue", label="new")

        if prev_atoms_list is not None:
            phismold, psismold = phi_psi_scatter(prev_atoms_list, phi_atoms, psi_atoms)
            ax.scatter(phismold, psismold, color="red", label="old")
        plt.legend()
        plt.savefig(f"{args.fname}/selected_samples.png")


if __name__ == "__main__":
    main()
