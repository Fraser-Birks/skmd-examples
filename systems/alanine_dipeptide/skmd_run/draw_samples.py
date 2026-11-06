import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from skmd.sampling import sample_diverse_structures
from systems.alanine_dipeptide.skmd_run.skmd_functions import phi_psi_scatter, load_pes

import argparse
from ase.io import read, write
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Run SKMD loop with configurable parameters.")
    p.add_argument("--fname", type=str,
                   help="Directory/prefix for plots and configs.")
    
    p.add_argument("--npz_path", type=str, default="phi_psi_relaxed_pes.npz")

    p.add_argument("--oracle_model", type=str, default=None)
    p.add_argument("-n", "--n_select", type=int, default=30,help="number of samples to draw")
    p.add_argument("--prev_samples", type=str, default=None, help="File containing all previously drawn samples")

    return p.parse_args()

args = parse_args()
if args.oracle_model is not None:
    from mace.calculators import MACECalculator

def make_calculator(model_path):
    return MACECalculator(
        model_paths=model_path,
        device="cuda",
        default_dtype="float64",
    )

folder_path = args.fname
npz_path = args.npz_path

prev_atoms_list=None
if args.prev_samples is not None:
    prev_atoms_list = read(args.prev_samples,index=":")


atoms_list = read(f"{folder_path}/final_points_processed.xyz", index=":")
selected_idx, selected = sample_diverse_structures(
    atoms_list,
    n_select=args.n_select,
    key="mace_descriptor",
    normalise=True,
    initial="medoid",
    previous_atoms_list=prev_atoms_list
)
if args.oracle_model is not None:
    calc = make_calculator(args.oracle_model)

for atoms in selected:
    symbols = atoms.get_chemical_symbols()
    atoms.set_chemical_symbols(["H" if s == "X" else s for s in symbols])
    if args.oracle_model is not None:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(np.max(forces))
        atoms.info["mace_energy"] = energy
        atoms.arrays["mace_forces"] = forces.copy()

write(f"{folder_path}/diverse_subset.xyz", selected)
print(selected_idx)
# plot selected samples on phi/psi projection

if prev_atoms_list is not None:
    write(f"{folder_path}/new_and_previous_samples.xyz", prev_atoms_list + selected)

PHI, PSI, Erel, phi_atoms, psi_atoms = load_pes(npz_path)
fig, ax = plt.subplots()
cf = ax.contourf(PHI, PSI, Erel, levels=50)
cbar = fig.colorbar(cf, ax=ax, label="Energy above min (eV)")

phism, psism = phi_psi_scatter(selected, phi_atoms, psi_atoms)
ax.scatter(phism, psism,color="blue",label="new")

if prev_atoms_list is not None:
    phismold, psismold = phi_psi_scatter(prev_atoms_list, phi_atoms, psi_atoms)
    ax.scatter(phismold, psismold,color="red",label="old")
plt.legend()

plt.savefig(f"{folder_path}/selected_samples.png")

