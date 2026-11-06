"""FPS diversity selection of unlabelled structures from an SKMD run (VASP labelling happens later)."""

import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from skmd.sampling import sample_diverse_structures
from skmd.visualise import pca_scatter

import argparse
from ase.io import read, write


def parse_args():
    p = argparse.ArgumentParser(description="Select diverse structures from an SKMD run.")
    p.add_argument("--fname", type=str,
                   help="Directory/prefix for plots and configs.")
    p.add_argument("-n", "--n_select", type=int, default=30, help="number of samples to draw")
    p.add_argument("--prev_samples", type=str, default=None, help="File containing all previously drawn samples")
    return p.parse_args()


def main():
    args = parse_args()
    folder_path = args.fname

    prev_atoms_list = None
    if args.prev_samples is not None:
        prev_atoms_list = read(args.prev_samples, index=":")

    atoms_list = read(f"{folder_path}/final_points_processed.xyz", index=":")
    selected_idx, selected = sample_diverse_structures(
        atoms_list,
        n_select=args.n_select,
        key="mace_descriptor",
        normalise=True,
        initial="medoid",
        previous_atoms_list=prev_atoms_list,
    )

    write(f"{folder_path}/diverse_subset.xyz", selected)
    print(selected_idx)

    if prev_atoms_list is not None:
        write(f"{folder_path}/new_and_previous_samples.xyz", prev_atoms_list + selected)

    # Reuse run_skmd_mace.py's burn-in PCA basis so axes are comparable across plots.
    pca_basis_path = Path(folder_path) / "traj_data" / "pca_basis.pkl"
    scaler = pca = None
    if pca_basis_path.is_file():
        with open(pca_basis_path, "rb") as fh:
            scaler, pca = pickle.load(fh)
    else:
        print(f"[draw_samples] no {pca_basis_path} found — fitting a fresh "
              "PCA basis for this plot only (axes won't match other plots)")

    labels = ["new"] if prev_atoms_list is None else ["previous", "new"]
    trajectories = [selected] if prev_atoms_list is None else [prev_atoms_list, selected]
    pca_scatter(trajectories, labels=labels, out_png=f"{folder_path}/selected_samples.png",
               scaler=scaler, pca=pca)


if __name__ == "__main__":
    main()
