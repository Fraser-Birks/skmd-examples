"""One-off setup: reorder the Si slab so frozen layers come first (IDs 1..N_FROZEN) and write
silicon_slab.data plus an N_FROZEN sidecar. Tag numbering runs top-to-bottom; check the printed z-ranges.
"""

import argparse

import numpy as np
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz", required=True, help="Input relaxed slab xyz (with a `tags` array)")
    p.add_argument("--data_out", required=True, help="Output LAMMPS .data file path")
    p.add_argument("--frozen_tags", type=int, nargs="+", default=[4,5,6,7,8],
                   help="tags of frozen substrate layers (higher tags = lower z)")
    return p.parse_args()


def main():
    args = parse_args()
    atoms = read(args.xyz)

    tags = atoms.get_tags()
    frozen_mask = np.array([t in args.frozen_tags for t in tags])
    n_frozen = int(frozen_mask.sum())

    z = atoms.positions[:, 2]
    frozen_z = z[frozen_mask]
    mobile_z = z[~frozen_mask]
    print(f"frozen_tags={args.frozen_tags}: frozen z-range "
          f"[{frozen_z.min():.3f}, {frozen_z.max():.3f}], mobile z-range "
          f"[{mobile_z.min():.3f}, {mobile_z.max():.3f}]")
    if frozen_z.max() > mobile_z.min():
        print("WARNING: frozen atoms are NOT the lowest-z group — "
              "--frozen_tags is probably selecting the wrong layers "
              "(check the tag-to-z convention for this structure).")

    order = [i for i, f in enumerate(frozen_mask) if f] + \
            [i for i, f in enumerate(frozen_mask) if not f]
    reordered = atoms[order]

    write(args.xyz, reordered, format="extxyz")
    write_lammps_data(args.data_out, reordered, velocities=False,
                      atom_style="atomic", specorder=["Si"])

    sidecar = f"{args.data_out}.n_frozen"
    with open(sidecar, "w") as f:
        f.write(f"{n_frozen}\n")

    print(f"Reordered {len(atoms)} atoms: {n_frozen} frozen (first), "
          f"{len(atoms) - n_frozen} mobile (rest).")
    print(f"Wrote reordered structure: {args.xyz}")
    print(f"Wrote LAMMPS data file: {args.data_out}")
    print(f"Wrote N_FROZEN sidecar: {sidecar}")
    print(f"N_FROZEN={n_frozen}")


if __name__ == "__main__":
    main()
