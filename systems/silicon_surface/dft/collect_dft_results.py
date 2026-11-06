#!/usr/bin/env python3
"""Parse VASP OUTCARs from frame_NNNN/ directories and write a cumulative extxyz.

Converged frames are appended; unconverged or missing OUTCARs are skipped with
a warning. Exits non-zero if no converged frames are found (training would fail).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import ase.io
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dft_runner import parse_outcar
from xyz_utils import write_xyz_frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dft-inputs', type=Path, required=True,
                   help='Directory containing frame_NNNN/ subdirectories')
    p.add_argument('--output', type=Path, required=True,
                   help='Output extxyz file (cumulative; created or appended)')
    p.add_argument('--prev-xyz', type=Path, default=None,
                   help='Previous iteration XYZ to seed cumulative file')
    p.add_argument('--iteration', type=int, required=True,
                   help='Current AL iteration number')
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.prev_xyz is not None and args.prev_xyz.is_file():
        shutil.copy(args.prev_xyz, args.output)
        print(f'[collect] seeded from {args.prev_xyz}')
    elif args.output.exists():
        args.output.unlink()

    frame_dirs = sorted(args.dft_inputs.glob('frame_????'))
    print(f'[collect] {len(frame_dirs)} frame directories in {args.dft_inputs}')

    n_ok = n_skip = 0
    for frame_dir in frame_dirs:
        result = parse_outcar(frame_dir)
        if result is None:
            print(f'  [skip] {frame_dir.name}: no usable OUTCAR')
            n_skip += 1
            continue
        if not result.converged:
            print(f'  [skip] {frame_dir.name}: unconverged')
            n_skip += 1
            continue
        if result.forces is None:
            print(f'  [skip] {frame_dir.name}: forces not parsed')
            n_skip += 1
            continue
        max_force = np.max(np.abs(result.forces))
        if max_force > 20.0:
            print(f'  [skip] {frame_dir.name}: max force {max_force} too large')
            n_skip += 1
            continue

        atoms = ase.io.read(str(frame_dir / 'POSCAR'), format='vasp')
        atoms.pbc = [True, True, True]

        write_xyz_frame(atoms, result, args.output, args.iteration)
        write_xyz_frame(atoms, result, args.output.parent / f"just_new_data_it_{args.iteration}.xyz", args.iteration)
        print(f'  [ok]   {frame_dir.name}: E={result.energy:.4f} eV, '
              f'{len(atoms)} atoms')
        n_ok += 1

    print(f'[collect] {n_ok} frames written, {n_skip} skipped -> {args.output}')
    if n_ok == 0:
        print('[collect] ERROR: no converged frames found', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
