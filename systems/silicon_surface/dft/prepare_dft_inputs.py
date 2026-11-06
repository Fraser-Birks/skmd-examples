#!/usr/bin/env python3
"""Write VASP single-point inputs (one frame_NNNN/ dir per frame) for a diverse_subset.xyz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import ase.io

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dft_runner import prepare_dft_inputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset', type=Path, required=True,
                   help='diverse_subset.xyz written by draw_samples.py')
    p.add_argument('--output', type=Path, required=True,
                   help='Root directory for frame_NNNN/ subdirectories')
    p.add_argument('--config', type=Path, required=True,
                   help='config.yaml containing a dft: section')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dft_cfg = cfg['dft']

    print(f'[prepare] reading: {args.subset}')
    frames = ase.io.read(str(args.subset), index=':')
    print(f'[prepare] {len(frames)} frames')

    args.output.mkdir(parents=True, exist_ok=True)

    for i, atoms in enumerate(frames):
        workdir = args.output / f'frame_{i:04d}'
        print(f'[prepare] frame {i:04d}: {len(atoms)} atoms -> {workdir}')
        prepare_dft_inputs(atoms, workdir, dft_cfg, mode='static')

    print(f'[prepare] done — {len(frames)} frame directories in {args.output}')


if __name__ == "__main__":
    main()
