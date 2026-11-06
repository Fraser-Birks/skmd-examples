"""Utilities for writing DFT-labelled frames to extxyz files for MACE training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ase.io
from ase import Atoms

from dft_runner import DFTResult


def write_xyz_frame(
    atoms: Atoms,
    result: DFTResult,
    xyz_path: Path,
    iteration: int,
) -> None:
    """Append one DFT-labelled frame (keys as in lora_input.yaml; eV, eV/Å, Voigt eV/Å³) to an extxyz."""
    out = atoms.copy()
    out.info['dft_energy'] = float(result.energy)
    out.info['config_type'] = 'si_slab_skmd'
    out.info['iteration'] = iteration
    out.arrays['dft_forces'] = np.asarray(result.forces)
    if result.stress is not None:
        out.info['dft_stresses'] = np.asarray(result.stress)
    ase.io.write(str(xyz_path), out, format='extxyz', append=True)
