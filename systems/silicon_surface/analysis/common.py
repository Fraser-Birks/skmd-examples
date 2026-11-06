"""Shared helpers for the silicon_surface analysis scripts; all paths are caller-supplied."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_mace_calculator(model_path, device: str = "cuda"):
    """Build an ASE MACECalculator from a .model file."""
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(model_path), device=device, default_dtype="float64")


def resolve_n_frozen(n_frozen_arg: int | None, lammps_data) -> int:
    """Explicit --n-frozen, else the .n_frozen sidecar next to --lammps-data; exits if neither (no silent 0)."""
    if n_frozen_arg is not None:
        return n_frozen_arg
    sidecar = Path(str(lammps_data) + ".n_frozen")
    if sidecar.is_file():
        n = int(sidecar.read_text().strip())
        print(f"[resolve_n_frozen] --n-frozen not given, read {n} from {sidecar}")
        return n
    print(f"[resolve_n_frozen] ERROR: --n-frozen not given and no sidecar at {sidecar} — "
          "pass --n-frozen explicitly (0 to relax every atom freely).", file=sys.stderr)
    raise SystemExit(1)


def lowest_z_indices(atoms, n: int) -> list[int]:
    """Indices of the n lowest-z atoms (frozen substrate) for structures not reordered by prepare_lammps_data.py."""
    return sorted(int(i) for i in np.argsort(atoms.positions[:, 2])[:n])


def relax_atoms(atoms, calc, fmax: float = 0.02, steps: int = 500, n_frozen: int = 0,
                frozen_indices=None):
    """LBFGS-relax a copy (fixed cell), freezing frozen_indices or else the first n_frozen atoms.

    Returns (relaxed_atoms, converged, nsteps, energy)."""
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    relaxed = atoms.copy()
    if frozen_indices is not None:
        relaxed.set_constraint(FixAtoms(indices=list(frozen_indices)))
    elif n_frozen > 0:
        relaxed.set_constraint(FixAtoms(indices=list(range(n_frozen))))
    relaxed.calc = calc
    # One restart with a fresh LBFGS history if the first run stalls.
    nsteps, converged = 0, False
    for _ in range(2):
        opt = LBFGS(relaxed, logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
        nsteps += opt.nsteps
        if converged:
            break
    energy = float(relaxed.get_potential_energy())
    return relaxed, bool(converged), int(nsteps), energy

