"""VASP/SCAN calculator construction (input writing only; jobs run via SLURM).

Fresh single points: ISTART=0/ICHARG=2, no WAVECAR/CHGCAR, Si _GW POTCAR.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def slab_kpoint_grid(atoms, kspacing: float, slab_axis: int):
    """Reproduce VASP's KSPACING scheme,
        N_i = max(1, ceil(2*pi*|b_i| / KSPACING)),
    with b_i the reciprocal vectors (b_i . a_j = delta_ij), then force the
    non-periodic slab axis to a single k-point.
    """
    recip = atoms.cell.reciprocal()            # rows are b_i, no 2*pi factor
    b_lengths = np.linalg.norm(recip, axis=1)
    N = np.maximum(1, np.ceil(2.0 * np.pi * b_lengths / kspacing).astype(int))
    N[slab_axis] = 1
    return [int(n) for n in N]


def make_vasp_calculator_for_writing(atoms, dft_cfg: dict, mode: str = "static"):
    """Build a write-only VASP calculator from a config 'dft' section; mode is 'static' or 'relax_ions'."""
    try:
        from ase.calculators.vasp import Vasp
    except ImportError as exc:
        raise ImportError("ASE VASP calculator not available.") from exc

    grid = slab_kpoint_grid(atoms, dft_cfg["kspacing"], dft_cfg.get("slab_axis", 2))

    base = dict(
        system="silicon_surface AL single-point",

        xc="SCAN",
        metagga="SCAN",
        lasph=dft_cfg.get("lasph", True),
        ldiag=dft_cfg.get("ldiag", False),
        lmixtau=dft_cfg.get("lmixtau", True),

        # ncore*kpar must divide the MPI rank count (see config_cpu.yaml).
        ncore=dft_cfg.get("ncore", 8),
        kpar=dft_cfg.get("kpar", 1),

        istart=0,
        icharg=2,
        ispin=1,
        isym=dft_cfg.get("isym", 1),

        ediff=dft_cfg["ediff"],
        nelm=dft_cfg.get("nelm", 300),

        prec=dft_cfg.get("prec", "Accurate"),
        encut=dft_cfg["encut"],

        kpts=grid,
        gamma=dft_cfg.get("gamma_centred", True),

        addgrid=dft_cfg.get("addgrid", True),
        lreal=dft_cfg.get("lreal", False),

        algo=dft_cfg.get("algo", "Fast"),

        ismear=dft_cfg["ismear"],
        sigma=dft_cfg["sigma"],

        lwave=False,
        lcharg=False,
        laechg=False,

        setups={"Si": "_GW"},
    )

    if mode == "static":
        base.update(ibrion=-1, nsw=-1, isif=2)
    elif mode == "relax_ions":
        base.update(ibrion=2, nsw=200, isif=2, ediffg=dft_cfg.get("ediffg", -0.01))
    else:
        raise ValueError(f"Unknown DFT mode: {mode}")

    calc = Vasp(**base)
    # Defeat any accidental subprocess call — inputs are written, never run.
    calc.command = "echo VASP_NOT_EXECUTED_HERE"
    return calc
