"""VASP/SCAN oracle management: parse a finished OUTCAR, else write inputs and optionally submit via SLURM."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculators import make_vasp_calculator_for_writing


@dataclass
class DFTResult:
    """Results extracted from a finished DFT calculation."""
    energy: float                       # eV
    forces: Optional[np.ndarray]        # (N, 3) eV/Å, or None if not parsed
    stress: Optional[np.ndarray]        # (6,) Voigt, eV/Å³, or None
    atoms_relaxed: Optional[object]     # ase.Atoms (CONTCAR) or None
    converged: bool
    source: str                         # path to OUTCAR


def _has_final_energy_block(text: str) -> bool:
    """Free-energy summary printed: completion marker when the GPU build crashes in MPI teardown before the timing trailer."""
    return "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in text and "energy(sigma->0)" in text


def find_outcar(workdir: Path) -> Optional[Path]:
    """Return the OUTCAR path if it exists and looks complete."""
    workdir = Path(workdir)
    outcar = workdir / "OUTCAR"
    if not outcar.exists():
        return None
    try:
        with open(outcar, "rb") as f:
            f.seek(-4096, os.SEEK_END) if outcar.stat().st_size > 4096 else f.seek(0)
            tail = f.read().decode(errors="ignore")
        if "General timing and accounting" in tail or "Total CPU time" in tail:
            return outcar
        if _has_final_energy_block(tail):
            return outcar
        return None
    except OSError:
        return None


def parse_outcar(workdir: Path, expected_natoms: Optional[int] = None) -> Optional[DFTResult]:
    """Parse an OUTCAR / CONTCAR via ASE."""
    from ase.io import read
    workdir = Path(workdir)
    outcar = find_outcar(workdir)
    if outcar is None:
        return None

    try:
        atoms = read(str(outcar), index=-1, format="vasp-out")
    except Exception:
        return None

    if expected_natoms is not None and len(atoms) != expected_natoms:
        raise RuntimeError(
            f"OUTCAR in {workdir} has {len(atoms)} atoms, expected {expected_natoms}."
        )

    try:
        energy = atoms.get_potential_energy()
    except Exception:
        return None

    try:
        forces = atoms.get_forces()
    except Exception:
        forces = None

    try:
        stress = atoms.get_stress(voigt=True)
    except Exception:
        stress = None

    contcar = workdir / "CONTCAR"
    atoms_relaxed = None
    if contcar.exists() and contcar.stat().st_size > 0:
        try:
            atoms_relaxed = read(str(contcar))
        except Exception:
            atoms_relaxed = None

    converged = _check_outcar_converged(outcar)

    return DFTResult(
        energy=float(energy),
        forces=np.asarray(forces) if forces is not None else None,
        stress=np.asarray(stress) if stress is not None else None,
        atoms_relaxed=atoms_relaxed,
        converged=converged,
        source=str(outcar),
    )


def _check_outcar_converged(outcar: Path) -> bool:
    """Best-effort convergence check from OUTCAR text."""
    import re
    try:
        text = Path(outcar).read_text(errors="ignore")
    except OSError:
        return False
    if "General timing and accounting" not in text and not _has_final_energy_block(text):
        return False
    if "EDDDAV: Call to ZHEGV failed" in text:
        return False
    # Static run (NSW<=0): completion alone is sufficient.
    m = re.search(r"NSW\s*=\s*(-?\d+)", text)
    if m and int(m.group(1)) <= 0:
        return True
    # Ionic relaxation: 'reached required accuracy' is the success marker.
    if "reached required accuracy" in text:
        return True
    return False


def prepare_dft_inputs(atoms, workdir: Path, dft_cfg: dict, mode: str = "static"):
    """Write VASP inputs; without VASP_PP_PATH, write INCAR/KPOINTS/POSCAR and a POTCAR_MISSING marker."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    calc = make_vasp_calculator_for_writing(atoms, dft_cfg, mode=mode)
    calc.directory = str(workdir)
    atoms_with_calc = atoms.copy()
    atoms_with_calc.calc = calc

    try:
        calc.write_input(atoms_with_calc)
        return workdir
    except Exception as e:
        print(f"  [warn] ASE write_input failed ({e}); writing INCAR/POSCAR/KPOINTS manually.")

    from ase.io.vasp import write_vasp
    write_vasp(str(workdir / "POSCAR"), atoms_with_calc, sort=True)

    incar_lines = []
    for d in (calc.float_params, calc.exp_params, calc.string_params,
              calc.int_params, calc.bool_params, calc.list_int_params,
              calc.list_float_params, calc.list_bool_params,
              calc.special_params, calc.dict_params):
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v_str = ".TRUE." if v else ".FALSE."
            elif isinstance(v, (list, tuple)):
                v_str = " ".join(str(x) for x in v)
            else:
                v_str = str(v)
            incar_lines.append(f"{k.upper()} = {v_str}")
    (workdir / "INCAR").write_text("\n".join(incar_lines) + "\n")

    kpts = calc.input_params.get("kpts", (1, 1, 1))
    gamma = calc.input_params.get("gamma", True)
    kpts_text = "Auto-generated\n0\n" + ("Gamma\n" if gamma else "Monkhorst-Pack\n")
    kpts_text += "{} {} {}\n0 0 0\n".format(*kpts)
    (workdir / "KPOINTS").write_text(kpts_text)

    (workdir / "POTCAR_MISSING").write_text(
        "POTCAR could not be written automatically.\n"
        "Set VASP_PP_PATH (pointing to PAW_PBE pseudopotential directory) and re-run,\n"
        "or assemble POTCAR manually using Si_GW (not plain Si):\n"
        "  cat $VASP_PP_PATH/PAW_PBE/Si_GW/POTCAR > POTCAR\n"
    )
    return workdir
