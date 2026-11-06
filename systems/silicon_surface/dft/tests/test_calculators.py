import sys
from pathlib import Path

import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calculators import slab_kpoint_grid, make_vasp_calculator_for_writing

DFT_CFG = dict(
    encut=900, ediff=1.0e-6, ediffg=-0.01, prec="Accurate",
    ismear=0, sigma=0.1, lreal=False, addgrid=True, lasph=True,
    ldiag=False, lmixtau=True, algo="Fast", nelm=300, ncore=8,
    isym=1, gamma_centred=True, kspacing=0.23, slab_axis=2,
)


def _slab(a=7.7321175927162855, c=43.7288986805749):
    # cell shape matches relaxed_struct_non_pandey.xyz
    cell = [[a, 0.0, 0.0], [a / 4, 3.3481051301704414, 0.0], [0.0, 0.0, c]]
    return Atoms("Si", positions=[[0, 0, 0]], cell=cell, pbc=True)


def test_slab_kpoint_grid_forces_single_kpoint_along_vacuum_axis():
    grid = slab_kpoint_grid(_slab(), kspacing=0.23, slab_axis=2)
    assert grid[2] == 1


def test_slab_kpoint_grid_at_least_one_kpoint_per_axis():
    grid = slab_kpoint_grid(_slab(), kspacing=0.23, slab_axis=2)
    assert all(n >= 1 for n in grid)


def test_slab_kpoint_grid_tighter_kspacing_gives_more_kpoints():
    coarse = slab_kpoint_grid(_slab(), kspacing=1.0, slab_axis=2)
    fine = slab_kpoint_grid(_slab(), kspacing=0.1, slab_axis=2)
    assert fine[0] >= coarse[0]
    assert fine[1] >= coarse[1]


def test_calculator_never_shells_out():
    calc = make_vasp_calculator_for_writing(_slab(), DFT_CFG, mode="static")
    assert calc.command != "vasp_std"
    assert "VASP_NOT_EXECUTED_HERE" in calc.command


def test_calculator_drops_output_files_and_oxygen_setup():
    calc = make_vasp_calculator_for_writing(_slab(), DFT_CFG, mode="static")
    assert calc.bool_params["lwave"] is False
    assert calc.bool_params["lcharg"] is False
    assert calc.bool_params["laechg"] is False
    assert calc.input_params["setups"] == {"Si": "_GW"}
    assert "O" not in calc.input_params["setups"]


def test_calculator_static_mode_no_ionic_relaxation():
    calc = make_vasp_calculator_for_writing(_slab(), DFT_CFG, mode="static")
    assert calc.int_params["ibrion"] == -1
    assert calc.int_params["nsw"] == -1
    assert calc.int_params["ispin"] == 1


def test_calculator_unknown_mode_rejected():
    with pytest.raises(ValueError):
        make_vasp_calculator_for_writing(_slab(), DFT_CFG, mode="bogus")
