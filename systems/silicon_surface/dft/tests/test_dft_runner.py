import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dft_runner import find_outcar, _check_outcar_converged


CONVERGED_STATIC = """\
 NSW    =     -1    number of steps for IOM
 General timing and accounting informations for this job:
 Total CPU time used (sec):        1.234
"""

UNCONVERGED_RELAX = """\
 NSW    =    200    number of steps for IOM
 General timing and accounting informations for this job:
 Total CPU time used (sec):        1.234
"""

CONVERGED_RELAX = """\
 NSW    =    200    number of steps for IOM
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
 Total CPU time used (sec):        1.234
"""

CRASHED = """\
 NSW    =     -1    number of steps for IOM
 EDDDAV: Call to ZHEGV failed. Returncode =  5  1  8
 General timing and accounting informations for this job:
"""

INCOMPLETE = """\
 NSW    =     -1    number of steps for IOM
 still running...
"""


def _write_outcar(tmp_path, text):
    workdir = tmp_path / "frame_0000"
    workdir.mkdir()
    (workdir / "OUTCAR").write_text(text)
    return workdir


def test_static_run_converged(tmp_path):
    workdir = _write_outcar(tmp_path, CONVERGED_STATIC)
    outcar = find_outcar(workdir)
    assert outcar is not None
    assert _check_outcar_converged(outcar) is True


def test_relax_run_needs_accuracy_marker(tmp_path):
    workdir = _write_outcar(tmp_path, UNCONVERGED_RELAX)
    outcar = find_outcar(workdir)
    assert outcar is not None
    assert _check_outcar_converged(outcar) is False


def test_relax_run_converged_with_accuracy_marker(tmp_path):
    workdir = _write_outcar(tmp_path, CONVERGED_RELAX)
    outcar = find_outcar(workdir)
    assert outcar is not None
    assert _check_outcar_converged(outcar) is True


def test_crashed_run_not_converged(tmp_path):
    workdir = _write_outcar(tmp_path, CRASHED)
    outcar = find_outcar(workdir)
    # find_outcar succeeds, but the ZHEGV crash marker must fail convergence.
    assert outcar is not None
    assert _check_outcar_converged(outcar) is False


def test_incomplete_run_has_no_outcar(tmp_path):
    workdir = _write_outcar(tmp_path, INCOMPLETE)
    assert find_outcar(workdir) is None


def test_missing_outcar(tmp_path):
    workdir = tmp_path / "frame_0001"
    workdir.mkdir()
    assert find_outcar(workdir) is None
