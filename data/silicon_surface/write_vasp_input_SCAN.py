import numpy as np
from ase.calculators.vasp import Vasp
from ase.io import read
import sys

# --- k-point settings ---
KSPACING = 0.23        # target reciprocal-space spacing, Angstrom^-1
GAMMA_CENTRED = True   # Gamma-centred mesh (safer default for slabs)
SLAB_AXIS = 2          # non-periodic direction: 0=a, 1=b, 2=c

struct_name = sys.argv[1]
atoms = read(struct_name)


def slab_kpoint_grid(atoms, kspacing, slab_axis):
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


grid = slab_kpoint_grid(atoms, KSPACING, SLAB_AXIS)
print(f"k-point grid: {grid}")

calc = Vasp(
    system="energy calculation",

    # MetaGGA setup
    xc="SCAN",
    metagga="SCAN",
    lasph=True,
    ldiag=False,
    lmixtau=True,

    # Parallelisation
    ncore=8,

    # Starting parameters
    istart=1,
    icharg=1,
    ispin=1,
    isym=1,

    # Electronic relaxation
    ediff=1e-6,
    nelm=300,

    # Ionic relaxation (single-point)
    ediffg=-0.01,
    nsw=-1,
    isif=2,
    ibrion=-1,

    # Plane waves
    prec="Accurate",
    encut=900,

    # k-points via explicit KPOINTS file (do not also set kspacing/kgamma).
    kpts=grid,
    gamma=GAMMA_CENTRED,

    # FFT grid
    addgrid=True,
    lreal=False,

    # Matrix diagonalisation
    algo="Fast",

    # DOS related
    ismear=0,
    sigma=0.1,
    lorbit=11,
    nedos=2000,

    # Output
    lwave=True,
    lcharg=True,
    laechg=True,

    # POTCAR setup
    setups={"Si": "_GW", "O": "_GW_new"},
)

atoms.calc = calc
calc.initialize(atoms)
calc.write_input(atoms)