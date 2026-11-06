"""
AD-specific geometry and PES utilities for alanine dipeptide.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from skmd.visualise import fill_nan_by_interp


def dihedral_deg(p0, p1, p2, p3) -> float:
    """Dihedral angle in degrees for four Cartesian positions."""
    p0 = np.asarray(p0); p1 = np.asarray(p1)
    p2 = np.asarray(p2); p3 = np.asarray(p3)
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n

    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def phi_psi_scatter(traj_atoms, phi_atoms, psi_atoms) -> tuple[np.ndarray, np.ndarray]:
    """Compute phi/psi dihedral angles for each frame.

    Args:
        traj_atoms: List of ASE Atoms objects.
        phi_atoms:  4-tuple of 0-based atom indices for phi.
        psi_atoms:  4-tuple of 0-based atom indices for psi.

    Returns:
        (phis, psis) — arrays of dihedral angles in degrees.
    """
    phi_atoms = tuple(int(i) for i in phi_atoms)
    psi_atoms = tuple(int(i) for i in psi_atoms)

    phis = np.empty(len(traj_atoms), dtype=float)
    psis = np.empty(len(traj_atoms), dtype=float)

    for t, a in enumerate(traj_atoms):
        pos = a.get_positions()
        phis[t] = dihedral_deg(pos[phi_atoms[0]], pos[phi_atoms[1]],
                               pos[phi_atoms[2]], pos[phi_atoms[3]])
        psis[t] = dihedral_deg(pos[psi_atoms[0]], pos[psi_atoms[1]],
                               pos[psi_atoms[2]], pos[psi_atoms[3]])

    return phis, psis


def load_pes(npz_path: str, absolute: bool = False):
    """Load and clean a phi/psi PES from an .npz file.

    Args:
        npz_path: Path to the .npz file containing PHI, PSI, E, phi_atoms, psi_atoms.
        absolute: If False (default), return E relative to its minimum.
                  If True, return the cleaned absolute E values.

    Returns:
        (PHI, PSI, E, phi_atoms, psi_atoms)
    """
    data = np.load(npz_path)
    PHI = data["PHI"]
    PSI = data["PSI"]
    E   = data["E"].astype(float)

    E[E < -20000.0] = np.nan
    E_min = np.nanpercentile(E, 1)
    E[np.abs(E - E_min) > 1.44] = np.nan
    E = fill_nan_by_interp(PHI, PSI, E, method="linear")

    if not absolute:
        E = E - np.nanmin(E)

    phi_atoms = data["phi_atoms"]
    psi_atoms = data["psi_atoms"]
    return PHI, PSI, E, phi_atoms, psi_atoms


def plot_contour_phi_psi(PHI, PSI, E, levels=200, title="E(φ, ψ)", out_png=None):
    """Ramachandran contour plot.

    Args:
        PHI, PSI: 2-D coordinate grids.
        E:        2-D energy grid (typically relative to minimum).
        levels:   Number of contour fill levels.
        title:    Plot title.
        out_png:  If given, save to this path.
    """
    Erel = E - np.nanmin(E)
    fig, ax = plt.subplots()
    cf = ax.contourf(PHI, PSI, Erel, levels=50)
    plt.colorbar(cf, ax=ax, label="Energy above min (eV)")
    ax.set_xlabel("φ (deg)")
    ax.set_ylabel("ψ (deg)")
    ax.set_title(title)
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200)
    plt.close(fig)
