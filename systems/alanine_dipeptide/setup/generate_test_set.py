"""
Build the held-out test set (paper App. E.2): raster (phi, psi), relax all other DOF with the
dihedrals fixed (FixInternals + BFGS), and label with the oracle (mace_energy/mace_forces).

Usage:
    python generate_test_set.py \
        --init_xyz ../../../data/alanine_dipeptide/ace-ala-nme.xyz \
        --oracle_model ../../../data/alanine_dipeptide/MACE-OFF23_small.model \
        --output ../../../data/alanine_dipeptide/test_set.xyz

--n_rattled > 0 also adds single-point-labelled rattled copies (config_type="rattled"; not used in the paper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS
from ase.constraints import FixInternals
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from systems.alanine_dipeptide.skmd_run.skmd_functions import dihedral_deg, plot_contour_phi_psi

# 0-based indices into ace-ala-nme.xyz (same convention as phi_psi_relaxed_pes_off23_small.npz).
PHI_ATOMS = (1, 3, 4, 6)   # ace-C, ala-N, ala-CA, ala-C
PSI_ATOMS = (3, 4, 6, 8)   # ala-N, ala-CA, ala-C, nme-N


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init_xyz", required=True, help="Seed ace-ala-nme structure (e.g. data/alanine_dipeptide/ace-ala-nme.xyz)")
    p.add_argument("--oracle_model", required=True, help="MACE .model file to label with (e.g. MACE-OFF23_small.model)")
    p.add_argument("--output", required=True, help="Output test_set.xyz path")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--nphi", type=int, default=10, help="Grid points along phi")
    p.add_argument("--npsi", type=int, default=10, help="Grid points along psi")
    p.add_argument("--phi_range", type=float, nargs=2, default=(-180.0, 180.0))
    p.add_argument("--psi_range", type=float, nargs=2, default=(-180.0, 180.0))
    p.add_argument("--fmax", type=float, default=0.05, help="BFGS force convergence (eV/A)")
    p.add_argument("--max_steps", type=int, default=400, help="Max BFGS steps per grid point")
    p.add_argument("--n_shards", type=int, default=1, help="Split the grid into N_SHARDS independent chunks (e.g. for a SLURM array)")
    p.add_argument("--shard_index", type=int, default=0, help="Which shard (0-based) this invocation writes; only meaningful with --n_shards > 1")
    p.add_argument("--pes_out", default=None, help="Optional diagnostic .npz + .png basename (default: alongside --output)")
    p.add_argument("--n_rattled", type=int, default=0,
                    help="Also generate this many rattled (Gaussian-perturbed, single-point "
                         "labelled) test points (0 = disabled).")
    p.add_argument("--rattle_stdev_min", type=float, default=0.01,
                    help="Minimum ase.Atoms.rattle stdev (Angstrom) for rattled points")
    p.add_argument("--rattle_stdev_max", type=float, default=0.15,
                    help="Maximum ase.Atoms.rattle stdev (Angstrom) for rattled points")
    p.add_argument("--rattle_seed", type=int, default=0, help="RNG seed for rattled-point selection/displacement")
    return p.parse_args()


def rotation_matrix(axis, angle_rad):
    """Standard axis-angle (Rodrigues) rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def build_bond_graph(atoms, cutoff_scale: float = 1.2) -> dict[int, set[int]]:
    """Adjacency list from natural bond cutoffs (computed once; rigid rotations preserve bonds)."""
    cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    nl = NeighborList(cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms)
    neighbors = {i: set() for i in range(len(atoms))}
    for i in range(len(atoms)):
        idxs, _ = nl.get_neighbors(i)
        for j in idxs:
            neighbors[i].add(int(j))
    return neighbors


def reachable_side(neighbors: dict[int, set[int]], start: int, blocked: int) -> list[int]:
    """Atoms reachable from `start` without crossing through `blocked`."""
    visited = {blocked}
    stack = [start]
    group = set()
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        group.add(u)
        stack.extend(v for v in neighbors[u] if v not in visited)
    return sorted(group)


def set_dihedral_rotate_k_side(atoms, i, j, k, l, target_deg, neighbors) -> None:
    """Rigidly rotate the k-side of the i-j-k-l dihedral to `target_deg` (pre-relaxation guess)."""
    pos = atoms.get_positions()
    current = dihedral_deg(pos[i], pos[j], pos[k], pos[l])
    delta = (float(target_deg) - current + 180.0) % 360.0 - 180.0
    if abs(delta) < 1e-12:
        return
    axis = pos[k] - pos[j]
    R = rotation_matrix(axis, np.radians(delta))
    pivot = pos[j]
    group = reachable_side(neighbors, start=k, blocked=j)
    newpos = pos.copy()
    for a in group:
        newpos[a] = pivot + R @ (pos[a] - pivot)
    atoms.set_positions(newpos)


def constrain_phi_psi(atoms, phi_atoms, psi_atoms, phi_deg: float, psi_deg: float) -> None:
    atoms.set_constraint(
        FixInternals(dihedrals_deg=[
            (float(phi_deg), tuple(int(i) for i in phi_atoms)),
            (float(psi_deg), tuple(int(i) for i in psi_atoms)),
        ])
    )


def raster_and_relax(atoms, phi_atoms, psi_atoms, calc, nphi, npsi, phi_range, psi_range,
                      fmax, max_steps, n_shards, shard_index):
    """Raster (phi, psi) and relax other DOF. Returns (relaxed_atoms, PHI, PSI, E); E is NaN where
    this shard skipped or relaxation failed."""
    phi_vals = np.linspace(phi_range[0], phi_range[1], nphi)
    psi_vals = np.linspace(psi_range[0], psi_range[1], npsi)
    total = nphi * npsi
    my_flat_indices = np.arange(shard_index, total, n_shards)
    neighbors = build_bond_graph(atoms)

    PHI, PSI = np.meshgrid(phi_vals, psi_vals, indexing="xy")
    E = np.full((npsi, nphi), np.nan)

    relaxed = []
    n_failed = 0
    for flat in tqdm(my_flat_indices, desc="raster+relax"):
        jpsi, iphi = divmod(int(flat), nphi)
        phi = float(phi_vals[iphi])
        psi = float(psi_vals[jpsi])

        a = atoms.copy()
        a.calc = calc
        set_dihedral_rotate_k_side(a, *phi_atoms, target_deg=phi, neighbors=neighbors)
        set_dihedral_rotate_k_side(a, *psi_atoms, target_deg=psi, neighbors=neighbors)
        constrain_phi_psi(a, phi_atoms, psi_atoms, phi_deg=phi, psi_deg=psi)

        opt = BFGS(a, logfile=None)
        try:
            opt.run(fmax=fmax, steps=max_steps)
            energy = float(a.get_potential_energy())
            forces = a.get_forces().copy()
        except Exception as exc:    # noqa: BLE001 -- relaxation can fail in several ASE/MACE-specific ways; skip and keep going
            print(f"  [skip] phi={phi:.1f} psi={psi:.1f}: relaxation failed ({exc})")
            n_failed += 1
            continue

        E[jpsi, iphi] = energy
        a.set_constraint()    # drop FixInternals before writing -- test_set.xyz shouldn't carry it
        a.info["mace_energy"] = energy
        a.arrays["mace_forces"] = forces
        a.info["config_type"] = "ramachandran_raster"
        relaxed.append(a)

    print(f"[raster] shard {shard_index}/{n_shards}: {len(relaxed)} relaxed, {n_failed} failed")
    return relaxed, PHI, PSI, E


def generate_rattled(relaxed, calc, n_rattled: int, stdev_min: float, stdev_max: float, seed: int):
    """Rattle random relaxed structures (stdev ~ U[stdev_min, stdev_max]) and single-point label them."""
    if not relaxed:
        print("[rattle] no relaxed structures available to rattle from, skipping")
        return []

    rng = np.random.default_rng(seed)
    rattled = []
    for k in tqdm(range(n_rattled), desc="rattle"):
        base = relaxed[int(rng.integers(len(relaxed)))]
        stdev = float(rng.uniform(stdev_min, stdev_max))
        a = base.copy()
        a.calc = calc
        a.rattle(stdev=stdev, seed=int(rng.integers(2**31 - 1)))
        energy = float(a.get_potential_energy())
        forces = a.get_forces().copy()
        a.info["mace_energy"] = energy
        a.arrays["mace_forces"] = forces
        a.info["config_type"] = "rattled"
        a.info["rattle_stdev"] = stdev
        rattled.append(a)

    print(f"[rattle] generated {len(rattled)} rattled structures "
          f"(stdev range [{stdev_min}, {stdev_max}] A)")
    return rattled


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.n_shards):
        raise ValueError(f"--shard_index must be in [0, {args.n_shards})")

    atoms = read(args.init_xyz)
    print(f"Seed structure: {args.init_xyz} ({len(atoms)} atoms)")
    print(f"phi_atoms={PHI_ATOMS}  psi_atoms={PSI_ATOMS}")

    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=args.oracle_model, device=args.device, default_dtype="float64")

    relaxed, PHI, PSI, E = raster_and_relax(
        atoms, PHI_ATOMS, PSI_ATOMS, calc,
        nphi=args.nphi, npsi=args.npsi,
        phi_range=tuple(args.phi_range), psi_range=tuple(args.psi_range),
        fmax=args.fmax, max_steps=args.max_steps,
        n_shards=args.n_shards, shard_index=args.shard_index,
    )

    rattled = []
    if args.n_rattled > 0:
        rattled = generate_rattled(
            relaxed, calc, n_rattled=args.n_rattled,
            stdev_min=args.rattle_stdev_min, stdev_max=args.rattle_stdev_max,
            seed=args.rattle_seed,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.n_shards > 1:
        output = output.with_name(f"{output.stem}_shard{args.shard_index}{output.suffix}")
        print(f"--n_shards > 1: writing this shard to {output} -- concatenate shards yourself once all are done")
    all_structures = relaxed + rattled
    write(str(output), all_structures, format="extxyz")
    print(f"Wrote {len(all_structures)} labelled structures -> {output} "
          f"({len(relaxed)} raster + {len(rattled)} rattled)")

    pes_base = Path(args.pes_out) if args.pes_out else output.with_suffix("")
    np.savez(f"{pes_base}_pes.npz", PHI=PHI, PSI=PSI, E=E,
             phi_atoms=np.array(PHI_ATOMS, dtype=int), psi_atoms=np.array(PSI_ATOMS, dtype=int))
    plot_contour_phi_psi(PHI, PSI, E, levels=50, title="Test-set raster: constrained-relaxed E(phi, psi)",
                          out_png=f"{pes_base}_pes.png")
    print(f"Wrote diagnostic PES -> {pes_base}_pes.npz / _pes.png")


if __name__ == "__main__":
    main()
