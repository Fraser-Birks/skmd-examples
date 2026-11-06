"""
skmd.lammps_setup — Generic LAMMPS initialisation for SKMD runs.

These functions set up LAMMPS with the symmetrix/mace pair style and
register the skmd fix. All system-specific quantities (element list,
data file, model path) are passed as arguments.
"""

from __future__ import annotations
import numpy as np
import lammps as lmpmod
from ase.io.lammpsdata import write_lammps_data


def make_lammps(kokkos: bool = False, kokkos_gpus: int = 1):
    """Construct a lammps instance, optionally with Kokkos/GPU (`-sf kk` picks /kk styles where they exist).

    Args:
        kokkos:      Enable Kokkos via `-k on g <kokkos_gpus> -sf kk -pk kokkos ...`.
        kokkos_gpus: GPU count passed to `-k on g` (one rank per GPU).
    """
    if not kokkos:
        return lmpmod.lammps()

    args = [
        "-k", "on", "g", f"{kokkos_gpus}",   # Kokkos GPU usage
        "-sf", "kk",                          # Use Kokkos style
        "-pk", "kokkos", "newton", "on", "neigh", "half",
    ]
    return lmpmod.lammps(cmdargs=args)


def setup_lammps(
    lmp,
    data_file: str,
    elements: list[str],
    model_json: str,
    plugin_so: str,
    comm_cutoff: float = 8.0,
    boundary: str = "f f f",
) -> None:
    """Initialise LAMMPS with the symmetrix/mace pair style.

    Args:
        lmp:         Active lammps.lammps instance.
        data_file:   Path to LAMMPS .data file.
        elements:    Ordered list of element symbols matching the data file types.
        model_json:  Path to the MACE model JSON file.
        plugin_so:   Path to fixskmdplugin.so.
        comm_cutoff: Ghost-atom communication cutoff in Angstrom.
        boundary:    LAMMPS `boundary` string: "f f f" for an isolated molecule,
                     "p p f" for a slab periodic in x/y.
    """
    elem_str = " ".join(elements)
    mass_map = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
                "Si": 28.086, "Fe": 55.845, "Al": 26.982}

    if plugin_so is not None:
        lmp.cmd(f"plugin load {plugin_so}")

    cmds = [
        "units           metal",
        "atom_style      atomic",
        f"boundary        {boundary}",
        "atom_modify     map yes",
        f"read_data       {data_file}",
    ]
    for i, el in enumerate(elements, start=1):
        mass = mass_map.get(el, 1.0)
        cmds.append(f"mass {i} {mass}")

    cmds += [
        f"comm_modify cutoff {comm_cutoff}",
        "pair_style symmetrix/mace no_domain_decomposition",
        f"pair_coeff    * * {model_json} {elem_str}",
        f"compute macedesc all symmetrix/mace/atom {model_json} {elem_str}",
        f"compute macedescgrad all symmetrix/maced/atom {model_json} {elem_str}",
        "compute peat all pe/atom",
    ]
    for c in cmds:
        lmp.command(c)


def setup_skmd_fix(
    lmp,
    compute_q_id: str,
    compute_jac_id: str,
    dataset_base: str,
    a_value: float,
    b_spec: str,
    thresh: float | None = None,
    b_max: float | None = None,
    b_min: float | None = None,
    buffer: int | None = None,
    beta: float | None = None,
) -> None:
    """Register the skmd fix with LAMMPS.

    Args:
        lmp:            Active lammps.lammps instance.
        compute_q_id:   ID of the descriptor compute (e.g. 'macedesc').
        compute_jac_id: ID of the Jacobian compute (e.g. 'macedescgrad').
        dataset_base:   Base path for .skdat dataset files.
        a_value:        Kernel amplitude.
        b_spec:         Kernel width — a float string or 'adaptive'.
        thresh:         Force-norm threshold (optional).
        b_max:          Upper bound on b when b_spec='adaptive' (optional).
        b_min:          Lower bound on b when b_spec='adaptive' (optional).
        buffer:         Min steps into the run before 'thresh' can fire (optional).
        beta:           Weight of the attractive term (optional; default 1.0 as in the paper).
    """
    cmd = (
        f"fix skmd all skmd {compute_q_id} {compute_jac_id} {dataset_base} "
        f"{a_value} {b_spec} "
        f"{f'thresh {thresh} ' if thresh is not None else ''}"
        f"{f'b_max {b_max} ' if b_max is not None else ''}"
        f"{f'b_min {b_min} ' if b_min is not None else ''}"
        f"{f'buffer {buffer} ' if buffer is not None else ''}"
        f"{f'beta {beta} ' if beta is not None else ''}"
    ).strip()
    lmp.command(cmd)


def dump_configs_md(
    lmp,
    nsteps: int,
    every: int,
    dumpfile: str,
    T: float,
    damping: float,
    nburnin: int = 4000,
    desc_compute_id: str = "macedesc",
    dyn_type: str = "overdamped",
    overdamped_timestep: float = 1e-4,
    underdamped_timestep: float = 0.001,
    dyn_group: str = "all",
    seed: int = 123456,
) -> None:
    """Run energy minimisation + Brownian/Langevin MD and dump configurations.

    Args:
        lmp:             Active lammps.lammps instance.
        nsteps:          Number of MD steps AFTER nburnin, with dumping active.
        every:           Dump frequency (steps).
        dumpfile:        Path for the dump file.
        T:               Brownian temperature (metal units).
        damping:         Brownian gamma_t damping coefficient.
        seed:            Thermostat RNG seed for the burn-in.
        nburnin:         Burn-in steps run before dumping starts (no output).
        desc_compute_id: Compute ID for descriptor (for dump output).
        dyn_type:        'overdamped' or 'underdamped'.
        overdamped_timestep:  Timestep (ps) used for Brownian dynamics.
        underdamped_timestep: Timestep (ps) used for Langevin+NVE dynamics.
        dyn_group:       Group the thermostat integrates; exclude frozen atoms
                         (`setforce 0` does not stop thermal kicks).
    """
    lmp.command("thermo 100")
    # CG rather than FIRE: MinFireKokkos crashes under Kokkos.
    lmp.command("min_style cg")
    lmp.command("min_modify line quadratic")
    lmp.command("minimize 0.0 1.0e-5 10000 100000")

    if dyn_type == "overdamped":
        lmp.command(f"timestep {overdamped_timestep}")
        lmp.command(f"fix dyn {dyn_group} brownian {T} {seed} gamma_t {damping}")
    else:
        lmp.command(f"timestep {underdamped_timestep}")
        lmp.command(f"fix nvefix {dyn_group} nve")
        lmp.command(f"fix dyn {dyn_group} langevin {T} {T} {damping} {seed} zero yes")

    lmp.command(f"run {nburnin}")
    lmp.command(
        f"dump init_dump all custom {every} {dumpfile} "
        f"id type mass x y z fx fy fz c_{desc_compute_id}[*]"
    )
    # Sort by ID so atom identities survive the dump -> ASE -> data-file round trip.
    lmp.command("dump_modify init_dump sort id")
    lmp.command(f"run {nsteps}")

    lmp.command("undump init_dump")
    lmp.command("unfix dyn")
    if dyn_type == "underdamped":
        lmp.command("unfix nvefix")


def load_config_into_lammps(lmp, atoms, tmpdir: str, elements: list[str]) -> None:
    """Write an ASE Atoms object to a temp file and reload it in LAMMPS.

    Args:
        lmp:      Active lammps.lammps instance.
        atoms:    ASE Atoms object to load.
        tmpdir:   Directory to write the temporary data file.
        elements: Element order matching LAMMPS type numbering.
    """
    import os
    tmp_path = os.path.join(tmpdir, "temp.data")
    write_lammps_data(tmp_path, atoms, velocities=True,
                      atom_style="atomic", specorder=elements)
    lmp.command("delete_atoms group all")
    lmp.command(f"read_data {tmp_path} add append")
    lmp.command("run 0")


def compute_q_jac_f(
    lmp,
    desc_compute_id: str = "macedesc",
    desc_grad_compute_id: str = "macedescgrad",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (q, jac, f) from the current LAMMPS state.

    Returns:
        q:   Global descriptor, shape (ncoeff,) — sum over atoms.
        jac: Jacobian, shape (3N, ncoeff).
        f:   Forces, shape (N, 3).

    id/f are truncated to nlocal (drops ghosts); raises if the computes are not nlocal-sized.
    """
    lmp.command("run 1")
    desc = np.array(
        lmp.numpy.extract_compute(desc_compute_id, lmpmod.LMP_STYLE_ATOM, lmpmod.LMP_TYPE_ARRAY),
        copy=True,
    )
    desc_grad = np.array(
        lmp.numpy.extract_compute(desc_grad_compute_id, lmpmod.LMP_STYLE_ATOM, lmpmod.LMP_TYPE_ARRAY),
        copy=True,
    )
    f   = np.array(lmp.numpy.extract_atom("f"),  copy=True)
    ids = np.array(lmp.numpy.extract_atom("id"), copy=True)

    nlocal = lmp.extract_global("nlocal")
    ids = ids[:nlocal]
    f = f[:nlocal]

    if desc.shape[0] != nlocal:
        raise RuntimeError(
            f"{desc_compute_id} returned {desc.shape[0]} rows, expected "
            f"nlocal={nlocal}."
        )
    if desc_grad.shape[0] != nlocal:
        raise RuntimeError(
            f"{desc_grad_compute_id} returned {desc_grad.shape[0]} rows, "
            f"expected nlocal={nlocal} (got {desc_grad.shape[0]}, likely "
            "nlocal+nghost). This compute is typically flagged 'ghost' in "
            "LAMMPS's neighbor-list info — its ghost-atom rows may need to "
            "be summed back onto their originating local atom rather than "
            "truncated. Check the symmetrix/skmd-lammps source for the "
            "intended reduction before changing this."
        )

    order = np.argsort(ids)
    desc      = desc[order]
    desc_grad = desc_grad[order]
    f         = f[order]

    return np.sum(desc, axis=0), desc_grad, f
