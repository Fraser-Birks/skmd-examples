"""SKMD on a Si slab with MACE descriptors via symmetrix: frozen substrate (--n_frozen), "p p f" boundary.

Optional a-SKMD online acquisition (Eq. 8) via --thresh/--target_samples, with zeta_0 auto-derived after exploration.
"""

import csv
import os
import pickle
import sys
import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from skmd.dataset import write_skdat
from skmd.loop import get_regime_params, median_pairwise_distance, select_b
from skmd.lammps_setup import (
    make_lammps,
    setup_lammps,
    setup_skmd_fix,
    dump_configs_md,
    load_config_into_lammps,
    compute_q_jac_f,
)
from skmd.visualise import pca_scatter


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SKMD loop on a Si slab with MACE descriptors."
    )
    p.add_argument("--lammps_data",  required=True, help="LAMMPS .data file for the Si slab")
    p.add_argument("--init_xyz",     required=True, help="Initial configuration .xyz file")
    p.add_argument("--model_json",   required=True, help="Path to MACE model .json file")
    p.add_argument("--plugin_so",    default=None, help="fix skmd plugin .so, if built as a LAMMPS plugin")
    p.add_argument("--kokkos", action="store_true",
                    help="Run LAMMPS with Kokkos/GPU (-k on g N -sf kk -pk kokkos)")
    p.add_argument("--kokkos_gpus", type=int, default=1,
                    help="GPU count for -k on g <N> (with --kokkos)")
    p.add_argument("--elements",     nargs="+", default=["Si"],
                   help="Element symbols in LAMMPS type order")
    p.add_argument("--n_frozen",     type=int, default=0,
                   help="Number of frozen substrate atoms (the first N_FROZEN; see prepare_lammps_data.py)")
    p.add_argument("--fname",        default="skmd_run", help="Output directory prefix")
    p.add_argument("--nsteps",       type=int,   default=300,  help="Max outer loop iterations")
    p.add_argument("--T",            type=float, default=50.0, help="Brownian temperature (K)")
    p.add_argument("--damping",      type=float, default=75.0, help="Brownian damping coefficient")
    p.add_argument("--a",            type=float, default=1.0,  help="Exploration kernel amplitude")
    p.add_argument("--b",            type=float, default=1.0,  help="Reference bandwidth b")
    p.add_argument("--b_mode",
                   choices=["fixed_user", "fixed_burnin", "adaptive", "adaptive-median"],
                   default="adaptive",
                   help="Exploration bandwidth: --b ('fixed_user'), burn-in median ('fixed_burnin'), "
                        "or per-iteration median clamped to [0.2*b0, b_max_mult*b0] ('adaptive'/'adaptive-median')")
    p.add_argument("--b_max_mult", type=float, default=3.0,
                   help="Upper clamp on adaptive b, as a multiple of burn-in b0")
    p.add_argument("--thresh_percentile", type=float, default=10.0,
                   help="Percentile of the raw (whole-slab) SKMD force norm used to auto-derive "
                        "zeta_0 when --thresh is unset in thresh mode (Eq. 8)")
    p.add_argument("--thresh_calib_window", type=int, default=20,
                   help="Trailing outer iterations used for --thresh_percentile and frozen b")
    p.add_argument("--target_samples", type=int, default=None,
                   help="Stop after this many online acquisitions; enables thresh mode (a-SKMD)")
    p.add_argument("--a_relax_mult", type=float, default=0.5,  help="Relaxation a multiplier")
    p.add_argument("--b_relax_mult", type=float, default=0.5,  help="Relaxation b multiplier")
    p.add_argument("--relax_start",  type=int,   default=100,
                   help="Iteration to begin relaxation (hard cap on exploration)")
    p.add_argument("--b_stability_tol", type=float, default=None,
                   help="End exploration early once adaptive b's max relative change over "
                        "--b_stability_window iterations is below this fraction")
    p.add_argument("--b_stability_window", type=int, default=10,
                   help="Iterations checked for b stability")
    p.add_argument("--thresh",       type=float, default=None, help="Force-norm threshold")
    p.add_argument("--buffer",       type=int,   default=50,
                   help="Minimum MD steps per outer iteration before --thresh may stop it")
    p.add_argument("--n_run",        type=int,   default=500,  help="MD steps per outer iteration")
    p.add_argument("--n_silent_burnin", type=int, default=4000,
                   help="Burn-in steps before dumping starts")
    p.add_argument("--n_burnin_md",  type=int,   default=6000,
                   help="Burn-in steps after --n_silent_burnin, dumped every --every_burnin")
    p.add_argument("--every_burnin", type=int,   default=200,  help="Dump frequency during burn-in")
    p.add_argument("--dyn_type", type=str, default="overdamped", help="underdamped or overdamped dynamics")
    p.add_argument("--initial_particle_positions", type=str, default=None, help="Initial particle positions (None for run MD)")
    p.add_argument("--seed", type=int, default=123456,
                   help="Thermostat RNG seed")
    return p.parse_args()


# MD timesteps (ps).
OVERDAMPED_TIMESTEP = 1e-4
UNDERDAMPED_TIMESTEP = 0.001

def apply_freeze(lmp, n_frozen: int) -> None:
    """Define frozen/mobile groups and zero frozen forces; call after every load_config_into_lammps.

    Thermostats must act on "mobile" only, since their noise bypasses setforce."""
    if n_frozen <= 0:
        return
    lmp.command("group frozen id " + f"1:{n_frozen}")
    lmp.command("group mobile subtract all frozen")
    lmp.command("fix freeze frozen setforce 0.0 0.0 0.0")


def _ensure_mace_descriptor_info(traj, desc_compute_id: str = "macedesc") -> None:
    """Sum per-atom c_macedesc[*] dump arrays into atoms.info['mace_descriptor'] (for plots)."""
    for atoms in traj:
        if "mace_descriptor" in atoms.info:
            continue
        dims = sorted(
            int(k[len("c_macedesc["):-1])
            for k in atoms.arrays
            if k.startswith("c_macedesc[") and k.endswith("]")
        )
        if not dims:
            continue
        desc = [np.sum(atoms.arrays[f"c_macedesc[{d}]"]) for d in dims]
        atoms.info["mace_descriptor"] = desc


_shared_scaler = None
_shared_pca = None


def plot_traj(traj, fname):
    """Diagnostic PCA scatter; the first call fits the basis, later calls reuse it for comparable axes."""
    global _shared_scaler, _shared_pca
    _ensure_mace_descriptor_info(traj)
    if not traj or "mace_descriptor" not in traj[0].info:
        print(f"  [skip plot] no mace_descriptor info available for {fname}")
        return
    X_pca, pca_out, scaler_out = pca_scatter(
        [traj], out_png=fname, scaler=_shared_scaler, pca=_shared_pca
    )
    if X_pca is not None:
        print(f"Wrote: {fname}")
        if _shared_pca is None:
            _shared_scaler, _shared_pca = scaler_out, pca_out


def main():
    args = parse_args()

    if args.relax_start < 0:
        raise ValueError("--relax_start must be >= 0")
    if args.relax_start > args.nsteps:
        raise ValueError(f"--relax_start={args.relax_start} exceeds --nsteps={args.nsteps}")
    if args.dyn_type not in ["overdamped", "underdamped"]:
        raise ValueError(f"--dyn_type must be 'overdamped' or 'underdamped'")

    thresh_mode_active = (args.thresh is not None) or (args.target_samples is not None)

    os.makedirs(args.fname, exist_ok=True)
    datapath = os.path.join(args.fname, "traj_data")
    os.makedirs(datapath, exist_ok=True)

    dataset_base   = os.path.join(datapath, "dataset")
    compute_q_id   = "macedesc"
    compute_jac_id = "macedescgrad"

    atoms0 = read(args.init_xyz)
    lmp    = make_lammps(kokkos=args.kokkos, kokkos_gpus=args.kokkos_gpus)

    print("Setting up LAMMPS...")
    # Periodic in-plane, non-periodic along the vacuum direction z.
    setup_lammps(lmp, args.lammps_data, args.elements, args.model_json, args.plugin_so,
                 boundary="p p f")
    apply_freeze(lmp, args.n_frozen)
    # Thermostat only mobile atoms ("mobile" exists only if n_frozen > 0).
    dyn_group = "mobile" if args.n_frozen > 0 else "all"

    if args.initial_particle_positions is None:
        print("Running initial burn-in MD...")
        burn_dump = os.path.join(datapath, "dump.lammpstrj")
        dump_configs_md(lmp, nsteps=args.n_burnin_md, every=args.every_burnin,
                        dumpfile=burn_dump, T=args.T, damping=args.damping,
                        nburnin=args.n_silent_burnin, dyn_type=args.dyn_type,
                        overdamped_timestep=OVERDAMPED_TIMESTEP,
                        underdamped_timestep=UNDERDAMPED_TIMESTEP,
                        dyn_group=dyn_group, seed=args.seed)
    else:
        burn_dump = args.initial_particle_positions

    qs, fs, jacs, configs = [], [], [], []
    print("Computing initial descriptors...")
    dumped = read(burn_dump, index=":")
    plot_traj(dumped, os.path.join(args.fname, "traj_initial.png"))
    # Save the PCA basis for draw_samples.py.
    if _shared_pca is not None:
        with open(os.path.join(datapath, "pca_basis.pkl"), "wb") as f:
            pickle.dump((_shared_scaler, _shared_pca), f)

    for atoms in dumped:
        load_config_into_lammps(lmp, atoms, datapath, args.elements)
        apply_freeze(lmp, args.n_frozen)
        q, jac, f = compute_q_jac_f(lmp)
        qs.append(q); jacs.append(jac); fs.append(f); configs.append(atoms)

    write_skdat(dataset_base, qs, fs, jacs)

    b0 = median_pairwise_distance(qs)
    if not (b0 > 0.0):
        print(f"  [b0] degenerate burn-in ensemble (b0={b0}) — falling back to --b={args.b}")
        b0 = args.b
    print(f"Burn-in baseline b0={b0:.6f}  (--b_mode={args.b_mode})")

    import lammps as lmpmod
    a_max        = args.a
    b_ref        = args.b
    frozen_b0    = None
    a_relax_mult_eff = args.a_relax_mult
    b_relax_mult_eff = args.b_relax_mult

    # Exploration end: --relax_start, possibly earlier via the b-stability check.
    relax_start_effective = args.relax_start
    b_history = []

    # Exploration b history; frozen b is its trailing-window mean.
    b_adaptive_history = []

    # Per-atom RMS SKMD force (diagnostic, logged).
    force_history = []

    # Raw whole-slab ||F^SKMD|| for zeta_0: fix_skmd compares against this scale (Eq. 8), not per-atom.
    raw_force_history = []

    selected_data = []

    print("Starting main SKMD loop...")
    for it in range(args.nsteps):
        write(os.path.join(datapath, f"configs_before_it_{it}.xyz"), configs, format="extxyz")

        qs.pop(0); fs.pop(0); jacs.pop(0)
        new_atoms = configs.pop(0)
        write_skdat(dataset_base, qs, fs, jacs)
        load_config_into_lammps(lmp, new_atoms, datapath, args.elements)
        apply_freeze(lmp, args.n_frozen)

        b_max = b_min = None
        if it < relax_start_effective:
            regime, relax_progress = "exploration", 0.0
            a_curr = a_max
            b_spec, b_max, b_min = select_b(args.b_mode, args.b, b0, args.b_max_mult)
        else:
            if frozen_b0 is None:
                # Trailing-window mean of exploration b; burn-in b0 if no exploration ran.
                if b_adaptive_history:
                    window = b_adaptive_history[-args.thresh_calib_window:]
                    frozen_b0 = float(np.mean(window))
                    print(f"Freezing relaxation b0 to {frozen_b0:.6f} "
                          f"(average over the last {len(window)} "
                          f"exploration iterations)")
                else:
                    frozen_b0 = b0 if b0 > 0.0 else b_ref
                    print(f"Freezing relaxation b0 to {frozen_b0:.6f}")

                if thresh_mode_active:
                    # a-SKMD (Eq. 8): no a/b decay after freezing.
                    if args.a_relax_mult != 1.0 or args.b_relax_mult != 1.0:
                        print(f"[thresh mode] overriding a_relax_mult/"
                              f"b_relax_mult to 1.0 (no decay once online "
                              f"acquisition is active; CLI values "
                              f"{args.a_relax_mult}/{args.b_relax_mult} ignored)")
                    a_relax_mult_eff = 1.0
                    b_relax_mult_eff = 1.0

                # Derive zeta_0 once, only in thresh mode and if --thresh was not given.
                if thresh_mode_active and args.thresh is None:
                    window = raw_force_history[-args.thresh_calib_window:]
                    args.thresh = float(np.percentile(window, args.thresh_percentile))
                    print(f"Auto-derived thresh (zeta_0) = {args.thresh:.6f} "
                          f"({args.thresh_percentile}th percentile over the "
                          f"last {len(window)} exploration iterations' "
                          f"raw SKMD force norm)")

            regime, a_curr, b_spec, relax_progress = get_regime_params(
                it=it, n_outer=args.nsteps, relax_start=relax_start_effective,
                a_max=a_max, a_relax_mult=a_relax_mult_eff,
                frozen_b0=frozen_b0, b_relax_mult=b_relax_mult_eff,
            )

        # Thresholding needs a fixed bandwidth: ||F^SKMD|| scales ~1/b under adaptive b (paper Section 6).
        effective_thresh = args.thresh if b_spec != "adaptive" else None

        print(f"[it={it:4d}] regime={regime:11s} a={a_curr:.6f} b={b_spec} "
              f"b_max={b_max} b_min={b_min} relax_progress={relax_progress:.3f} "
              f"thresh_active={effective_thresh is not None}")

        setup_skmd_fix(lmp, compute_q_id, compute_jac_id, dataset_base,
                       a_value=a_curr, b_spec=b_spec, thresh=effective_thresh,
                       b_max=b_max, b_min=b_min, buffer=args.buffer)

        if args.dyn_type == "overdamped":
            lmp.command("timestep 1e-4")
            lmp.command(f"fix dyn {dyn_group} brownian {args.T} {args.seed} gamma_t {args.damping}")
        else:
            lmp.command("timestep 0.001")
            lmp.command(f"fix nvefix {dyn_group} nve")
            lmp.command(f"fix dyn {dyn_group} langevin {args.T} {args.T} {args.damping} {args.seed} zero yes")

        lmp.command("fix fstore all store/force")
        lmp.command("thermo 100")
        lmp.command(
            f"dump dumpy all custom 20 {os.path.join(datapath, str(it))}.lammpstrj "
            "id type mass x y z vx vy vz f_fstore[*] c_peat c_macedesc[*]"
        )
        # Sort by ID so LAMMPS atom reordering cannot scramble atom identities.
        lmp.command("dump_modify dumpy sort id")
        step_before = lmp.get_thermo("step")
        lmp.command(f"run {args.n_run}")
        n_steps_run = int(lmp.get_thermo("step") - step_before)

        skmd_fnorm    = lmp.extract_fix("skmd", lmpmod.LMP_STYLE_GLOBAL, lmpmod.LMP_TYPE_SCALAR)
        current_b_val = lmp.extract_fix("skmd", lmpmod.LMP_STYLE_GLOBAL, lmpmod.LMP_TYPE_VECTOR, 1)
        triggered     = bool(lmp.extract_fix("skmd", lmpmod.LMP_STYLE_GLOBAL, lmpmod.LMP_TYPE_VECTOR, 2))
        # Mean fnorm2 over the run (less noisy than the end-of-run skmd_fnorm).
        mean_fnorm    = lmp.extract_fix("skmd", lmpmod.LMP_STYLE_GLOBAL, lmpmod.LMP_TYPE_VECTOR, 3)

        n_atoms   = lmp.get_natoms()
        rms_force = mean_fnorm / np.sqrt(n_atoms) if n_atoms > 0 else mean_fnorm
        force_history.append(rms_force)
        raw_force_history.append(mean_fnorm)

        if regime == "exploration":
            b_adaptive_history.append(current_b_val)

            if b_spec == "adaptive" and args.b_stability_tol is not None:
                b_history.append(current_b_val)
                window = args.b_stability_window
                # Only check until relaxation start has been pulled earlier.
                if relax_start_effective == args.relax_start and len(b_history) > window:
                    recent = b_history[-(window + 1):]
                    rel_changes = [abs(recent[i] - recent[i-1]) / recent[i-1]
                                   for i in range(1, len(recent))]
                    max_rel_change = max(rel_changes)
                    if max_rel_change < args.b_stability_tol:
                        relax_start_effective = it + 1
                        print(f"[it={it:4d}] adaptive b stabilized (max relative "
                              f"change {max_rel_change:.4f} < {args.b_stability_tol} "
                              f"over last {window} iterations) — switching to "
                              f"relaxation at it={relax_start_effective}")

        lmp.command("unfix skmd")
        lmp.command("run 0")
        lmp.command(
            f"write_dump all custom {os.path.join(datapath, 'output.dump')} "
            "id type mass x y z vx vy vz f_fstore[*] c_peat c_macedesc[*] "
            "modify sort id"
        )
        new_cfg = read(os.path.join(datapath, "output.dump"))

        print(f"  skmd_fnorm={skmd_fnorm:.6f}  mean_fnorm={mean_fnorm:.6f}  "
              f"rms_force_per_atom={rms_force:.6f}  b={current_b_val:.6f}  "
              f"triggered={triggered}  n_steps_run={n_steps_run}")

        stopping_log_path = os.path.join(args.fname, "stopping_log.csv")
        stopping_log_is_new = not os.path.exists(stopping_log_path)
        with open(stopping_log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if stopping_log_is_new:
                writer.writerow(["it", "regime", "triggered", "n_steps_run",
                                  "final_fnorm", "mean_fnorm",
                                  "rms_force_per_atom", "a", "b", "thresh"])
            writer.writerow([it, regime, triggered, n_steps_run,
                              skmd_fnorm, mean_fnorm, rms_force, a_curr,
                              current_b_val, effective_thresh])

        lmp.command("undump dumpy")
        lmp.command("unfix dyn")
        if args.dyn_type == "underdamped":
            lmp.command("unfix nvefix")

        q, jac, f = compute_q_jac_f(lmp)
        qs.append(q); jacs.append(jac); fs.append(f); configs.append(new_cfg)

        plot_traj(configs, os.path.join(datapath, f"{it}.png"))

        # Offline: collect every config (FPS later). Thresh mode: only triggered configs during relaxation.
        if (not thresh_mode_active) or (regime == "relaxation" and triggered):
            lmp.command(
                f"write_dump all custom {args.fname}/selected_data.dump "
                "id type mass x y z f_fstore[*] c_peat c_macedesc[*] "
                "modify append yes sort id"
            )
            selected_data.append(new_cfg)
            plot_traj(selected_data, f"{args.fname}/selected_data.png")
            # Rewrite the xyz each time so it survives the teardown crash.
            write(f"{args.fname}/selected_data.xyz", selected_data, format="extxyz")

        lmp.command("unfix fstore")

        if args.target_samples is not None and len(selected_data) >= args.target_samples:
            print(f"[it={it:4d}] reached --target_samples={args.target_samples} "
                  f"({len(selected_data)} collected) -- stopping.")
            break

    lmp.close()
    print("SKMD loop complete.")


if __name__ == "__main__":
    main()
