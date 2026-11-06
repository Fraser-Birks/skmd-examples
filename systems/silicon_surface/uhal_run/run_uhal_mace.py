"""uHAL (ase_uhal) exploration on a Si slab: N burn-in walkers advanced round-robin under one shared committee.

Same nsteps*n_run budget as run_skmd_mace.py; frozen substrate via FixAtoms. Online mode selects on HAL score;
--offline folds every endpoint into the committee and picks --n_select by FPS. Output: diverse_subset.xyz.
"""

import csv
import os
import sys
import time
import argparse
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.units import fs
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from common import load_mace_calculator, resolve_n_frozen
from skmd.visualise import pca_scatter

from ase_uhal.bias_calculators import HALBiasCalculator
from ase_uhal.committee_calculators import MACEHALCalculator
from ase_uhal import StructureSelector

from skmd.sampling import sample_diverse_structures


def parse_args():
    p = argparse.ArgumentParser(
        description="Run an N-walker, shared-committee UHAL loop on a Si slab with MACE, "
                    "as a comparison driver to run_skmd_mace.py."
    )
    p.add_argument("--lammps_data",  required=True, help="LAMMPS .data file for the Si slab (n_frozen sidecar)")
    p.add_argument("--init_xyz",     required=True, help="Initial configuration .xyz file")
    p.add_argument("--mace_model",   required=True, help="Path to a MACE .model file (foundation or fine-tuned)")
    p.add_argument("--n_frozen",     type=int, default=None,
                   help="Number of frozen substrate atoms (default: --lammps_data.n_frozen sidecar)")
    p.add_argument("--fname",        default="uhal_run", help="Output directory prefix")
    p.add_argument("--device",       default="cuda")

    # Committee
    p.add_argument("--committee_size", type=int, default=10)
    p.add_argument("--prior_weight",   type=float, default=0.5)
    p.add_argument("--energy_weight",  type=float, default=1.0)
    p.add_argument("--forces_weight",  type=float, default=10.0)
    p.add_argument("--lowmem",         action="store_true")
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--seed",           type=int, default=42)

    # Bias calculator (tau) -- shared across all N walkers
    p.add_argument("--tau_rel",    type=float, default=0.03)
    p.add_argument("--tau_hist",   type=float, default=10)
    p.add_argument("--tau_delay",  type=float, default=None)

    # Structure selector -- ignored entirely if --offline is set
    p.add_argument("--offline", action="store_true",
                   help="No online selector: fold every walker endpoint into the committee, FPS at the end")
    p.add_argument("--selector_threshold",  default="adaptive",
                   help="'adaptive' or a fixed float score threshold")
    p.add_argument("--selector_delay",      type=int,   default=10)
    p.add_argument("--selector_mixing",     type=float, default=0.1)
    p.add_argument("--selector_thresh_mul", type=float, default=1.2)
    p.add_argument("--selector_interval",   type=int,   default=2,
                   help="MD steps between HAL score checks (ignored if --offline)")

    # Dynamics
    p.add_argument("--T",            type=float, default=700.0, help="Langevin temperature (K)")
    p.add_argument("--friction",     type=float, default=0.01,  help="Langevin friction (1/fs)")
    p.add_argument("--timestep_fs",  type=float, default=1.0,   help="MD timestep (fs)")

    # Burn-in: unbiased MD giving N = n_burnin_md // every_burnin walkers (as in run_skmd_mace.py).
    p.add_argument("--n_silent_burnin", type=int, default=4000,
                   help="Unbiased MD steps run BEFORE dumping starts (no walkers collected here)")
    p.add_argument("--n_burnin_md",  type=int, default=6000,
                   help="MD steps run AFTER --n_silent_burnin, dumping a walker every --every_burnin steps")
    p.add_argument("--every_burnin", type=int, default=200,
                   help="Dump frequency during burn-in -- N walkers = n_burnin_md // every_burnin")

    # Step budget: each outer iteration advances one walker --n_run steps (total nsteps * n_run).
    p.add_argument("--nsteps",  type=int, default=200, help="Outer iterations (one walker advanced --n_run steps each)")
    p.add_argument("--n_run",   type=int, default=500, help="Steps per outer iteration for the active walker")
    p.add_argument("--n_select", type=int, default=30,
                   help="Online: stop after this many selections (keep top by HAL score); "
                        "--offline: number picked by FPS")

    # Trajectory / diagnostics output
    p.add_argument("--traj_interval", type=int, default=50,
                   help="MD steps between frames in trajectory_walker_<i>.traj")

    return p.parse_args()


def plot_traj(traj, fname, s=6, alpha=0.35):
    """Cheap raw-position PCA scatter (no descriptor evaluation)."""
    if len(traj) < 2:
        print(f"[plot_traj] only {len(traj)} frame(s) so far, skipping PCA plot "
              f"(need >= 2) -- {fname} not written yet")
        return
    X = np.array([a.get_positions().flatten() for a in traj])
    X = X - X.mean(axis=0)
    X_pca = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=s, alpha=alpha)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    print(f"Wrote: {fname}")


def attach_progress_printer(dyn, total_steps, label, interval=100):
    """Attach a 'step N/total' progress printer to dyn; returns the observer for detach()."""
    start_time = time.time()
    start_step = dyn.get_number_of_steps()

    def _progress():
        step = dyn.get_number_of_steps() - start_step
        elapsed = time.time() - start_time
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = total_steps - step
        eta_s = remaining / rate if rate > 0 else float("inf")
        eta_str = str(timedelta(seconds=int(eta_s))) if eta_s != float("inf") else "?"
        print(f"[UHAL] {label}: step {step}/{total_steps}  "
              f"({elapsed:.1f}s elapsed, {rate:.1f} steps/s, ETA {eta_str})", flush=True)

    dyn.attach(_progress, interval=interval)
    return _progress


def attach_global_progress_printer(dyn, total_steps, label, global_offset, start_time, interval=100):
    """Progress printer for the round-robin loop: steps counted from global_offset, timed from start_time."""
    start_step = dyn.get_number_of_steps()

    def _progress():
        step = global_offset + (dyn.get_number_of_steps() - start_step)
        elapsed = time.time() - start_time
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = total_steps - step
        eta_s = remaining / rate if rate > 0 else float("inf")
        eta_str = str(timedelta(seconds=int(eta_s))) if eta_s != float("inf") else "?"
        print(f"[UHAL] {label}: step {step}/{total_steps}  "
              f"({elapsed:.1f}s elapsed, {rate:.1f} steps/s, ETA {eta_str})", flush=True)

    dyn.attach(_progress, interval=interval)


def detach(dyn, observer_fn):
    dyn.observers = [o for o in dyn.observers if o[0] is not observer_fn]


def report_gpu_peak_memory(device: str) -> None:
    """Print the process's peak torch CUDA memory."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"[UHAL] peak GPU memory (torch, exact): allocated={allocated:.1f} MiB  reserved={reserved:.1f} MiB")


def run_burnin(atoms0, mace_calc, n_frozen, args):
    """Unbiased ASE burn-in returning N = n_burnin_md // every_burnin walkers (FixAtoms carried by copies)."""
    atoms = atoms0.copy()
    if n_frozen > 0:
        atoms.set_constraint(FixAtoms(indices=list(range(n_frozen))))
    atoms.calc = mace_calc
    dyn = Langevin(atoms, args.timestep_fs * fs, temperature_K=args.T,
                    friction=args.friction / fs)

    if args.n_silent_burnin > 0:
        print(f"Running {args.n_silent_burnin} silent burn-in steps...")
        silent_progress = attach_progress_printer(dyn, args.n_silent_burnin, "silent burn-in")
        dyn.run(args.n_silent_burnin)
        detach(dyn, silent_progress)

    walkers = []

    def _dump():
        walkers.append(atoms.copy())

    print(f"Running {args.n_burnin_md} burn-in steps, dumping a walker every "
          f"{args.every_burnin} steps...")
    attach_progress_printer(dyn, args.n_burnin_md, "burn-in dump")
    dyn.attach(_dump, interval=args.every_burnin)
    dyn.run(args.n_burnin_md)

    if not walkers:
        raise RuntimeError(
            f"Burn-in produced zero walkers (n_burnin_md={args.n_burnin_md}, "
            f"every_burnin={args.every_burnin}) -- every_burnin must be <= n_burnin_md."
        )
    for i, w in enumerate(walkers):
        w.info["walker_id"] = i
    print(f"Burn-in complete: {len(walkers)} walker(s) collected "
          f"(n_burnin_md // every_burnin = {args.n_burnin_md // args.every_burnin}).")
    return walkers


def main():
    args = parse_args()

    os.makedirs(args.fname, exist_ok=True)

    n_frozen = resolve_n_frozen(args.n_frozen, args.lammps_data)

    atoms0 = read(args.init_xyz)

    print("Loading mean MACE calculator...")
    mace_calc = load_mace_calculator(args.mace_model, device=args.device)

    walkers = run_burnin(atoms0, mace_calc, n_frozen, args)
    n_walkers = len(walkers)
    plot_traj(walkers, os.path.join(args.fname, "walkers_initial.png"))

    print("Building committee...")
    comm_calc = MACEHALCalculator(
        mace_calc,
        committee_size=args.committee_size,
        prior_weight=args.prior_weight,
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        lowmem=args.lowmem,
        batch_size=args.batch_size,
        random_seed=np.random.RandomState(args.seed),
    )
    # One committee shared by all walkers: their only coupling.
    comm_calc.resample_committee()
    print(f"  committee device: {comm_calc.torch_device}   committee_size: {args.committee_size}")
    print(f"  N walkers: {n_walkers}  (shared committee/tau across all of them)")

    hal_calc = HALBiasCalculator(
        mean_calc=mace_calc,
        committee_calc=comm_calc,
        adaptive_tau=True,
        tau_rel=args.tau_rel,
        tau_hist=args.tau_hist,
        tau_delay=args.tau_delay,
    )
    hal_calc.reset_tau()

    for w in walkers:
        w.calc = hal_calc

    traj_writers = [
        Trajectory(os.path.join(args.fname, f"trajectory_walker_{w.info['walker_id']}.traj"), "w", w)
        for w in walkers
    ]

    offline_configs = []
    offline_draw_steps = []
    selection_records = []
    live_selected_snapshots = []
    diag_steps, diag_taus, diag_scores, diag_thresholds, diag_walker_ids = [], [], [], [], []

    diagnostics_path = os.path.join(args.fname, "uhal_diagnostics.csv")
    diagnostics_fh = open(diagnostics_path, "w", newline="")
    diagnostics_writer = csv.writer(diagnostics_fh)

    # Callbacks read `atoms`/`dyn` at call time, so they follow the active walker.
    if args.offline:
        print(f"--offline: skipping the online HAL selector -- automatically drawing "
              f"and folding in each walker's endpoint every --n_run={args.n_run} steps, "
              f"then picking --n_select={args.n_select} diverse structures via FPS.")

        # Log the HAL score as a diagnostic only.
        diagnostics_writer.writerow(["step", "walker_id", "tau", "hal_score"])

        def _log_diagnostics():
            step = global_step + dyn.get_number_of_steps()
            score = hal_calc.get_score()
            diagnostics_writer.writerow([step, atoms.info["walker_id"], hal_calc.tau, score])
            diag_steps.append(step)
            diag_taus.append(hal_calc.tau)
            diag_scores.append(score)
            diag_walker_ids.append(atoms.info["walker_id"])
            diagnostics_fh.flush()
    else:
        try:
            selector_threshold = float(args.selector_threshold)
        except ValueError:
            selector_threshold = args.selector_threshold

        selector = StructureSelector(
            bias_calc=hal_calc,
            threshold=selector_threshold,
            auto_resample=True,
            delay=args.selector_delay,
            mixing=args.selector_mixing,
            thresh_mul=args.selector_thresh_mul,
        )

        # selection_records[i] <-> comm_calc.selected_structures[i].
        def _on_selection():
            step = global_step + dyn.get_number_of_steps()
            selection_records.append({
                "step": step,
                "walker_id": atoms.info["walker_id"],
                "hal_score": selector._prev_score,
                "tau": hal_calc.tau,
            })
            print(f"[UHAL] SELECTED structure at step {step} (walker {atoms.info['walker_id']})  "
                  f"(hal_score={selector._prev_score:.4f}, tau={hal_calc.tau:.4f}, "
                  f"n_selected_so_far={len(selection_records)})", flush=True)

            # Own snapshot so the live plot is never a step behind.
            live_selected_snapshots.append(atoms.copy())
            plot_traj(live_selected_snapshots, os.path.join(args.fname, "diverse_subset.png"))

        selector.add_selection_callback(_on_selection)

        diagnostics_writer.writerow(["step", "walker_id", "tau", "hal_score", "threshold", "n_selected_so_far"])

        def _log_diagnostics():
            step = global_step + dyn.get_number_of_steps()
            diagnostics_writer.writerow([
                step, atoms.info["walker_id"], hal_calc.tau, selector._prev_score,
                selector.threshold, len(comm_calc.selected_structures),
            ])
            diag_steps.append(step)
            diag_taus.append(hal_calc.tau)
            diag_scores.append(selector._prev_score)
            diag_thresholds.append(selector.threshold)
            diag_walker_ids.append(atoms.info["walker_id"])
            diagnostics_fh.flush()

    # Round-robin main loop (mirrors run_skmd_mace.py).
    walker_queue = deque(walkers)
    total_steps = args.nsteps * args.n_run
    start_time = time.time()
    global_step = 0

    if args.offline:
        print(f"Running {total_steps} biased MD steps total (nsteps={args.nsteps} x n_run={args.n_run}), "
              f"round-robin across {n_walkers} walkers...")
    else:
        print(f"Running up to {total_steps} biased MD steps total (nsteps={args.nsteps} x n_run={args.n_run}), "
              f"round-robin across {n_walkers} walkers, stopping early once {args.n_select} "
              f"structures have been selected...")

    for it in range(args.nsteps):
        atoms = walker_queue.popleft()
        dyn = Langevin(atoms, args.timestep_fs * fs, temperature_K=args.T,
                        friction=args.friction / fs)
        dyn.attach(hal_calc.update_tau)
        dyn.attach(traj_writers[atoms.info["walker_id"]].write, interval=args.traj_interval)
        attach_global_progress_printer(dyn, total_steps, "biased MD", global_step, start_time)

        if args.offline:
            dyn.attach(_log_diagnostics, args.selector_interval)
        else:
            dyn.attach(selector, args.selector_interval)
            dyn.attach(_log_diagnostics, args.selector_interval)

        dyn.run(args.n_run)
        global_step += args.n_run

        if args.offline:
            new_config = atoms.copy()
            offline_configs.append(new_config)
            offline_draw_steps.append(global_step)
            print(f"[UHAL] offline: collected endpoint {len(offline_configs)}/{args.nsteps} "
                  f"from walker {atoms.info['walker_id']} at step {global_step}", flush=True)
            # Fold into the committee so later walkers are biased away from it.
            comm_calc.select_structure(new_config)
            comm_calc.resample_committee()
            plot_traj(offline_configs, os.path.join(args.fname, "endpoints_overview.png"))
        else:
            if len(comm_calc.selected_structures) >= args.n_select:
                print(f"[UHAL] reached target n_select={args.n_select} after outer iteration "
                      f"{it + 1}/{args.nsteps} (step {global_step}) -- stopping early")
                walker_queue.append(atoms)
                break

        walker_queue.append(atoms)

    diagnostics_fh.close()
    for tw in traj_writers:
        tw.close()

    # Overview of all walkers' frames.
    full_traj = []
    for w in walkers:
        full_traj.extend(read(os.path.join(args.fname, f"trajectory_walker_{w.info['walker_id']}.traj"), index=":"))
    print(f"Read back {len(full_traj)} frames total across {n_walkers} walker trajectory files")
    plot_traj(full_traj, os.path.join(args.fname, "trajectory_overview.png"))

    if args.offline:
        walker_cmap = plt.get_cmap("tab20")
        walker_norm = plt.Normalize(vmin=0, vmax=max(n_walkers - 1, 1))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9))
        ax2.sharex(ax1)
        sc1 = ax1.scatter(diag_steps, diag_taus, c=diag_walker_ids, cmap=walker_cmap, norm=walker_norm, s=10)
        fig.colorbar(sc1, ax=ax1, label="walker_id")
        ax1.set_xlabel("MD step"); ax1.set_ylabel("tau")

        ax2.scatter(diag_steps, diag_scores, c=diag_walker_ids, cmap=walker_cmap, norm=walker_norm, s=10)
        for step, w_id in zip(offline_draw_steps, [a.info["walker_id"] for a in offline_configs]):
            ax2.axvline(step, color=walker_cmap(walker_norm(w_id)), alpha=0.5, linewidth=1)
        ax2.set_xlabel("MD step"); ax2.set_ylabel("HAL score")

        draw_counts = np.bincount([a.info["walker_id"] for a in offline_configs], minlength=n_walkers)
        ax3.bar(range(n_walkers), draw_counts,
                color=[walker_cmap(walker_norm(i)) for i in range(n_walkers)])
        ax3.set_xlabel("walker_id"); ax3.set_ylabel("endpoints drawn")

        fig.tight_layout()
        diag_png = os.path.join(args.fname, "uhal_diagnostics.png")
        fig.savefig(diag_png); plt.close(fig)
        print(f"Wrote: {diag_png}")

        print(f"UHAL (offline) loop complete. {len(offline_configs)} trajectory endpoint(s) collected "
              f"across {n_walkers} walkers.")

        for a in offline_configs:
            a.info["mace_descriptor"] = comm_calc.get_descriptor_energy(a).tolist()
            a.info["unbiased_energy"] = float(mace_calc.get_potential_energy(a))

        n_select_eff = min(args.n_select, len(offline_configs))
        if n_select_eff < args.n_select:
            print(f"  --n_select={args.n_select} > {len(offline_configs)} collected endpoints "
                  f"(nsteps={args.nsteps} too small for this budget?) -- capping to {n_select_eff}")

        selected_idx, selected = sample_diverse_structures(
            offline_configs, n_select=n_select_eff, key="mace_descriptor",
            normalise=True, initial="medoid",
        )
        print(f"  FPS selected {len(selected)}/{len(offline_configs)} endpoints -- indices {selected_idx}")

        write(os.path.join(args.fname, "diverse_subset.xyz"), selected, format="extxyz")
        print(f"Wrote {len(selected)} structures to {args.fname}/diverse_subset.xyz")

        plot_traj(selected, os.path.join(args.fname, "diverse_subset.png"))

        pca_scatter([selected], labels=["uhal-offline FPS selections"],
                    out_png=os.path.join(args.fname, "traj_overview.png"))
        print(f"Wrote: {args.fname}/traj_overview.png")
        report_gpu_peak_memory(args.device)
        return

    # -- Online mode only, from here on --

    # Diagnostics: tau, HAL score/threshold vs step, selections per walker.
    walker_cmap = plt.get_cmap("tab20")
    walker_norm = plt.Normalize(vmin=0, vmax=max(n_walkers - 1, 1))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9))
    ax2.sharex(ax1)
    sc1 = ax1.scatter(diag_steps, diag_taus, c=diag_walker_ids, cmap=walker_cmap, norm=walker_norm, s=10)
    fig.colorbar(sc1, ax=ax1, label="walker_id")
    ax1.set_xlabel("MD step"); ax1.set_ylabel("tau")

    ax2.scatter(diag_steps, diag_scores, c=diag_walker_ids, cmap=walker_cmap, norm=walker_norm, s=10, label="hal_score")
    ax2.plot(diag_steps, diag_thresholds, color="black", linestyle="--", linewidth=1, label="threshold")
    for rec in selection_records:
        ax2.axvline(rec["step"], color=walker_cmap(walker_norm(rec["walker_id"])), alpha=0.5, linewidth=1)
    ax2.set_xlabel("MD step"); ax2.set_ylabel("HAL score")
    ax2.legend()

    selection_counts = np.bincount([rec["walker_id"] for rec in selection_records], minlength=n_walkers)
    ax3.bar(range(n_walkers), selection_counts,
            color=[walker_cmap(walker_norm(i)) for i in range(n_walkers)])
    ax3.set_xlabel("walker_id"); ax3.set_ylabel("selections")

    fig.tight_layout()
    diag_png = os.path.join(args.fname, "uhal_diagnostics.png")
    fig.savefig(diag_png); plt.close(fig)
    print(f"Wrote: {diag_png}")

    n_online_selections = len(comm_calc.selected_structures)
    print(f"UHAL loop complete. {n_online_selections} structure(s) selected online "
          f"across {n_walkers} walkers.")

    if n_online_selections == 0:
        print("[run_uhal_mace] WARNING: no structures were ever selected — "
              "HAL score never peaked above threshold. Writing an empty "
              "diverse_subset.xyz; consider recalibrating --tau_rel / "
              "--selector_thresh_mul.")
        write(os.path.join(args.fname, "diverse_subset.xyz"), [], format="extxyz")
        report_gpu_peak_memory(args.device)
        return

    order = sorted(range(n_online_selections),
                    key=lambda i: selection_records[i]["hal_score"], reverse=True)
    keep_idx = order[:args.n_select]
    if len(order) > args.n_select:
        print(f"  capping {len(order)} online selections to top --n_select={args.n_select} by HAL score")

    kept = []
    for i in keep_idx:
        src = comm_calc.selected_structures[i]
        a = src.copy()
        rec = selection_records[i]
        a.info["mace_descriptor"] = comm_calc.get_descriptor_energy(src).tolist()
        a.info["unbiased_energy"] = float(mace_calc.get_potential_energy(src))
        a.info["hal_score"] = float(rec["hal_score"])
        a.info["tau_at_selection"] = float(rec["tau"])
        a.info["selection_step"] = int(rec["step"])
        a.info["walker_id"] = int(rec["walker_id"])
        kept.append(a)

    write(os.path.join(args.fname, "diverse_subset.xyz"), kept, format="extxyz")
    print(f"Wrote {len(kept)} structures to {args.fname}/diverse_subset.xyz")

    pca_scatter([kept], labels=["uhal selections"],
                out_png=os.path.join(args.fname, "traj_overview.png"))
    print(f"Wrote: {args.fname}/traj_overview.png")
    report_gpu_peak_memory(args.device)


if __name__ == "__main__":
    main()
