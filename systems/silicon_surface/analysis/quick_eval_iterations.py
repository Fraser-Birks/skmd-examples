#!/usr/bin/env python3
"""Reconstruction energy E(reconstructed) - E(reference) under the base model and every AL iteration's model.

MACECalculator only (no LAMMPS); per-model results cached in --run-dir/.reconstruction_energy_cache.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_mace_calculator, lowest_z_indices, relax_atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dft"))
from dft_runner import parse_outcar

ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")
BASE_MODEL_NAME = "mace-mpa-0-medium"


def discover_models(run_dir: Path, model_pattern: str = "si_slab_skmd_lora_it_{n}.model") -> list[tuple[str, Path]]:
    """Return [(label, model_path), ...], base model first; model_pattern is relative to training/iteration_{n}/."""
    found = []

    base_model = run_dir.parent / f"{BASE_MODEL_NAME}.model"
    if base_model.is_file():
        found.append(("it0_base", base_model))
    else:
        print(f"[quick_eval_iterations] WARNING: base model not found at {base_model}, "
              "skipping it0_base", file=sys.stderr)

    training_dir = run_dir / "training"
    tuned = []
    if training_dir.is_dir():
        for iter_dir in training_dir.iterdir():
            m = ITERATION_DIR_RE.match(iter_dir.name)
            if not m:
                continue
            n = int(m.group(1))
            model_path = iter_dir / model_pattern.format(n=n)
            if model_path.is_file():
                tuned.append((n, model_path))
    tuned.sort(key=lambda t: t[0])
    found += [(f"it{n}", p) for n, p in tuned]
    return found


def eval_energy(atoms, calc) -> float:
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy())


def dft_energy(workdir: Path, expected_natoms: int) -> float:
    result = parse_outcar(workdir, expected_natoms=expected_natoms)
    if result is None:
        print(f"[quick_eval_iterations] ERROR: no parseable OUTCAR found in {workdir}",
              file=sys.stderr)
        raise SystemExit(1)
    if not result.converged:
        print(f"[quick_eval_iterations] WARNING: {workdir} is not converged (relaxation "
              "unfinished or SCF/ionic limit hit) -- using its last ionic step anyway",
              file=sys.stderr)
    return result.energy


def relax_protocol_tag(relax: dict | None) -> str:
    """Cache-signature suffix for the relaxation protocol, so fixed/relaxed results never mix."""
    if not relax:
        return ""
    return f";relax:fmax={relax['fmax']},steps={relax['steps']},n_frozen_lowest_z={relax['n_frozen']}"


def _relaxed_energy(atoms, calc, relax: dict, label: str, kind: str):
    frozen = lowest_z_indices(atoms, relax["n_frozen"])
    relaxed, converged, nsteps, energy = relax_atoms(
        atoms, calc, fmax=relax["fmax"], steps=relax["steps"], frozen_indices=frozen)
    if not converged:
        print(f"\n    WARNING: {label} {kind} relaxation NOT converged in {nsteps} steps", flush=True)
    geom_dir = relax.get("geom_dir")
    if geom_dir is not None:
        Path(geom_dir).mkdir(parents=True, exist_ok=True)
        relaxed.info.update(label=label, kind=kind, relax_converged=converged, relax_nsteps=nsteps)
        relaxed.write(str(Path(geom_dir) / f"{label}_{kind}.extxyz"))
    return energy, nsteps, converged


def _geom_signature(atoms) -> str:
    """Short hash of symbols + positions, to invalidate the cache if a structure changes."""
    h = hashlib.sha1()
    h.update(repr(atoms.get_chemical_symbols()).encode())
    h.update(np.round(atoms.get_positions(), 8).tobytes())
    return h.hexdigest()[:16]


def _cache_geom_sig_line(reference, reconstructed, protocol: str = "") -> str:
    return f"# geom_sig={_geom_signature(reference)}:{_geom_signature(reconstructed)}{protocol}\n"


def _load_cache(cache_path: Path, reference, reconstructed,
                protocol: str = "") -> dict[str, tuple[float, float, float, float]]:
    """model_path -> (mtime, e_ref, e_recon, delta) in eV; {} if missing or for different geometries."""
    if not cache_path.is_file():
        return {}
    with open(cache_path, newline="") as fh:
        first_line = fh.readline()
        if first_line != _cache_geom_sig_line(reference, reconstructed, protocol):
            print(f"[quick_eval_iterations] cache at {cache_path} is for a different "
                  "reference/reconstructed geometry (or is stale) -- ignoring it", file=sys.stderr)
            return {}
        cache: dict[str, tuple[float, float, float, float]] = {}
        for row in csv.DictReader(fh):
            try:
                cache[row["model_path"]] = (
                    float(row["model_mtime"]), float(row["e_reference_eV"]),
                    float(row["e_reconstructed_eV"]), float(row["delta_eV"]),
                )
            except (KeyError, ValueError):
                continue
        return cache


def _write_cache(cache_path: Path, reference, reconstructed,
                  rows: list[tuple[str, str, float, float, float, float]], protocol: str = "") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as fh:
        fh.write(_cache_geom_sig_line(reference, reconstructed, protocol))
        writer = csv.writer(fh)
        writer.writerow(["label", "model_path", "model_mtime", "e_reference_eV", "e_reconstructed_eV", "delta_eV"])
        for label, model_path, mtime, e_ref, e_recon, delta in rows:
            writer.writerow([label, model_path, mtime, e_ref, e_recon, delta])


def compute_deltas(run_dir: Path, reference, reconstructed, device: str = "cpu",
                    model_pattern: str = "si_slab_skmd_lora_it_{n}.model",
                    cache_path: Path | None = None, refresh_cache: bool = False,
                    relax: dict | None = None):
    """Return [(label, e_ref, e_recon, delta)] in eV for every model under run_dir, at fixed geometries.

    relax ({"fmax", "steps", "n_frozen", "geom_dir"?}) relaxes both structures per model instead (use a
    separate cache_path). cache_path caches by model path + mtime."""
    models = discover_models(run_dir, model_pattern=model_pattern)
    if not models:
        print(f"[quick_eval_iterations] ERROR: no models found (base model or "
              f"training/iteration_N/{model_pattern}) under {run_dir}",
              file=sys.stderr)
        raise SystemExit(1)

    protocol = relax_protocol_tag(relax)
    cache = (_load_cache(cache_path, reference, reconstructed, protocol)
             if (cache_path is not None and not refresh_cache) else {})

    rows = []
    cache_rows = []
    n_models = len(models)
    for i, (label, model_path) in enumerate(models, start=1):
        model_key = str(model_path.resolve())
        mtime = model_path.stat().st_mtime
        cached = cache.get(model_key)
        if cached is not None and cached[0] == mtime:
            _, e_ref, e_recon, delta = cached
            print(f"[{i}/{n_models}] {label}: cached (delta = {delta:+.6f} eV)", flush=True)
        else:
            print(f"[{i}/{n_models}] {label}: loading {model_path} ...", end=" ", flush=True)
            calc = load_mace_calculator(model_path, device=device)
            if relax:
                print("loaded. relaxing reference ...", end=" ", flush=True)
                e_ref, n_ref, _ = _relaxed_energy(reference, calc, relax, label, "reference")
                print(f"{n_ref} steps. relaxing reconstructed ...", end=" ", flush=True)
                e_recon, n_rec, _ = _relaxed_energy(reconstructed, calc, relax, label, "reconstructed")
                print(f"{n_rec} steps.", end=" ", flush=True)
            else:
                print("loaded. evaluating reference ...", end=" ", flush=True)
                e_ref = eval_energy(reference, calc)
                print("done. evaluating reconstructed ...", end=" ", flush=True)
                e_recon = eval_energy(reconstructed, calc)
            delta = e_recon - e_ref
            print(f"done. delta = {delta:+.6f} eV", flush=True)
        rows.append((label, e_ref, e_recon, delta))
        cache_rows.append((label, model_key, mtime, e_ref, e_recon, delta))

    if cache_path is not None:
        _write_cache(cache_path, reference, reconstructed, cache_rows, protocol)

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                    help="Run directory containing training/iteration_N/ subdirs "
                         "(e.g. data/silicon_surface/skmd_offline_seed_1)")
    p.add_argument("--reference-xyz", type=Path, required=True,
                    help="Single-frame reference structure (e.g. the plain surface)")
    p.add_argument("--reconstructed-xyz", type=Path, required=True,
                    help="Single-frame structure to compare against the reference "
                         "(e.g. the Pandey 2x1 reconstruction)")
    p.add_argument("--reference-dft-dir", type=Path, default=None,
                    help="Finished VASP calc dir for --reference-xyz (requires "
                         "--reconstructed-dft-dir too)")
    p.add_argument("--reconstructed-dft-dir", type=Path, default=None,
                    help="Finished VASP calc dir for --reconstructed-xyz (requires "
                         "--reference-dft-dir too)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="default cpu")
    p.add_argument("--refresh-cache", action="store_true",
                    help="Ignore the cache and re-evaluate every model")
    args = p.parse_args()

    if (args.reference_dft_dir is None) != (args.reconstructed_dft_dir is None):
        print("[quick_eval_iterations] ERROR: --reference-dft-dir and "
              "--reconstructed-dft-dir must be given together", file=sys.stderr)
        raise SystemExit(1)

    reference = read(str(args.reference_xyz), index=0)
    reconstructed = read(str(args.reconstructed_xyz), index=0)
    print(f"[quick_eval_iterations] reference: {args.reference_xyz.name} ({len(reference)} atoms)   "
          f"reconstructed: {args.reconstructed_xyz.name} ({len(reconstructed)} atoms)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Under --run-dir so plot_reconstruction_energy_curves.py shares this cache.
    cache_path = args.run_dir / ".reconstruction_energy_cache.csv"
    rows = compute_deltas(args.run_dir, reference, reconstructed, device=args.device,
                           cache_path=cache_path, refresh_cache=args.refresh_cache)

    dft_delta = None
    if args.reference_dft_dir is not None:
        print("[quick_eval_iterations] parsing DFT OUTCARs ...", flush=True)
        e_ref_dft = dft_energy(args.reference_dft_dir, len(reference))
        e_recon_dft = dft_energy(args.reconstructed_dft_dir, len(reconstructed))
        dft_delta = e_recon_dft - e_ref_dft
        print(f"[quick_eval_iterations] DFT delta = {dft_delta:+.6f} eV", flush=True)

    csv_path = args.output_dir / "quick_eval_iterations.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "e_reference_eV", "e_reconstructed_eV", "delta_eV"])
        for label, e_ref, e_recon, delta in rows:
            writer.writerow([label, e_ref, e_recon, delta])
        if dft_delta is not None:
            writer.writerow(["DFT", e_ref_dft, e_recon_dft, dft_delta])
    print(f"[quick_eval_iterations] wrote {csv_path}")

    bar_labels = [r[0] for r in rows]
    bar_values = [r[3] for r in rows]

    fig, ax = plt.subplots()
    ax.bar(bar_labels, bar_values, color="tab:blue")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    if dft_delta is not None:
        ax.axhline(dft_delta, color="black", linewidth=1.5, linestyle="--", label="DFT")
        ax.legend(fontsize="small")
    ax.set_xticklabels(bar_labels, rotation=45, ha="right")
    ax.set_ylabel("E(reconstructed) - E(reference) (eV)")
    ax.set_title(f"{args.reconstructed_xyz.name} vs. {args.reference_xyz.name}")
    fig.tight_layout()
    png_path = args.output_dir / "quick_eval_iterations.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[quick_eval_iterations] wrote {png_path}")


if __name__ == "__main__":
    main()
