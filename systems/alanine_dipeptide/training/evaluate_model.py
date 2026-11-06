#!/usr/bin/env python3
"""Evaluate a MACE model on a test set (mace_energy/mace_forces labels); write energy/force RMSE/MAE to JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import ase.io


def compute_metrics(e_errors: list[float], f_errors: list[float]) -> dict:
    e = np.array(e_errors)
    f = np.array(f_errors)
    return {
        "n_frames": len(e_errors),
        "energy_rmse_meV_per_atom": round(float(np.sqrt(np.mean(e ** 2))) * 1000, 4),
        "energy_mae_meV_per_atom":  round(float(np.mean(np.abs(e))) * 1000, 4),
        "force_rmse_eV_per_A":      round(float(np.sqrt(np.mean(f ** 2))), 6),
        "force_mae_eV_per_A":       round(float(np.mean(np.abs(f))), 6),
    }


def evaluate_frames(model_path: Path, frames: list, device: str = "cuda") -> dict:
    """Score labelled `frames` under `model_path`; returns {"overall", "by_config_type"} metrics."""
    from mace.calculators import MACECalculator
    calc = MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype="float64",
    )

    e_errors: list[float] = []
    f_errors: list[float] = []
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: {"e": [], "f": []})
    n_missing = 0

    for atoms in frames:
        if "mace_energy" not in atoms.info or "mace_forces" not in atoms.arrays:
            n_missing += 1
            continue
        ref_e = float(atoms.info["mace_energy"])
        ref_f = np.array(atoms.arrays["mace_forces"])

        atoms_clean = atoms.copy()
        atoms_clean.calc = calc
        pred_e = float(atoms_clean.get_potential_energy())
        pred_f = atoms_clean.get_forces()

        n = len(atoms_clean)
        e_err = (pred_e - ref_e) / n
        f_err = (pred_f - ref_f).flatten().tolist()

        e_errors.append(e_err)
        f_errors.extend(f_err)

        ct = atoms.info.get("config_type", "unknown")
        by_type[ct]["e"].append(e_err)
        by_type[ct]["f"].extend(f_err)

    if not e_errors:
        raise ValueError("no evaluable frames found (missing mace_energy/mace_forces on every frame)")

    overall = compute_metrics(e_errors, f_errors)
    overall["n_skipped"] = n_missing

    per_type = {
        ct: compute_metrics(data["e"], data["f"])
        for ct, data in sorted(by_type.items())
    }

    return {"overall": overall, "by_config_type": per_type}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True,
                   help="Path to .model file")
    p.add_argument("--test-xyz", type=Path, required=True,
                   help="Test set extxyz with mace_energy / mace_forces keys")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON for metrics")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    frames = ase.io.read(str(args.test_xyz), index=":")
    print(f"[eval] {len(frames)} test frames in {args.test_xyz}")

    try:
        metrics = evaluate_frames(args.model, frames, device=args.device)
    except ValueError as e:
        print(f"[eval] ERROR: {e}", flush=True)
        raise SystemExit(1)

    overall = metrics["overall"]
    per_type = metrics["by_config_type"]
    n_missing = overall["n_skipped"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")

    e_rmse = overall["energy_rmse_meV_per_atom"]
    e_mae  = overall["energy_mae_meV_per_atom"]
    f_rmse = overall["force_rmse_eV_per_A"]
    f_mae  = overall["force_mae_eV_per_A"]

    print(f"[eval] {overall['n_frames']} frames evaluated ({n_missing} skipped)")
    print(f"[eval] overall  energy RMSE: {e_rmse:.2f} meV/atom  MAE: {e_mae:.2f} meV/atom")
    print(f"[eval] overall  force  RMSE: {f_rmse:.4f} eV/Å      MAE: {f_mae:.4f} eV/Å")
    for ct, m in sorted(per_type.items()):
        print(f"[eval]   {ct:35s}  E-RMSE: {m['energy_rmse_meV_per_atom']:7.2f} meV/atom  "
              f"F-RMSE: {m['force_rmse_eV_per_A']:.4f} eV/Å  (n={m['n_frames']})")
    print(f"[eval] metrics → {args.output}")


if __name__ == "__main__":
    main()
