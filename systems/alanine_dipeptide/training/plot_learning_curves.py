"""
Plot test-set energy/force RMSE vs AL iteration per arm (median and 25-75th percentile over seeds),
from <file_root>/<prefix>_seed_*/training/iteration_*/test_metrics.json; also writes markdown tables.
--regenerate re-scores every model against --test-xyz first (e.g. after the test set changes).

Usage:
    python systems/alanine_dipeptide/training/plot_learning_curves.py
    python systems/alanine_dipeptide/training/plot_learning_curves.py --out learning_curves.png
    python systems/alanine_dipeptide/training/plot_learning_curves.py --file-root /path/to/alanine_dipeptide
    python systems/alanine_dipeptide/training/plot_learning_curves.py --arm "My arm=my_prefix" --arm "Other=other_prefix"
    python systems/alanine_dipeptide/training/plot_learning_curves.py --regenerate
    python systems/alanine_dipeptide/training/plot_learning_curves.py --regenerate --test-xyz /path/to/new_test_set.xyz --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluate_model import evaluate_frames
from skmd import plot_style

# label -> (seed-dir prefix, model filename pattern); legend order = paper Fig. 3f,g arms.
# (U) underdamped Langevin, (O) overdamped (Brownian).
DEFAULT_ARMS = {
    "Langevin 300K (U)": ("langevin_300k", "mace_off_32_skmd_it_{n}.model"),
    "Langevin 700K (U)": ("langevin_700k", "mace_off_32_skmd_it_{n}.model"),
    "Langevin 50K (O)":  ("baseline",      "mace_off_32_skmd_it_{n}.model"),
    "uHAL (U)":          ("uhal",          "mace_off_32_uhal_it_{n}.model"),
    "SKMD (O)":          ("skmd_offline",  "mace_off_32_skmd_it_{n}.model"),
    "a-SKMD (O)":        ("askmd",         "mace_off_32_askmd_it_{n}.model"),
}

ITER_RE = re.compile(r"iteration_(\d+)$")


def regenerate_metrics(file_root: Path, arms: dict, test_xyz: Path, device: str) -> None:
    """Re-evaluate every discovered model against `test_xyz`, overwriting test_metrics.json."""
    import ase.io
    frames = ase.io.read(str(test_xyz), index=":")
    print(f"[regenerate] {len(frames)} test frames in {test_xyz}")

    n_written = 0
    for label, (prefix, model_pattern) in arms.items():
        seed_dirs = sorted(file_root.glob(f"{prefix}_seed_*"))
        if not seed_dirs:
            print(f"  [skip] {label} ({prefix}_seed_*): no seed directories found")
            continue
        for seed_dir in seed_dirs:
            for iter_dir in sorted((seed_dir / "training").glob("iteration_*")):
                m = ITER_RE.search(iter_dir.name)
                if not m:
                    continue
                n = int(m.group(1))
                model_path = iter_dir / model_pattern.format(n=n)
                if not model_path.is_file():
                    print(f"    [warn] {model_path} not found, skipping")
                    continue
                print(f"  [eval] {label} / {seed_dir.name} / {iter_dir.name} "
                      f"({model_path.name}) ...", end=" ", flush=True)
                metrics = evaluate_frames(model_path, frames, device=device)
                out_path = iter_dir / "test_metrics.json"
                out_path.write_text(json.dumps(metrics, indent=2) + "\n")
                n_written += 1
                print(f"E-RMSE={metrics['overall']['energy_rmse_meV_per_atom']:.2f} meV/atom "
                      f"F-RMSE={metrics['overall']['force_rmse_eV_per_A']:.4f} eV/A -> {out_path}")

    print(f"[regenerate] wrote {n_written} test_metrics.json file(s)")


def load_arm(file_root: Path, prefix: str) -> dict[int, dict[str, list[float]]] | None:
    """{iteration: {"energy": [...], "force": [...]}} over all seeds; None if no seed dirs exist."""
    seed_dirs = sorted(file_root.glob(f"{prefix}_seed_*"))
    if not seed_dirs:
        return None

    data: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"energy": [], "force": []})
    for seed_dir in seed_dirs:
        for metrics_path in sorted((seed_dir / "training").glob("iteration_*/test_metrics.json")):
            m = ITER_RE.search(metrics_path.parent.name)
            if not m:
                continue
            it = int(m.group(1))
            try:
                overall = json.loads(metrics_path.read_text())["overall"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    [warn] skipping unreadable {metrics_path}: {e}")
                continue
            data[it]["energy"].append(overall["energy_rmse_meV_per_atom"])
            data[it]["force"].append(overall["force_rmse_eV_per_A"])

    return dict(data)


def summarize(data: dict[int, dict[str, list[float]]], key: str):
    """(iterations, medians, p25, p75) arrays, sorted by iteration."""
    its = sorted(data.keys())
    medians, lo, hi = [], [], []
    for it in its:
        vals = np.array(data[it][key])
        medians.append(np.median(vals))
        lo.append(np.percentile(vals, 25))
        hi.append(np.percentile(vals, 75))
    return np.array(its), np.array(medians), np.array(lo), np.array(hi)


def _write_markdown_table(lines: list[str], title: str, curves: list[dict], value_key: str, fmt: str) -> None:
    """Append a GFM table (rows=arms, columns=even iterations) to `lines`."""
    all_its = sorted({it for c in curves for it in c["its"] if it % 2 == 0})
    lines.append(title)
    lines.append("")
    header = ["Method"] + [f"It. {it}" for it in all_its]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for c in curves:
        by_it = dict(zip(c["its"], c[value_key]))
        row = [f"{c['label']} (n={c['n']})"]
        for it in all_its:
            row.append(fmt.format(by_it[it]) if it in by_it else "--")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def write_markdown_tables(path: Path, curves: list[dict]) -> None:
    """Write median energy (meV/atom) and force (eV/A) RMSE tables to markdown."""
    lines = ["<!-- Auto-generated by plot_learning_curves.py -->", ""]
    _write_markdown_table(lines, "Energy RMSE (meV/atom) vs. AL iteration, by method.",
                           curves, "med_energy", "{:.3f}")
    _write_markdown_table(lines, "Force RMSE (eV/Å) vs. AL iteration, by method.",
                           curves, "med_force", "{:.4f}")
    path.write_text("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file-root", default=None,
                   help="Path to data/alanine_dipeptide/ (default: auto-resolved as "
                        "this script's ../../../data/alanine_dipeptide)")
    p.add_argument("--out", default="learning_curves.png")
    p.add_argument("--table-out", default="learning_curves_table.md",
                   help="Markdown file with two plain GFM tables (Energy RMSE, Force RMSE), "
                        "one row per arm, one column per iteration")
    p.add_argument("--arm", action="append", default=None, metavar="LABEL=PREFIX[:MODEL_PATTERN]",
                   help="Repeatable; overrides the default arm list entirely if given at all. "
                        "MODEL_PATTERN only needed if --regenerate is also used")
    p.add_argument("--regenerate", action="store_true",
                   help="Re-evaluate every discovered model against --test-xyz, overwriting "
                        "each iteration's test_metrics.json before plotting")
    p.add_argument("--test-xyz", default=None,
                   help="Test set to re-evaluate against with --regenerate (default: <file-root>/test_set.xyz)")
    p.add_argument("--device", default="cpu", help="Device for --regenerate (default cpu)")
    p.add_argument("--iter0-metrics", default=None, metavar="PATH",
                   help="test_metrics.json of the initial (pre-fine-tuning) model, written by "
                        "evaluate_model.py. Every arm starts from this same model, so its value "
                        "is added to every curve as iteration 0 (no seed spread).")
    p.add_argument("--logy", action="store_true", help="Log-scale y axes")
    return p.parse_args()


def main():
    args = parse_args()

    if args.file_root is not None:
        file_root = Path(args.file_root).resolve()
    else:
        file_root = Path(__file__).resolve().parents[3] / "data" / "alanine_dipeptide"
    if not file_root.is_dir():
        raise SystemExit(f"ERROR: {file_root} not found")

    if args.arm:
        arms = {}
        for a in args.arm:
            label, rest = a.split("=", 1)
            if ":" in rest:
                prefix, model_pattern = rest.split(":", 1)
            else:
                prefix, model_pattern = rest, "mace_off_32_skmd_it_{n}.model"
            arms[label] = (prefix, model_pattern)
    else:
        arms = DEFAULT_ARMS

    if args.regenerate:
        test_xyz = Path(args.test_xyz) if args.test_xyz else file_root / "test_set.xyz"
        if not test_xyz.is_file():
            raise SystemExit(f"ERROR: {test_xyz} not found")
        regenerate_metrics(file_root, arms, test_xyz, args.device)

    print(f"Reading from: {file_root}")

    iter0 = None
    if args.iter0_metrics:
        overall = json.loads(Path(args.iter0_metrics).read_text())["overall"]
        iter0 = (overall["energy_rmse_meV_per_atom"], overall["force_rmse_eV_per_A"])
        print(f"Iteration 0 (initial model): {iter0[0]:.4f} meV/atom, {iter0[1]:.5f} eV/Å")

    fig, (ax_e, ax_f) = plot_style.new_figure(2)

    curves: list[dict] = []
    max_it = 0
    for i, (label, (prefix, _model_pattern)) in enumerate(arms.items()):
        data = load_arm(file_root, prefix)
        if data is None:
            print(f"  [skip] {label} ({prefix}_seed_*): no seed directories found")
            continue
        if not data:
            print(f"  [skip] {label} ({prefix}_seed_*): seed directories found, but no "
                  f"test_metrics.json inside yet")
            continue

        if iter0 is not None:
            data[0] = {"energy": [iter0[0]], "force": [iter0[1]]}
        max_it = max(max_it, max(data))

        color = plot_style.arm_color(label, i)
        n_seeds_max = max(len(v["energy"]) for v in data.values())
        print(f"  [ok]   {label}: {len(data)} iteration(s) found, up to {n_seeds_max} seed(s) each")

        its, med_e, lo_e, hi_e = summarize(data, "energy")
        plot_style.plot_curve(ax_e, its, med_e, lo_e, hi_e, color=color, label=label)

        its, med_f, lo_f, hi_f = summarize(data, "force")
        plot_style.plot_curve(ax_f, its, med_f, lo_f, hi_f, color=color, label=label)

        curves.append({"label": label, "n": n_seeds_max, "its": its,
                        "med_energy": med_e, "med_force": med_f})

    if not curves:
        raise SystemExit("ERROR: no arms had any test_metrics.json to plot -- "
                          "check --file-root and that the seeds scripts have actually run")

    plot_style.style_axis(ax_e, "Test error in energies", "RMSE (meV/atom)")
    plot_style.style_axis(ax_f, "Test error in forces", "RMSE (eV/Å)")
    first_it = 0 if iter0 is not None else 1
    for ax in (ax_e, ax_f):
        ax.set_xticks(range(first_it, max_it + 1))
        if args.logy:
            ax.set_yscale("log")
    plot_style.add_legend(fig, ax_f)
    plot_style.save(fig, args.out)

    write_markdown_tables(Path(args.table_out), curves)
    print(f"Wrote: {args.table_out}")


if __name__ == "__main__":
    main()
