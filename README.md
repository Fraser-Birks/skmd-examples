# skmd-examples

Experiments for *Stein Kernelized Molecular Dynamics for Active Learning of
Interatomic Potentials* (Zou, Birks, Foster, Marzouk, NeurIPS 2026).
`fix skmd` for LAMMPS is in [skmd-lammps](https://github.com/Fraser-Birks/skmd-lammps).

| Directory | Paper |
|---|---|
| [`systems/muller_brown`](systems/muller_brown/README.md) | Sec. 5.1, Figs. 1–2 |
| [`systems/alanine_dipeptide`](systems/alanine_dipeptide/README.md) | Sec. 5.2, Fig. 3 |
| [`systems/silicon_surface`](systems/silicon_surface/README.md) | Sec. 5.3, Fig. 4 |

`skmd/` is the shared Python package; [`docs/usage.md`](docs/usage.md) lists
the SKMD parameters.

## Installation

Müller–Brown only: `pip install -e ".[muller-brown]"`.

MACE experiments (GPU cluster):

1. LAMMPS with `fix skmd` and symmetrix (tag `skmd-v1.0` of
   [symmetrix-public](https://github.com/Fraser-Birks/symmetrix-public)); see skmd-lammps.
2. `pip install -e ".[pipelines,dev]"`. Paper: Python 3.11, torch 2.13, mace-torch 0.3.16.
3. VASP 6 (Si only).

The drivers submit the jobs in `systems/*/slurm/example/`. Edit their
`#SBATCH` lines for your cluster and export:

| Variable | |
|---|---|
| `SKMD_VENV` | venv from step 2 |
| `SKMD_MODULES` | optional; modules loaded in every job |
| `VASP_STD`, `VASP_STD_CPU` | GPU / CPU `vasp_std` (Si) |
| `VASP_PP_PATH` | PAW potentials (Si) |
| `VASP_LD_LIBRARY_PATH` | optional; extra libraries for VASP |

## Data

```bash
bash data/fetch_data.sh
```

Downloads the large inputs from Zenodo
([10.5281/zenodo.24011550](https://doi.org/10.5281/zenodo.24011550)); see
[`data/MANIFEST.md`](data/MANIFEST.md).

`bash data/fetch_data.sh results` adds the paper's per-run test metrics and
ΔE_recon values, from which the plotting commands below redraw Figs. 3f,g and 4
without rerunning anything. Fine-tuned models, trajectories and labelled
structures are not archived; rerun the pipelines to regenerate them.

## Figure 3 (alanine dipeptide)

```bash
cd systems/alanine_dipeptide
for arm in langevin_300k langevin_700k baseline uhal skmd_offline askmd; do
  ./run_${arm}_seeds.sh 10 5 --n-select 100
done
cd ../..
python systems/alanine_dipeptide/training/evaluate_model.py \
    --model data/alanine_dipeptide/mace_off_32_0.1_percent.model \
    --test-xyz data/alanine_dipeptide/test_set.xyz \
    --output data/alanine_dipeptide/iter0_test_metrics.json --device cuda
python systems/alanine_dipeptide/training/plot_learning_curves.py \
    --iter0-metrics data/alanine_dipeptide/iter0_test_metrics.json --logy
```

## Figure 4 (Si(111))

```bash
cd systems/silicon_surface
for arm in langevin_300k langevin_700k uhal skmd_offline askmd; do
  ./run_${arm}_seeds.sh 5 3 --n-select 30
done
cd ../..
python systems/silicon_surface/training/plot_reconstruction_energy_curves.py --relax \
    --reference-dft-dir     data/silicon_surface/DFT_relaxed/ref \
    --reconstructed-dft-dir data/silicon_surface/DFT_relaxed/pandey --device cuda
```

Each model relaxes both slabs (LBFGS, fmax 0.01 eV/Å, 18 lowest atoms fixed);
ΔE_recon is compared with relaxed SCAN (−0.279 J/m²). Model energies are
cached in each run directory, so replotting needs no GPU.

## Tests

```bash
pytest
```

## Citation

```bibtex
@inproceedings{zou2026skmd,
  title     = {Stein Kernelized Molecular Dynamics for Active Learning of Interatomic Potentials},
  author    = {Zou, Joanna and Birks, Fraser and Foster, Dallas and Marzouk, Youssef},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT
