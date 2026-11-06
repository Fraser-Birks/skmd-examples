# Data

Files marked Zenodo ([10.5281/zenodo.24011550](https://doi.org/10.5281/zenodo.24011550))
are fetched by `bash data/fetch_data.sh [alanine|silicon|results]`.

## `muller_brown/`

| File | | Purpose |
|---|---|---|
| `dataset0.torch` | repo | initial training set / SKMD ensemble |
| `model0.torch` | repo | initial model |
| `dataset_test_1000.torch` | repo | test set (Boltzmann samples) |

## `alanine_dipeptide/`

| File | | Purpose |
|---|---|---|
| `ace-ala-nme.{data,xyz}` | repo | starting structure |
| `surrogate_fit_input.yaml` | repo | `mace_run_train` input for the surrogate |
| `MACE-OFF23_small.model` | Zenodo | oracle |
| `mace_off_32_0.1_percent.model` | Zenodo | surrogate (iteration 0) |
| `training_set_0.1_percent.xyz` | Zenodo | surrogate training set |
| `phi_psi_relaxed_pes_off23_small.npz` | Zenodo | oracle Ramachandran surface (Fig. 3a) |
| `test_set.xyz` | Zenodo | test set (Fig. 3f,g) |

## `silicon_surface/`

| File | | Purpose |
|---|---|---|
| `relaxed_struct_{non_pandey,pandey}.xyz` | repo | unreconstructed / Pandey slabs |
| `silicon_slab.data{,.n_frozen}` | repo | LAMMPS data file, 18 frozen atoms |
| `write_vasp_input_SCAN.py` | repo | reference VASP/SCAN settings |
| `mace-mpa-0-medium.model` | Zenodo | foundation model (iteration 0) |
| `DFT_relaxed/{ref,pandey}/OUTCAR` | Zenodo | relaxed SCAN slabs, ΔE_recon = −0.279 J/m² (Fig. 4) |
| `DFT_reference/{ref,pandey}/OUTCAR` | Zenodo | SCAN single points at the fixed geometries, −0.382 J/m² |

The OUTCARs are deposited as `silicon_dft_outcars.tar.gz`. POTCARs are not
redistributed (VASP licence).

## Paper results (`skmd_results.tar.gz`, `fetch_data.sh results`)

Per-run outputs read by the figure scripts, unpacked into the seed directories:

| File | Purpose |
|---|---|
| `alanine_dipeptide/<arm>_seed_<N>/training/iteration_<k>/test_metrics.json` | test-set RMSEs (Fig. 3f,g) |
| `alanine_dipeptide/iter0_test_metrics.json` | iteration-0 point |
| `silicon_surface/<arm>_seed_<N>/.reconstruction_energy{,_relaxed}_cache.csv` | ΔE_recon per model (Fig. 4) |

Not archived: fine-tuned models, trajectories, sampled and DFT-labelled
structures. Rerun the pipelines to regenerate them.
