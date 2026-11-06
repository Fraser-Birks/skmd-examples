# Si(111) surface reconstruction (Sec. 5.3, Fig. 4, App. E.3)

Fine-tunes MACE-MPA-0-medium on SCAN data to predict the Pandey (2 × 1)
reconstruction energy of Si(111). 28-atom slab, lowest 18 atoms fixed. Per
iteration: J = 30 particles, 200 outer steps × 500 MD steps, 30 configurations
labelled with VASP.

| Arm | Launch script | Settings |
|---|---|---|
| Langevin 300 K / 700 K | `run_langevin_{300k,700k}_seeds.sh` | underdamped, `--a 0`, FPS |
| SKMD | `run_skmd_offline_seeds.sh` | overdamped 50 K, a = 4.0, FPS |
| a-SKMD | `run_askmd_seeds.sh` | a = 4.0, ζ₀ = 10th percentile, ℓ₀ = 250 |
| uHAL | `run_uhal_seeds.sh` | ase_uhal |

```bash
./run_<arm>_seeds.sh NUM_ITERS N_SEEDS --n-select 30      # paper: 5 3
```

Each iteration is a SLURM chain in `slurm/example/`: sampling (phase 1), a VASP
array (`gpu_dft_array.slurm`, or `dft_array.slurm` with `--cpu-dft`), then
collection and LoRA fine-tuning (phase 2), which submits the next iteration.
Resume with the loop driver's `--start-from K` (`--skip-md`, `--skip-dft`).
