# Alanine dipeptide (Sec. 5.2, Fig. 3, App. E.2)

Fine-tunes a surrogate MACE model (`mace_off_32_0.1_percent.model`) towards
MACE-OFF23-small. Per iteration: J = 30 particles, 300 outer steps × 500 MD
steps, 100 configurations labelled, LoRA fine-tuning.

| Arm | Launch script | Settings |
|---|---|---|
| Langevin 300 K / 700 K | `run_langevin_{300k,700k}_seeds.sh` | underdamped, `--a 0`, FPS |
| Overdamped 50 K | `run_baseline_seeds.sh` | `--a 0`, FPS |
| SKMD | `run_skmd_offline_seeds.sh` | overdamped 50 K, a = 3.0, FPS |
| a-SKMD | `run_askmd_seeds.sh` | a = 3.0, ζ₀ = 10th percentile, ℓ₀ = 250 |
| uHAL | `run_uhal_seeds.sh` | ase_uhal, 700 K, 10-member committee |

```bash
./run_<arm>_seeds.sh NUM_ITERS N_SEEDS --n-select 100     # paper: 10 5
```

Each iteration is one SLURM job (`slurm/example/iteration*.slurm`) that samples,
labels, fine-tunes, evaluates and submits the next iteration. Outputs go to
`data/alanine_dipeptide/<arm>_seed_<N>/`. Resume with the loop driver's
`--start-from K`.

Test set (on Zenodo; to regenerate):

```bash
python setup/generate_test_set.py --init_xyz ../../data/alanine_dipeptide/ace-ala-nme.xyz \
    --oracle_model ../../data/alanine_dipeptide/MACE-OFF23_small.model \
    --output ../../data/alanine_dipeptide/test_set.xyz
```
