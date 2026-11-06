# SKMD parameters

Flags of `systems/*/skmd_run/run_skmd_mace.py` (full list: `--help`). The
`run_al_loop*.sh` drivers set the paper values.

| Flag | Meaning | Alanine | Si(111) |
|---|---|---|---|
| `--nsteps` | outer-loop steps | 300 | 200 |
| `--n_run` | MD steps per outer step | 500 | 500 |
| `--a` | kernel amplitude (0 = unbiased MD) | 3.0 | 4.0 |
| `--relax_start` | end of exploration; b frozen here | 250 (SKMD), 50 (a-SKMD) | 160 (SKMD), 50 (a-SKMD) |
| `--a_relax_mult`, `--b_relax_mult` | a, b decay linearly to these fractions after `relax_start` | 1.0 | 0.5 (SKMD) |
| `--dyn_type`, `--T`, `--damping` | dynamics | overdamped, 50 K, 75 | same |
| `--b_mode` | `adaptive-median` (paper), `fixed_user`, `fixed_burnin` | | |
| `--b_max_mult` | upper clamp on adaptive b, × initial median | 3.0 (SKMD), 8.0 (a-SKMD) | 15.0 |
| `--n_burnin_md`, `--every_burnin` | burn-in length and snapshot interval; J = ratio | 6000, 200 | same |

a-SKMD (online acquisition) is enabled by `--target_samples N`. After
exploration, ζ₀ (`--thresh`) is set to the `--thresh_percentile`-th (10)
percentile of recent ‖F^SKMD‖; each particle then stops and is acquired once
‖F^SKMD‖ < ζ₀ after at least `--buffer` (ℓ₀ = 250) steps.

Offline arms select N configurations by farthest point sampling in descriptor
space (`skmd_run/draw_samples.py`).
