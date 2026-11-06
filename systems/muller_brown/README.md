# Müller–Brown (Sec. 5.1, Figs. 1–2, App. E.1)

Active learning of a small neural network potential against the Müller–Brown
potential. Pure PyTorch, one CPU core.

```bash
for i in $(seq 1 50); do
  python run_al_trial_lang.py --trial-id $i                          # overdamped Langevin
  python run_al_trial_udd.py  --trial-id $i                          # UDD
  python run_al_trial_skmd.py --trial-id $i                          # SKMD
  python run_al_trial_skmd.py --trial-id $i --trigger stein-force    # a-SKMD
done
```

Inputs (`model0.torch`, `dataset0.torch`, `dataset_test_1000.torch`) are read
from `data/muller_brown/` (`--data-dir`). Each trial writes its models, test
RMSEs (`test_error.npz`), runtime and memory, and per-iteration plots to
`results/`. See `--help` for the sampler parameters.
