# Biodiesel with the KAN-ODE RBF kernel (clean data, seed 0)

`clean_seed0/` was trained with `chemkan/scripts/train_biodiesel.py` at its paper defaults
(156 parameters, clean data, seed 0, 10,000 epochs, `--eval-every 100`) while
`chemkan/src/chemkan/kan/_common.py` was **temporarily** changed from the repository's RBF
kernel `exp(-r² / (2h²))` to the KAN-ODE reference implementation's `exp(-(r/h)²)` — the
same Gaussian with its width divided by √2. The edit was reverted after the run.

**The kernel is not recorded** in `config.json` or in the checkpoint, and `config.json`
names the clean commit `b4be13a`. Reloading `checkpoint_final.pt` with the repository's
kernel therefore builds the wrong model. Checked on 2026-09-11:

| kernel used to reload `checkpoint_final.pt` | train MSE | test MSE |
|---|---:|---:|
| KAN-ODE `exp(-(r/h)²)` | 0.014882 | 0.038514 |
| repository `exp(-r² / (2h²))` | 55.808281 | 53.073536 |
| `metrics.json`, written at the end of the run | 0.014882 | 0.038514 |

`history.csv` and `metrics.json` were written during the run with the KAN-ODE kernel in
place and are valid as recorded. Any re-evaluation of this checkpoint must use the KAN-ODE
kernel.
