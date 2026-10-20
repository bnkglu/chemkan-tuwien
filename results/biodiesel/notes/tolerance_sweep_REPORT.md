# Evaluation-tolerance sweep, biodiesel_v2 Fig. 3 runs (read-only diagnostic)

Same checkpoints, data, initial conditions, model and normalization; only the evaluation
`rtol`/`atol` change. Script: `tolerance_sweep.py` (CPU, 1 thread, `torch.no_grad`, no
backward, nothing written outside this directory). Repository at commit 379d92c.

## Setup

- Figure 3 route: `fig03_biodiesel_trajectories.author_trajectory_data` ->
  `author_runs.chemkan_predict`: `chemkan.solver.integrate` (torchdiffeq `odeint`, Tsit5),
  state-only, one trajectory per solve, rtol/atol read from each run's `config.json`
  (the training values, rtol 1e-2, atol 1e-6), RHS / 50.
- Checkpoints: `figures/fig05/chemkan_noise{00,05,10,15}_seed0/checkpoint_final.pt`.
- Fig. 3 condition: test row 24 (`FIG3_TEST_INDEX = 3`), TG 1.941, ROH 1.433, T 334.75 K.
- Losses: time-averaged MSE in the runs' normalized species coordinates (`Flux.mse`
  convention, mean over 6 species and 30 times); Eq. 18 (sum over time) = 30 x.
- Controls (`controls.json`): at the stored tolerance this script reproduces
  `chemkan_predict` exactly (max abs diff 0.0); parameter and checkpoint-file sha256 are
  identical before and after for all four checkpoints.

## Fig. 3 condition (`sweep_fig3.csv`)

Clean MSE (time-averaged); max |diff| vs E on the 301-point dense grid, normalized units.

| noise | A rtol 1e-2 | B 1e-4 | C 1e-6 | D 1e-8 | E 1e-10 | max diff A vs E | max diff B / C vs E |
|---|---|---|---|---|---|---|---|
| 0 % | 7.24e-5 | 8.12e-5 | 8.13e-5 | 8.14e-5 | 8.14e-5 | 5.3e-3 | 7.6e-5 / 1.7e-6 |
| 5 % | 6.64e-5 | 6.93e-5 | 6.95e-5 | 6.95e-5 | 6.95e-5 | 3.2e-3 | 8.0e-5 / 1.7e-6 |
| 10 % | **2.56e-3** | 1.54e-4 | 1.54e-4 | 1.54e-4 | **1.54e-4** | **1.28e-1** | 9.9e-5 / 1.3e-6 |
| 15 % | **6.95e-4** | 3.44e-4 | 3.44e-4 | 3.44e-4 | **3.44e-4** | **4.31e-2** | 3.7e-5 / 1.0e-6 |

RHS evaluations on the dense grid: A ~50, B ~70, C ~125, D ~235, E ~540.

## All 10 test trajectories (`sweep_test.csv`), clean MSE, time-averaged

| noise | rtol | mean | median | worst | Fig. 3 row |
|---|---|---|---|---|---|
| 0 % | 1e-2 / 1e-6 / 1e-8 / 1e-10 | 3.15e-4 / 3.17e-4 / 3.17e-4 / 3.17e-4 | 1.62e-4 (all) | 9.71e-4 / 9.50e-4 (C-E) | 7.24e-5 / 8.13e-5 |
| 5 % | same | 2.62e-4 / 2.67e-4 (C-E) | 1.54e-4 / 1.51e-4 | 8.40e-4 / 8.62e-4 | 6.64e-5 / 6.95e-5 |
| 10 % | same | **5.15e-4 / 2.67e-4** (C-E) | 2.74e-4 / 2.19e-4 | **2.56e-3 / 4.61e-4** | **2.56e-3 / 1.54e-4** |
| 15 % | same | 5.73e-4 / 5.82e-4 (C-E) | 5.71e-4 / 4.34e-4 | 1.32e-3 / 1.67e-3 | 6.95e-4 / 3.44e-4 |

The rtol 1e-2 and 1e-10 means equal the trainers' own `metrics.json` final evaluations
(e.g. 15 %: 5.72e-4 and 5.82e-4).

## Reading

- Convergence criterion: another 100x tightening changes the trajectory by < 1e-5
  (normalized) and the MSE by < 0.1 %. Met from **rtol 1e-6 (C)** on for all four
  checkpoints; rtol 1e-4 already fixes the MSE to 3 digits (trajectory change ~1e-4).
- rtol 1e-2 is not converged for any checkpoint. For 10 % (Fig. 3 row) the loose solve
  bends ROH / GL / RCO2R near t = 10-15 and is 17x worse than the converged prediction;
  for 15 % it adds small kinks (MG, DG, RCO2R near t = 13-17) and doubles the MSE. These
  kinks are integration error, not the learned model.
- Tighter is not always lower: at 0 % / 5 % the loose solve happens to score slightly
  better (7.24e-5 vs 8.14e-5).
- What remains at convergence is learned-model error: the 15 % model's TG falls too fast
  early (t < 10) and GL dips slightly below 0 near t = 2; the 10 % model tracks the truth.
- One seed per noise level; Fig. 3 is one condition (the 10 % worst-case test row is this
  row).
