# Biodiesel with Xavier initialization — Figure 5b

One variable changed: how the weights are drawn before training. Everything else — data,
156-parameter architecture, Adam at `lr = 0.002`, Tsit5 `rtol=1e-6`/`atol=1e-8`, direct
autograd, 10,000 epochs, seed 0, train-only min-max normalization — is identical to the
committed runs, and the notebook asserts that field by field.

## What Xavier changes

With the base activation off, the kinetic core has exactly two trainable tensors, both RBF
coefficient grids. The default draw is `randn(...) * 0.1` — a fixed standard deviation
that ignores layer shape (`chemkan/src/chemkan/kan/rbf.py:57`). Xavier scales by fan-in and
fan-out instead: `U(-a, a)` with `a = sqrt(6 / (fan_in + fan_out))`.

| parameter | shape | fan_in | fan_out | default std | Xavier bound | Xavier std | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `add.edges.w_rbf` | (4, 7, 3) | 21 | 12 | 0.1013 | 0.4264 | 0.2364 | 2.33× |
| `lean.edges.w_rbf` | (6, 4, 3) | 12 | 18 | 0.0960 | 0.4472 | 0.2496 | 2.60× |

Parameter count is unchanged at 156. There are no bias vectors, so nothing is left at its
default. Epoch-0 loss rises from `1276.571044921875` to `6075.73876953125`.

## Result — Xavier did not help

Median over the **last 20 % of epochs** (the primary comparison; see below for why):

| noise | train ratio | clean-test ratio |
|---:|---:|---:|
| 0 % | 1.847 | 1.381 |
| 2 % | 1.785 | 1.391 |
| 7 % | 1.475 | 1.350 |
| 15 % | 1.177 | 1.431 |

Ratios are Xavier / default, so **> 1 is worse**. Xavier is worse at **every** noise level
on both training and noise-free test loss, median ratios 1.630 and 1.386. The figures show
why: the wider initial weights cost a long early phase in which Xavier tracks well above
the default curve, only partly recovered by 10,000 epochs.

Neither initialization meets the repository's overfitting criterion at any noise level.

### Why the final checkpoint alone is not the verdict

These runs oscillate strongly late in training. The default run's own epoch-to-epoch swing
over the last 20 % of epochs is 0.10–0.19, while the gap between the two initializations at
the final checkpoint is only 0.004–0.028 — **0.04× to 0.26× the oscillation span**. A
single-checkpoint comparison therefore mostly measures where each run happened to stop.
Read at the final checkpoint alone, Xavier appears better on training loss at 0 % noise
(0.0576 vs 0.0620); that difference is 0.04× the span and is not meaningful.
`tables/xavier_late_window_comparison.csv` carries both readings and the ratio.

## Scope

Single seed. Initialization changes the entire optimization trajectory, so this shows
Xavier did not help in this configuration; it does not establish a general ordering, and it
is not a claim about the paper, which does not specify an initialization scheme.

## Reproducing (from the repository root)

```bash
PY=~/uni_projects/chemkan-venv/bin/python
EXP=results/experiments/biodiesel_xavier_init

$PY chemkan/scripts/train_biodiesel.py --seed 0 --epochs 10000 --eval-every 1 \
    --init xavier --experiment-name xavier_init --run-dir $EXP/clean_seed0

for pct in 2 7 15; do
  $PY chemkan/scripts/train_biodiesel.py --seed 0 --epochs 10000 --eval-every 1 \
      --noise-percent $pct --init xavier --experiment-name xavier_init \
      --run-dir $EXP/noise$(printf %02d $pct)_seed0
done
```

`--eval-every 1` is what produces the per-epoch noise-free test column Figure 5b needs.
The default-init curves are the existing committed runs and were **not** retrained:
`results/reproduction/chemkan/biodiesel/noise/clean_replay_seed0` for the 0 % panel (the
committed figure's own source, bitwise identical to the main run) and
`.../noise/noise{02,07,15}_seed0` for the rest.

## The `--init` flag

Added to the unchanged trainer `chemkan/scripts/train_biodiesel.py`, alongside the
`--use-base-act` and `--input-scaling` ablations it already carried. `--init default` draws
nothing extra from the RNG, so unflagged runs remain bit-identical to everything trained
before the flag existed — verified against the committed run at epochs 0/1/2
(`1276.571044921875`, `1099.385498046875`, `934.4246826171875`). The choice is recorded in
`config.json` and in the checkpoint provenance. Applying Xavier to a parameter with fewer
than two dimensions has no defined fan, so the helper raises rather than inventing a policy;
this architecture has no such parameter.

## Layout

```
clean_seed0/  noise{02,07,15}_seed0/   the four Xavier runs
figures/fig05b_xavier_vs_default_chemkan.{pdf,png}   Xavier vs default ChemKAN
figures/fig05b_xavier_three_way.{pdf,png}            + DeepONet, committed layout
tables/xavier_init_statistics.csv                    fan/std table above
tables/xavier_final_losses.csv                       final-checkpoint losses
tables/xavier_late_window_comparison.csv             late-window medians + oscillation
tables/xavier_fig5b_overfit_assessment.json          both initializations
```

Analysis: `chemkan/notebooks/12_biodiesel_xavier_init.ipynb`.
