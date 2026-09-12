# Biodiesel: observed-state interval training (experiment)

An **additive** experiment. Nothing in the existing reproduction was changed: no source
file, script, notebook, dataset, checkpoint, history, figure, table, or document outside
this directory was modified, renamed, or reinterpreted. `chemkan/scripts/train_biodiesel.py`
remains the original trainer and gained no mode switch.

## What is different

| | original trainer | this experiment |
|---|---|---|
| solve per update | one, `t[0] → t[-1]`, started at the observed state at `t = 0` | one **per interval**, `t[j] → t[j+1]`, each started at the **observed** state at `t[j]` |
| supervision | the whole predicted trajectory | each interval's **endpoint** only |
| state carried forward | yes — the model's own prediction | no — every interval restarts from its observation |
| optimizer updates | one per epoch | one per epoch (gradients from all intervals accumulate first) |
| loss formula | `chemkan.losses.trajectory_mse` (Eq. 18) | `chemkan.losses.trajectory_mse` (Eq. 18), **unchanged** |

The error formula is identical. `trajectory_mse` is called once per interval on `(1, B, m)`
tensors — mean over species, mean over trajectories — and the per-interval values are
summed over the endpoints. No new loss, penalty, weight, derivative target, or division by
the interval count was introduced. The objective differs from full-trajectory training
only because its predictions start from fresh observations.

**Provenance.** Restarting from observed states at each interval is an
additional experiment, not a procedure the ChemKAN paper describes
(Sec. II C 3–5, III A 1, Eq. 18). Nothing here claims it is a proven correction of the
reproduction, nor the authors' undocumented procedure.

## Settings (verified against the checkout, not assumed)

156 parameters, hidden width 4, three RBF bases, `n_mu = 2`, base activation off,
train-only min-max scaling, Adam at `lr = 0.002`, Tsit5 with `rtol = 1e-6` / `atol = 1e-8`,
direct autograd, 10,000 optimizer updates, clean biodiesel only. Identical for both
procedures in every pair.

**Time grid.** `biodiesel.npz` stores 30 saved states including `t = 0` and `t = 30 s`,
hence **29 intervals** at `dt ≈ 1.03448 s`. The paper states 1 s sampling. The archive is
used **as is**; no data was regenerated. All counts are derived from the loaded data.

**Paired initialization.** Both scripts seed the global RNG and then construct the same
`KineticCore`, so the two runs of a seed start from identical parameter tensors with
independent optimizer states. The first row of each run's `history.csv` shows it: the
observed-interval run's update-0 full-rollout loss equals the original run's epoch-0 loss.

## Reproducing the runs (from the repository root)

```bash
PY=~/uni_projects/chemkan-venv/bin/python
EXP=results/experiments/biodiesel_observed_intervals

# observed-interval runs
for s in 0 1 2; do
  $PY chemkan/scripts/train_biodiesel_observed_intervals.py \
      --seed $s --epochs 10000 --eval-every 100 --snapshot-epochs 5000 \
      --run-dir $EXP/seed$s
done

# paired original-training runs, produced by the UNCHANGED trainer into fresh directories
for s in 1 2; do
  $PY chemkan/scripts/train_biodiesel.py \
      --seed $s --epochs 10000 --eval-every 100 --snapshot-epochs 5000 \
      --experiment-name paired_baseline \
      --run-dir $EXP/baseline_original_seed$s
done

# focused checks
cd chemkan && $PY -m pytest tests/test_observed_interval_training.py -q
```

Both scripts refuse to replace a completed run: `checkpoint_final.pt` means finished, and
`checkpoint_resume.pt` means an interrupted run that `--resume` can continue. The
experiment script has no `--overwrite` flag at all.

**Seed 0's original-training arm is reused, not re-run.** It is
`results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0` — same data, same
settings, and its initialization was confirmed to reproduce bit-exactly under the current
checkout (epoch-0 loss `1276.571044921875`). Notebook 11 reads it through
`results/reproduction/chemkan/biodiesel/noise/clean_replay_seed0`, a re-run of the same
configuration whose final weights are identical to the main run's and which also logged the
test loss every epoch.

## Layout

```
seed{0,1,2}/                     observed-interval runs
baseline_original_seed{1,2}/     paired runs from the unchanged original trainer
figures/                         comparison figures written by notebook 11
tables/                          comparison tables written by notebook 11
RESULTS.md                       measured outcome and interpretation
```

Each run directory holds `config.json`, `run.log`, `history.csv`,
`checkpoint_epoch_5000.pt`, and `checkpoint_final.pt`.

`history.csv` for an observed-interval run has columns
`epoch, total_loss, full_rollout_train_mse, full_rollout_test_mse_clean, eval_seconds,
elapsed_seconds`, where `total_loss` is the **observed-interval objective** and the two
`full_rollout_*` columns are the comparable quantity: a rollout from the original initial
conditions over the whole grid, evaluated at the parameter state that produced that row's
objective. The final row, at `epoch == 10000`, is the state of the saved checkpoint.

## Analysis

`chemkan/notebooks/11_biodiesel_observed_intervals.ipynb` — all comparison numbers,
figures, and tables. See `RESULTS.md` for the outcome.

Additional figures focusing on the **interval objective** are described in
[INTERVAL_OBJECTIVE_FIGURES.md](INTERVAL_OBJECTIVE_FIGURES.md): convergence and species
errors. They are generated by
`chemkan/scripts/figures/plot_biodiesel_interval_objective.py`, which reconstructs the
saved RBF widths and checks the recomputed losses against the archived results.
