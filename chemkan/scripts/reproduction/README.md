# Per-figure reproduction scripts

> **Plotting has moved.** Every paper figure is now produced by a single readable Python
> script under [`chemkan/scripts/figures/`](../figures/README.md); the notebooks import
> those functions and supply run paths. The shell scripts in this directory still own
> *training and data preparation* and then call the notebooks, which is unchanged.
>
> To regenerate a figure without training anything:
>
> ```bash
> python chemkan/scripts/figures/fig03_biodiesel_trajectories.py     # Figure 3
> python chemkan/scripts/figures/fig04_biodiesel_scaling.py          # Figure 4
> python chemkan/scripts/figures/fig05_biodiesel_noise.py            # Figure 5A
> python chemkan/scripts/figures/fig05_biodiesel_loss.py             # Figure 5B
> python chemkan/scripts/figures/fig06_biodiesel_profiles.py         # Figure 6
> python chemkan/scripts/figures/fig07_hydrogen_trajectories.py      # Figure 7
> python chemkan/scripts/figures/fig08_hydrogen_generalization.py    # Figure 8A
> python chemkan/scripts/figures/fig08_hydrogen_ignition.py          # Figure 8B
> python chemkan/scripts/figures/plot_all.py                         # all of them
> ```
>
> A notebook plots another run by importing the same function and changing paths:
>
> ```python
> from fig05_biodiesel_loss import make_figure
> fig, results = make_figure(chemkan_runs={0: RUN_DIR}, output_path=RUN_DIR / "fig05b")
> ```
>
> There is no second plotting implementation in the notebooks.

Notebook 07 and the figure wrappers now use `reference_final_trunk_relu` DeepONet
checkpoints. All 14 corrected runs and the five fixed-`n_mu=2` ChemKAN points are
verified and plotted. Legacy checkpoints remain untouched; previous reports are
labelled under `legacy_final_trunk_linear/` in the figures and tables directories.

## Completed biodiesel runs

Run from the repository root, using the project Python environment:

```bash
export CHEMKAN_PYTHON="$HOME/uni_projects/chemkan-venv/bin/python3"
bash chemkan/scripts/reproduction/biodiesel/deeponet_reference.sh
bash chemkan/scripts/reproduction/biodiesel/fig04_nmu2.sh
```

Append `--dry-run` to inspect either plan without writing files or starting training.
Both commands support `--seed`, `--device`, `--output-root`, and `--help`.

- **DeepONet:** eight noise runs (0/1/2/3/5/7/10/15%, 10,000 epochs) and six scaling
  widths (3/5/6/8/10/13, 50,000 epochs). Only `reference_final_trunk_relu` is trained.
  Adam, lr=1e-3, seed 0, data, scaling and Eq. 18 loss are preserved. Output root:
  `results/reproduction/baselines/deeponet/biodiesel/reference_final_trunk_relu/`.
  Use `--only noise` or `--only scaling` to select a subset; `--noise-epochs` and
  `--scaling-epochs` change the budgets for a separately labelled experiment.
- **Fixed n_mu=2:** widths 2/3/4/10/17, 5,000 epochs, matching the Figure-4 ChemKAN
  budget. At the default seed/device, h=3 reuses the completed 5,000-epoch run and h=4
  reuses the clean replay's epoch-5,000 snapshot. Only h=2/10/17 need new training.
  New runs go to `results/reproduction/chemkan/biodiesel/scaling_nmu2/`.
  `--epochs 10000 --output-root <another-directory>` creates a separate longer-budget
  experiment; it does not reuse the 5,000-step checkpoints.

The runner validates the whole plan before launching any job. It compares the full
config except timestamps, git/run identifiers and runtime device, checks checkpoint
architecture, steps, normalization and parameter count, and refuses mismatches. Reuse
of h=3/h=4 additionally permits a different experiment name and evaluation-only logging;
the snapshot's **actual** step replaces its enclosing run's 10,000-epoch budget. A
completed matching result is skipped; an interrupted matching run is resumed. Files in
an incomplete directory without a resume checkpoint are preserved and reported.

Each successful command writes `manifest_*_seed0.json` under its output root, with
checkpoint paths, configs, and checkpoint/data hashes. Reused checkpoints remain in
their original directories. Notebook 07 validates these manifests before rebuilding
corrected Figures 4/5A/5B/6 and the additional fixed-`n_mu=2` Figure 4 comparison.
`biodiesel_completed_run_audit.csv` records the validated sources and training times. These two commands perform
training only; they do not render notebooks or run hydrogen.

## Existing figure wrappers

One wrapper per paper result, ending with the corresponding row in
`results/reproduction/tables/reproduction_comparison.csv`. Biodiesel DeepONet results
rendered by these wrappers use **reference_final_trunk_relu**. Missing corrected
checkpoints produce an instruction to run `deeponet_reference.sh`.

```
chemkan/scripts/reproduction/
├── all.sh                             everything, biodiesel then hydrogen
├── _common.sh                         shared helpers (sourced, not executed)
├── biodiesel/
│   ├── 00_data.sh                     3 % noise level + Figure-3 condition
│   ├── _runs.sh                       runs shared by Figs. 3/5A/5B/6 (sourced)
│   ├── deeponet_reference.sh          corrected DeepONet only: 8 noise + 6 scaling runs
│   ├── fig04_nmu2.sh                  fixed-n_mu=2 comparison data: 3 new + 2 reused
│   ├── fig03_trajectories.sh          Fig. 3   noise columns on the published condition
│   ├── fig04_neural_scaling.sh        Fig. 4   width sweep + paper digitization
│   ├── fig05a_noise_robustness.sh     Fig. 5A  three metrics × 8 noise levels × 2 models
│   ├── fig05b_loss_dynamics.sh        Fig. 5B  per-epoch loss curves
│   ├── fig06_15pct_profiles.sh        Fig. 6   15 % profiles, no new training
│   └── all.sh
└── hydrogen/
    ├── 00_data.sh                     441-condition fine grid
    ├── _checkpoints.sh                checkpoint guard (sourced)
    ├── fig07_trajectories.sh          Fig. 7   published conditions
    ├── fig08a_generalization.sh       Fig. 8A  441-condition map
    ├── fig08b_ignition_delay.sh       Fig. 8B  ignition delay
    ├── table1_efficiency.sh           Table I  counts + local inference benchmark
    └── all.sh
```

## Running

```bash
export CHEMKAN_PYTHON=~/uni_projects/chemkan-venv/bin/python   # needs torch + cantera

./chemkan/scripts/reproduction/biodiesel/fig05a_noise_robustness.sh    # one figure
./chemkan/scripts/reproduction/hydrogen/all.sh                         # all hydrogen results
./chemkan/scripts/reproduction/all.sh --dry-run                        # plan only, change nothing
```

Flags: `--dry-run` prints the plan and touches nothing; `--no-render` skips notebook
execution. `DRY_RUN=1` / `NO_RENDER=1` work as environment variables. Scripts run from any
working directory.

## What these guarantee

For the older per-figure wrappers, a run with `checkpoint_final.pt` is skipped — never retrained, never
overwritten; a completed run is a result, not a cache. A run with only
`checkpoint_resume.pt` is resumed. No script here passes `--overwrite`. Re-running a
figure after an interruption is safe, and running several figures never trains a shared
model twice.

**No hydrogen training.** Figures 7, 8A, 8B and Table I are evaluated from completed
checkpoints. If a checkpoint is missing the script **fails with a pointer to the workflow
doc** rather than silently starting a multi-hour two-stage run. `train_hydrogen.py` is
never invoked from this directory.

**`B0` is reused, never retrained.** It supplies the 0 % ChemKAN point for Figures 3 and
5A. The separate clean replay exists only to give Figure 5B's 0 % panel the per-epoch
test history `B0`'s own history lacks; it reproduces `B0` bitwise, so that panel plots
`B0`'s trajectory rather than a substitute.

**Evaluation ≠ agreement.** Every script ends with the row for its figure from the
comparison table, which keeps *evaluation completed* separate from *paper result matched*.
Of 20 compared results, 1 is matched.

## Where the plots come from

The figures are rendered by the notebooks (`chemkan/notebooks/07_*.ipynb`,
`08_*.ipynb`), which stay the single source of the plotting code. Each script executes
its owning notebook at the end, so there is no second copy of
the plotting code to drift out of sync. `--no-render` skips that step when you only want
the artifacts. Figure 4 is the exception: `assemble_fig4_scaling.py` renders it directly,
because its fit masks and per-point oscillation bands are computed there.

## Refreshing completed results

The training commands above skip compatible completed runs. To validate all 19 points
and rebuild the biodiesel tables and both Figure-4 plots without training:

```bash
"$CHEMKAN_PYTHON" chemkan/scripts/diagnostics/refresh_biodiesel_reports.py
```

Execute Notebook 07 to refresh its inline outputs and Figures 5A/5B/6 too. Training
times are stored in each run's history and summarized in
`results/reproduction/tables/biodiesel_completed_run_audit.csv`; rendering time is separate.

Run `table1_efficiency.sh` on an otherwise idle machine — it is a wall-clock measurement,
and a loaded CPU makes it meaningless.
