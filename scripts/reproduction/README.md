# Per-figure reproduction scripts

One script per paper result. Each drives the whole chain for that result — data →
training → evaluation → artifacts → figure — and finishes by printing the verdict for
that figure from `results/reproduction/tables/reproduction_comparison.csv`.

```
scripts/reproduction/
├── all.sh                             everything, biodiesel then hydrogen
├── _common.sh                         shared helpers (sourced, not executed)
├── biodiesel/
│   ├── 00_data.sh                     3 % noise level + Figure-3 condition
│   ├── _runs.sh                       runs shared by Figs. 3/5A/5B/6 (sourced)
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

./scripts/reproduction/biodiesel/fig05a_noise_robustness.sh    # one figure
./scripts/reproduction/hydrogen/all.sh                         # all hydrogen results
./scripts/reproduction/all.sh --dry-run                        # plan only, change nothing
```

Flags: `--dry-run` prints the plan and touches nothing; `--no-render` skips notebook
execution. `DRY_RUN=1` / `NO_RENDER=1` work as environment variables. Scripts run from any
working directory.

## What these guarantee

**Idempotent.** A run with `checkpoint_final.pt` is skipped — never retrained, never
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
its owning notebook at the end — 15 s for 07, 50 s for 08 — so there is no second copy of
the plotting code to drift out of sync. `--no-render` skips that step when you only want
the artifacts. Figure 4 is the exception: `assemble_fig4_scaling.py` renders it directly,
because its fit masks and per-point oscillation bands are computed there.

## Cost

| script | new training | wall clock on an idle laptop |
|---|---|---|
| `fig03` | 3 ChemKAN runs (shared) | ~35 min from scratch |
| `fig05a` | 7 ChemKAN + 8 DeepONet | ~1.6 h from scratch |
| `fig05b` | clean replay + the 2/7/15 % pairs | ~35 min beyond `fig05a` |
| `fig06` | none beyond the 15 % pair | seconds if `fig05a` has run |
| `fig04` | 4 ChemKAN + 6 DeepONet | ~25 min |
| all hydrogen | **none** | ~2 min |

With every run already present, `all.sh` re-evaluates and re-renders in a few minutes.

Run `table1_efficiency.sh` on an otherwise idle machine — it is a wall-clock measurement,
and a loaded CPU makes it meaningless.
