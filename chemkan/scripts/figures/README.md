# Paper figure scripts

One script per paper figure. Each is the **single implementation** of that figure: the
notebooks import these functions and supply paths, and no figure is drawn anywhere else.

**No figure script trains anything.** They load finished checkpoints, histories and
tables, integrate an already-trained model where the figure needs a curve, and compute
that figure's metric.

## Commands

Run from the repository root with the project environment.

| Figure | Command | Reads |
|---|---|---|
| Figure 3 — biodiesel trajectories under noise | `python chemkan/scripts/figures/fig03_biodiesel_trajectories.py` | ChemKAN Fig. 5 runs |
| Figure 4 — biodiesel neural scaling | `python chemkan/scripts/figures/fig04_biodiesel_scaling.py` | Fig. 4 runs (metrics) |
| Figure 5A — noise robustness | `python chemkan/scripts/figures/fig05_biodiesel_noise.py` | Fig. 5 runs (final checkpoints) |
| Figure 5B — loss dynamics | `python chemkan/scripts/figures/fig05_biodiesel_loss.py` | Fig. 5 runs (histories) |
| Figure 6 — 15 % noise profiles | `python chemkan/scripts/figures/fig06_biodiesel_profiles.py` | Fig. 5 runs, 15 % noise |
| Figure 7 — hydrogen trajectories | `python chemkan/scripts/figures/fig07_hydrogen_trajectories.py` | hydrogen 200k Stage-2 checkpoint |
| Figure 8A — generalization map | `python chemkan/scripts/figures/fig08_hydrogen_generalization.py` | `evaluate_hydrogen_grid.py` table |
| Figure 8B — ignition delay | `python chemkan/scripts/figures/fig08_hydrogen_ignition.py` | `evaluate_hydrogen_ignition.py` table |
| **All of them** | `python chemkan/scripts/figures/plot_all.py` | |

The biodiesel figures read the biodiesel_v2 figure runs (ChemKAN:
`chemkan/scripts/author_repo_match/run_figures.py`; DeepONet: `deeponet/run_figure_runs.sh`)
and write to `results/experiments/biodiesel/author_repo_match/figures/plots/`. Their losses
are time-averaged MSE (Eq. 18 / 30); they take no options.

The hydrogen figures default to the main run's 200k Stage-2 checkpoint and its tables
(notebook 08b's arguments) and write to `results/experiments/hydrogen/author_matched/main/`
`figures/` and `tables/`. `--run LABEL=DIR` (Fig. 7) or `--table LABEL=CSV` (Figs. 8A, 8B)
plots other runs instead; `--output-dir` / `--output` writes elsewhere; `--summed` (Figs. 7,
8A) shows the Eq. 18 loss summed over the time points instead of the time-averaged one;
Fig. 7 also takes `--rtol`, `--atol` and `--float64` for its integration. `--help` lists
each script's options.

`plot_all.py --only fig07 fig08a` regenerates a subset; `--keep-going` continues after a
failure and reports at the end.

## Using them from a notebook

Import the figure function and pass the paths for the run you care about. The notebook
chooses *which run*; the script decides *how the figure is made*.

```python
import sys; sys.path.insert(0, str(ROOT / "chemkan" / "scripts" / "figures"))
import fig07_hydrogen_trajectories as fig07

figs, results = fig07.make_figure(run_dirs={"my_run": RUN_DIR}, output_dir=OUT, show=True)
```

The biodiesel scripts expose `make_author_figure(output_path=None, show=False)`; the
hydrogen ones `make_figure(...)` with run or table arguments as above. Each returns the
figure(s) together with the numbers it computed, so a notebook can interpret them without
recomputing anything. To point a figure at a different run, change paths — never copy
plotting code into the notebook.

## Files

```
common.py                            repo paths, checkpoint loading, figure saving
author_runs.py                       the biodiesel figure runs: plan, histories, predictions, losses
author_paper_table.py                paper-vs-ours comparison table for Figs. 4 and 5
fig03_biodiesel_trajectories.py      Figure 3
fig04_biodiesel_scaling.py           Figure 4
fig05_biodiesel_noise.py             Figure 5A (+ the palette shared with 5B)
fig05_biodiesel_loss.py              Figure 5B
fig06_biodiesel_profiles.py          Figure 6
fig07_hydrogen_trajectories.py       Figure 7 (+ the true-scale companion)
fig08_hydrogen_generalization.py     Figure 8A (paper scale and full range)
fig08_hydrogen_ignition.py           Figure 8B
plot_all.py                          regenerates every figure above
```

`common.py` is deliberately small: paths, `load_checkpoint`, `save_figure`,
`require_file`, `relative_to_root`, `loss_reduction` and `use_headless_backend`.
Figure-specific science stays in the figure's own file.

### Why `use_headless_backend()` exists

Called from `main()` only, never on import. With `bbox_inches="tight"` the crop comes from
the renderer's text metrics, and macOS's native backend rounds it one pixel differently
from Agg — so without it a command-line run produces a PNG one pixel wider than the
notebook's. Importing a figure script inside a notebook must not change the notebook's
backend, which is why it is not called at import time.
