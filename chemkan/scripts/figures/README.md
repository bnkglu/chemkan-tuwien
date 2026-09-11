# Paper figure scripts

One script per paper figure. Each is the **single implementation** of that figure: the
notebooks import these functions and supply paths, and no figure is drawn anywhere else.

Open a script to understand a figure — each reads top to bottom as
load → validate → evaluate → plot → `make_figure` → `main`.

**No figure script trains anything.** They load finished checkpoints, histories and
tables, integrate an already-trained model where the figure needs a curve, and compute
that figure's metric.

## Commands

The clean observed-interval experiment uses these same entry points:

```bash
python chemkan/scripts/figures/fig03_biodiesel_trajectories.py --observed-intervals
python chemkan/scripts/figures/fig05_biodiesel_loss.py --observed-intervals
```

They write `fig3` and `fig5b` (PDF/PNG) inside the experiment's figures directory.
The interval-trained procedure is explicit in each title; Figure 5B uses full-rollout
losses, not the local interval objective. The all-in-one shell wrapper is
`chemkan/scripts/reproduction/biodiesel/observed_intervals_figures.sh`.
The extra interval diagnostics are documented in
`results/experiments/biodiesel_observed_intervals/INTERVAL_OBJECTIVE_FIGURES.md`.

Run from the repository root, with the project environment
(`~/uni_projects/chemkan-venv/bin/python`, shown as `python` below).

| Figure | Command |
|---|---|
| Figure 3 — biodiesel trajectories under noise | `python chemkan/scripts/figures/fig03_biodiesel_trajectories.py` |
| Figure 4 — biodiesel neural scaling | `python chemkan/scripts/figures/fig04_biodiesel_scaling.py` |
| Figure 4 — fixed `n_mu=2` variant | `python chemkan/scripts/figures/fig04_biodiesel_scaling.py --n-mu 2` |
| Figure 5A — noise robustness | `python chemkan/scripts/figures/fig05_biodiesel_noise.py` |
| Figure 5B — loss dynamics | `python chemkan/scripts/figures/fig05_biodiesel_loss.py` |
| Figure 6 — 15 % noise profiles | `python chemkan/scripts/figures/fig06_biodiesel_profiles.py` |
| Figure 7 — hydrogen trajectories | `python chemkan/scripts/figures/fig07_hydrogen_trajectories.py` |
| Figure 8A — generalization map | `python chemkan/scripts/figures/fig08_hydrogen_generalization.py` |
| Figure 8B — ignition delay | `python chemkan/scripts/figures/fig08_hydrogen_ignition.py` |
| **All of them** | `python chemkan/scripts/figures/plot_all.py` |

`plot_all.py --only fig07 fig08a` regenerates a subset. Each script also takes
`--output` (or `--output-dir`) to write elsewhere; `--help` lists the rest.

## The two loss reductions (`--time-averaged`)

Eq. 18 as optimized **sums** the per-time squared error over the `N_t` observation times.
Dividing by `N_t` gives the conventional **time-averaged** MSE, which is the reduction the
paper's axes appear to use. Every figure defaults to the summed form — the quantity that
is actually optimized — and `--time-averaged` writes a `_time_averaged` companion:

```bash
python chemkan/scripts/figures/plot_all.py --time-averaged      # all of them
python chemkan/scripts/figures/fig05_biodiesel_noise.py --time-averaged
```

`N_t` is read from the dataset, never hard-coded: **30 for biodiesel, 50 for hydrogen**,
so this is not a single "÷30" across all figures. Applies to Figures 3, 4, 5A, 5B, 7 and
8A; Figures 6 and 8B display no loss, so they have no variant.

This is a **display convention only**: one constant applied after training. It changes no
model, no stored table, and no ranking between runs — and it does *not* explain the gap to
the paper, whose post-÷30 residual spans 2.1x-1382x across the Figure-4 points.

## Using them from a notebook

Import `make_figure` and pass the paths for the run you care about. The notebook chooses
*which run*; the script decides *how the figure is made*.

```python
import sys; sys.path.insert(0, str(ROOT / "chemkan" / "scripts" / "figures"))
from fig05_biodiesel_loss import make_figure as make_fig05b

RUN = ROOT / "results/experiments/my_experiment/seed0"
fig, results = make_fig05b(
    chemkan_runs={0: RUN},                        # override just the 0 % panel
    output_path=RUN / "fig05b",
    show=True,
)
print(results[("ChemKAN", 0)]["test_late_span"])
```

To point a figure at a different experiment, change paths — never copy plotting code into
the notebook.

Every `make_figure` returns `(fig, results)`; `results` holds the numbers that figure
computed, so a notebook can interpret them without recomputing anything. Figures that
produce more than one panel set (Figure 7's two initializations, Figure 8A's two colour
scales) return a dict of figures instead of a single one, because those views share
axis or colour limits and must be built together.

## Files

```
common.py                            repo paths, checkpoint loading, figure saving
fig03_biodiesel_trajectories.py      Figure 3 (+ the earlier clean-column view)
fig04_biodiesel_scaling.py           Figure 4 (both n_mu conventions)
fig05_biodiesel_noise.py             Figure 5A (+ the palette shared with 5B)
fig05_biodiesel_loss.py              Figure 5B
fig06_biodiesel_profiles.py          Figure 6 (+ the test-case companion)
fig07_hydrogen_trajectories.py       Figure 7 (+ the true-scale companion)
fig08_hydrogen_generalization.py     Figure 8A (paper scale and full range)
fig08_hydrogen_ignition.py           Figure 8B
plot_all.py                          runs every make_figure above
```

`common.py` is deliberately small: paths, `load_checkpoint`, `save_figure`,
`require_file`, `relative_to_root`, and `use_headless_backend`. Figure-specific science
stays in the figure's own file.

### Why `use_headless_backend()` exists

Called from `main()` only, never on import. With `bbox_inches="tight"` the crop comes from
the renderer's text metrics, and macOS's native backend rounds it one pixel differently
from Agg — so without it a command-line run produces a PNG one pixel wider than the
notebook's. Importing a figure script inside a notebook must not change the notebook's
backend, which is why it is not called at import time.

## Note on `assemble_fig4_scaling.py`

`chemkan/scripts/diagnostics/assemble_fig4_scaling.py` is now a shim that forwards to
`fig04_biodiesel_scaling.py`. It is kept because `refresh_biodiesel_reports.py` and the
reproduction shell wrappers invoke that path.
