# Figures for the observed-interval training objective

These two figures use the existing clean training data and saved runs. No model was
trained or updated. Each figure is available as PDF and PNG in `figures/`.

| Figure | What it shows |
|---|---|
| [Objective convergence](figures/interval_objective_convergence.pdf) | The recorded interval objective for seeds 0, 1, and 2, in one panel over all epochs. The last-2,000-epoch panel was removed. |
| [Interval error by species](figures/interval_objective_per_species.pdf) | Final models from both training methods evaluated with the **same observed-state resets**. Bars are means over three seeds. |

Both figures concern **training conditions**. They describe the interval diagnostic;
they do not demonstrate reproduction of the paper's complete-trajectory accuracy.

## Loss and reductions

There are 30 observed states and 29 intervals. Each interval starts from its own
observed state. Only the predicted endpoint is scored. Squared errors use the archive's
train-only min-max normalization. The objective sums over the 29 endpoints and averages
over the 20 conditions and six species.

For the species bars, each value sums over endpoints and averages over conditions.
Taking the mean of the six species values gives the overall interval objective. The
individual interval terms are listed in `tables/interval_objective_error_by_time_species.csv`. No division by 29 or 30 is applied
to the objective.

The baseline bars are **new interval evaluations of the original final
models**. They are not the full-trajectory training losses logged by those runs.

| Seed | Original model: interval evaluation | Interval-trained model: interval evaluation |
|---:|---:|---:|
| 0 | 0.00212749 | 0.000245433 |
| 1 | 0.00113835 | 0.000389095 |
| 2 | 0.00274789 | 0.000204941 |

At 10,000 epochs, interval training lowers this local metric in all three paired
comparisons. This does not change the previously reported full-rollout results.

## Verification and regeneration

The plotting script checks all three recomputed interval objectives against the final
history rows and all six full-rollout training losses against the archived comparison
table. It also checks its endpoint-error decomposition against the trainer's
`accumulate_interval_gradients(..., backward=False)` routine. Checks use
`rtol=1e-5, atol=1e-9` and must pass before figures are written.

Legacy checkpoints store RBF centers but omit the width `h`, which is a Python float.
The script reconstructs `h` from each layer's saved uniform center spacing. All six
models here have centers `[-1, 0, 1]` and `h=1` in both layers. This avoids applying the
current AddKAN default width to older models. The reconstruction is in memory; the
checkpoints and shared library are unchanged.

From the repository root, using the existing project environment:

```bash
python \
  chemkan/scripts/figures/plot_biodiesel_interval_objective.py
```

Source artifacts:

- [Plotting script](../../../../../chemkan/scripts/figures/plot_biodiesel_interval_objective.py)
- [Final metrics and recorded checks](tables/interval_objective_summary.csv)
- [Interval/species error table](tables/interval_objective_error_by_time_species.csv)

## Additional figures at the paper's Figure 3 condition

These two separate figures use the same clean interval-trained seed-0 model at
10,000 epochs, evaluated at `TG=1.94, ROH=1.43, DG=MG=GL=RCO2R=0, T=334.8 K`.
This condition is distinct from the canonical training and test splits. The reference
comes from the existing `biodiesel_fig3_condition.npz` mechanistic simulation.

- [Endpoint comparison at Figure 3's condition](figures/fig3_endpoint_diagnostic.pdf)
  ([PNG](figures/fig3_endpoint_diagnostic.png)): the same type
  of diagnostic as the training-case example, now at the published initial condition.
  The orange endpoints condition on intermediate observations from this unseen case.
  Their error is a conditional evaluation metric, not a training loss or an autonomous
  prediction from the initial condition.
- [Figure 3 for the interval-trained models, all four noise columns](figures/fig3_noise_columns.pdf)
  ([PNG](figures/fig3_noise_columns.png)): six species in six rows, 0 / 5 / 10 / 15 %
  training noise in four columns, with the clean reference, that column's observations,
  and the model's complete rollout. No intermediate observation is supplied to these
  rollouts. The evaluation procedure corresponds to Figure 3; the training procedure is
  explicitly identified as the observed-interval experiment. The single-column version
  of this figure (`fig3.pdf`) was superseded by this one and removed.

For this single condition, the conditional interval loss is **0.000122895**, and the
full-rollout loss is **0.004982419**. Both average over species and sum over observation
endpoints; neither is divided by 30. The full-rollout error is lower here than in the
earlier training-case example. This is variation across initial conditions for the same
model, not an improvement from additional training. It does not change the aggregate
training/test metrics or establish quantitative reproduction across the paper's cases.

Dense curves use 601 display times, while the loss uses the original 30 observation
times. The plotting solve includes all observation times and is checked against the
separate observation-grid solve. No model predictions are clipped.

Regenerate the noise-column figure, the endpoint diagnostic and the verified seed-0
metrics tables:

```bash
python \
  chemkan/scripts/figures/fig03_biodiesel_trajectories.py --observed-intervals
```

Section 4 of `chemkan/notebooks/11_biodiesel_observed_intervals.ipynb` draws the same
four-column figure inline.

Sources: [script](../../../../../chemkan/scripts/figures/plot_biodiesel_interval_fig3.py),
[metrics](tables/interval_fig3_seed0_metrics.json),
[predictions at observation times](tables/interval_fig3_seed0_predictions.csv),
[dense plotted curves](tables/interval_fig3_seed0_dense.csv).

### The Figure 3 noise columns

**Done (2026-09-17).** The interval trainer takes `--noise-percent`, which uses the
stored noisy observations as both the reset states and the endpoint targets, keeping the
same train-only scaling, architecture, RBF grids, learning rate and epoch budget, and
reports complete-rollout errors against the clean held-out trajectories. Seed-0 interval
runs exist at 1, 2, 3, 5, 7, 10 and 15 % in `noise/noiseNN_seed0/`, so the figure's four
columns (0 / 5 / 10 / 15 %) are drawn from four separately trained models.

At this single condition the interval models degrade with noise far faster than the
original-method models (Eq. 18 loss against the clean reference, seed 0):

| training noise | observed-interval | original |
|---|---|---|
| 0 % | 0.00498 | 0.07357 |
| 5 % | 0.03803 | 0.02832 |
| 10 % | 0.21634 | 0.03340 |
| 15 % | 0.49440 | 0.02566 |

Interval training is the better model on clean data here and the worse one at every
noise level, which is consistent with its procedure: each of the 29 intervals restarts
**from an observed state**, so noisy observations enter as corrupted initial conditions
rather than only as comparison targets. One condition and one seed — the aggregate
comparison is the Fig. 5A layout across noise levels, which has not been assembled for
the interval models yet. Numbers: `tables/fig3_noise_columns_method_comparison.csv`.

## Standard entry points and paper numbering

Generate the interval experiment's figures through the existing figure scripts:

```bash
CHEMKAN_PYTHON=python \
  bash chemkan/scripts/reproduction/biodiesel/observed_intervals_figures.sh
```

The wrapper draws the supplemental interval diagnostics, calls the standard Figure 3
and Figure 5B scripts with `--observed-intervals`.
It performs no training. Add `--dry-run` to inspect its commands.

- [fig3_noise_columns](figures/fig3_noise_columns.pdf): the paper's initial condition at
  0 / 5 / 10 / 15 % training noise, one interval-trained model per column. Drawn by
  section 4 of `chemkan/notebooks/11_biodiesel_observed_intervals.ipynb`, which calls
  `fig03_biodiesel_trajectories.make_figure` with `run_dirs` pointed at the interval
  runs. The wrapper's `--observed-intervals` flag writes the same figure; it no longer
  produces the superseded single-column `fig3.pdf`.
- [fig5b](figures/fig5b.pdf): two panels distinguish the actual interval objective for
  all three seeds from full-rollout training/test evaluation for seed-0 original
  ChemKAN, interval-trained ChemKAN, and reference DeepONet. Raw evaluation times
  are retained. Separate markers at epoch 10,000 show final-checkpoint metrics;
  the original ChemKAN history itself ends at epoch 9,999.
- The per-species bars and interval objective remain labelled
  biodiesel diagnostics because they have no direct paper figure number.
- Figure 4 is a model-size sweep; these interval runs have only one model size.

## Generate the three email attachments in Notebook 11

Run [Notebook 11](../../../../../chemkan/notebooks/11_biodiesel_observed_intervals.ipynb)
with the `chemkan-venv` kernel. Sections 4, 6, and 7 call the existing Figure 3,
Figure 5B, and Figure 5A scripts directly as Python functions. Each call displays
the same figure that it saves as PDF and PNG. The notebook also explains exactly
how the full-rollout training loss is calculated and verifies the final comparison
against the archived results. No model is trained or updated.

The third attachment, [fig5a](figures/fig5a.pdf), shows final training and clean
held-out full-rollout losses at 0% noise after 10,000 epochs. All three paired
ChemKAN seeds are shown individually, with one DeepONet seed-0 reference.
The x-axis names the methods, with all seeds of one method at the same x position;
grey lines connect matched ChemKAN seeds. Numeric labels show final losses to six
decimal places. Every point uses 0% noise, so this is a clean-data comparison, not a noise sweep.
At 0%, clean-reference and noisy-reference test losses coincide and are shown once.

The callable functions are `fig03_biodiesel_trajectories.make_interval_figure`,
`fig05_biodiesel_loss.make_interval_clean_figure`, and
`fig05_biodiesel_noise.make_interval_clean_figure`. Each accepts `output_path`
and `show`, returns `(figure, results)`, and writes no artifacts when `output_path`
is omitted. The existing default reproduction modes are retained.

Source data for the updated comparisons are exported to
[final losses](tables/fig5a_clean_final_losses.csv),
[full-rollout histories and final markers](tables/fig5b_clean_trajectory_losses.csv),
and [actual interval objectives](tables/fig5b_interval_objective_losses.csv),
including metric conventions and original source paths.

Figure 5B includes tables of the final interval objectives (seeds 0–2) and final
full-rollout training/held-out losses (three models, seed 0). Figure 3 labels its
full-rollout loss, scored on the 30 observation times rather than the dense display grid.
Displayed values are rounded; exported metric tables retain their original precision.

## Time-averaged Figure 5 companions

Notebook 11 also calls the existing Figure 5A and 5B functions with
`time_averaged=True` and separate output stems. The generated
[Figure 5A](figures/fig5a_time_averaged.pdf) divides both full-rollout panels by
N_t = 30; [Figure 5B](figures/fig5b_time_averaged.pdf) divides its right-hand
full-rollout curves, final markers, and table by 30. Its left-hand interval
objective remains the sum over 29 independently restarted endpoints.

The divisor comes from the saved observation grid, including the initial
zero-error state. Averaging changes neither model rankings nor relative errors.
These are derived diagnostics of the time-summed loss. Numeric rollout labels
use eight decimal places; full precision is retained in the companion CSVs.
Original figures and tables remain available under their existing names.
