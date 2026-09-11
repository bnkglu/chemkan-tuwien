# Figures for the observed-interval training objective

These four figures use the existing clean training data and saved runs. No model was
trained or updated. Each figure is available as PDF and PNG in `figures/`.

| Figure | What it shows |
|---|---|
| [Objective convergence](figures/interval_objective_convergence.pdf) | The recorded interval objective for seeds 0, 1, and 2, in one panel over all updates. The last-2,000-update panel was removed. |
| [Interval error by species](figures/interval_objective_per_species.pdf) | Final models from both training methods evaluated with the **same observed-state resets**. Bars are means over three seeds. Individual markers denote seed 0 (circle), seed 1 (square), and seed 2 (triangle). |
| [Interval error across time and species](figures/interval_objective_time_species.pdf) | Mean squared normalized endpoint error for each interval and species, averaged over 20 training conditions and three seeds. Both panels use one logarithmic color scale. |
| [Endpoint concentration predictions](figures/interval_objective_endpoint_profiles.pdf) | Observed concentrations, independent interval endpoints, and a complete rollout for the same interval-trained model. Seed 0 and training condition 0 were fixed without ranking prediction errors. |

The endpoint concentration figure is especially useful for explaining why a small
interval objective can coexist with a larger rollout error. Orange crosses restart
from the observed state at the preceding observation time. They are separate
predictions, not a continuous trajectory. The blue curve uses only the original initial
condition. In this example, the GL rollout becomes negative even though its independently
reset endpoints stay much closer to the observations. The plotted predictions are not
clipped to hide negative concentrations. The blue line connects predictions at the
archive's 30 saved times.

All four figures concern **training conditions**. The concentration example is not an
unseen test case, and its errors need not represent all 20 conditions. These figures
describe the interval diagnostic; they do not demonstrate reproduction of the paper's
complete-trajectory accuracy.

## Loss and reductions

There are 30 observed states and 29 intervals. Each interval starts from its own
observed state. Only the predicted endpoint is scored. Squared errors use the archive's
train-only min-max normalization. The objective sums over the 29 endpoints and averages
over the 20 conditions and six species.

For the species bars, each value sums over endpoints and averages over conditions.
Taking the mean of the six species values gives the overall interval objective. The
heatmap contains the individual interval terms: sum all its cells and divide by six
to recover the objective averaged over three seeds. No division by 29 or 30 is applied
to the objective.

The baseline bars and heatmap are **new interval evaluations of the original final
models**. They are not the full-trajectory training losses logged by those runs.

| Seed | Original model: interval evaluation | Interval-trained model: interval evaluation |
|---:|---:|---:|
| 0 | 0.00212749 | 0.000245433 |
| 1 | 0.00113835 | 0.000389095 |
| 2 | 0.00274789 | 0.000204941 |

At 10,000 updates, interval training lowers this local metric in all three paired
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
/Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/figures/plot_biodiesel_interval_objective.py
```

Source artifacts:

- [Plotting script](../../../chemkan/scripts/figures/plot_biodiesel_interval_objective.py)
- [Final metrics and recorded checks](tables/interval_objective_summary.csv)
- [Interval/species error table](tables/interval_objective_error_by_time_species.csv)

## Additional figures at the paper's Figure 3 condition

These two separate figures use the same clean interval-trained seed-0 model at
10,000 updates, evaluated at `TG=1.94, ROH=1.43, DG=MG=GL=RCO2R=0, T=334.8 K`.
This condition is distinct from the canonical training and test splits. The reference
comes from the existing `biodiesel_fig3_condition.npz` mechanistic simulation.

- [Endpoint comparison at Figure 3's condition](figures/fig3_endpoint_diagnostic.pdf)
  ([PNG](figures/fig3_endpoint_diagnostic.png)): the same type
  of diagnostic as the training-case example, now at the published initial condition.
  The orange endpoints condition on intermediate observations from this unseen case.
  Their error is a conditional evaluation metric, not a training loss or an autonomous
  prediction from the initial condition.
- [Figure 3 clean column for the interval-trained model](figures/fig3.pdf)
  ([PNG](figures/fig3.png)): six species in six rows,
  with the clean reference, clean observations, and the model's complete rollout.
  No intermediate observation is supplied to this rollout. The evaluation procedure
  corresponds to Figure 3; the training procedure is explicitly identified as the
  observed-interval experiment.

For this single condition, the conditional interval loss is **0.000122895**, and the
full-rollout loss is **0.004982419**. Both average over species and sum over observation
endpoints; neither is divided by 30. The full-rollout error is lower here than in the
earlier training-case example. This is variation across initial conditions for the same
model, not an improvement from additional training. It does not change the aggregate
training/test metrics or establish quantitative reproduction across the paper's cases.

Dense curves use 601 display times, while the loss uses the original 30 observation
times. The plotting solve includes all observation times and is checked against the
separate observation-grid solve. No model predictions are clipped.

Regenerate these two figures only:

```bash
/Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/figures/fig03_biodiesel_trajectories.py --observed-intervals
```

Sources: [script](../../../chemkan/scripts/figures/plot_biodiesel_interval_fig3.py),
[metrics](tables/interval_fig3_seed0_metrics.json),
[predictions at observation times](tables/interval_fig3_seed0_predictions.csv),
[dense plotted curves](tables/interval_fig3_seed0_dense.csv).

### Other Figure 3 noise columns

The original full-trajectory training method already has completed seed-0 models at
0%, 5%, 10%, and 15% noise. The observed-interval experiment currently has completed
clean models only. Its 5%, 10%, and 15% columns would need separate models trained on
those noisy observations. Adding noisy dots to this clean model's prediction would
not demonstrate learning from noisy data.

The interval trainer currently loads clean data explicitly. A noise extension must use
the selected noisy observations as both the reset states and endpoint targets, retain
the same train-only scaling, architecture, RBF grids, learning rate, and update budget,
and report complete-rollout errors against clean held-out trajectories. A paired 5%
seed-0 pilot would be a useful next diagnostic before extending to 10%, 15%, and more
seeds. It is not necessary to postpone reporting the completed clean experiment.

## Standard entry points and paper numbering

Generate the interval experiment's figures through the existing figure scripts:

```bash
CHEMKAN_PYTHON=/Users/berke/uni_projects/chemkan-venv/bin/python \
  bash chemkan/scripts/reproduction/biodiesel/observed_intervals_figures.sh
```

The wrapper draws the supplemental interval diagnostics, calls the standard Figure 3
and Figure 5B scripts with `--observed-intervals`.
It performs no training. Add `--dry-run` to inspect its commands.

- [fig3](figures/fig3.pdf): clean column at the paper's initial condition.
- [fig5b](figures/fig5b.pdf): clean complete-trajectory loss curves for interval-trained
  ChemKAN seed 0 and the existing clean DeepONet. The interval objective is excluded
  from these trajectory-loss curves. ChemKAN uses the recorded evaluation times.
- The per-species bars, interval objective, and time/species heatmap remain labelled
  biodiesel diagnostics because they have no direct paper figure number.
- Figure 4 is a model-size sweep; these interval runs have only one model size.
