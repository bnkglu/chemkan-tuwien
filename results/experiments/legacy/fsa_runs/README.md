# Forward sensitivity analysis (FSA) runs

The ChemKAN paper trains with Forward Sensitivity Analysis (p. 8). The reproduction's
default runs use direct autograd (backprop through the Tsit5 solve). `--sensitivity fsa`
(`chemkan/src/chemkan/fsa.py`, `chemkan/src/chemkan/ASSUMPTIONS.md` §9) forms the same
training gradient by integrating the continuous forward sensitivity equations. The FSA runs
here repeat the saved direct-autograd configurations with only the sensitivity backend
changed. They never overwrite a `direct_autograd` run.

**Status.** FSA is implemented and numerically validated. In the completed biodiesel and
hydrogen Stage-1 experiments, changing the sensitivity backend from direct autograd to FSA
does not materially change the achieved loss, although FSA required substantially more
training time in these runs (recorded wall time, not a controlled benchmark). Hydrogen
Stage-2 FSA training is currently incomplete, so the thermodynamic/temperature
reproduction with FSA remains unresolved.

## Validation

[`results/experiments/validation/fsa/README.md`](../../validation/fsa/README.md): core validation passed. Checks C (FSA vs
direct-autograd gradients) and E (trajectory independence) passed after the documented
refinement extension (amendment 1, one finer tolerance level). The optional
finite-difference check G is inconclusive for one very small gradient (magnitude 3.67e-08,
a `thermo.correction` coefficient), where cancellation error dominates.

## Completed runs

Seed 0; single runs, not a seed study. Time is recorded wall time on a shared machine,
not a controlled benchmark.

| experiment | loss statistic | direct autograd | FSA | training time, DA → FSA |
|---|---|---|---|---|
| biodiesel, 0 % noise | median Eq. 18 training loss, epochs 8,000–10,000 | 0.0358 | 0.0356 | 628 s → 3,842 s |
| hydrogen Stage 1 (N=4, base ON) | final Stage-1 loss | 0.3149 | 0.3202 | 1.43 h → 11.4 h |

- **Biodiesel.** In the completed 0 %-noise seed-0 run, FSA reaches essentially the same
  late-training optimization loss as direct autograd, so replacing direct autograd with FSA
  does not close the observed loss-scale gap in this run. Final-checkpoint values are in
  [`results/experiments/legacy/biodiesel/fsa/tables/biodiesel_B0_fsa_comparison.csv`](../biodiesel/fsa/tables/biodiesel_B0_fsa_comparison.csv).
  Single end points are not used for the comparison because the loss oscillates late in
  training; see notebook 14.
- **Hydrogen Stage 1.** The final losses of the two backends are close.
  The 11.4 h is the cumulative recorded training time of a run resumed once: the sum of
  its two non-overlapping segments, 7,882 s + 33,181 s.

Sources:
- direct autograd biodiesel: `results/reproduction/legacy/biodiesel/chemkan/main/direct_autograd_seed0`
- FSA biodiesel: `results/experiments/legacy/biodiesel/fsa/fsa_seed0`
- direct autograd hydrogen Stage 1: `results/reproduction/legacy/hydrogen/chemkan/diagnostics/base_on_n4/stage1_seed0`
- FSA hydrogen Stage 1: `results/experiments/legacy/hydrogen/fsa/stage1_fsa_seed0`
- comparison tables: [`biodiesel/fsa/tables/`](../biodiesel/fsa/tables/), [`hydrogen/fsa/tables/`](../hydrogen/fsa/tables/)
  and the cross-domain [`fsa_comparison.json`](fsa_comparison.json)

## Hydrogen Stage 2 (incomplete)

Hydrogen Stage-2 FSA training is incomplete, so the thermodynamic/temperature reproduction
with FSA is not yet established. The values below are not benchmark results.

**Interim observation, random initialization (H0), incomplete run.** At the same available
horizon, from each run's `history_stage2.csv` (`total_loss`):

| | direct autograd | FSA |
|---|---|---|
| loss at epoch 5,970 | 2.1683 | 2.1526 |
| median loss, epochs 5,000–5,970 | 2.1739 | 2.1571 |

By the current ~5,970-epoch horizon, the direct-autograd and FSA runs have reached very
similar Stage-2 training-loss levels. At epoch 5,970 the losses are 2.1683 and 2.1526, and
the medians over epochs 5,000–5,970 are 2.1739 and 2.1571. The FSA run remains incomplete,
so the thermodynamic/temperature reproduction with FSA is not yet established. The
norm-matched (Hnorm1) FSA run stopped much earlier and is not used for a comparison.

- **Committed artifacts.** The committed Stage-2 FSA artifacts contain the partial
  histories/probes and retained epoch-500 model snapshots, in
  `results/experiments/legacy/hydrogen/fsa/random_stage2_10000_fsa_seed0/` (random thermo initialization, H0) and
  `results/experiments/legacy/hydrogen/fsa/normmatched_dir1_stage2_10000_fsa/` (norm-matched initialization, Hnorm1).
- **End-to-end comparison.** The FSA Stage-2 runs start from the FSA Stage-1 checkpoint
  (`results/experiments/legacy/hydrogen/fsa/stage1_fsa_seed0`). The direct-autograd Stage-2 runs start from
  `base_on_n4/stage1_seed0`. The existing setup is therefore an end-to-end backend
  comparison, not a controlled Stage-2-only ablation of the sensitivity backend.

## See also

- [`chemkan/notebooks/14_forward_sensitivity_analysis.ipynb`](../../../../chemkan/notebooks/14_forward_sensitivity_analysis.ipynb):
  method, validation and training curves.
- [`docs/reproduction_workflow.md`](../../../../docs/reproduction_workflow.md), "Forward
  sensitivity analysis": commands.
