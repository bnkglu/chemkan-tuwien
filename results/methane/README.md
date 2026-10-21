# Methane ChemKAN extension (seed 0)

This project extension applies the two-stage combustion model to methane; it is not
an experiment from the original ChemKAN paper.

## Settings and checkpoints

- Data: `chemkan/data/generated/methane.npz`, 52 species mass fractions plus temperature,
  35 training conditions and one held-out condition (1550 K, φ 1.3), 100 observation
  times over 5 ms. GRI-Mech 3.0 is used with AR excluded from the modeled state.
- Architecture: core width 3, five RBF basis functions, n_mu = 3, no base activation,
  1,892 parameters in the complete model.
- Training: Adam, learning rate 2e-3; float32 Tsit5, rtol 1e-6 / atol 1e-8,
  direct autograd, t_ref = 5 ms.
- Stage 1: history ends at epoch 54,790. Stage 2 starts from the archived epoch-50,000
  checkpoint, as recorded in `S2_seed0/config.json`, and runs for 200,000 epochs.
- Selected evaluation checkpoint: `S2_seed0/archive/stage2_S2_seed0_epoch180000.pt`,
  selected using training-condition errors, not the held-out condition. The model
  is not converged; the final 200k checkpoint is retained for comparison.

The history CSVs retain every row and the columns
`epoch, loss, pinn, mse_time_avg, elapsed_seconds`. Full diagnostic columns and
historical paths are described in [`../PATHS.md`](../PATHS.md).
Stage 1 integrates species with an observed temperature interpolant; Stage 2
integrates species and temperature together. Their losses are not directly comparable.

## Reproduction

Run from the repository root with the project environment:

```bash
bash chemkan/scripts/combustion/run_methane.sh
python chemkan/scripts/figures/methane_species.py
python chemkan/scripts/combustion/evaluate_methane_grid.py
python chemkan/scripts/interpretability/combustion.py --system methane
```

The current `run_methane.sh` launcher schedules 100,000 Stage-1 epochs, whereas the recorded Stage-1 run stopped at epoch 54,790. Stage 2 is initialized from the epoch-50,000 Stage-1 archive in both cases. The launcher therefore reproduces the intended Stage-2 initialization and configuration, but not the exact historical training schedule.

Training can be expensive; consult the launcher and root README before rerunning.
The figure, evaluation and interpretability commands load existing checkpoints.

## Results

- [`figures/`](figures/): main-species trajectories at 1550 K / φ 1.3, combined
  linear/logarithmic panels and separate `_linear` and `_log` PNG/PDF exports.
  Species use actual mass fractions, with no display multipliers. Negative predictions
  remain visible on the linear axis; the logarithmic axis can show only positive values.
- [`generalization/README.md`](generalization/README.md): 441-condition evaluation;
  selected-checkpoint held-out normalized time-averaged MSE is 1.1626e-3.
- [`interpretability/README.md`](interpretability/README.md): learned-edge and rate-parity
  diagnostics. These describe learned model behavior and do not
  establish an independently identified chemical mechanism.
- `chemkan/notebooks/methane_extension.ipynb`: training history and checkpoint comparison.
