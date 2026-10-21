# Methane generalization over a 21 x 21 condition grid

Hydrogen Fig. 8A analogue for methane (the paper has no methane results). Reproduce with:

    python chemkan/scripts/data_gen/generate_methane.py \
        --out chemkan/data/generated/methane_fine.npz \
        --T0s 1400 1412.5 1425 1437.5 1450 1462.5 1475 1487.5 1500 1512.5 1525 1537.5 1550 1562.5 1575 1587.5 1600 1612.5 1625 1637.5 1650 \
        --phis 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50
    python chemkan/scripts/combustion/evaluate_methane_grid.py

## Setup

- **Reference data:** `methane_fine.npz`, made by the methane generator with the settings of `methane.npz`:
  GRI-Mech 3.0, 1 atm, Cantera rtol 1e-9 / atol 1e-15, 100 points over 5 ms.
  - It contains 441 conditions: T0 = 1400–1650 K in 12.5 K steps, φ = 0.5–1.5 in 0.05 steps.
  - The 36 conditions it shares with `methane.npz` are reproduced exactly (max |Δ| = 0).
- **Groups:** the 35 training conditions of `methane.npz`, the held-out 1550 K / φ 1.3, and 405 new conditions.
  "Unseen" = 406 = the new ones plus the held-out one.
- **Model:** the selected Stage-2 snapshot, epoch 180,000 (`chemkan/notebooks/methane_extension.ipynb`). Each condition is integrated from its initial state alone (full trajectory) with the training solver (Tsit5, rtol 1e-6, atol 1e-8, float32).
- **Error:** time-averaged MSE over all 53 states on the 100 observation times, in the checkpoint's training normalization. The fine file's own normalization is not used.
- **Ignition delay:** argmax dT/dt on a 601-point grid. The reference value is the generator's 601-point diagnostic.

## Results (`methane_grid_441.json`, per condition in `methane_grid_441.csv`)

| group | MSE median | MSE mean | MSE max | ignition-delay \|error\| median (max) |
|---|---|---|---|---|
| 35 training conditions | 9.72e-4 | 1.22e-3 | 3.19e-3 | 1.1 % (6.0 %) |
| 406 unseen conditions | 9.21e-4 | 1.77e-3 | 1.27e-2 | 1.4 % (7.3 %) |
| held-out 1550 K / φ 1.3 | 1.16e-3 | | | 3.2 % |

The training mean (1.220e-3) equals the training loss of the 180k weights (history row 180,001: 1.2198e-3). This confirms the evaluation matches the trainer.

Three MSE maps over the 21 × 21 conditions (counterparts of hydrogen Fig. 8A):
- `methane_generalization_441.{png,pdf}`: the hydrogen Fig. 8A colour range (linear,
  0–10 × 10⁻⁴), so the two systems can be read on the same scale. 201 of the 441 methane
  conditions are above that limit and share the darkest colour.
- `methane_generalization_441_full_range.{png,pdf}`: the full measured range, linear from 0 to
  the largest error (1.27e-2), like hydrogen's full-range map. The φ 1.40–1.45 strip
  dominates it.
- `methane_generalization_441_full_range_log.{png,pdf}`: the full range on a log colour
  scale, which shows the structure both below and above 10⁻³.

`evaluate_methane_grid.py --plot-only` redraws the three maps from the saved CSV and JSON. The per-condition ignition delays are
only tabulated in the CSV, not plotted.

## Reading

- **Interpolation works over most of the grid.** The unseen median (9.2e-4) is not worse than the training median (9.7e-4).
- **The exception is a narrow strip at φ ≈ 1.40–1.45**, between the trained φ 1.3 and 1.5 columns (both fine). There the MSE reaches 5e-3–1.3e-2.
  - Temperature is not the main contributor; its MSE there is ~2e-3.
  - The strip is dominated by normalized errors in trace and radical species, e.g. at 1550 K / φ 1.45: NNH (true peak ~7e-8), H2, H, NO2.
  - It is a generalization weakness in composition space between two training columns, at a single seed.
- **The lowest temperatures (1400–1425 K) have higher MSE at all φ.** These are the late-igniting conditions with the pre-ignition temperature dip seen in `chemkan/notebooks/methane_extension.ipynb`.
