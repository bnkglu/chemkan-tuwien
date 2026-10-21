# Interpretability of the hydrogen ChemKAN (read-only analysis)

Run from the repository root to regenerate the analysis figures (`.png` and `.pdf`) and `summary.json`. This README describes the edge, temperature-structure and rate-parity diagnostics:

    python chemkan/scripts/interpretability/combustion.py --system hydrogen

Nothing was trained. The checkpoint parameters are identical before and after the analysis
(`parameters_unchanged: true` in `summary.json`). The same script is used for methane
(`--system methane`; outputs in `results/methane/interpretability/`); the biodiesel analysis is in
`chemkan/scripts/interpretability/biodiesel.py`.

## What was analysed

- **Checkpoint:** `results/hydrogen/S2_seed0_200k/stage2_final.pt`, 200,000 Stage-2 epochs.
- **State:** 9 species + T (10 inputs).
- **Data:** 36 conditions × 100 times: 35 training conditions, held-out 1150 K / φ 1.3.
- **Model** (from the checkpoint config). z = [Y, T] is min-max normalized with the training statistics.
  - **Layer 1:** h_o = Σ_j φ_oj(z_j), additive, 3 hidden nodes, no input tanh.
  - **Layer 2:** species rate r_i = φ_i1(h1) · φ_i2(h2) · φ_i3(h3). This is LeanKAN with n_mu = 3 = width: a pure product, with tanh inside each edge.
  - **Temperature:** dT̂/dt = Σ_i w_i r_i + KAN_cor(z).
  - **Physical units:** du/dt = span · output / t_ref, with t_ref = 0.6 ms.
- **Reference:** Cantera's instantaneous rates of the data generator's reactor (`h2o2.yaml`, adiabatic, constant pressure, 1 atm) at the same state:
  - dY_k/dt = ω̇_k W_k / ρ;
  - dT/dt = −Σ h_k ω̇_k / (ρ c_p).

  These agree with a short reactor step to 1e-4 relative.

## Documented figures

| figure | content |
|---|---|
| `fig1_edges_layer1` | all 30 layer-1 edge functions, solid over the range seen in training. (%) = share of the node's spread over the 3,500 training states |
| `fig2_edges_layer2` | the three factors of each of the 9 species rates. (%) = factor's share of the spread of log\|rate\| |
| `fig3_temperature_structure` | thermo coefficients w_i·span_T/span_i against Cantera's −h_i/c_p (median and range over the training states); the five largest KAN_cor edges |
| `fig4_rate_parity` | ChemKAN against Cantera rates at every reference state (symlog axes). Titles: rank correlation, share within a factor 2, relative RMSE |

## Numerical summary

**Rates, training states:** rank correlation / share within a factor of 2.

| | dT/dt | H2 | O2 | H2O | OH | H |
|---|---|---|---|---|---|---|
| at the same state (fig. 4) | 0.54 / 54 % | 0.18 / 5 % | 0.05 / 18 % | 0.93 / 29 % | 0.77 / 57 % | 0.95 / 48 % |
| along the model's own trajectory, same time | 0.74 / **92 %** | 0.06 / 66 % | −0.02 / 68 % | 0.95 / **83 %** | 0.87 / **83 %** | 0.98 / **86 %** |

- **Relative RMSE at the same state is large** (dT/dt 27). It is dominated by a few states on the steep ignition fronts, where ChemKAN's rate can even have the opposite sign.
- **"Along the model's own trajectory"** means ChemKAN's rate on its own prediction, compared with Cantera's rate on the reference trajectory at the same time.

**Temperature structure (fig. 3):**
- dT̂/dt is driven almost entirely by the linear term. Spread over the training states: linear 62 vs KAN_cor 0.19.
- **Coefficients vs physics:**
  - H2O lies inside Cantera's −h_i/c_p range;
  - H2 and H2O are within 50 % of it;
  - the sign agrees for 7 of 9 species.
- **Not physical:** a large coefficient on the inert N2 (3e5 K). It acts on ChemKAN's spurious nonzero N2 rate.

**Element conservation of the predicted rates.** Per state:
|Σ_k a_ek dY_k/dt / W_k| / Σ_k |…|, median (p90). Cantera: 1e-16.
- H: 0.05 (0.55);
- O: 0.36 (0.72);
- N: 1.0 (1.0), because ChemKAN gives the inert N2 a nonzero rate.

The conservation term in the loss is only a soft penalty.

**Hidden nodes (descriptive Pearson r over the training states):**
- h1 rises with T (0.94) and H2O (0.89);
- h2 falls with them (−0.80, −0.81).

## Interpretation (conservative)

**Edge level:**
- The edges are smooth over the data range.
- 18 of the 27 layer-2 factors change sign over the data. The product structure therefore does not keep each species rate's sign fixed.
- The share percentages measure variation over correlated trajectory data, not causal effects.

**Full network:**
- The model reproduces the *time evolution*. Its trajectory errors are small, and along its own trajectory its rates agree with Cantera's at the same time (dT/dt within a factor 2 at 92 % of the points).
- It does *not* reproduce the *rate law as a function of state*. At the reference states themselves the agreement is much weaker, especially for H2 and O2.
- It is a trajectory surrogate, accurate along the paths it was trained on, but not a locally faithful chemical rate model.
- This is a stronger version of the biodiesel result. There, the rates at reference states matched within ~10 % for the main species, with a 3-reaction mechanism and no stiff ignition front.
- The temperature part has the right sign for most species, and its H2O coefficient is physical. Element conservation is only approximate.

**Hidden nodes:** they track temperature and reaction progress, as expected for any variable that follows the reaction progress. No hidden node is interpreted as a reaction.

**Scope:** one model and seed, with rates evaluated over the reference trajectory states and the model's own trajectories. These diagnostics do not establish a reaction mechanism or guarantee accurate rates at other states.
