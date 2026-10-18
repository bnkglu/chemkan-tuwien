# Interpretability of the methane ChemKAN (read-only analysis)

`interpretability.py` reproduces everything here (figures as `.png` and `.pdf`, `summary.json`):

    ~/uni_projects/chemkan-venv/bin/python results/experiments/methane/interpretability/interpretability.py

Nothing was trained. The checkpoint parameters are identical before and after the analysis
(`parameters_unchanged: true` in `summary.json`). The same script is used for hydrogen
(`results/experiments/hydrogen/interpretability/`); the biodiesel analysis is in
`results/experiments/biodiesel/interpretability/`.

## What was analysed

- **Checkpoint:** `methane/author_matched/main/S2_seed0/archive/stage2_S2_seed0_epoch180000.pt`, the selected snapshot (best 5k snapshot by training loss; notebook 15).
- **State:** 52 species + T (53 inputs).
- **Data:** 36 conditions × 100 times: 35 training conditions, held-out 1550 K / φ 1.3.
- **Model** (from the checkpoint config). z = [Y, T] is min-max normalized with the training statistics.
  - **Layer 1:** h_o = Σ_j φ_oj(z_j), additive, 3 hidden nodes, no input tanh.
  - **Layer 2:** species rate r_i = φ_i1(h1) · φ_i2(h2) · φ_i3(h3). This is LeanKAN with n_mu = 3 = width: a pure product, with tanh inside each edge.
  - **Temperature:** dT̂/dt = Σ_i w_i r_i + KAN_cor(z).
  - **Physical units:** du/dt = span · output / t_ref, with t_ref = 5 ms.
- **Reference:** Cantera's instantaneous rates of the data generator's reactor (`gri30.yaml`, adiabatic, constant pressure, 1 atm) at the same state:
  - dY_k/dt = ω̇_k W_k / ρ;
  - dT/dt = −Σ h_k ω̇_k / (ρ c_p).

  These agree with a short reactor step to 4e-5 relative.

## Figures

| figure | content |
|---|---|
| `fig1_edges_layer1` | the 8 largest layer-1 edge functions per hidden node (of 53), solid over the range seen in training. (%) = share of the node's spread over the 3,500 training states |
| `fig2_edges_layer2` | the three factors of 9 main species rates (CH4, O2, H2O, CO2, CO, H2, OH, CH2O, HO2). (%) = factor's share of the spread of log\|rate\| |
| `fig3_temperature_structure` | thermo coefficients w_i·span_T/span_i against Cantera's −h_i/c_p (the 6 lowest and 6 highest; median and range over the training states); the five largest KAN_cor edges |
| `fig4_rate_parity` | ChemKAN against Cantera rates at every reference state (symlog axes). Titles: rank correlation, share within a factor 2, relative RMSE |
| `fig5_perturbations` | T (±60 K), CH4 and O2 (×0.8–1.2, N2 keeps ΣY = 1) varied around four states: induction, ignition and post-ignition of 1550 K / φ 1.1, and ignition of the held-out condition. Bold = within 3 median nearest-neighbour spacings of a training state |
| `fig6_arrhenius_induction` | CH4 consumption against 1/T at the induction state |

## Numerical summary

**Rates, training states:** rank correlation / share within a factor of 2.

| | dT/dt | CH4 | O2 | H2O | CO2 | CO | OH | H |
|---|---|---|---|---|---|---|---|---|
| at the same state (fig. 4) | 0.22 / 38 % | −0.15 / 50 % | 0.33 / 36 % | 0.23 / 41 % | 0.16 / 27 % | 0.49 / 22 % | 0.06 / 0 % | 0.24 / 0 % |
| along the model's own trajectory, same time | 0.38 / 68 % | 0.27 / 77 % | 0.50 / 59 % | 0.59 / 63 % | 0.56 / 45 % | 0.58 / 35 % | 0.75 / 35 % | 0.84 / 3 % |

- **Relative RMSE at the same state is about 1 for most species.**
- **"Along the model's own trajectory"** means ChemKAN's rate on its own prediction, compared with Cantera's rate on the reference trajectory at the same time.

**Temperature structure (fig. 3):**
- dT̂/dt is mainly the linear term. Spread over the training states: linear 2.5 vs KAN_cor 1.1.
- **Coefficients vs physics:**
  - O2, CH3 and CH4 lie inside Cantera's −h_i/c_p range;
  - O2, CH3 and C2H2 are within 50 % of it;
  - the sign agrees for 25 of 52 species.

**Element conservation of the predicted rates.** Per state:
|Σ_k a_ek dY_k/dt / W_k| / Σ_k |…|, median (p90). Cantera: 1e-16.
- H: 0.08 (0.29);
- O: 0.06 (0.21);
- C: 0.16 (0.35);
- N: 0.46 (1.0).

The conservation term in the loss is only a soft penalty.

**Perturbations (fig. 5):**
- ChemKAN's rate changes in the same direction as Cantera's in 8–12 of 16–18 sweeps per state.
- Changes in CH4 or O2 at fixed everything else leave the training region quickly. Temperature changes at fixed composition stay near the data longer.

**Apparent activation energy at the induction state (fig. 6):** CH4 consumption gives 68.8 kcal/mol for ChemKAN against 8.7 for Cantera. This is an apparent overall value at that state, not a single reaction's.

**Hidden nodes:**
- **Correlations** (descriptive Pearson r over the training states): h2 rises with T (0.89) and falls with CH4 (−0.90).
- **Largest layer-1 edges:** they come from T, H2O, CH4, O2, but also from trace nitrogen species (N2O, HNO, NO2, N, NO). After min-max normalization these get the same [0, 1] range as the main species.

## Interpretation (conservative)

**Edge level:**
- The edges are smooth over the data range.
- 93 of the 156 layer-2 factors change sign over the data.
- The hidden nodes draw heavily on trace nitrogen species. That reflects the min-max normalization, which gives every species the same range, not chemical importance.
- The share percentages measure variation over correlated trajectory data, not causal effects.

**Full network:**
- The model reproduces the *time evolution* reasonably: along its own trajectory its rates agree with Cantera's at the same time far better than at fixed states (dT/dt within a factor 2 at 68 % vs 38 % of the points).
- It does *not* reproduce the *rate law as a function of state*. Rank correlations at the reference states are 0.06–0.5, and the local temperature sensitivity is wrong (apparent Ea 69 vs 8.7 kcal/mol).
- It is a trajectory surrogate, not a locally faithful chemical rate model. The gap is larger than for hydrogen (52 species, 325 reactions).
- Most temperature coefficients are not physical, and element conservation is approximate.

**Hidden nodes:** they track temperature and reaction progress. No hidden node is interpreted as a reaction.

**Scope:** one model and seed (the 180k snapshot); four reference states; one-at-a-time perturbations. Composition perturbations leave the trajectory manifold quickly (one-dimensional paths in a 53-dimensional space), so their non-bold parts are extrapolation.
