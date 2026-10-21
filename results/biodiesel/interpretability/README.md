# Interpretability of the final biodiesel ChemKAN (read-only analysis)

Run from the repository root with the project environment to reproduce the figures
(as `.png` and `.pdf`) and `summary.json`:

```bash
python chemkan/scripts/interpretability/biodiesel.py
```

Nothing was trained; the parameter sha256 (`e83046d3…`, in `summary.json`) is identical
before and after the analysis.

## What was analysed

- **Checkpoint:** `results/biodiesel/chemkan_figure_runs/fig05/chemkan_noise00_seed0/checkpoint_final.pt`.
  This is the noise-free run of chemkan/notebooks/biodiesel_reproduction.ipynb: 156 parameters, 10,000 epochs, the same model as the inference benchmark.
- **Model:**
  - z = [TG, ROH, DG, MG, GL, R′CO₂R, T] is min-max normalized with the training statistics (`summary.json` → `normalization`).
  - **Layer 1** (additive, 7 → 4): hidden node h_o = Σ_j φ_oj(z_j).
  - **Layer 2** (LeanKAN, 4 → 6): out_i = φ_i1(h1)·φ_i2(h2) + φ_i3(h3) + φ_i4(h4), with tanh applied inside each layer-2 edge.
  - Every φ is a sum of three Gaussians.
  - Physical rate: dy_i/dt = (y_max,i − y_min,i)·out_i / 50, in mol L⁻¹ s⁻¹ (the released example's divisor).
- **Reference:** the data generator's known kinetics with the released-code ordering of Arrhenius constants:
  - r1 = k1[TG][ROH], r2 = k2[DG][ROH], r3 = k3[MG][ROH];
  - k = exp(lnA − Ea/RT), with Ea = 14.54 / 14.42 / 6.47 kcal/mol.
- **Data domain:** the 20 clean training trajectories (600 states) and the 10 test trajectories (300 states) of `biodiesel_v2.npz`.

## Documented figures

| figure | content |
|---|---|
| `fig1_edges_layer1` | all 28 layer-1 edges φ_oj(z_j). Solid over the range of z_j seen in training, dotted outside it; ticks mark the training states; (%) = the edge's share of the hidden node's spread over the training states |
| `fig2_edges_layer2` | all 24 layer-2 edges against the hidden-node values. Orange edges are multiplied (h1·h2 → one product term), blue edges are added; (%) = the term's share of the output's spread |
| `fig3_rate_parity` | full-network species rates against the known kinetics at all 900 reference states |

## Numerical summary

**Rates at the reference states (fig. 3).**

| species | relative RMSE, train / test | R², train / test |
|---|---|---|
| TG | 0.09 / 0.10 | 0.99 / 0.98 |
| ROH | 0.09 / 0.13 | 0.98 / 0.96 |
| DG | 0.10 / 0.09 | 0.99 / 0.99 |
| MG | 0.24 / 0.28 | 0.90 / 0.87 |
| GL | 0.12 / 0.14 | 0.96 / 0.94 |
| R′CO₂R | 0.09 / 0.12 | 0.98 / 0.96 |

Relative RMSE is RMSE divided by the RMS of the known rate.

**Structure the known kinetics imposes exactly:**
- At t = 0 (DG = MG = 0), MG and GL are not produced. ChemKAN still gives MG production of 0.0049 (train) / 0.0061 (test) mol L⁻¹ s⁻¹ (median). That is the vertical streak in the MG parity panel.
- In 7 of 120 cases, a species at zero concentration gets a negative rate.

**Layer-2 term shares (fig. 2):**
- The h1·h2 product term carries most of the spread of the TG, ROH, DG, MG and R′CO₂R rates (60–84 %).
- The additive h3 edge carries most of the GL rate (82 %).

**Layer 1 (fig. 1):** every hidden node takes substantial input from several variables. Temperature contributes 13–18 % to each node; TG, ROH and DG together contribute 54–79 %; MG, GL and R′CO₂R contribute ≤ 17 % each.

## Interpretation (conservative)

**Edge-level observations** (descriptions of the learned functions, not chemistry):
- All edges are smooth and mostly monotone over the training range. A few are non-monotone, e.g. TG → h2, MG → h3, R′CO₂R → h4.
- Outside the training range (dotted), several edges bend, so the edge shapes say nothing about extrapolation.
- Temperature enters every hidden node through a monotone edge (nearly linear for h1 and h3).
- The share percentages describe how much each edge varies over the training data. They are not effect sizes of a chemical variable, because the inputs are strongly correlated along the trajectories.

**Full network** (this is where chemical meaning can be judged; the reference is the known model):
- Near the data, ChemKAN reproduces the main rates (TG, ROH, DG, R′CO₂R) within about 10 %.
- **Where it is not chemically faithful:**
  - It does not reproduce the sequential structure TG → DG → MG → GL: the model can predict production of later intermediates at t = 0.
  - It slightly violates non-negativity at zero concentrations.
  - The MG rate is the least accurate (relative RMSE 0.24–0.28).
- Agreement with integrated trajectories does not establish that the instantaneous rates or reaction mechanism have been identified.

**Hidden nodes:**
- Over the training states, h1 correlates with the true r1 (Pearson r = −0.89) and h3 with r3 (0.88). Other nodes correlate with several rates at once (`summary.json`, labelled descriptive).
- Every rate in this system is driven by the same trajectories, so these correlations are confounded. They are not evidence that a hidden node represents a reaction, and no node is interpreted as one.

**Scope:** one model and one seed (the noise-free Fig. 5 run), evaluated at the 900 reference states. The edge functions and rate comparisons describe the learned model over those states; they do not establish causal feature importance or identify individual chemical reactions.
