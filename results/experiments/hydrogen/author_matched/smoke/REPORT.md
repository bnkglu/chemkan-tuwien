# Author-matched hydrogen: smoke report

Single runs (seed 0), 2,000 epochs per stage: a pipeline and settings check, not results. Everything below is read from the files in this directory.

## 1. Solve mode and precision (`solve_mode_dtype.json`)

| comparison at the initial parameters | prediction rel. L2 | gradient rel. L2 | gradient cosine |
|---|---|---|---|
| float32: per_trajectory vs batched | 8.79e-07 | 3.84e-05 | 0.9999999995 |
| float64: per_trajectory vs batched | 4.36e-07 | 1.43e-05 | 0.9999999999 |
| batched: float32 vs float64 | 6.62e-07 | 7.67e-06 | 1.0000000000 |

| 50 Adam steps | s/epoch | loss step 1 | loss step 50 |
|---|---|---|---|
| batched_float32 | 0.166 | 16.269026 | 14.360485 |
| per_trajectory_float32 | 2.020 | 16.269028 | 14.360500 |
| batched_float64 | 0.219 | 16.269029 | 14.360485 |

Choice for all runs below: **batched, float32** (same curve to ~1e-6, fastest).

## 2. Scale references

- Largest normalized species derivative in the training data (100-point grid, `np.gradient`): **8.1e+04 /s**; temperature: 2.85e+04 /s.
- Cantera -h_i/c_p at the (1050 K, phi 0.9) training initial composition, converted to normalized units w_i * Y_range_i / T_range (scale reference only, not a target):

| H2 | H | O | O2 | OH | H2O | HO2 | H2O2 | N2 | norm |
|---|---|---|---|---|---|---|---|---|---|
| -0.167 | -0.58 | -0.112 | -0.0628 | -0.028 | 1.02 | -8.29e-05 | 1.08e-05 | -0.00631 | 1.2 |

## 3. Stage 1

| run | t_ref | epochs | NaN | Eq. 18 @1 / 500 / 1000 / 2000 | time-avg @2000 | PINN @2000 | max abs KAN_kin raw / scaled @2000 | raw growth last 200 / 500 ep | s/epoch (last 500) |
|---|---|---|---|---|---|---|---|---|---|
| S1a | 0.0006 | 2000 | no | 16.2 / 1.89 / 0.985 / 0.573 | 0.00573 | 0.52 | 15.5 / 2.58e+04 | +40.4% / +58.0% | 0.405 |
| S1b | 1.0 | 2000 | no | 16.2 / 15.5 / 11.8 / 4.01 | 0.0401 | 0.85 | 1.66e+03 / 1.66e+03 | +17.6% / +65.9% | 0.164 |

The scaled column is comparable with the data's 8.1e+04 /s.

## 4. Stage 2 arms

| arm | from | warm-up | epochs (warm-up + S2) | NaN | Eq. 18 @1 / 1000 / 2000 | time-avg @2000 | max abs KAN_kin scaled @2000 | w_norm @S2 start / end | s/epoch (last 500) |
|---|---|---|---|---|---|---|---|---|---|
| A | S1a | 0 | 0 + 2000 | no | 31.9 / 0.517 / 0.433 | 0.00433 | 3.05e+04 | 3e-05 / 0.712 | 0.368 |
| B | S1a | 1000 | 1000 + 2000 | no | 14.3 / 2.3 / 0.663 | 0.00663 | 1.96e+04 | 2.36 / 2.34 | 0.342 |
| C | S1b | 0 | 0 + 2000 | no | 7.55 / 2.23 / 1.59 | 0.0159 | 4.01e+03 | 3e-05 / 1.01 | 0.148 |
| D | S1b | 1000 | 1000 + 2000 | no | 4.25 / 6.47 / 2.37 | 0.0237 | 2.96e+03 | 2.13 / 1.77 | 0.103 |

Final thermo.linear weights (normalized units) vs the Cantera scale reference:

| run | H2 | H | O | O2 | OH | H2O | HO2 | H2O2 | N2 | norm |
|---|---|---|---|---|---|---|---|---|---|---|
| Cantera ref. | -0.167 | -0.58 | -0.112 | -0.0628 | -0.028 | 1.02 | -8.29e-05 | 1.08e-05 | -0.00631 | 1.2 |
| A | -0.187 | 0.613 | 0.0581 | -0.203 | 0.145 | 0.167 | 0.00819 | 0.0222 | -0.0449 | 0.712 |
| B | -0.276 | -1.58 | 0.237 | -0.228 | 0.409 | 0.322 | 0.0739 | -0.878 | -1.32 | 2.34 |
| C | -0.285 | 0.32 | 0.271 | -0.205 | 0.302 | 0.241 | 0.174 | 0.182 | 0.706 | 1.01 |
| D | -0.533 | -0.464 | 0.68 | 0.168 | 0.67 | 0.337 | -0.719 | -0.777 | 0.669 | 1.77 |

Warm-up B: w_norm 3e-05 -> 2.36 over 1000 epochs; Eq. 18 31.9 -> 14.4.

Warm-up D: w_norm 3e-05 -> 2.13 over 1000 epochs; Eq. 18 7.55 -> 4.25.

## 5. Evaluation (`eval/*.json`, all 36 conditions)

| run | Eq. 18 median train / test | held-out (1150 K, 1.3) Eq. 18 | held-out rise pred / ref [K] | held-out peak T pred / ref [K] | held-out delay pred / ref [s] | held-out max dT/dt pred / ref [K/s] | Fig. 8B median rise pred / ref [K] | Fig. 8B median abs rel delay error | 950 K rises pred [K] | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| S1a | 0.534 / 0.585 | 0.585 | Stage 1: T is observed | | | | | | | 0 |
| S1b | 4.02 / 7.63 | 7.63 | Stage 1: T is observed | | | | | | | 0 |
| A | 0.405 / 0.411 | 0.411 | 1.77e+03 / 1.61e+03 | 2915.2 / 2761.6 | 6.7e-05 / 6.3e-05 | 4.23e+07 / 8.28e+07 | 1.63e+03 / 1.58e+03 | 0.0421 | 11.6, 0, 0, 0, 0, 0 | 0 |
| B | 0.596 / 0.618 | 0.618 | 1.81e+03 / 1.61e+03 | 2957.6 / 2761.6 | 4.9e-05 / 6.3e-05 | 2.74e+07 / 8.28e+07 | 1.66e+03 / 1.58e+03 | 0.135 | 30.4, 4.53, 0.877, 3.53, 16.6, 30.6 | 0 |
| C | 1.67 / 2.24 | 2.24 | 1.74e+03 / 1.61e+03 | 2892.5 / 2761.6 | 3.9e-05 / 6.3e-05 | 8.2e+06 / 8.28e+07 | 1.58e+03 / 1.58e+03 | 0.349 | 0, 0, 0, 0, 0, 0 | 0 |
| D | 2.52 / 3.81 | 3.81 | 1.82e+03 / 1.61e+03 | 2972.5 / 2761.6 | 8.8e-05 / 6.3e-05 | 6.03e+06 / 8.28e+07 | 1.61e+03 / 1.58e+03 | 0.298 | 0, 0, 0, 0, 0, 0 | 0 |

## 6. Selection rule for the long run

Eligible only if all hold: (a) all Stage-2 epochs (and warm-up) completed, no NaN; (b) final w_norm >= 0.1; (c) Fig. 8B median predicted rise >= 20 % of the reference median; (d) median predicted rise of the six 950 K cases <= 20 K. Pick: lowest median total loss (`loss` column) over the last 200 Stage-2 epochs.

| arm | (a) complete, no NaN | (b) final w_norm | (c) rise ratio | (d) 950 K median rise [K] | eligible | median loss last 200 |
|---|---|---|---|---|---|---|
| A | pass | 0.712 (pass) | 1.03 (pass) | 0 (pass) | yes | 0.43716 |
| B | pass | 2.34 (pass) | 1.05 (pass) | 10.6 (pass) | yes | 0.67912 |
| C | pass | 1.01 (pass) | 0.999 (pass) | 0 (pass) | yes | 1.6371 |
| D | pass | 1.77 (pass) | 1.02 (pass) | 0 (pass) | yes | 2.4702 |

**Pick: A** (t_ref 0.0006, warm-up 0).
Projection for 1e5 + 1e5 epochs at the measured rates (Stage 1 0.405 s/epoch from S1a, Stage 2 0.368 s/epoch from the last 500 epochs of A): **21.5 h (0.89 days)**; budget 6.0 days -> within budget. Rates measured at 2,000 epochs; later epochs may be slower if the dynamics stiffen.

## 7. Recommendation (from these results only)

- **Settings:** batched float32 solves (same training as per-trajectory and float64 to ~1e-6, 12x / 1.3x faster).
- **t_ref:** 6e-4 s over 1 s. With the same budget, t_ref 6e-4 reaches lower Stage-1 loss (0.573 vs 4.01 Eq. 18 at 2,000 epochs) and lower Stage-2 loss (A 0.433 / B 0.663 vs C 1.59 / D 2.37), and its kinetic output gets closer to the data's derivative scale (2.6e4 vs 1.7e3 /s scaled, data 8.1e4 /s).
- **Warm-up:** not needed at this budget. Warm-up raised w_norm to ~2 quickly, but without it A reaches a lower loss, a smaller ignition-delay error (4.2 % vs 13.5 % median over the Fig. 8B set) and no 950 K false ignitions (B: median 10.6 K, up to 30.6 K).
- **Long run:** A's settings. Expect the per-epoch cost to rise: Stage-1 max |KAN_kin| was still growing ~40 % per 200 epochs at epoch 2,000, so the 21.5 h projection (0.405 / 0.368 s per epoch) is a lower bound.
- **Caveats:** one seed per arm and 2,000 epochs per stage, against the authors' 1e5-2e5. All arms under-predict the held-out peak dT/dt (A 4.2e7 vs 8.3e7 K/s) while over-predicting the rise (1.77e3 vs 1.61e3 K); these are early-training values.
