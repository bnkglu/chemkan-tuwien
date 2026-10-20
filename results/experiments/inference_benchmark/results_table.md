# ChemKAN inference benchmark: results

Speed-up = reference runtime / method runtime (< 1: slower than the reference). Two baselines: the Python reference at its data-generation tolerance (Cantera rtol 1e-9 or LSODA rtol 1e-10) and the Julia reference at the same tolerance (Arrhenius.jl CVODE_BDF rtol 1e-9, or the released kinetics with CVODE_BDF rtol 1e-10). Throughput speed-up uses the method's faster mode (sequential or batched) against the baseline solving the conditions sequentially. MSE: time-averaged, model-normalized, 100 output points, against the Python reference at its generation tolerance. Latency: one trajectory per call, median and IQR over conditions x repetitions.

## hydrogen

Checkpoint `results/experiments/hydrogen/author_matched/main/S2_seed0_200k/stage2_final.pt` (epochs 200000); 36 conditions (35 train / 1 test); 100 points over 0.0006 s. Apple M1, 1 thread, 3 warm-up + 25 timed repetitions.

| implementation | method | solver | rtol / atol | precision | latency median [s] | latency IQR [s] | throughput sequential [traj/s] | throughput batched [traj/s] | speed-up latency vs Cantera ref. | speed-up throughput vs Cantera ref. | speed-up latency vs Julia ref. | speed-up throughput vs Julia ref. | MSE median (all) | MSE median (test) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python | reference | Cantera CVODES | 1e-9 / 1e-15 | float64 | 3.94e-03 | 6.87e-04 | 286 | – | 1 | 1 | 2.56 | 2.56 | 0 | 0 |  |
| Python | reference | Cantera CVODES | 1e-6 / 1e-12 | float64 | 2.04e-03 | 2.99e-04 | 562 | – | 1.93 | 1.97 | 4.93 | 5.05 | 4.07e-10 | 1.26e-10 |  |
| Python | reference | Cantera CVODES | 1e-6 / 1e-10 | float64 | 1.91e-03 | 3.10e-04 | 611 | – | 2.06 | 2.14 | 5.26 | 5.49 | 3.73e-06 | 6.84e-07 |  |
| Python | reference | Cantera CVODES | 1e-2 / 1e-6 | float64 | 1.98e-04 | 1.68e-05 | 4.74e+03 | – | 19.9 | 16.6 | 50.8 | 42.5 | 0.197 | 0.315 |  |
| Python | ChemKAN | PyTorch/torchdiffeq Tsit5 | 1e-6 / 1e-8 | float32 | 0.112 | 0.0126 | 10.3 | 61.3 | 0.0353 | 0.215 | 0.0903 | 0.55 | 4.64e-04 | 3.51e-04 | peak-T err 10.6 K, delay rel. err 0 |
| Python | ChemKAN | PyTorch/torchdiffeq Tsit5 | 1e-2 / 1e-6 | float32 | 0.0489 | 0.0138 | 24 | 119 | 0.0805 | 0.418 | 0.206 | 1.07 | 5.64e-04 | 4.45e-04 | peak-T err 13.2 K, delay rel. err 0 |
| Python | reference | Cantera CVODES | 1e-6 / 1e-8 | – | – | – | – | – | – | – | – | – | – | – | failed (non-finite rates / solver error) for 3 conditions; not timed |
| Julia | ChemKAN | OrdinaryDiffEq Tsit5 | 1e-2 / 1e-6 | Float32 | 7.11e-04 | 3.32e-04 | 1.28e+03 | 420 | 5.54 | 4.47 | 14.2 | 11.5 | 4.77e-04 | 3.52e-04 |  |
| Julia | ChemKAN | OrdinaryDiffEq Tsit5 | 1e-6 / 1e-8 | Float32 | 1.60e-03 | 2.52e-04 | 652 | 168 | 2.46 | 2.28 | 6.28 | 5.86 | 4.64e-04 | 3.51e-04 |  |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-9 / 1e-15 | Float64 | 0.0101 | 2.18e-03 | 111 | – | 0.391 | 0.39 | 1 | 1 | 2.81e-15 | 1.24e-15 |  |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-6 / 1e-8 | – | – | – | – | – | – | – | – | – | – | – | failed (non-finite rates / solver error) for 8 conditions; not timed |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-6 / 1e-10 | Float64 | 4.49e-03 | 7.60e-04 | 236 | – | 0.876 | 0.828 | 2.24 | 2.12 | 2.21e-05 | 1.10e-05 |  |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-2 / 1e-6 | Float64 | 1.41e-04 | 1.73e-05 | 7.12e+03 | – | 27.8 | 24.9 | 71.2 | 63.9 | 0.197 | 0.315 |  |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-6 / 1e-12 | Float64 | 4.87e-03 | 9.80e-04 | 217 | – | 0.809 | 0.758 | 2.07 | 1.94 | 3.06e-10 | 7.58e-12 |  |

Python reference vs stored dataset: {"same_grid": true, "max_abs_diff_species": 3.054822506065591e-10, "max_abs_diff_T_K": 5.866041192348348e-06}

## methane

Checkpoint `results/experiments/methane/author_matched/main/S2_seed0/archive/stage2_S2_seed0_epoch180000.pt` (epochs 180000); 36 conditions (35 train / 1 test); 100 points over 0.005 s. Apple M1, 1 thread, 3 warm-up + 25 timed repetitions.

| implementation | method | solver | rtol / atol | precision | latency median [s] | latency IQR [s] | throughput sequential [traj/s] | throughput batched [traj/s] | speed-up latency vs Cantera ref. | speed-up throughput vs Cantera ref. | speed-up latency vs Julia ref. | speed-up throughput vs Julia ref. | MSE median (all) | MSE median (test) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python | reference | Cantera CVODES | 1e-9 / 1e-15 | float64 | 0.0396 | 3.88e-03 | 25.3 | – | 1 | 1 | 3.39 | 3.4 | 0 | 0 |  |
| Python | reference | Cantera CVODES | 1e-6 / 1e-12 | float64 | 0.0219 | 2.14e-03 | 45.5 | – | 1.81 | 1.8 | 6.13 | 6.13 | 3.86e-12 | 6.82e-13 |  |
| Python | reference | Cantera CVODES | 1e-6 / 1e-10 | float64 | 0.0182 | 1.77e-03 | 54.8 | – | 2.17 | 2.17 | 7.38 | 7.37 | 1.07e-10 | 6.38e-11 |  |
| Python | reference | Cantera CVODES | 1e-6 / 1e-8 | float64 | 0.0146 | 1.45e-03 | 67.7 | – | 2.72 | 2.68 | 9.22 | 9.11 | 1.16e-07 | 1.70e-09 |  |
| Python | reference | Cantera CVODES | 1e-2 / 1e-6 | float64 | 7.73e-03 | 1.04e-03 | 127 | – | 5.12 | 5.01 | 17.4 | 17.1 | 1.91e-03 | 3.74e-05 |  |
| Python | ChemKAN | PyTorch/torchdiffeq Tsit5 | 1e-6 / 1e-8 | float32 | 0.117 | 0.0276 | 8.34 | 25.7 | 0.338 | 1.02 | 1.15 | 3.46 | 9.94e-04 | 1.16e-03 | peak-T err 21.4 K, delay rel. err 0 |
| Python | ChemKAN | PyTorch/torchdiffeq Tsit5 | 1e-2 / 1e-6 | float32 | 0.0474 | 0.0121 | 20 | 90.2 | 0.835 | 3.56 | 2.84 | 12.1 | 1.00e-03 | 1.17e-03 | peak-T err 24 K, delay rel. err 0 |
| Julia | ChemKAN | OrdinaryDiffEq Tsit5 | 1e-2 / 1e-6 | Float32 | 2.87e-03 | 4.32e-04 | 338 | 101 | 13.8 | 13.4 | 46.7 | 45.5 | 9.94e-04 | 1.16e-03 |  |
| Julia | ChemKAN | OrdinaryDiffEq Tsit5 | 1e-6 / 1e-8 | Float32 | 9.93e-03 | 2.64e-03 | 99.6 | 25.5 | 3.98 | 3.94 | 13.5 | 13.4 | 9.94e-04 | 1.16e-03 |  |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-9 / 1e-15 | Float64 | 0.134 | 0.0126 | 7.43 | – | 0.295 | 0.294 | 1 | 1 | 2.13e-18 | 1.35e-17 |  |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-2 / 1e-6 | – | – | – | – | – | – | – | – | – | – | – | failed (non-finite rates / solver error) for 1 conditions; not timed |
| Julia | reference | Arrhenius.jl CVODE_BDF | 1e-6 / 1e-12 | Float64 | 0.0732 | 9.81e-03 | 13.6 | – | 0.541 | 0.536 | 1.83 | 1.82 | 4.85e-12 | 7.34e-13 |  |

Python reference vs stored dataset: {"same_grid": true, "max_abs_diff_species": 0.0, "max_abs_diff_T_K": 0.0}

## biodiesel

Checkpoint `results/experiments/biodiesel/author_repo_match/figures/fig05/chemkan_noise00_seed0/checkpoint_final.pt` (epochs 10000); 30 conditions (20 train / 10 test); 100 points over 30 s. Apple M1, 1 thread, 3 warm-up + 25 timed repetitions.

| implementation | method | solver | rtol / atol | precision | latency median [s] | latency IQR [s] | throughput sequential [traj/s] | throughput batched [traj/s] | speed-up latency vs LSODA ref. | speed-up throughput vs LSODA ref. | speed-up latency vs Julia ref. | speed-up throughput vs Julia ref. | MSE median (all) | MSE median (test) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python | reference | SciPy LSODA | 1e-10 / 1e-12 | float64 | 1.20e-03 | 1.52e-04 | 793 | – | 1 | 1 | 0.169 | 0.163 | 0 | 0 |  |
| Python | reference | SciPy LSODA | 1e-6 / 1e-8 | float64 | 7.31e-04 | 1.76e-04 | 1.39e+03 | – | 1.65 | 1.75 | 0.279 | 0.286 | 4.28e-15 | 3.77e-15 |  |
| Python | ChemKAN | PyTorch/torchdiffeq Tsit5 | 1e-10 / 1e-12 | float64 | 0.0672 | 0.0184 | 15.3 | 371 | 0.0179 | 0.468 | 3.03e-03 | 0.0763 | 7.98e-05 | 1.61e-04 | final FAME rel. err 0.0266 |
| Python | ChemKAN | PyTorch/torchdiffeq Tsit5 | 1e-2 / 1e-6 | float64 | 8.50e-03 | 1.42e-03 | 117 | 3.26e+03 | 0.142 | 4.11 | 0.024 | 0.67 | 8.68e-05 | 1.62e-04 | final FAME rel. err 0.0256 |
| Julia | ChemKAN | OrdinaryDiffEq Tsit5 | 1e-2 / 1e-6 | Float64 | 1.14e-04 | 2.09e-05 | 8.75e+03 | 8.89e+03 | 10.5 | 11.2 | 1.78 | 1.83 | 7.98e-05 | 1.61e-04 |  |
| Julia | ChemKAN | OrdinaryDiffEq Tsit5 | 1e-10 / 1e-12 | Float64 | 6.90e-04 | 2.00e-04 | 1.49e+03 | 1.32e+03 | 1.74 | 1.88 | 0.295 | 0.307 | 7.98e-05 | 1.61e-04 |  |
| Julia | reference | released kinetics, CVODE_BDF | 1e-10 / 1e-12 | Float64 | 2.04e-04 | 5.09e-05 | 4.86e+03 | – | 5.91 | 6.13 | 1 | 1 | 7.04e-19 | 6.05e-19 |  |
| Julia | reference | released kinetics, AutoTsit5(Rosenbrock23) | solver default | Float64 | 4.95e-05 | 6.53e-06 | 2.01e+04 | – | 24.3 | 25.4 | 4.12 | 4.14 | 5.18e-08 | 5.36e-08 |  |

Python reference vs stored dataset: {"same_grid": false, "dataset_grid_points": 30, "max_abs_diff_on_dataset_grid_mol_per_L": 3.0193625377705757e-12}

