# ChemKAN inference-time benchmark (final, IDP report)

ChemKAN against the detailed-chemistry reference on identical initial-value problems, in
two implementations each: Python (PyTorch ChemKAN; Cantera / SciPy LSODA) and Julia
(a port of the same trained ChemKAN; Arrhenius.jl / the released biodiesel kinetics).
Measured on an Apple M1 (8 cores, 1 thread used): hydrogen and biodiesel on 2026-10-15,
methane on 2026-10-16 after its training had finished. Single seed, one checkpoint per system
(methane: the selected 180k snapshot, and the final 200k model for comparison).

| file | content |
|---|---|
| `run_all.sh` | hydrogen and biodiesel timing runs in sequence, then the report |
| `run_methane.sh` | methane timing runs (180k in Python and Julia, Julia references, final 200k in Python), then the report |
| `benchmark.py` | Python benchmark (one system per call): timing and accuracy in one run |
| `julia/export_inputs.py` | weights, normalization, initial states, grid, tolerances and the Python reference trajectories for Julia (`julia/inputs_<system>.npz`) |
| `julia/chemkan_julia.jl` | ChemKAN in Julia (allocation-free port of the PyTorch model, OrdinaryDiffEq Tsit5) |
| `julia/make_arrhenius_mech.py` | Arrhenius.jl mechanism files (`julia/mechanism/`) from Cantera's `h2o2.yaml` / `gri30.yaml` |
| `julia/reference_julia.jl` | Arrhenius.jl reactor (hydrogen, methane) and the released-example kinetics (biodiesel) |
| `julia/Project.toml`, `Manifest.toml` | Julia environment (Arrhenius.jl from GitHub master, Sundials, OrdinaryDiffEq) |
| `make_report.py` | `results_table.{md,csv}` and `runtime_vs_accuracy.{png,pdf}` |
| `<system>/results.json`, `latency_samples.csv`, `per_condition.csv` | Python results (all settings, timings, accuracy, versions, sha256); `methane_final200k/` = the final 200k methane model |
| `julia/julia_*_<system>.json`, `*_pred.npz` | Julia timings and trajectories |
| `run_<system>.log`, `julia/run_*.log` | console output of the measured runs |

Re-run: `bash results/experiments/inference_benchmark/run_all.sh` from the repository root
(about 10 minutes; Julia 1.12 with `julia --project=results/experiments/inference_benchmark/julia -e 'using Pkg; Pkg.instantiate()'` once).

## Problems, models and references

Same initial states, the same 100 uniformly spaced output times and full-state outputs on
every side.

| system | conditions, window | ChemKAN | references |
|---|---|---|---|
| hydrogen | 36 (35 train, 1 held-out), 0.6 ms | author-matched Stage 2, 200k epochs (`hydrogen/author_matched/main/S2_seed0_200k/stage2_final.pt`), float32 | Cantera 3.0.0 IdealGasConstPressureReactor with `h2o2.yaml` (the data's generator: adiabatic, 1 atm, CVODES); Arrhenius.jl (GitHub master) with the same mechanism and the constant-pressure reactor of its ignition example, Sundials CVODE_BDF |
| methane | 36 (35 train, 1 held-out: 1550 K, phi 1.3), 5 ms | author-matched Stage 2, the **epoch-180,000 snapshot** (`methane/author_matched/main/S2_seed0/archive/stage2_S2_seed0_epoch180000.pt`), float32; the final 200k model (`stage2_final.pt`) sits on a training-loss spike (training-condition full-trajectory MSE 2.89e-3 vs 1.22e-3), and 180k is the best 5k snapshot by training-condition MSE (notebook 15) | Cantera with `gri30.yaml` (GRI-Mech 3.0, 53 species, 325 reactions; Ar dropped from the state), same reactor as hydrogen; Arrhenius.jl with the same mechanism, CVODE_BDF |
| biodiesel | 30 (20 train, 10 test), 30 s | 7b noise-free run (`biodiesel/author_repo_match/figures/fig05/chemkan_noise00_seed0`), float64 | the known mass-action kinetics: SciPy LSODA with analytic Jacobian (the data generator); in Julia the released example's `trueODEfunc` with CVODE_BDF, and with the example's own data-generation solver AutoTsit5(Rosenbrock23) at default tolerances. No Cantera mechanism exists for biodiesel |

Checks (all in the result files):
- the Python reference reproduces the stored datasets (hydrogen max |ΔY| 3e-10, |ΔT| 6e-6 K;
  methane exactly; biodiesel 3e-12 mol/L on the dataset's 30-point grid);
- Arrhenius.jl reproduces Cantera (hydrogen max |ΔT| 0.026 K, MSE 3e-15; methane max |ΔT|
  3e-4 K, MSE 2e-18; both at rtol 1e-9); its mechanism
  `.npz` written by `make_arrhenius_mech.py` equals the package's own `gri30.yaml.npz` in all
  rate arrays (the package's bundled `h2o2.yaml` has no N2, so it is not used);
- the Julia ChemKAN reproduces the PyTorch trajectories (max deviation 2e-9 for biodiesel at
  rtol 1e-10, 8e-5 for hydrogen and 1.1e-4 for methane in float32 at rtol 1e-6; at rtol 1e-2 the two Tsit5 step
  controllers take different steps, so trajectories differ within that tolerance);
- the PyTorch ChemKAN reproduces the earlier evaluations (hydrogen held-out 3.51e-4 = Fig. 7;
  biodiesel 30-point test mean 3.1652e-4 = the 7b value at rtol 1e-10).

Not usable: the registered Arrhenius.jl release 0.1.0 returns zero reaction rates (its
rate-of-progress function never stores its result); the GitHub master is used.

## Results

Latency: one trajectory per call, median over conditions × 25 repetitions. Speed-up =
reference time / ChemKAN time (< 1: ChemKAN slower). MSE: time-averaged, model-normalized,
median over conditions, against the Python reference at its generation tolerance. Full
table: `results_table.md`.

**Hydrogen** (references at the data-generation tolerance: Cantera 3.94 ms, Arrhenius.jl
10.1 ms per trajectory)

| ChemKAN | rtol / atol | latency | vs Cantera | vs Arrhenius.jl | MSE all / held-out |
|---|---|---|---|---|---|
| PyTorch | 1e-6 / 1e-8 (trained) | 111.5 ms | 0.035x | 0.090x | 4.64e-4 / 3.51e-4 |
| PyTorch | 1e-2 / 1e-6 (biodiesel tol.) | 48.9 ms | 0.081x | 0.21x | 5.64e-4 / 4.45e-4 |
| Julia | 1e-6 / 1e-8 (trained) | **1.60 ms** | **2.5x** | **6.3x** | 4.64e-4 / 3.51e-4 |
| Julia | 1e-2 / 1e-6 (biodiesel tol.) | **0.71 ms** | **5.5x** | **14x** | 4.77e-4 / 3.52e-4 |

The references at looser tolerances: Cantera and Arrhenius.jl at rtol 1e-6 / atol 1e-12 take
2.04 / 4.87 ms (MSE 4e-10 / 3e-10), so the Julia ChemKAN is 1.3x / 3.0x faster than those.
At rtol 1e-2 / atol 1e-6 both references are fast (0.20 / 0.14 ms) but lose the ignition
(MSE 0.20, temperature errors up to 1,680 K), while ChemKAN's accuracy barely changes. At
the ChemKAN tolerance atol 1e-8 on mass fractions, Cantera fails for 3 and Arrhenius.jl for
8 of 36 conditions (non-finite rates).

**Methane** (references at the data-generation tolerance: Cantera 39.6 ms, Arrhenius.jl
134 ms per trajectory)

| ChemKAN (180k) | rtol / atol | latency | vs Cantera | vs Arrhenius.jl | MSE all / held-out |
|---|---|---|---|---|---|
| PyTorch | 1e-6 / 1e-8 (trained) | 117 ms | 0.34x | 1.15x | 9.94e-4 / 1.16e-3 |
| PyTorch | 1e-2 / 1e-6 (biodiesel tol.) | 47.4 ms | 0.84x | 2.8x | 1.00e-3 / 1.17e-3 |
| Julia | 1e-6 / 1e-8 (trained) | **9.93 ms** | **4.0x** | **13.5x** | 9.94e-4 / 1.16e-3 |
| Julia | 1e-2 / 1e-6 (biodiesel tol.) | **2.87 ms** | **13.8x** | **47x** | 9.94e-4 / 1.16e-3 |

At looser reference tolerances: Cantera at rtol 1e-6 / atol 1e-12 takes 21.9 ms (MSE 4e-12)
and Arrhenius.jl 73.2 ms (MSE 5e-12), so the Julia ChemKAN is still 2.2x / 7.4x faster.
Unlike hydrogen, Cantera integrates every methane condition at the ChemKAN tolerance
(1e-6 / 1e-8: 14.6 ms, MSE 1.2e-7) and keeps the ignition at rtol 1e-2 / atol 1e-6 (7.7 ms,
MSE 1.9e-3, worse than ChemKAN's 9.9e-4); at that setting the Julia ChemKAN is 2.7x faster
than Cantera and more accurate. Arrhenius.jl fails at rtol 1e-2 for 1 of 36 conditions.
Physical accuracy (180k, median): peak-temperature error 21 K, ignition delay equal on the
50 µs output grid, element mass drift 4.4 %. PyTorch batched over all 36 conditions reaches
the Cantera throughput (1.02x). The final 200k model costs the same (114 ms in PyTorch) but is
less accurate: MSE 1.47e-3 / held-out 1.27e-3, ignition-delay error 7.3 % (median),
largest instantaneous |ΔT| 433 K vs 200 K (`methane_final200k/`).

**Biodiesel** (references: LSODA 1.20 ms, Julia CVODE_BDF 0.20 ms at rtol 1e-10; the
example's AutoTsit5(Rosenbrock23) at default tolerances 0.049 ms, MSE 5e-8)

| ChemKAN | rtol / atol | latency | vs LSODA | vs Julia CVODE_BDF | MSE all / test |
|---|---|---|---|---|---|
| PyTorch | 1e-10 / 1e-12 (7b evaluation) | 67.2 ms | 0.018x | 0.003x | 7.98e-5 / 1.61e-4 |
| PyTorch | 1e-2 / 1e-6 (trained) | 8.50 ms | 0.14x | 0.024x | 8.68e-5 / 1.62e-4 |
| Julia | 1e-10 / 1e-12 | 0.69 ms | 1.7x | 0.29x | 7.98e-5 / 1.61e-4 |
| Julia | 1e-2 / 1e-6 | 0.114 ms | 10.5x | 1.8x | 7.98e-5 / 1.61e-4 |

Throughput (all conditions per call): batching all conditions into one solve helps PyTorch
(5-6x for hydrogen, 3-5x for methane, 24-28x for biodiesel over sequential calls) but not Julia, where one
solve per trajectory is already cheap and a batched solve must follow the stiffest
trajectory's steps (hydrogen Julia: 652 traj/s sequential vs 168 traj/s batched).

Physical accuracy (median over conditions; the same with either implementation): hydrogen
peak-temperature error 10.6 K (13.2 K at rtol 1e-2), ignition delay equal on the 6.06 µs
output grid for the median condition, element mass drift 6 % (the conservation term is only
a soft penalty); biodiesel final FAME error 2.7 %.

## Reading

- The implementation decides the sign of the speed-up. The same trained ChemKAN, giving the
  same trajectories, takes 111.5 ms per hydrogen trajectory in PyTorch and 1.60 ms in Julia
  (70x): in PyTorch every right-hand-side call of a 344-parameter model is dominated by
  framework dispatch, not arithmetic.
- In Julia, ChemKAN is faster than both hydrogen references at the data-generation
  tolerance (2.5x Cantera, 6.3x Arrhenius.jl) and still faster at a practical reference
  tolerance (1.3x / 3.0x at rtol 1e-6, atol 1e-12), with the model's own accuracy (MSE
  4.6e-4). The paper reports 2.0x against Arrhenius.jl; our setting differs (hardware,
  tolerances, timing scope unknown), so the numbers are not directly comparable.
- The advantage grows with the mechanism. From hydrogen (9 species, 29 reactions) to methane
  (52 species, 325 reactions) the reference cost rises 10x (Cantera 3.9 to 39.6 ms) and 13x
  (Arrhenius.jl 10.1 to 134 ms), while the Julia ChemKAN rises 6x (1.6 to 9.9 ms); its speed-up
  goes from 2.5x to 4.0x against Cantera and from 6.3x to 13.5x against Arrhenius.jl. In PyTorch
  the methane ChemKAN costs about the same as for hydrogen (117 vs 112 ms), because framework
  overhead, not the 52-species arithmetic, dominates; it is still 3x slower than Cantera per
  trajectory.
- ChemKAN tolerates loose solver tolerances (rtol 1e-2: MSE 4.8e-4 in Julia), the stiff
  references do not; a tolerance-matched comparison at rtol 1e-2 would be meaningless for the
  references.
- For biodiesel (3 reactions) there is nothing for a surrogate to save: the released
  kinetics in Julia are faster than ChemKAN at matched tolerance (0.20 vs 0.69 ms), and the
  example's own solver at default tolerances (0.049 ms, MSE 5e-8) is faster than any ChemKAN
  setting. The 10x against LSODA reflects SciPy's per-call overhead, not the chemistry.

Limits: one machine, one thread, one seed and checkpoint per system; two gas mechanisms
(29 and 325 reactions), so the trend with mechanism size rests on two points; the methane
model is not converged (training loss still decreasing at 200k). The machine was busy during the runs (1-min load
average 9 to 15 on 8 cores, mostly iCloud and Spotlight indexing); repeat measurements
agreed within about 3 % (Cantera 4.07 vs 3.94 ms, PyTorch ChemKAN 108 vs 111.5 ms between
the 09:21 and 09:52 runs). Wall-clock times include each language's call overhead (the
Cantera reactor loop and SciPy's `solve_ivp` are driven from Python).

## Methods paragraph (for the report)

Inference cost was measured against the detailed-chemistry reference on identical
initial-value problems. For hydrogen, the author-matched Stage-2 ChemKAN (200,000 epochs,
float32) was compared with an adiabatic constant-pressure reactor using the `h2o2`
mechanism, solved with Cantera 3.0.0 (the data generator) and with Arrhenius.jl and Sundials
CVODE_BDF; all were given the same initial temperature, pressure (1 atm) and composition
(from the equivalence ratio, air as O2 + 3.76 N2) for the 36 training and held-out
conditions and returned the full state at the same 100 uniformly spaced times over 0.6 ms.
For methane, the same comparison used the GRI-Mech 3.0 mechanism (53 species, 325 reactions),
36 conditions over 5 ms and the Stage-2 ChemKAN snapshot at 180,000 epochs, selected as the
5,000-epoch snapshot with the lowest training-condition error (the final model at 200,000
epochs lay on a training-loss spike).
For biodiesel, the noise-free ChemKAN model of Section X (float64) was compared with the known
mass-action kinetics, solved with SciPy's LSODA (the data generator) and, in Julia, with
CVODE_BDF and with the released example's AutoTsit5/Rosenbrock23 solver, for the 30
conditions over 30 s at 100 output times. Each ChemKAN model was run in PyTorch
(torchdiffeq) and in a direct Julia port of the same trained weights (OrdinaryDiffEq), both
with the Tsit5 method at the training tolerances (hydrogen rtol 1e-6, atol 1e-8; biodiesel
rtol 1e-2, atol 1e-6) and at one additional tolerance (hydrogen rtol 1e-2, atol 1e-6;
biodiesel rtol 1e-10, atol 1e-12); the Julia port reproduced the PyTorch trajectories to
8e-5 (hydrogen, float32) and 2e-9 (biodiesel) in normalized units. The references were
solved at their data-generation tolerances (rtol 1e-9, atol 1e-15 for hydrogen and methane; rtol 1e-10,
atol 1e-12 for biodiesel), which define the accuracy target, and at looser tolerances. The
timed region contained the per-trajectory initialization (thermodynamic state and reactor
network, or rate constants; normalization of the initial state; ODE problem construction),
the full integration and the conversion of the outputs to physical units; model, mechanism
and data loading, compilation, training, reference-data generation, accuracy evaluation,
plotting and disk I/O were excluded. All runs used one thread of an Apple M1 CPU. After
three warm-up calls, each measurement was repeated 25 times, and we report medians and
interquartile ranges of the single-trajectory latency and of the throughput for all
conditions (sequential calls, and for ChemKAN also one batched solve). The speed-up was
computed as reference runtime divided by ChemKAN runtime. Accuracy was evaluated on the same
solutions as the time-averaged mean-squared error in the model's min-max normalized
coordinates against the reference, with peak temperature, ignition delay (argmax dT/dt) and
element conservation for hydrogen and methane and the final biodiesel yield for biodiesel. For
hydrogen, the Julia ChemKAN (1.60 ms per trajectory at its training tolerance) was 2.5 times
faster than Cantera and 6.3 times faster than Arrhenius.jl at the generation tolerance, with
a time-averaged MSE of 4.6e-4; the PyTorch implementation of the same model was 70 times
slower and slower than both references. For methane, the Julia ChemKAN (9.9 ms) was 4.0 times
faster than Cantera and 13.5 times faster than Arrhenius.jl, with an MSE of 9.9e-4, so the
speed-up grew with the size of the mechanism. For biodiesel, the three-reaction kinetics solved in
Julia remained faster than ChemKAN at matched tolerance.
