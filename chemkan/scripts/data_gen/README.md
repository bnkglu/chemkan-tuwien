# Data generation

Scripts that generate the ground-truth trajectories for reproducing the ChemKAN
paper (Koenig, Kim & Deng, 2025). They integrate the reference kinetic models,
run some sanity checks, and save each system as a compressed `.npz`.

**Biodiesel and hydrogen are the reproduction datasets. Methane is an optional
extension** (see the end of this file). Every `.npz` is regenerated from source; the
canonical `biodiesel.npz` / `hydrogen.npz` and the production
`hydrogen_temperature_20000.npz` cache are tracked in Git (small, and needed to reproduce
reported results), while diagnostic dense caches (50k/100k/200k) and any other `.npz`
stay untracked. See the root `.gitignore` negation rules.

## Files

| Script | What it does |
|---|---|
| `common.py` | small shared utilities (RNG, normalization, noise, diagnostics, I/O) |
| `reactor.py` | shared 0-D constant-pressure Cantera reactor helper |
| `generate_biodiesel.py` | biodiesel transesterification trajectories |
| `generate_hydrogen.py` | hydrogen–air combustion trajectories |
| `generate_all.py` | runs the biodiesel + hydrogen generators |
| `verify_data.py` | sanity checks on a generated `.npz` |
| `extensions/generate_methane.py` | optional methane extension (not run by default) |
| `data_generation_walkthrough.ipynb` | notebook that traces each step with intermediate outputs and plots |

## Quickstart

Run from this directory (`chemkan/scripts/data_gen`):

```bash
pip install -r ../../requirements.txt

# generate the two reproduction datasets (biodiesel + hydrogen)
python generate_all.py --out-dir ../../data/generated

# optionally also generate the hydrogen fine grid (hydrogen_fine.npz)
python generate_all.py --out-dir ../../data/generated --include-fine

# check them
python verify_data.py ../../data/generated/hydrogen.npz
python verify_data.py ../../data/generated/biodiesel.npz --system biodiesel
```

Individual generators accept CLI arguments (e.g. `--out`, `--seed`, `--n-train`,
`--n-test`, `--phis`) for reproducibility and quick experiments:

```bash
python generate_biodiesel.py --out ../../data/generated/biodiesel.npz --seed 0
python generate_hydrogen.py  --out ../../data/generated/hydrogen.npz
python generate_hydrogen.py  --out ../../data/generated/hydrogen_fine.npz --grid fine # Figure 8 (A) 441 total data. 406 of which were unseen, 35 were seen during training.
```

### Dense Stage-1 temperature cache (supervisor-approved)

Hydrogen Stage 1 integrates species only, with temperature supplied externally as
`T(t)`. The paper reads Stage-1 temperature from the training data but does not
specify how it is evaluated at the adaptive ODE solver's internal times. The
original reproduction used the sparse 50-point trajectory through the linear
`ObservedTemperature` provider (`src/chemkan/temperature.py`); the
supervisor-approved final implementation instead precomputes a **dense** Cantera
temperature trajectory and reads it through the *same* linear `ObservedTemperature`.

`--temperature-only` reuses the canonical hydrogen Cantera setup (mechanism,
fuel/oxidizer, pressure, tolerances, coarse IC grid, 35/1 split, ordering) and
saves **only** the temperature trajectory — no dense species, normalization, or
ignition data. It does not touch `hydrogen.npz`.

```bash
# default dense resolution: 20000 points over the same 0.6 ms interval
python generate_hydrogen.py --temperature-only --n-points 20000 \
    --out ../../data/generated/hydrogen_temperature_20000.npz
```

Archive schema: `t (N,)`, `train_T (N,35,1)`, `test_T (N,1,1)`, `train_ics (35,2)`,
`test_ics (1,2)`, plus `n_points`, `t_end`, `mechanism`, `pressure`, `rtol`, `atol`,
`metadata`. `train_T[:, b, :]` follows the exact ordering of the 50-point hydrogen
training conditions. Load it with `_data.load_hydrogen_temperature(split, n_points)`,
which validates the file against the canonical archive and fails loudly on mismatch.

Training consumes it via `train_hydrogen.py --stage1-temperature-source dense-cantera`
(the default). Important: this is still **linear interpolation** on a fine grid — it
reduces interpolation error, it does not eliminate interpolation, because the solver's
internal query times almost never land exactly on the grid. The Stage-1 output grid
and species targets stay on the canonical 50 points; only `T(t)` becomes dense.
`20000` is a reproduction choice, not a paper-specified value. The original 50-point
provider remains available as an ablation via
`--stage1-temperature-source training-data`.

### Interactive walkthrough

`data_generation_walkthrough.ipynb` runs the *same* functions as the scripts and
shows each intermediate step (rate constants, a single trajectory, the full
train/test split, normalization, noise, ignition delay, verification) with inline
plots. It reuses the generators — it does not re-implement them. It needs the
runtime plus a small plotting/notebook stack:

```bash
pip install -r ../../requirements.txt -r ../../requirements-dev.txt
jupyter lab data_generation_walkthrough.ipynb   # or: jupyter notebook / VS Code
```

### Cantera version

ChemKAN cites Cantera 3.0.0. This repository pins `cantera==3.0.0` for
reproduction. The environment also pins `numpy==1.26.4`, since Cantera 3.0.0
uses the NumPy 1.x ABI.

## Generated files

| File | Cases | Shape of `states` | `state_layout` |
|---|---|---|---|
| `biodiesel.npz` | 20 train + 10 test | `(cases, 30, 6)` | `species_only` |
| `hydrogen.npz` | 35 train + 1 test | `(cases, n_points, 10)`, default `n_points=50` | `species_then_temperature` |
| `hydrogen_fine.npz` *(optional)* | 35 train + 406 test | `(441, n_points, 10)`, default `n_points=50` | `species_then_temperature` |
| `methane.npz` *(optional)* | 35 train + 1 test | `(cases, 1001, 53)` | `species_then_temperature` |

For hydrogen the last axis is `10 = 9 species + temperature`; the default
`n_points=50` follows ChemNODE-style saved trajectories, and `--n-points 601`
gives the 1 µs high-resolution grid.

**The two layouts are not interchangeable — check `state_layout` before
indexing.** For the combustion systems, `u = [Y_1, ..., Y_m, T]` with
temperature last.

Biodiesel is isothermal, so the saved `train_states` has shape `(20, 30, 6)`
(6 species only), and the per-case constant temperature is stored separately in
`train_T` / `test_T`. The model input can be made `(20, 30, 7)` by appending the
constant temperature column with `common.with_temperature`:

```python
from common import load, with_temperature

d = load("../../data/generated/biodiesel.npz")
u_in = with_temperature(d["train_states"], d["train_T"])
# shape: (n_train, n_times, 7)
# defaults: (20, 30, 7)
# 7 = 6 biodiesel species + one constant temperature column
# temperature is appended as a model input but is not integrated for biodiesel
```

For biodiesel, the saved `u_min` and `u_max` correspond to the six species
profiles used as training targets. If later model code normalizes the appended
temperature input as well, it should compute a separate input scaler from
`train_T` or from the full appended training input.

Each archive also carries `t`, `species`, `mechanism`, `u_min`, `u_max`, and a
concise `metadata` JSON string (system, generator, seed, mechanism, species,
time grid, normalization method, library versions):

```python
import json, numpy as np
print(json.loads(str(np.load("../../data/generated/hydrogen.npz")["metadata"])))
```

Explanations of the modelling choices live in this README, not in the `.npz`.

## Biodiesel setup

Three-reaction transesterification (ChemKAN Eqs. 19–21), integrated with LSODA
and an analytic Jacobian. `Ea = [14.54, 6.47, 14.42]` kcal/mol,
`ln A = [18.60, 7.93, 19.13]`, isothermal at `T ~ U(323, 343)` K,
`[TG]_0, [ROH]_0 ~ U(0.5, 2)`, other species starting at zero, 30 s window,
30 samples.

The ChemKAN paper gives the three biodiesel reactions and Arrhenius rate
constants, but it does not explicitly write the full ODE system. This
implementation treats the written bimolecular reactions as irreversible
second-order mass-action steps:

```
r1 = k1 * TG * ROH
r2 = k2 * DG * ROH
r3 = k3 * MG * ROH
```

This is the standard interpretation of reactions with one glyceride and one
methanol reactant on the left-hand side, but it is documented because a
first-order implementation would generate different trajectories with the same
Arrhenius constants. This point can be confirmed with the supervisor if exact
chemical interpretation becomes important.

The activation energies are reported in kcal/mol, so the code uses
R = 1.987e-3 kcal/(mol*K).

(`verify_data.py` checks stoichiometric consistency — the glyceride backbone
`TG + DG + MG + GL` stays constant and methanol consumed equals ester produced.
These are consistency checks on the implemented system; they do not by
themselves prove that the rate order is correct.)

## Hydrogen setup

Adiabatic constant-pressure 0-D reactors at 1 atm. Cantera's `h2o2.yaml` is the
H₂/O₂ submechanism of GRI-Mech 3.0; dropping Ar (inert, absent from air) gives
the 9 species and 29 reactions the paper quotes:
`[H2, H, O, O2, OH, H2O, HO2, H2O2, N2]`, plus temperature. 0.6 ms window with 50
saved time points by default (see the sample-count note below; `--n-points 601`
gives the 1 µs high-resolution grid). The initial conditions are a grid of six temperatures
(950–1200 K) and six equivalence ratios; `(1150 K, phi = 1.3)` is withheld as
the test case, giving 35 train / 1 test.

**On the equivalence-ratio grid.** ChemKAN reports 36 hydrogen cases and 35
training cases after withholding one case, but the printed list of equivalence
ratios contains only five values (0.7, 0.9, 1.1, 1.3, 1.5).

- Those counts require six equivalence-ratio values, and Fig. 8 shows the range starting at 0.5, so this implementation uses `phi = {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}`. The omitted value appears to be `phi = 0.5`.
- Use `--phis 0.7 0.9 1.1 1.3 1.5` to take the printed list literally instead.

Saved hydrogen trajectories use 50 time points by default. Ignition delay is
computed on a denser diagnostic grid, controlled by `--ignition-points`
defaulting to 601, to avoid quantizing the ignition-delay diagnostic.

The fine grid (`--grid fine`) is a 21×21 sweep over the same ranges for a
generalization study; the 35 coarse training cases stay in the training set, so
its normalization matches the coarse file. (Koenig et al. ChemKAN, p. 15, last paragraph)

## Verification

`verify_data.py` runs sanity checks before any training:

```bash
python verify_data.py ../../data/generated/hydrogen.npz
python verify_data.py ../../data/generated/biodiesel.npz --system biodiesel
```

- **Combustion:**
  - mass fractions non-negative and summing to 1
  - H/O/N element conservation along each trajectory
  - non-negative temperature; normalized training states inside [0, 1].
  - The Cantera mechanism used for element conservation is read from the archive's `mechanism` key, not guessed from the species count.
- **Biodiesel:**
  - non-negativity
  - two stoichiometric consistency checks
  - empirical noise sigma per level

## Implementation notes and paper ambiguities

Where the paper is explicit, the code follows it. Where it is silent or
ambiguous, the choice is labelled below in plain wording.

- **Noise model — implementation choice.**
  ChemKAN reports experiments with synthetic noise up to 15%, but the exact
  noise distribution is not explicitly specified. This implementation uses
  multiplicative Gaussian noise as a documented implementation choice. This is
  consistent with the plotted scatter in the ChemKAN noise figures and with
  related LeanKAN/KAN-ODE work, where multiplicative noise was used in a
  noisy-data experiment. A range-scaled alternative is available via
  `--noise-mode range` for sensitivity checks.

- **Hydrogen equivalence-ratio grid — resolved from the paper.**
  Section II.D.2 reports 36 hydrogen initial-condition combinations and 35
  training cases after withholding one case, but the printed equivalence-ratio
  list contains only five values: 0.7, 0.9, 1.1, 1.3, and 1.5. Those counts
  require six equivalence-ratio values. Fig. 8 also shows the equivalence-ratio
  range starting at 0.5. This implementation therefore uses
  `phi = {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}`. The omitted value appears to be
  `phi = 0.5`.

- **Hydrogen sample count — implementation choice with related-paper support.**
  ChemKAN states the hydrogen time span but does not state the number of saved
  time points. Because ChemKAN says the setup is largely identical to ChemNODE,
  and ChemNODE saved each time series at 50 points, this implementation uses
  50 points by default. Use `--n-points 601` for a 1 µs high-resolution grid.

- **Train-only normalization — implementation choice.**
  Eq. 18 states that each thermochemical state \(u_k\) is normalized to \([0,1]\)
  by subtracting a minimum and dividing by a range. The paper does not specify
  whether these min/max values are computed from all data, training data only,
  or per trajectory. This implementation computes one min/max pair per state
  variable using the clean training trajectories only, across all training cases
  and saved time points. The same stored `u_min` and `u_max` are then reused for
  training, testing, noisy data, and later model predictions. This avoids test-set
  leakage and preserves differences between initial conditions.

- **Biodiesel time grid — implementation choice.**
  ChemKAN describes 30 total data points over the biodiesel time history. This
  implementation uses `linspace(0, 30, 30)`, preserving the 30 s window and the
  30-point count. The resulting spacing is approximately 1.034 s.

- **Mechanism and reactor — directly from ChemKAN plus related-paper support.**
  - The H2/O2 mechanism with 9 species and 29 reactions is directly from ChemKAN. The constant-pressure 1 atm reactor setup follows ChemNODE, which ChemKAN says it closely follows. The `c_p` temperature equation in ChemKAN is consistent with this constant-pressure setup.

## Optional methane extension

Methane is optional and is not part of the original ChemKAN reproduction. It is included as an extension dataset using the same style of 0-D constant-pressure reactor generation. The setup mirrors the hydrogen generator where possible, but methane is a larger and different chemical system, not a controlled one-variable comparison.

```bash
cd extensions
python generate_methane.py --out ../../../data/generated/methane.npz
```

Notes:

- It uses full GRI-Mech 3.0. Methane did not ignite within the short
  hydrogen-style time window in preliminary tests, so this extension uses a
  higher temperature range (1400–1650 K) and a 5 ms window.
- Full GRI-Mech includes NOx chemistry, so unlike the reduced H₂/O₂ setup, N₂ is
  kept in the methane state vector. Argon is a constant-zero inert here and is
  dropped from the stored state.
- A reduced methane mechanism such as DRM19 could be tested later if full
  GRI-Mech is too large (`--mech drm19.yaml`, once the file is on Cantera's data
  path).
