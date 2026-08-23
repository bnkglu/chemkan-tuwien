# ChemKAN reproduction — implementation assumptions log

Paper = Koenig, Kim & Deng (2025), *ChemKAN*. This file separates what the paper
specifies from what this PyTorch port had to decide. Nothing here is a silent fix.

## 1. Paper / specification facts (implemented as stated)

- Kinetic core is `KAN_kin = Psi^lean_1 ∘ Psi^add_0` (AddKAN then LeanKAN). — `model.KineticCore`
- LeanKAN multiplies the first `n_mu` **inputs** and adds the rest (Eq. 8–10). — `kan/layers.LeanKANLayer`
- Temperature-rate structure `dT/dt = Linear(dY/dt) + KAN_cor(u)` (Eq. 14–15). — `model.ThermodynamicSuperstructure`
- The thermo linear map holds `m` scalar coefficients ⇒ `nn.Linear(m, 1, bias=False)`.
- `KAN_cor` is a single-output additive KAN over the full state (Eq. 17): `AddKANLayer(m+1, 1)`.
- Gaussian RBF denominator is `2 * h**2` (Eq. 12). — `kan/_common.gaussian`
- Two-stage training; **all** parameters updated in Stage 2 (kinetic core not frozen). — `training.py`
- Optimizer Adam, `lr = 2e-3` (paper default; CLI-overridable in the scripts).
- `alpha_PINN = 1e-4` (paper default; CLI-overridable in `scripts/train_hydrogen.py`).
- Dataset states normalized to `[0,1]` with train-only min-max for the Eq. 18 MSE. — `normalization.MinMaxNormalizer`
- Paper integrates with **Tsit5** and differentiates with **Forward Sensitivity Analysis**.

## 2. Reproduction ambiguity — base activation (Eq. 11)

Eq. 11 includes a Swish/SiLU base path `w^b b(x)` on every edge, but the reported
parameter counts only match with the base path **OFF**:

| system    | base OFF (reported) | base ON |
|-----------|---------------------|---------|
| biodiesel | **156** ✅          | 208     |
| hydrogen  | **344** ✅          | 411     |

Counts: biodiesel `(7·4 + 4·6)·3 = 156`; hydrogen `(10·3 + 3·9 + 10·1)·5 + 9 = 344`.

Decision: `use_base_act` is a **required, keyword-only** argument with no default.
Main count-matching reproduction uses `use_base_act=False` (no trainable `w_base` is
created — `register_parameter("w_base", None)`). `use_base_act=True` is retained only
as a literal-Eq.-11 sensitivity experiment. — `kan/rbf.py`

## 3. Reproduction inference — grid size

`num_basis = 5` for hydrogen is **inferred** (default in `scripts/train_hydrogen.py`,
CLI-overridable) from the reported architecture plus the
344-parameter count, not an explicitly stated grid size. Biodiesel `num_basis = 3`
likewise reproduces 156. — script CLI defaults in `scripts/train_{hydrogen,biodiesel}.py`

## 4. PyTorch implementation assumptions (NOT paper-specified)

- **Method**: `tsit5` — the **same integrator as the paper**, provided by the pinned
  GitHub `torchdiffeq` commit (stock PyPI `torchdiffeq` does not expose Tsit5). See §9
  for the exact pin. — `solver.py`, `scripts/train_{biodiesel,hydrogen}.py`
- **Gradients**: `torchdiffeq.odeint` with **direct autograd** (backprop through the
  solver). This is **NOT** the paper's Forward Sensitivity Analysis and is **not claimed
  equivalent** to it; `odeint_adjoint` is also deliberately not used. **FSA remains a
  reproduction gap** (see §9). — `solver.py`
- **Tolerances**: `rtol=1e-6`, `atol=1e-8` are implementation choices, not paper values.
- **Observed Stage-1 temperature** is **linearly interpolated** between saved training
  times (endpoints clamped); the paper does not specify the scheme at adaptive solver
  times. — `temperature.ObservedTemperature`
- **Loss reduction (Eq. 18)**: `(1/n*) Σ_k` (mean over state axis) then `Σ` over
  timesteps, then **mean over the batch** of trajectories. The PINN term is summed over
  elements and timesteps and averaged over the batch, so both terms share the same
  per-trajectory-then-mean normalization before `alpha_PINN`. — `losses.py`
- **Element table** (atom counts, atomic masses, molar masses for the h2o2 species) are
  standard chemistry constants, not paper values, needed only for the PINN term. —
  `scripts/_chemistry.py`
- **PINN stage usage**: whether PINN is applied in Stage 1 / Stage 2 is an explicit
  **CLI choice** in `scripts/train_hydrogen.py` (`--pinn-stage1`, `--no-pinn-stage2`),
  not baked into the library. Main interpretation: Stage 1 OFF, Stage 2 ON. Biodiesel:
  MSE only. The library's training functions take a caller-provided `loss_fn`.
- **Init scale** `randn * 0.1` for `w_rbf` is an implementation choice.

## 5. Deliberate structural choices

- Model math (`model.py`) is free of solver, optimizer, temperature, and mode flags.
  Kinetics-only use is expressed by driving `model.kinetic` through
  `dynamics.KineticDynamics` — there is **no `use_thermo` flag**.
- `KineticDynamics` returns `dY/dt` with the same shape as the integrated species state
  `(B, m)`; temperature is supplied externally. Stage 1 never integrates `[Y,T]` while
  returning only `dY/dt`.
- One shared `gaussian` primitive; layers use composition over `RBFEdgeFunctions`.

## 6. Data / model / experiment boundary (refactor)

- **Dataset dimensions are inferred from the generated data** (`species_dim =
  data["species_TBm"].shape[-1]`), never hard-coded. Species names, batch size, time
  grid, initial conditions, temperatures, and normalization statistics all come from
  `scripts/_data.py` — one source of truth.
- **Architecture hyperparameters are experiment choices** (`hidden_dim`, `num_basis`,
  `n_mu`, `use_base_act`) exposed via each script's argparse; the paper counts are the
  CLI defaults. `use_base_act` has **no default** on the reusable classes.
- **No global `config.py`.** The former `ChemKANConfig`/`TrainingConfig`/`biodiesel_config`/
  `hydrogen_config` and the duplicate `SolverConfig` were removed. The single
  `SolverConfig` lives in `solver.py`.
- **The reusable library is experiment-agnostic**: `training.py` exposes
  `train_kinetic_stage` / `train_full_chemkan` (no dataset-named functions), each taking
  a caller-provided `loss_fn`. No reusable code branches on "biodiesel"/"hydrogen".
- **Checkpoints store the actual run settings** — `model_state` plus `architecture`,
  `data` (species + species_dim), `training`, and `solver` dicts — so a model is
  reconstructed from *dataset dimension + checkpoint architecture*, with evaluation
  validating that checkpoint species match the dataset.
- **Paper benchmark counts (156 / 344 / 208 / 411) are regression tests**, not
  restrictions on the generic model.
- **`MinMaxNormalizer` is an `nn.Module`** with `u_min`/`u_max`/`range` buffers, so it
  follows `.to(device)`; it is still distinct from the KAN-internal `tanh`.

## 7. Explicit-choice hardening (no arbitrary capacity/method defaults)

Scientifically/numerically meaningful choices are never defaulted in reusable code;
only implementation-convenience defaults remain.

- **Removed defaults** (now required): `num_basis` on `RBFActivation`,
  `RBFEdgeFunctions`, `AddKANLayer`, `LeanKANLayer`; `n_mu` on `LeanKANLayer`; `lr` and
  `solver` on `train_kinetic_stage`/`train_full_chemkan`; all four `SolverConfig`
  fields (`method`, `rtol`, `atol`, `sensitivity`); `integrate(config=...)`; `use_pinn`
  on `chemkan_loss` (and `alpha_pinn` is validated as required when `use_pinn=True`).
- **Retained defaults** (harmless): `grid=(-1,1)` / `x_min=-1` / `x_max=1` (canonical
  KAN domain paired with the input `tanh`), `MinMaxNormalizer eps=1e-12` (numeric
  guard), `log_every=100`, `device="cpu"`, checkpoint filenames, and the experiment
  scripts' CLI defaults (paper architecture + `lr=2e-3`, `alpha_PINN=1e-4`, and the
  visible solver defaults `tsit5`/`1e-6`/`1e-8` — `tsit5` matches the paper's integrator,
  while the tolerances remain implementation choices).
- **Seeds**: both training scripts take `--seed` (default 0), call `torch.manual_seed`
  (+ `cuda.manual_seed_all`), and record the seed in checkpoint `training` metadata.
- **Biodiesel epochs**: script default is now `10000` (paper's 1e4); use `--epochs 100`
  for a smoke run. Hydrogen stage epoch counts remain implementation choices (not
  specified in the available text).
- **Train/test split**: loaders take `split="train"|"test"`. Normalization statistics
  are ALWAYS the train-only `u_min`/`u_max` stored in the archive; test states are
  normalized with train stats. Training uses train; evaluation defaults to test and
  reconstructs `SolverConfig` from the checkpoint (failing loudly if it is missing).
- **Hydrogen species order** is validated (`assert_species_order`) before the PINN term,
  since the element/molar-weight arrays are positional.

## 8. Pre-KAN input scaling (reproduction assumption)

The ChemKAN paper explicitly specifies `[0,1]` min-max normalization for Eq. 18 loss
evaluation and `tanh` normalization at each KAN layer input, but it does not explicitly
state the representation supplied to Eq. 13 immediately before the first KAN layer.
Direct dimensional combustion temperatures are numerically saturated by the stated
`tanh`/fixed-grid implementation (`tanh(323 K) = tanh(2800 K) = 1`). Therefore this
PyTorch reproduction applies a **global training-set min-max transformation to the
complete thermochemical state immediately before the KAN**, while retaining physical
ODE coordinates and physical derivative outputs. **This is a reproduction assumption,
not confirmed author preprocessing.**

- **Primary run**: `state_representation = physical`, `input_scaling = minmax`.
- The scaling lives in `dynamics.py` (`KineticDynamics` / `ChemKANDynamics` take a
  required `input_normalizer`), NOT in `model.py`, `RBFEdgeFunctions`, or
  `temperature.py`. The internal per-layer `tanh` and the RBF grid are unchanged; the
  solver still integrates physical `[Y, T]`; PINN still uses physical species.
- The full-state input normalizer is fitted on **training trajectories only**, is
  **global** (not per-trajectory), and is **not clipped** — held-out values may map
  outside `[0,1]`. Biodiesel appends T's train min/max to the archive's species stats;
  hydrogen uses the archive's `m+1` stats directly. Stage 1 / Stage 2 / held-out inputs
  all use this same normalizer.
- **`input_scaling=none`** is retained as an explicit raw-input diagnostic/ablation.
  Two witnesses (`tests/test_input_scaling.py`) prove the practical consequence on
  CPU/float64: raw physical T gives bit-identical model outputs for 323 K vs 343 K and
  `d(out)/d(T) == 0` exactly; under min-max scaling `d(out)/d(T) != 0`.
- Checkpoints record `state_representation` and `input_scaling` (with the exact
  `u_min`/`u_max` for `minmax`); evaluation reconstructs the transform from the
  checkpoint and refuses missing/unsupported representation or scaling metadata.

Out of scope (separate follow-ups): normalized-state ODE integration, component-wise
`atol`, OLS/thermo `θ_thermo` init, output-derivative scaling, Cantera-scale
diagnostics — none are implemented here.

## 9. Solver provenance and the Forward-Sensitivity gap

**Integrator (matches the paper).** Both training scripts default to `method="tsit5"` —
the same Tsitouras 5(4) explicit Runge–Kutta method the paper uses. Tsit5 is **not** in
the released PyPI `torchdiffeq`; it is provided by the GitHub source, pinned to:

```
torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq.git@657943acefa826ef04c025ebeb1ff5e9d60dc268
```

This commit was reported by the repository maintainer (the original training venv was no
longer available to read back its `direct_url.json`); a fresh install from this commit
was verified to expose `tsit5`. `requirements.txt` and `pyproject.toml` carry the same
pin, so a clean install guarantees `method="tsit5"` is available.

**Sensitivity (still a gap).** The paper differentiates the ODE solve with **Forward
Sensitivity Analysis (FSA)**. This reproduction instead backpropagates directly through
`odeint` (`sensitivity="direct_autograd"`). FSA is **not implemented**, and direct
autograd is a distinct mechanism that is **not claimed equivalent** to FSA.
`SolverConfig` rejects any sensitivity value other than `"direct_autograd"`. A true FSA
path is deliberately out of scope for now and remains a **TODO / open reproduction gap**.

## 10. Hydrogen two-stage training procedure (terminology)

Preserved exactly; do not conflate the stages.

**Stage 1**
- The ODE state being integrated contains **species `Y` only**.
- The kinetic core still receives the full `m+1` input `[Y, T]`.
- Temperature `T(t)` is **supplied from the training data** during Stage 1 (with
  interpolation in this PyTorch implementation when the adaptive solver requests
  intermediate times). — `temperature.ObservedTemperature`
- Only the species trajectory is predicted/integrated and used as the Stage-1 MSE target.

**Stage 2**
- Restart integration from the **observed initial full state** `u0 = [Y0, T0]`.
- Do **NOT** supply the future observed temperature trajectory to the dynamics.
- After initialization, `T` is part of the integrated ODE state and the current
  solver-generated `[Y(t), T(t)]` is passed into ChemKAN.
- ChemKAN computes **both** `dY/dt` and `dT/dt`.
- The observed temperature trajectory is used as a **Stage-2 loss target**, not as the
  future temperature input to the dynamics.
- The kinetic core is initialized from its converged Stage-1 parameters but is **NOT
  frozen** in Stage 2; the entire ChemKAN is updated.
