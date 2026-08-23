# How the ChemKAN code works

A walkthrough of the `chemkan/` package: what each module does, how the pieces fit
together, and how the two experiments (biodiesel, hydrogen) flow end-to-end.

ChemKAN reproduces Koenig, Kim & Deng (2025) — a **Kolmogorov–Arnold Network trained as
a neural ODE** for chemical kinetics. In one sentence: a KAN maps the physical
thermochemical state `[Y, T]` to its time derivative `[dY/dt, dT/dt]`, an ODE solver
integrates that derivative over time, and we train the network by comparing the
integrated trajectories against reference data — differentiating **through the solver**.

> Companion learning material: `notebooks/02_einsum_edge_functions.ipynb` (the one
> `einsum` that powers every KAN edge) and `notebooks/03_chemkan_from_scratch.ipynb`
> (this whole pipeline rebuilt inline, step by step, on biodiesel).

---

## 1 · Layout

```
chemkan/
  src/chemkan/            reusable, EXPERIMENT-AGNOSTIC library (no "biodiesel"/"hydrogen")
    kan/
      _common.py          the shared Gaussian RBF primitive          (Eq. 12)
      rbf.py              RBFActivation, RBFEdgeFunctions (the einsum) (Eq. 11–12)
      layers.py           AddKANLayer, LeanKANLayer                   (Eq. 7–10)
    model.py              KineticCore, ThermodynamicSuperstructure, ChemKAN (Eq. 13–17)
    normalization.py      MinMaxNormalizer (dataset [0,1] scaling, Eq. 18)
    temperature.py        ConstantTemperature, ObservedTemperature
    dynamics.py           KineticDynamics, ChemKANDynamics (the ODE right-hand side)
    solver.py             SolverConfig + integrate (torchdiffeq.odeint, direct autograd)
    losses.py             trajectory_mse, element_conservation_loss, chemkan_loss (Eq. 18)
    training.py           train_kinetic_stage, train_full_chemkan
  scripts/                EXPERIMENT-SPECIFIC (names, benchmark configs, chemistry live here)
    _data.py              thin loader for the generated .npz archives
    _chemistry.py         element table for the hydrogen PINN term
    train_biodiesel.py / evaluate_biodiesel.py
    train_hydrogen.py  / evaluate_hydrogen.py
    data_gen/             the reference-data generators (Cantera-based)
  data/generated/         biodiesel.npz, hydrogen.npz  (produced by data_gen)
```

**The one rule that shapes the whole design:** `src/chemkan/` is generic and knows
nothing about any specific chemical system. Dataset dimensions come *from the data*;
architecture and training choices are passed *in* by the caller. Everything
system-specific — species names, the element table, paper hyperparameters — lives in
`scripts/`. That is why, for example, `train_biodiesel.py` reads `species_dim` from the
array shape rather than hard-coding `6`.

---

## 2 · The data path (bottom-up)

Read the modules in dependency order; each one only needs the one above it.

### 2.1 `kan/_common.py` — the Gaussian bump (Eq. 12)

One function, so the formula lives in exactly one place:

```python
gaussian(x, centers, h) = exp(-(x - c_k)^2 / (2 h^2))
```

It appends a trailing basis axis, so it serves any input rank
(`(B, in) -> (B, in, K)`). `centers` is a fixed grid on `[-1, 1]`; `h` is the spacing
between centers (the bump width).

### 2.2 `kan/rbf.py` — edge functions and **the einsum**

A KAN replaces every scalar weight with a **learned function on an edge**. Each edge
function is a weighted sum of `K` Gaussian bumps (Eq. 11):

$$\phi_{o,i}(x_i) = \sum_{k=1}^{K} w^{\psi}_{o,i,k}\,\psi_k(x_i)$$

`RBFEdgeFunctions.forward` evaluates *every* edge for *every* item in the batch at once:

```python
x   = torch.tanh(x)                              # KAN-internal normalization
psi = gaussian(x, self.centers, self.h)          # (B, in, K)
edge = torch.einsum("bik,oik->boi", psi, self.w_rbf)   # (B, out, in)
```

The einsum reads: multiply, **sum away the basis index `k`**, keep batch `b`, output
`o`, input `i`. That summed-over-`k` *is* the equation above.
(`notebook 02` derives this four independent ways.)

Two subtleties that matter later:

- **`tanh` is the network's *internal* normalization** — it squashes each layer's inputs
  onto the fixed `[-1, 1]` center grid. It is **not** a dataset statistic. Raw Kelvin
  saturates it (`tanh(1000 K) = 1`); that is the problem §2.7 fixes.
- **`use_base_act`** (required, keyword-only) toggles the optional Swish base path
  `w_base · SiLU(x)` from Eq. 11. It is **OFF** in the main reproduction because the
  paper's reported parameter counts only match without it (see §4). When off, no
  `w_base` parameter is created at all (`register_parameter("w_base", None)`).

`RBFActivation` is the scalar, single-edge sibling — for teaching, tests, and plotting a
single learned activation; the production layers use `RBFEdgeFunctions`.

### 2.3 `kan/layers.py` — Add and Lean layers

Both share an `RBFEdgeFunctions` grid and differ only in how they **collapse the input
axis `i`** of `edge (B, out, in)`:

- **`AddKANLayer`** (Eq. 7): `y_o = Σ_i φ_{o,i}(x_i)` — sum over inputs.
- **`LeanKANLayer`** (Eq. 8–10): **multiply** the first `n_mu` inputs, **add** the rest:
  `y_o = (Π_{i<n_mu} φ) + (Σ_{i≥n_mu} φ)`. `n_mu` partitions the *input* axis. With
  `n_mu = 0` it reduces to AddKAN — and the code returns the additive term alone, because
  an empty product in PyTorch yields `1` and would otherwise add a spurious constant.

### 2.4 `model.py` — the ChemKAN math (Eq. 13–17)

Pure math: **no** solver, optimizer, temperature interpolation, or mode flags. State in,
derivative out.

- **`KineticCore`**: the species-rate network `Ψ^lean ∘ Ψ^add`.
  `u = [Y₁..Yₘ, T] (B, m+1) → dY/dt (B, m)`.
- **`ThermodynamicSuperstructure`** (Eq. 14–17): the temperature rate
  `dT/dt = Linear(dY/dt) + KAN_cor(u)`. The linear map holds the paper's `m` scalar
  coefficients (≈ −hᵢ/c_p, hence `bias=False`); `KAN_cor` is a single-output AddKAN over
  the full state.
- **`ChemKAN`**: the full model `u = [Y, T] (B, m+1) → [dY/dt, dT/dt] (B, m+1)`. It is
  autonomous and stateless — there is **no `use_thermo` flag**. Kinetics-only use is
  expressed by driving `model.kinetic` through `KineticDynamics` (next), not by a mode
  switch.

### 2.5 `normalization.py` — dataset scaling (Eq. 18)

`MinMaxNormalizer` scales physical states to `[0, 1]` using **train-only** `u_min`/`u_max`
(stored in the `.npz`). This is the Eq. 18 loss normalization and is **distinct from the
internal `tanh`**. It is an `nn.Module` with buffers, so `.to(device)` moves its stats
with the model; `.subset(cols)` slices out, e.g., species-only columns for Stage 1.

### 2.6 `temperature.py` — where does T come from when we don't integrate it?

For kinetics-only training the solver still needs a temperature at each internal step:

- **`ConstantTemperature`** — isothermal biodiesel; each trajectory has its own fixed T,
  returned as `(B, 1)`.
- **`ObservedTemperature`** — hydrogen Stage 1; `T(t)` read from data and **linearly
  interpolated** at the solver's adaptive times (interpolation scheme is our
  implementation choice, not paper-specified). It gathers on-device to avoid a per-step
  GPU→CPU sync.

### 2.7 `dynamics.py` — the ODE right-hand side (and the input-scaling fix)

The solver calls `f(t, state)`. These adapters own **no physics**; they assemble the
state the model expects and forward `t` (which the autonomous model ignores).

- **`KineticDynamics`**: integrates species only. `Y (B, m) → dY/dt (B, m)`. It calls the
  temperature provider, concatenates to form the physical `u = [Y, T]`, and drives the
  kinetic core.
- **`ChemKANDynamics`**: integrates the full `[Y, T] (B, m+1)` with the complete model.

**The representation boundary (important).** The ODE solver always integrates **physical**
coordinates and these adapters always return **physical** derivatives. The
`input_normalizer` (a required argument) transforms only the *copy* of the state handed
to the KAN:

```
physical [Y,T]  ──min-max──▶  tanh ──▶ KAN ──▶ physical dY/dt
                (model input only)              (ODE state & output stay physical)
```

Why: raw dimensional temperature (~10³ K) saturates the internal `tanh`, so
`tanh(320 K) == tanh(340 K) == 1` and the network becomes **blind to temperature**.
Min-max scaling the state into a sane range *before* the `tanh` restores a nonzero
`d(output)/d(T)`. Passing `input_normalizer=None` is the explicit raw-input ablation
(`--input-scaling none`). This is a documented **reproduction assumption**, not confirmed
author preprocessing.

### 2.8 `solver.py` — integration

`integrate(func, y0, t, config)` wraps `torchdiffeq.odeint`: `y0 (B, dim)`, `t (T,)` →
`(T, B, dim)`. `SolverConfig` has **no defaults** — `method` / `rtol` / `atol` /
`sensitivity` are all supplied by the caller.

- `method="tsit5"` — the **same integrator as the paper** (Tsitouras 5(4)), provided by
  the pinned GitHub `torchdiffeq` commit (stock PyPI `torchdiffeq` does not expose Tsit5).
  `rtol=1e-6` / `atol=1e-8` are implementation choices, not paper values.
- `sensitivity="direct_autograd"` — gradients flow by backprop **through** `odeint`. This
  is **not** the paper's Forward Sensitivity Analysis (FSA) and is **not claimed
  equivalent** to it; `odeint_adjoint` is also not used, and the config rejects any other
  value. **FSA remains an open reproduction gap** (see `ASSUMPTIONS.md` §9).

### 2.9 `losses.py` — the objective (Eq. 18)

`L = L_MSE + α_PINN · L_PINN`, with the two terms deliberately on different scales:

- **`trajectory_mse`** — on `[0,1]`-**normalized** states: mean over the state axis
  (the `1/n*`), sum over timesteps, mean over the batch.
- **`element_conservation_loss`** (the PINN term) — on **physical** species. For each
  element it computes the mixture mass fraction `z_i = Σ_k N_iᵏ · W_i · Y_k / W_k` and
  penalizes its **drift from the initial state**, summed over elements and time, meaned
  over the batch. (Closed reactor ⇒ each element's total mass is conserved.)
- **`chemkan_loss`** — combines them; `use_pinn` is required, and `alpha_pinn` + the
  element/weight tensors are required only when `use_pinn=True`.

### 2.10 `training.py` — the two stages

Generic, experiment-agnostic loops that take a caller-supplied `loss_fn(pred) → scalar`:

- **`train_kinetic_stage`** — integrate species with an external temperature; optimize
  **only** the kinetic core. Serves biodiesel and hydrogen Stage 1.
- **`train_full_chemkan`** — integrate the full `[Y, T]`; optimize **all** parameters
  (the kinetic core is **not** frozen in Stage 2). Serves hydrogen Stage 2.

Both require explicit `lr` and `solver` — the library never invents them. The shared
inner loop is: `integrate → loss_fn(pred) → backward → Adam.step`.

---

## 3 · The two experiments, end-to-end

### 3.1 Biodiesel (`scripts/train_biodiesel.py`) — isothermal, kinetics only, MSE

```
load_biodiesel(split="train")            # species_dim inferred from the array
  → KineticCore(m, hidden=4, num_basis=3, n_mu=2, use_base_act=False)   # 156 params
  → full-state MinMaxNormalizer: species stats (from .npz) + T's train min/max
  → KineticDynamics(core, ConstantTemperature(T_const), input_normalizer)
  → train_kinetic_stage: odeint(Y0, t) → trajectory_mse on normalized species
  → checkpoint {model_state, architecture, data, training, solver,
                state_representation="physical", input_scaling}
```

Isothermal ⇒ we integrate species only; temperature is a fixed per-trajectory constant.
Loss is MSE-only (no PINN). The loss normalizer is the **species subset** of the
full-state input normalizer, so training targets and model inputs use consistent stats.

### 3.2 Hydrogen (`scripts/train_hydrogen.py`) — two-stage, thermo + PINN

Nine species `[H2, H, O, O2, OH, H2O, HO2, H2O2, N2]` (m = 9); the full state is
`m+1 = 10`.

- **Stage 1** — species only, temperature supplied by `ObservedTemperature`; optimize the
  kinetic core (`train_kinetic_stage`). PINN is off by default here.
- **Stage 2** — integrate the full `[Y, T]` with the complete `ChemKAN`; optimize **all**
  parameters (`train_full_chemkan`). The default loss adds the **PINN** element-drift term
  (`α_PINN = 1e-4`), using the physical species and the `_chemistry.py` element table.

`_chemistry.py` holds the positional element table (`ELEMENT_COUNTS`, `ATOMIC_WEIGHTS`,
`MOLAR_WEIGHTS`) for the PINN term, plus `assert_species_order`, which fails loudly if the
dataset's species order ever diverges from the table (the arrays are positional, so a
silent reorder would compute wrong chemistry).

### 3.3 Evaluation (`evaluate_*.py`)

Evaluation **reconstructs the exact preprocessing from the checkpoint** rather than
refitting anything: it rebuilds the model from `data.species_dim + architecture`,
rebuilds the `SolverConfig` from `solver`, and rebuilds the input normalizer from
`input_scaling` (`load_input_scaling`, which refuses missing/unknown metadata). It
defaults to the **test** split, normalized with **train** statistics.

---

## 4 · Parameter counts — the reproduction anchor

The paper reports parameter counts, so we treat them as a hard check. With the base path
**off**:

| system    | arithmetic                                   | params |
|-----------|----------------------------------------------|:------:|
| biodiesel | `(7·4 + 4·6)·3`                              | **156** |
| hydrogen  | `(10·3 + 3·9 + 10·1)·5 + 9`                  | **344** |

Reading the hydrogen sum: AddKAN edges `(m+1)·hidden = 10·3`, LeanKAN edges
`hidden·m = 3·9`, `KAN_cor` edges `(m+1)·1 = 10·1`, all `× num_basis=5`, plus the thermo
linear's `m = 9` coefficients. Turning the base path **on** gives 208 / 411 instead — a
literal-Eq.-11 sensitivity variant, not the main run. (`num_basis=5` for hydrogen is
*inferred* from this count; biodiesel's `num_basis=3` likewise reproduces 156.)

---

## 5 · Design decisions worth knowing

- **Physical everywhere except the KAN input.** The ODE state, the returned derivatives,
  the temperature providers, and the PINN term are all physical. Only the *copy* fed to
  the KAN is min-max scaled. (§2.7)
- **Two different normalizations, never conflated.** `tanh` = internal, onto the RBF grid;
  `MinMaxNormalizer` = dataset `[0,1]`, train-only, for the loss and the input scaling.
- **No global config object.** There is no `config.py`; every knob is a function argument
  or a script CLI flag. Scientifically meaningful choices (`num_basis`, `n_mu`, `lr`,
  solver settings, `use_pinn`) have **no defaults** in the reusable code — the experiment
  scripts choose them explicitly, and the paper values are the scripts' CLI defaults.
- **Self-describing checkpoints.** Each checkpoint stores `architecture`, `data`,
  `training` (incl. `seed`), `solver`, `state_representation`, and `input_scaling`, so a
  run can be reconstructed exactly for evaluation.

---

## 6 · Running it

```bash
# from the chemkan/ directory; scripts add ../src to the path automatically
python scripts/train_biodiesel.py --epochs 100        # quick smoke run (paper: 1e4)
python scripts/evaluate_biodiesel.py                  # test split, from the checkpoint

python scripts/train_hydrogen.py --stage1-epochs 200 --stage2-epochs 200
python scripts/evaluate_hydrogen.py

# ablation: raw physical input (no pre-KAN scaling) vs. the default min-max
python scripts/train_biodiesel.py --epochs 100 --input-scaling none
```

Data defaults to `chemkan/data/generated/`; override with the `CHEMKAN_DATA_DIR`
environment variable. Generate the archives with the notebooks/scripts in
`scripts/data_gen/` if they are missing.

---

## 7 · State of the package (things to know)

- **Installable, standard `src/` layout.** `pyproject.toml` lives at the `chemkan/` root;
  `pip install -e .` makes `import chemkan` available without any `sys.path` hack.
- **Tests live at `chemkan/tests/`** — the full suite (`test_rbf`, `test_layers`,
  `test_model`, `test_dynamics`, `test_input_scaling`, `test_solver`, …) plus the
  repo-specific `test_chemistry_constants.py` (validated against Cantera). Run
  `pytest` from the `chemkan/` directory.
- **`ASSUMPTIONS.md` / `CHECK.md`** live under `src/chemkan/` and record every
  paper-vs-implementation decision. In particular, `ASSUMPTIONS.md` §9 documents the
  pinned Tsit5-capable `torchdiffeq` commit and the still-open Forward-Sensitivity gap,
  and §10 the hydrogen two-stage terminology.
- **Solver:** `method="tsit5"` (matches the paper's integrator) with `direct_autograd`
  sensitivity (FSA still TODO) — see §2.8.
