# chemkan — implementation status

The ChemKAN reproduction, now an installable package in the `chemkan-tuwien` repo
(`src/` layout, `pip install -e .`). See `ASSUMPTIONS.md` for every paper-vs-implementation
decision — including §9 (pinned Tsit5-capable `torchdiffeq` commit + the open
Forward-Sensitivity gap) and §10 (hydrogen two-stage terminology).

## Layout

```
src/chemkan/            reusable, experiment-AGNOSTIC library
  kan/{_common,rbf,layers}.py   Gaussian primitive, RBF activations, Add/Lean layers
  model.py              KineticCore, ThermodynamicSuperstructure, ChemKAN (math only)
  normalization.py      MinMaxNormalizer (nn.Module, buffers, device-aware)
  temperature.py        ConstantTemperature, ObservedTemperature (batch-aware -> (B,1))
  dynamics.py           KineticDynamics (B,m), ChemKANDynamics (B,m+1)
  solver.py             the single SolverConfig + integrate (odeint, direct autograd)
  losses.py             trajectory_mse, element_conservation_loss, chemkan_loss
  training.py           train_kinetic_stage, train_full_chemkan  (take a loss_fn)
scripts/                experiment-SPECIFIC (names + benchmark configs live here)
  _data.py, _chemistry.py, train_{biodiesel,hydrogen}.py, evaluate_{biodiesel,hydrogen}.py
tests/                  test_{rbf,layers,model,normalization,temperature,dynamics,losses,solver,refactor}.py
```

There is **no** `src/chemkan/config.py`.

## Data / model / experiment boundary (this refactor)

- **Dataset facts inferred from data** (`species_dim = data["species_TBm"].shape[-1]`),
  never hard-coded. Single source of truth: `scripts/_data.py`.
- **Architecture is a CLI choice** in the scripts (`--hidden-dim`, `--num-basis`,
  `--n-mu`, `--use-base-act`); paper counts are the defaults. `use_base_act` has **no
  default** on reusable classes.
- **One `SolverConfig`** (in `solver.py`); the old duplicate + conversion is gone.
  Benchmark solver is `method="tsit5"` (paper's integrator, from the pinned GitHub
  `torchdiffeq`) with `direct_autograd` sensitivity; FSA is not implemented (gap).
- **Library is generic**: `training.py` has no dataset-named functions and takes a
  caller-supplied `loss_fn`; PINN stage usage is a script CLI flag, not library logic.
- **Checkpoints are self-describing**: `{model_state, architecture, data, training,
  solver}` — models are rebuilt from *dataset dim + checkpoint architecture*; eval
  validates checkpoint species vs. dataset.
- **Paper counts are regression tests** (156 / 344 / 208 / 411), not model restrictions.

## Explicit-choice hardening (latest pass)

- **No arbitrary capacity/method defaults** in reusable code: `num_basis` (all KAN
  classes), `n_mu` (LeanKAN), `lr`+`solver` (training fns), all `SolverConfig` fields,
  `integrate(config)`, and `use_pinn` (`chemkan_loss`) are now required.
- **Seeds**: `--seed` in both train scripts → `torch.manual_seed` + recorded in ckpt.
- **Train/test split**: `load_*(split=...)`; eval defaults to `test`; normalization
  stats stay train-only; eval reconstructs `SolverConfig` from the checkpoint.
- **Hydrogen species order** validated before the PINN term (`assert_species_order`).
- **`ObservedTemperature`** validates shapes + strictly-increasing times; avoids a
  per-step device→host sync.
- **Biodiesel epochs** default `10000` (paper 1e4); `--epochs 100` for smoke runs.
- **Deps** declared in `pyproject.toml` at the repo root: `torch`, pinned
  `torchdiffeq @ …@657943a`, `numpy==1.26.4`, `scipy==1.17.1`, `cantera==3.0.0`, `tqdm`;
  extras `[dev]` (pytest, ruff) and `[notebooks]`. `requirements*.txt` mirror this.

## Pre-KAN input scaling (raw-temperature fix)

Physical dimensional temperature saturated the KAN's internal `tanh`
(`tanh(1000 K)=1`), erasing temperature dependence. Fixed by scaling the model input
only — the ODE state and derivatives stay physical:

```
old:  physical [Y,T] ------------------> tanh -> KAN -> physical dY/dt
new:  physical [Y,T] -> train-minmax -> tanh -> KAN -> physical dY/dt
```

- Scaling lives in `dynamics.py` (`input_normalizer`, required kw-only; `None` = raw
  ablation via `--input-scaling none`). `model.py`, `RBFEdgeFunctions`, `temperature.py`
  unchanged; internal `tanh` and RBF grid unchanged; solver still integrates physical.
- Full-state train-only normalizer (biodiesel appends T's train range to species stats;
  hydrogen uses archive `m+1` stats). No clipping; not per-trajectory.
- Checkpoints store `state_representation="physical"` + `input_scaling` (exact
  `u_min`/`u_max` for `minmax`); eval reconstructs it and rejects missing/unknown metadata.
- Witnesses (`tests/test_input_scaling.py`): raw path → `d(out)/d(T)=0` exactly;
  scaled path → `d(out)/d(T)≠0`. Parameter counts unchanged (156/208, 344/411).

## Preserved from before (unchanged math)

Shared `gaussian`; `RBFActivation`; `RBFEdgeFunctions`; `AddKANLayer`; `LeanKANLayer`;
keyword-only `use_base_act`; `KineticCore` (AddKAN→LeanKAN); `ThermodynamicSuperstructure`
(`Linear(m,1,bias=False)` + `KAN_cor`); `ChemKAN` `[Y,T]→[dY,dT]` with **no `use_thermo`
flag**; batch-aware temperature; Stage-1 species-only vs. Stage-2 full `[Y,T]` ODE state.

## Reproduce

```bash
cd chemkan && pip install -e ".[dev]"                     # editable install + test deps
pytest -q                                                 # unit + regression + refactor
python scripts/train_biodiesel.py --epochs 200            # e.g. override arch via CLI
```
