# ChemKAN reproduction

A PyTorch reproduction of the datasets and models from Koenig, Kim & Deng (2025),
*ChemKANs for combustion chemistry modeling and acceleration*.

## Current status

Data generation **and** the model/training stack are implemented:

- **Data generation** (`scripts/data_gen`): biodiesel and hydrogen reference datasets
  (Cantera); methane is an optional extension.
- **KAN building blocks** (`src/chemkan/kan`): Gaussian RBF edge functions, `AddKAN`,
  `LeanKAN`.
- **Model** (`src/chemkan/model.py`): `KineticCore` (AddKAN→LeanKAN) and the
  `ThermodynamicSuperstructure` (`dT/dt = Linear(dY/dt) + KAN_cor`).
- **Biodiesel**: isothermal, kinetics-only training/evaluation (MSE).
- **Hydrogen**: two-stage training/evaluation (Stage 1 species-only with data-supplied
  temperature; Stage 2 full `[Y, T]` with the thermodynamic superstructure) and an
  optional element-conservation **PINN** loss term.
- **Pre-KAN input scaling** (reproduction assumption): the physical state is min-max
  scaled before the KAN to avoid `tanh` saturation on dimensional temperature; the ODE
  state and derivatives stay physical.
- **Integration**: `torchdiffeq.odeint` with **Tsit5** (the paper's integrator, from a
  pinned GitHub `torchdiffeq` commit).
- **Sensitivity**: **direct autograd** through the solver. The paper uses Forward
  Sensitivity Analysis; FSA is **not implemented yet** and is a known reproduction gap
  (see `src/chemkan/ASSUMPTIONS.md` §9).

See **[`HOW_THE_CODE_WORKS.md`](HOW_THE_CODE_WORKS.md)** for a module-by-module
walkthrough and **[`src/chemkan/ASSUMPTIONS.md`](src/chemkan/ASSUMPTIONS.md)** for every
paper-vs-implementation decision.

## Layout

```
chemkan/
├── pyproject.toml           # installable package (src/ layout) + pinned deps
├── requirements.txt         # runtime lock; requirements-dev.txt adds test/notebook tools
├── src/chemkan/             # the importable library (KAN, model, solver, losses, …)
├── scripts/                 # experiment scripts: train_/evaluate_{biodiesel,hydrogen}, data_gen/
├── tests/                   # pytest suite
├── notebooks/               # einsum + from-scratch walkthroughs
└── data/generated/          # generated .npz datasets (not committed)
```

## Install & run

```bash
cd chemkan
pip install -e ".[dev]"                 # editable install + test deps (needs the pinned torchdiffeq)
pytest -q                               # run the test suite

python scripts/train_biodiesel.py --epochs 100      # smoke run (paper uses ~1e4)
python scripts/evaluate_biodiesel.py
python scripts/train_hydrogen.py --stage1-epochs 200 --stage2-epochs 200
python scripts/evaluate_hydrogen.py
```

Data generation setup and commands: **[`scripts/data_gen/README.md`](scripts/data_gen/README.md)**.

## Reference

Koenig, B. C., Kim, S., & Deng, S. (2025). ChemKANs for combustion chemistry modeling
and acceleration. *Physical Chemistry Chemical Physics*, 27, 17313–17330.
