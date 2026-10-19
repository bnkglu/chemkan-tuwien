# Tested environment

The environments in which the final results were produced and tested. Earlier, superseded
runs are not reconstructed; their `config.json` records what they used where available.

## Python (ChemKAN, data generation, DeepONet baseline, benchmark)

| | MacBook | TU Wien JupyterHub |
|---|---|---|
| hardware / OS | Apple M1, macOS 26.3 (arm64) | Linux, CPU only |
| Python | 3.11.3 | 3.11.16 |
| torch | **2.13.0** | **2.13.0+cu130** (CUDA build, run on CPU) |
| used for | data generation, hydrogen and methane main runs, all evaluation, figures, notebooks, inference benchmark | biodiesel Fig. 4 / Fig. 5 runs, methane pilot; methane main Stage 1 up to epoch 28,000, then continued on the Mac (`host_transfer.json`) |

Core packages (both machines): numpy 1.26.4, scipy 1.17.1, Cantera 3.0.0, torchdiffeq at
GitHub commit `657943a` (reports 0.2.5), matplotlib 3.8.4.

- `chemkan/requirements.txt` / `chemkan/pyproject.toml`: the pinned runtime dependencies.
- `chemkan/requirements-lock.txt`: `pip freeze` of the Mac venv (all packages, exact
  versions), 2026-10-20.
- `chemkan/third_party/torchdiffeq/`: source distribution and MIT license of the pinned
  torchdiffeq commit, for installation without GitHub.
- Every run's `config.json` records the git commit; the author-matched runs also record the
  Python and torch versions. All configs that record a torch version have 2.13.0
  (22 runs on the Mac, 16 on JupyterHub, and the 17 adaptive-FGD runs as `torch_version`).

Install:

    cd chemkan
    pip install -r requirements.txt && pip install -e . --no-deps     # runtime
    pip install -r requirements-lock.txt && pip install -e . --no-deps  # exact Mac environment

The lock file is from macOS arm64; on Linux, pip resolves the platform wheels of the same
versions.

## DeepONet

The biodiesel DeepONet baseline (`deeponet/biodiesel_deeponet*.py`,
`train_biodiesel_deeponet*.py`, `evaluate_biodiesel_deeponet.py`) is PyTorch and runs in the
ChemKAN environment above. `deeponet/requirements.txt` and `deeponet/src/` belong to the
original DeepONet code (TensorFlow, DeepXDE) and are not needed for our results.
`deeponet_param_audit.py` optionally imports DeepXDE (1.15.0 in the lock file) to count the
reference model's parameters.

## Julia

| | version | environment |
|---|---|---|
| inference benchmark (ChemKAN port, Arrhenius.jl, Sundials) | Julia 1.12.7 | `results/experiments/inference_benchmark/julia/{Project,Manifest}.toml`; Arrhenius.jl from GitHub master (release 0.1.0 returns zero rates) |
| authors' released biodiesel example | Julia 1.11.1 | recorded in `results/experiments/biodiesel/author_repo_match/julia_reference/` |

Exact timings depend on the hardware; the benchmark README states the machine and thread
settings.
