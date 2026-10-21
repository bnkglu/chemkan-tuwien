# ChemKAN: reproduction and extension

A PyTorch implementation of **ChemKAN** (Koenig, Kim & Deng, *ChemKANs for combustion chemistry
modeling and acceleration*, PCCP 2025): Kolmogorov–Arnold networks used as neural ODEs for
chemical kinetics. The repository reproduces the paper's biodiesel and hydrogen results,
extends ChemKAN to methane oxidation, and contains the DeepONet comparison baseline and a
validated forward sensitivity analysis (FSA).

Interdisciplinary project at TU Wien, course
[194.147 Interdisciplinary Project in Data Science](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?dswid=6763&dsrid=17&semester=2026S&courseNr=194147),
summer semester 2026.

| | |
|---|---|
| Author | Hasan Berke Bankoglu (MSc Data Science, TU Wien) |
| Main supervisor | Ass. Prof. DI Dr. Markus Bösenhofer |
| Co-supervisor | Univ.-Prof. Dr. Thomas Gärtner |

**Contents:** [Overview](#overview) · [Main results](#main-results) ·
[Repository structure](#repository-structure) · [Installation](#installation) ·
[Quick start](#quick-start) · [Training](#training) ·
[Reproducing the experiments](#reproducing-the-experiments) ·
[Evaluation and figures](#evaluation-and-figures) · [Notebooks](#notebooks) ·
[Custom datasets](#custom-datasets) · [Reproducibility](#reproducibility) ·
[Limitations](#limitations) · [References](#references)

## Overview

| Part | What it is | Status |
|---|---|---|
| Biodiesel (paper Sec. III A) | isothermal transesterification, 6 species; kinetic core only; noise robustness (Figs. 3, 5, 6) and neural scaling (Fig. 4) | reproduced, 13 ChemKAN runs |
| Hydrogen (paper Sec. III B) | H₂/air autoignition, 9 species + temperature, 344-parameter ChemKAN; two-stage training; Figs. 7, 8A, 8B | reproduced, main run to 200,000 Stage-2 epochs |
| Methane (project extension) | CH₄/air, GRI-Mech 3.0, 52 species + temperature; the hydrogen procedure on a larger mechanism | trained to 200,000 Stage-2 epochs; not converged |
| DeepONet baseline | the paper's comparison model for biodiesel, a self-contained PyTorch implementation | 14 runs |
| Forward sensitivity analysis | the paper's training method: sensitivity equations solved with the ODE, no backpropagation through the solver | implemented and validated; used by the biodiesel runs |
| Inference benchmark | ChemKAN (PyTorch and a Julia port) against detailed chemistry (Cantera, Arrhenius.jl) | hydrogen, methane, biodiesel |

The ChemKAN runs are single-seed (seed 0). Losses are time-averaged MSE in normalized
coordinates (the paper's Eq. 18 divided by the number of time points) unless stated otherwise.

## Main results

Each row gives the paper's claim and ours; the record behind each number is linked.

**Biodiesel** ([`biodiesel_reproduction.ipynb`](chemkan/notebooks/biodiesel_reproduction.ipynb), comparison table at its end)

| Quantity | Paper | Ours | |
|---|---|---|---|
| ChemKAN noise-free test MSE, 15 % / 0 % training noise | ~2× | 1.84× | reproduced |
| DeepONet noise-free test MSE, 15 % / 0 % | ~5× | 6.90× | reproduced |
| noise-free test MSE at 15 %, DeepONet / ChemKAN | 4.4× | 2.73× | same direction, smaller gap |
| ChemKAN training-MSE increase, 0 → 1 % / 0 → 5 % noise | 3.78e-5 / 9.64e-4 | 4.19e-5 / 1.05e-3 | reproduced, close to the noise floor (4.46e-5 / 1.08e-3) |
| ChemKAN neural-scaling order, training / testing | 1.0 / 0.6 | 1.49 / 0.37 | not reproduced numerically |
| DeepONet training loss below ChemKAN's above ~200 parameters | yes | mixed (see the notebook's table) | partly |

**Hydrogen**, 200,000-epoch Stage-2 model ([`hydrogen_reproduction.ipynb`](chemkan/notebooks/hydrogen_reproduction.ipynb), [`results/hydrogen/`](results/hydrogen/README.md))

| Quantity | Paper | Ours |
|---|---|---|
| model size | 344 parameters | 344 parameters |
| Fig. 7 conditions: training 1050 K / φ 0.9, held-out 1150 K / φ 1.3 | accurate reconstructions | MSE 4.5e-4, 3.5e-4 |
| Fig. 8A, 441 conditions | MSE mostly in the 1e-4 range, worse below 1000 K | median 4.9e-4; 3e-4 to 1e-3 from 1012.5 K up, 3–5e-2 at 975–987.5 K |
| Fig. 8B, ignition delay on the 30 igniting conditions | accurate | median relative error 1.4 %, max 8.7 % |
| speed-up over detailed chemistry | 2× (Arrhenius.jl) | the Julia port of ChemKAN at its training tolerance: 6.3× vs Arrhenius.jl and 2.5× vs Cantera at the data's tolerance; the PyTorch ChemKAN is slower than both ([`results/inference_benchmark/`](results/inference_benchmark/README.md)) |

**Methane** (extension; the paper has no methane case), selected Stage-2 snapshot at epoch 180,000
([`methane_extension.ipynb`](chemkan/notebooks/methane_extension.ipynb), [`results/methane/`](results/methane/README.md)):
training-condition MSE median 9.7e-4; held-out 1550 K / φ 1.3: 1.16e-3; 406 unseen grid conditions:
median 9.2e-4, ignition-delay error median 1.4 %. The training loss was still falling at 200,000
epochs. The Julia ChemKAN was 4.0× faster than Cantera and 13.5× faster than Arrhenius.jl.

**Not done:** the ChemNODE baseline named in the project proposal; repeated seeds of the
ChemKAN runs; hydrogen or methane runs trained with FSA (their recorded runs use direct
autograd).

## Repository structure

```text
chemkan/
├── src/chemkan/            the library: KAN layers (kan/), ChemKAN model, ODE dynamics, solver,
│                           losses, normalization, temperature providers, FSA (fsa.py)
├── scripts/
│   ├── train.py            one-file kinetic-only training example, read top to bottom
│   ├── train_biodiesel.py  biodiesel trainer, one run per call
│   ├── train_hydrogen.py   hydrogen entry point, one stage per call
│   ├── train_methane.py    methane entry point, one stage per call
│   ├── reference_components.py   model pieces matching the authors' released code
│   ├── biodiesel/          figure-run launcher, released-example datasets, equivalence check
│   ├── combustion/         shared hydrogen/methane trainer, predictor, evaluations, launchers
│   ├── data_gen/           dataset generators (SciPy, Cantera) and their README
│   ├── figures/            one script per figure, plot_all.py
│   ├── interpretability/   analyses of the trained models' learned functions
│   └── benchmark/          launcher of the inference benchmark
├── notebooks/              reproduction and FSA notebooks; extra/: explanation notebooks
├── data/generated/         the datasets the scripts read (tracked; regenerable)
├── tests/                  pytest suite
└── third_party/torchdiffeq the pinned ODE-solver source, for installation without GitHub
docs/                       the from-scratch guide of notebook extra/01
deeponet/                   the DeepONet baseline (separate from ChemKAN, see its README)
└── upstream_reference/     unmodified copy of the original DeepONet code, for reference only
results/                    the recorded experiments: biodiesel/, hydrogen/, methane/,
                            inference_benchmark/; PATHS.md maps original record paths
```

**Where to start reading the code**

| Order | File | Why |
|---|---|---|
| 1 | [`chemkan/scripts/train.py`](chemkan/scripts/train.py) | the whole workflow in one file: data, model, dynamics, solver, loss, optimization |
| 2 | [`chemkan/src/chemkan/model.py`](chemkan/src/chemkan/model.py), [`kan/`](chemkan/src/chemkan/kan) | the kinetic core (AddKAN → LeanKAN) and the thermodynamic superstructure |
| 3 | [`chemkan/scripts/reference_components.py`](chemkan/scripts/reference_components.py) | how the final runs configure the library like the authors' released example |
| 4 | [`chemkan/scripts/combustion/train_combustion.py`](chemkan/scripts/combustion/train_combustion.py) | the two-stage hydrogen/methane trainer |
| 5 | [`chemkan/src/chemkan/fsa.py`](chemkan/src/chemkan/fsa.py), [`forward_sensitivity_analysis.ipynb`](chemkan/notebooks/forward_sensitivity_analysis.ipynb) | forward sensitivity analysis and its validation |

## Installation

Python 3.11, PyTorch 2.13.0, Cantera 3.0.0, NumPy 1.26.4, SciPy 1.17.1 and torchdiffeq at
GitHub commit `657943a` (the PyPI release lacks the Tsit5 method). The results were produced
with these versions on macOS arm64 (Python 3.11.3) and on a Linux CPU JupyterHub
(Python 3.11.16).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r chemkan/requirements.txt      # the exact environment of the results
pip install -e chemkan --no-deps             # the chemkan package itself
```

`chemkan/requirements.txt` is the `pip freeze` of that environment (it includes Jupyter, pytest
and ruff); `chemkan/pyproject.toml` declares the runtime dependencies, with extras `[dev]` and
`[notebooks]`. Without GitHub access, install torchdiffeq from `chemkan/third_party/torchdiffeq/`
(see its README). The inference benchmark also needs Julia 1.12
(`results/inference_benchmark/julia/Project.toml`, `Manifest.toml`).

## Quick start

```bash
cd chemkan && python -m pytest tests -q && cd ..        # about 1 minute
python chemkan/scripts/train.py                         # 1,000 epochs of biodiesel, about 20 s
python chemkan/scripts/train_hydrogen.py stage1 --run-dir runs/check/S1 --epochs 2
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/check/S2 --epochs 2 \
    --stage1-archive runs/check/S1/archive/stage1_S1_epoch2.pt
```

New runs write to `runs/` (ignored by Git); nothing above touches `results/`.

## Training

| Script | Use | System |
|---|---|---|
| `chemkan/scripts/train.py` | simple, editable example of the full workflow; any constant-temperature `.npz` (`--data`) | kinetic-only |
| `chemkan/scripts/train_biodiesel.py` | final biodiesel trainer: noise levels, model sizes, FSA, resume, run records | kinetic-only |
| `chemkan/scripts/train_hydrogen.py`, `train_methane.py` | final two-stage trainers: entry points to the shared `combustion/train_combustion.py` | combustion |
| `deeponet/train_deeponet.py` | the DeepONet baseline ([`deeponet/README.md`](deeponet/README.md)) | biodiesel |

The final scripts configure the library like the authors' released example
(`reference_components.py`). The explanation notebooks and `chemkan.training` use the
library's generic configuration instead: they teach the building blocks, not the recorded runs.

### Kinetic-only systems (biodiesel)

The species are integrated and temperature is a constant model input:
`dz/dt = [KAN_kin(z), 0] / 50 s` on normalized `z = [Ŷ, T̂]`. `train.py` keeps every setting in a
dictionary at its top and, on the biodiesel data, computes exactly what `train_biodiesel.py`
computes (tested in `tests/test_train_example.py`). A paper-figure configuration (Fig. 5, 5 % noise):

```bash
python chemkan/scripts/train_biodiesel.py --data canonical --noise-percent 5 --hidden 4 \
    --solve-mode batched --sensitivity fsa --rhs-divisor 50 --epochs 10000 \
    --run-dir runs/biodiesel/noise05
```

| `train_biodiesel.py` option | Default | Figure runs | Meaning |
|---|---|---|---|
| `--data` | `author_repo` | `canonical` | `canonical`: `biodiesel_v2.npz` (or `--data-file`); `author_repo`: the released example's arrays, built by `biodiesel/build_released_dataset.py` |
| `--noise-percent` | 0 | 0, 1, 2, 3, 5, 7, 10, 15 | stored noise level of the training targets |
| `--hidden` | 4 | 2, 3, 4, 9, 11 | hidden width H; 39·H parameters (156 at H = 4) |
| `--sensitivity` | `direct_autograd` | `fsa` | gradient backend: FSA (the paper's method) or backpropagation through the solver |
| `--solve-mode` | `per_trajectory` | `batched` | one solve per trajectory (as the released Julia code) or one for all |
| `--rhs-divisor` | 50 | 50 | time scale [s] dividing the right-hand side (the data window is 30 s) |
| `--epochs` | 10000 | 10,000 (Fig. 5), 5,000 (Fig. 4) | |
| `--seed`, `--threads`, `--checkpoint-every` | 0, 1, 500 | | `--resume`, `--init-from`: see `--help` |

### Two-stage combustion models (hydrogen, methane)

1. **Stage 1** trains the kinetic core on the species, with the observed temperature `T(t)`
   (a dense Cantera trajectory, 20,000 points) as an input:
   `dŶ/dt = KAN_kin([Ŷ, T̂_obs(t)]) / t_ref`. It writes permanent archives
   `archive/stage1_<run>_epoch<N>.pt`.
2. **Stage 2** starts from one Stage-1 archive, adds the thermodynamic superstructure and
   trains the whole model on the full state, temperature included:
   `dû/dt = [r, Linear(r) + KAN_cor(û)] / t_ref` with `r = KAN_kin(û)`. The kinetic core is
   not frozen. The loss adds an element-conservation term (weight 1e-4).

```bash
python chemkan/scripts/train_hydrogen.py stage1 --run-dir runs/hydrogen/S1_seed0 --epochs 100000
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/hydrogen/S2_seed0 --epochs 100000 \
    --stage1-archive runs/hydrogen/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt
```

`train_methane.py` takes the same commands. The defaults are the settings of the recorded runs:

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 100000 | total epochs of the stage (`--resume --epochs N` extends it) |
| `--t-ref` | the time window: 6e-4 s (H₂), 5e-3 s (CH₄) | time scale dividing the right-hand side |
| `--lr` | 2e-3 | Adam learning rate (the paper's) |
| `--rtol`, `--atol` | 1e-6, 1e-8 | Tsit5 tolerances |
| `--dtype` | `float32` | |
| `--solve-mode` | `batched` | one solve for all 35 training conditions |
| `--sensitivity` | `direct_autograd` | or `fsa` |
| `--checkpoint-every`, `--archive-every` | 500, 5000 | resume checkpoints and permanent archives |
| `--stage1-archive` (stage 2) | — | the Stage-1 archive to branch from |
| `--warmup-epochs` (stage 2) | 0 | optional epochs that train only the thermodynamic linear layer first |

### Resuming

Biodiesel: rerun an interrupted run with the same command plus `--resume`. Hydrogen and
methane: `--resume`, with a larger `--epochs` to extend a stage. To continue a committed run,
copy its folder, copy `stage2_final.pt` to `stage2_working.pt` (identical for a finished run)
and resume in the copy:

```bash
cp -Rp results/hydrogen/S2_seed0_200k runs/hydrogen/S2_more
cp runs/hydrogen/S2_more/stage2_final.pt runs/hydrogen/S2_more/stage2_working.pt
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/hydrogen/S2_more --resume --epochs 250000
```

## Reproducing the experiments

| Experiment | Command | Records and checkpoints |
|---|---|---|
| biodiesel ChemKAN figure runs (Figs. 3–6), 13 runs | `python chemkan/scripts/biodiesel/run_figure_runs.py` (`--list`, `--dry-run`, `--only`, `--shard K/N`) | `results/biodiesel/chemkan_figure_runs/fig04/`, `fig05/`, one folder per run with `checkpoint_final.pt` |
| biodiesel DeepONet figure runs, 14 runs | `bash deeponet/run_figure_runs.sh` | `results/biodiesel/deeponet_figure_runs/fig04/`, `fig05/` |
| hydrogen main run (S1 100k; S2 100k → 150k → 200k) | `bash chemkan/scripts/combustion/run_hydrogen.sh` | `results/hydrogen/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt`; `S2_seed0*/stage2_final.pt`, 5,000-epoch snapshots in `archive/` |
| methane main run (S1; S2 to 200k from the 50k Stage-1 archive) | `bash chemkan/scripts/combustion/run_methane.sh` | `results/methane/S1_seed0/archive/`; `S2_seed0/stage2_final.pt`, `archive/` (selected: `stage2_S2_seed0_epoch180000.pt`) |
| inference benchmark | `bash chemkan/scripts/benchmark/run_inference_benchmark.sh` | `results/inference_benchmark/` |

The biodiesel launchers write into `results/` and skip every run whose `checkpoint_final.pt`
exists, so on a fresh clone they report everything as done. `BIODIESEL_RESULTS=<dir>`
redirects both launchers and the biodiesel figures to `<dir>`, so an independent rerun leaves
the recorded results untouched:

```bash
export BIODIESEL_RESULTS=runs/biodiesel_rerun
python chemkan/scripts/biodiesel/run_figure_runs.py && bash deeponet/run_figure_runs.sh
python chemkan/scripts/figures/plot_all.py --only fig03 fig04 fig05a fig05b fig06
```

The combustion launchers write to `runs/<system>/` (`OUT=<dir>` to change). The inference
benchmark overwrites its folder's timings.

## Evaluation and figures

| Task | Command | Output |
|---|---|---|
| a hydrogen checkpoint on the 36 conditions | `python chemkan/scripts/combustion/evaluate_checkpoint.py --checkpoint <pt> --out <json>` | losses; Stage 2 also temperature rise and ignition delay |
| hydrogen 441-condition grid (Fig. 8A) | `python chemkan/scripts/combustion/evaluate_hydrogen_grid.py --run-dir results/hydrogen/S2_seed0_200k --out runs/eval --label main_S2_seed0_200k` | per-condition table |
| hydrogen ignition delays (Fig. 8B) | `python chemkan/scripts/combustion/evaluate_hydrogen_ignition.py` with the same arguments | ignition-delay table |
| methane 441-condition grid | `python chemkan/scripts/combustion/evaluate_methane_grid.py` | `results/methane/generalization/` |
| trajectory plots of any checkpoint | `python chemkan/scripts/combustion/plot_trajectory.py --checkpoint <pt> --out-dir runs/plots` | one figure per condition |
| paper Figs. 3–8 | `python chemkan/scripts/figures/plot_all.py` | `results/biodiesel/figures/`, `results/hydrogen/figures/` |
| methane main species (extension figure) | `python chemkan/scripts/figures/methane_species.py` | `results/methane/figures/methane_main_species_S2_epoch180000_T1550_phi1p3.{png,pdf}` |
| interpretability analyses | `python chemkan/scripts/interpretability/biodiesel.py`; `combustion.py --system hydrogen` (or `methane`) | `results/<system>/interpretability/` |

Biodiesel runs evaluate themselves at the end of training (`metrics.json`: training solver and a
near-exact rtol 1e-10 solve). `plot_all.py` reproduces the committed figures (PNG identical, PDF
apart from its creation date) and rewrites `results/hydrogen/tables/fig07_per_state_mse_200k.csv`
with the same numbers and the checkpoint's current path.
[`chemkan/scripts/figures/README.md`](chemkan/scripts/figures/README.md) lists each figure's
script and options.

## Notebooks

| Notebook | Contents |
|---|---|
| [`biodiesel_reproduction.ipynb`](chemkan/notebooks/biodiesel_reproduction.ipynb) | biodiesel Figs. 3–6 and the paper-vs-ours table, with the DeepONet baseline |
| [`hydrogen_reproduction.ipynb`](chemkan/notebooks/hydrogen_reproduction.ipynb) | hydrogen Stage 1 and 2, evaluation, Figs. 7, 8A, 8B |
| [`methane_extension.ipynb`](chemkan/notebooks/methane_extension.ipynb) | methane training, snapshot selection, errors, trajectories, ignition delay |
| [`forward_sensitivity_analysis.ipynb`](chemkan/notebooks/forward_sensitivity_analysis.ipynb) | the sensitivity equations and their implementation; an analytic check; FSA vs direct-autograd gradients on the final models over solver tolerances; trajectory independence; the FSA unit tests |
| [`extra/01`–`04`](chemkan/notebooks/extra) | explanations: a KAN from scratch, the edge-function `einsum`, ChemKAN from the paper's equations, the library's kinetic path |
| [`data_generation_walkthrough.ipynb`](chemkan/scripts/data_gen/data_generation_walkthrough.ipynb) | dataset generation step by step, checked against the committed files |

[`chemkan/notebooks/README.md`](chemkan/notebooks/README.md) describes each one.

**Forward sensitivity analysis.** `--sensitivity fsa` integrates the state and its parameter
sensitivities `S = ∂x/∂θ` (`dS/dt = (∂F/∂x) S + ∂F/∂θ`, `S(t₀) = 0`) in one Tsit5 solve and
assembles `dL/dθ = Σ_j S(t_j)ᵀ ∂L/∂x(t_j)`, differentiating the trainers' own right-hand side
with `torch.func`. In the FSA notebook, its gradients converge to the direct-autograd ones as
the tolerance tightens (agreement 2e-7 to 4e-6 at rtol 1e-10 on the final models). At the
biodiesel training tolerance (rtol 1e-2, the released example's) the gradients of both
backends are far from converged, while the loss is accurate.

## Custom datasets

For a system at constant temperature, write one `.npz` with `t` (n_t,), `train_states` and
`test_states` (cases, n_t, m species), `train_T` and `test_T` (one temperature per trajectory),
then run `python chemkan/scripts/train.py --data my_system.npz`. The script checks the file,
normalizes with the training statistics and builds the kinetic core for m species. Its
docstring states the format and the limits (kinetic core only, constant temperature, one shared
time grid); `rhs_divisor` should be of the order of the time window. A system whose temperature
evolves needs the two-stage trainer; `configure_system` in `combustion/train_combustion.py`
shows how methane was added next to hydrogen.

The eight datasets in `chemkan/data/generated/` are regenerable; each command in
[`chemkan/scripts/data_gen/README.md`](chemkan/scripts/data_gen/README.md) reproduces its
file's arrays exactly.

## Reproducibility

- **Short reruns:** all 31 recorded main runs, rerun for 5 epochs with their recorded
  settings, matched their histories exactly for hydrogen Stage 1 and 2, methane Stage 2 and the
  14 DeepONet runs.
- **Across platforms:** the biodiesel ChemKAN runs and methane Stage 1 (to epoch 28,000) were
  trained on a Linux JupyterHub. A rerun on the Mac starts from the same weights and agrees to
  rounding (1e-12 to 1e-16 relative in float64; one float32 rounding step in methane Stage 1).
  Over long training, the adaptive solver and Adam amplify these differences: a 5,000-epoch
  rerun of one biodiesel run ended within 7e-4 relative of its record.
- **Resumed runs:** the recorded runs were interrupted and resumed several times (their
  `run.log`); on the same platform, a resumed run is bit-identical to an uninterrupted one.
- **Records:** `results/` holds the configs, metrics, logs, checkpoints, figures, tables,
  historical launch scripts and READMEs as produced on the development branch, where they name
  its paths; [`results/PATHS.md`](results/PATHS.md) maps them to the current ones. The
  training histories keep every row but only the columns used here; the complete files are on
  the development branch, and `PATHS.md` lists the dropped columns. Labels recorded in run
  configs (for example `"experiment": "hydrogen/author_matched"`) are kept in new runs so that
  the recorded runs stay resumable.

## Limitations

- One seed per ChemKAN run; differences between models are single-run observations.
- The biodiesel neural-scaling orders differ from the paper's (1.49 / 0.37 vs 1.0 / 0.6).
- The methane model is not converged, and its final 200k model sits on a loss spike, so the
  epoch-180,000 snapshot is reported. After ignition it predicts slightly negative CH₄ and O₂
  at the held-out condition (`methane_species.py`).
- Hydrogen generalization is weak at 975–987.5 K.
- The PyTorch ChemKAN is slower at inference than the detailed-chemistry references; the
  speed-up comes from the Julia port of the same weights, measured on one machine and one thread.
- The paper states neither the DeepONet activation nor its learning rate; the figure runs use
  tanh and Adam lr 1e-2, chosen by a 10-seed comparison ([`deeponet/README.md`](deeponet/README.md)).
- The authors' clarifications by email, cited in configs and docstrings as items G*, B*, H*, are
  summarized in `docs/authors_materials.md` on the development branch.

## References

- B. C. Koenig, S. Kim, S. Deng. *ChemKANs for combustion chemistry modeling and acceleration.*
  Phys. Chem. Chem. Phys. 27, 17313–17330 (2025); [arXiv:2504.12580](https://arxiv.org/abs/2504.12580).
  Released code: DENG-MIT/ChemKAN.
- L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. *Learning nonlinear operators via DeepONet
  based on the universal approximation theorem of operators.* Nat. Mach. Intell. 3, 218–229 (2021).
- Cantera 3.0.0, <https://cantera.org>; GRI-Mech 3.0, <http://combustion.berkeley.edu/gri-mech/>.
- torchdiffeq, <https://github.com/rtqichen/torchdiffeq> (vendored in `chemkan/third_party/`).
- SciMLSensitivity.jl, "Sensitivity Math Details",
  <https://docs.sciml.ai/SciMLSensitivity/stable/sensitivity_math/> (the FSA formulation).

```bibtex
@article{koenig2025chemkans,
  title={ChemKANs for combustion chemistry modeling and acceleration},
  author={Koenig, Benjamin C and Kim, Suyong and Deng, Sili},
  journal={Physical Chemistry Chemical Physics},
  volume={27},
  number={33},
  pages={17313--17330},
  year={2025},
  publisher={Royal Society of Chemistry}
}
```
