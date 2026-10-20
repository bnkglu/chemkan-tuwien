# chemkan-tuwien

A PyTorch implementation of **ChemKAN** (Koenig, Kim & Deng, 2025: chemistry
Kolmogorov–Arnold networks as neural ODEs for chemical kinetics), with the reproduction of the
paper's biodiesel and hydrogen results, a methane extension and the DeepONet comparison
baseline. Interdisciplinary project at TU Wien:
[194.147 Interdisciplinary Project in Data Science](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?dswid=6763&dsrid=17&semester=2026S&courseNr=194147).

## Repository structure

```text
chemkan/
├── src/chemkan/            the library: KAN layers, ChemKAN model, ODE dynamics, solver,
│                           losses, forward sensitivity analysis (fsa.py)
├── scripts/
│   ├── train.py            one-file training example (biodiesel), read top to bottom
│   ├── train_biodiesel.py  biodiesel trainer, one run per call
│   ├── train_hydrogen.py   hydrogen trainer, one stage per call
│   ├── train_methane.py    methane trainer, one stage per call
│   ├── reference_components.py   model pieces matching the authors' released code
│   ├── biodiesel/          figure-run launcher, released-example datasets, equivalence check
│   ├── combustion/         shared hydrogen/methane trainer, predictor, evaluations, launchers
│   ├── data_gen/           dataset generators (Cantera) and their README
│   ├── figures/            one script per paper figure, plot_all.py
│   ├── interpretability/   analyses of the trained models' learned functions
│   └── benchmark/          launcher of the inference benchmark
├── notebooks/              reproduction notebooks; extra/: explanation notebooks
├── data/generated/         the datasets the scripts read (tracked; regenerable)
├── tests/                  pytest suite
└── third_party/torchdiffeq the pinned ODE-solver source, for installation without GitHub
deeponet/                   the DeepONet baseline (separate from ChemKAN, see its README)
results/                    the recorded experiments: biodiesel/, hydrogen/, methane/,
                            inference_benchmark/; PATHS.md maps original record paths
```

## Environment

Python 3.11, PyTorch 2.13.0, Cantera 3.0.0, NumPy 1.26.4, SciPy 1.17.1 and torchdiffeq at
GitHub commit `657943a` (the PyPI release lacks the Tsit5 method). The final results were
produced on macOS arm64 (Python 3.11.3) and on a Linux CPU JupyterHub (Python 3.11.16) with
these versions.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r chemkan/requirements.txt      # the exact environment of the results
pip install -e chemkan --no-deps             # the chemkan package itself
```

`chemkan/requirements.txt` is the `pip freeze` of that environment (it includes Jupyter,
pytest and ruff). `chemkan/pyproject.toml` declares the runtime dependencies, with extras
`[dev]` and `[notebooks]`. If GitHub is unavailable, install torchdiffeq from
`chemkan/third_party/torchdiffeq/` (see its README). The inference benchmark also needs
Julia 1.12 (`results/inference_benchmark/julia/Project.toml`, `Manifest.toml`).

## Quick check

```bash
cd chemkan && python -m pytest tests -q && cd ..        # about 1 minute
python chemkan/scripts/train.py                         # 1,000 epochs of biodiesel, about 20 s
python chemkan/scripts/train_hydrogen.py stage1 --run-dir runs/check/S1 --epochs 2
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/check/S2 --epochs 2 \
    --stage1-archive runs/check/S1/archive/stage1_S1_epoch2.pt
```

New runs write to `runs/` (ignored by Git); nothing above touches `results/`.

## Training

**The example.** `chemkan/scripts/train.py` is the whole workflow in one file: data, model,
dynamics and solver, loss, optimization, saving. Its settings are a dictionary at the top.
With the same settings it computes exactly what the biodiesel trainer computes (tested in
`tests/test_train_example.py`).

**Biodiesel.** One configuration per call; `--help` lists every option. A paper-figure
configuration (Fig. 5, 5 % noise):

```bash
python chemkan/scripts/train_biodiesel.py --data canonical --noise-percent 5 --hidden 4 \
    --solve-mode batched --sensitivity fsa --rhs-divisor 50 --epochs 10000 \
    --run-dir runs/biodiesel/noise05
```

`--data canonical` trains on `chemkan/data/generated/biodiesel_v2.npz` (or `--data-file`);
`--noise-percent` selects a stored noise level (0, 1, 2, 3, 5, 7, 10, 15); `--hidden` the
model size (39·H parameters); `--sensitivity` the gradient backend (`fsa`, forward
sensitivity analysis as in the paper, or `direct_autograd`, backpropagation through the
solver); `--solve-mode batched` solves all trajectories at once. The right-hand side is
divided by 50 s by default, as in the authors' released example and every recorded biodiesel
run (`--rhs-divisor` changes it; the data's time window is 30 s). The default `--data author_repo`
trains on the released example's own arrays, built first with
`python chemkan/scripts/biodiesel/build_released_dataset.py`.

**Hydrogen and methane.** Two stages, one per call; `stage1 --help` and `stage2 --help` list
the options (`--sensitivity`, `--dtype`, `--solve-mode`, `--lr`, `--rtol`, `--atol`, `--seed`,
`--t-ref`, `--warmup-epochs`, `--archive-every`, …). The defaults are the settings of the
recorded runs; `t_ref` is the system's time window (6e-4 s hydrogen, 5e-3 s methane).

```bash
python chemkan/scripts/train_hydrogen.py stage1 --run-dir runs/hydrogen/S1_seed0 --epochs 100000
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/hydrogen/S2_seed0 --epochs 100000 \
    --stage1-archive runs/hydrogen/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt
```

`train_methane.py` takes the same commands. Stage 1 writes permanent archives
(`archive/stage1_<run>_epoch<N>.pt`); Stage 2 starts from one of them.

**Resuming.** Biodiesel: rerun an interrupted run with the same command plus `--resume`.
Hydrogen and methane: `--resume`, with a larger `--epochs` to extend a stage. To continue one
of the committed runs, copy its folder, copy `stage2_final.pt` to `stage2_working.pt` (for a
finished run the two are identical) and resume in the copy:

```bash
cp -Rp results/hydrogen/S2_seed0_200k runs/hydrogen/S2_more
cp runs/hydrogen/S2_more/stage2_final.pt runs/hydrogen/S2_more/stage2_working.pt
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/hydrogen/S2_more --resume --epochs 250000
```

## Reproducing the experiments

| Experiment | Command | Recorded result |
|---|---|---|
| biodiesel ChemKAN figure runs (Figs. 3–6), 13 runs | `python chemkan/scripts/biodiesel/run_figure_runs.py` (`--list`, `--dry-run`, `--only`, `--shard K/N`) | `results/biodiesel/chemkan_figure_runs/` |
| biodiesel DeepONet figure runs, 14 runs | `bash deeponet/run_figure_runs.sh` | `results/biodiesel/deeponet_figure_runs/` |
| hydrogen main run (S1 100k, S2 100k → 150k → 200k) | `bash chemkan/scripts/combustion/run_hydrogen.sh` | `results/hydrogen/` |
| methane main run (S1, S2 200k) | `bash chemkan/scripts/combustion/run_methane.sh` | `results/methane/` |
| inference benchmark | `bash chemkan/scripts/benchmark/run_inference_benchmark.sh` | `results/inference_benchmark/` |

The two biodiesel launchers write into `results/` and skip every run whose
`checkpoint_final.pt` exists, so on a fresh clone they report everything as done; move a run
folder away to retrain it. The combustion launchers write to `runs/<system>/`
(`OUT=<dir>` to change). The recorded runs were interrupted and resumed several times
(their `run.log`), so a rerun is not claimed to be bit-identical to them; a rerun of a
biodiesel figure configuration matched its recorded losses to 1e-12 relative (20 epochs
checked, different machine). The inference benchmark overwrites its folder's timings.

## Evaluation

```bash
# hydrogen checkpoint on all 36 conditions (losses; Stage 2 also temperature rise, ignition delay)
python chemkan/scripts/combustion/evaluate_checkpoint.py \
    --checkpoint results/hydrogen/S2_seed0_200k/stage2_final.pt --out runs/eval/S2_200k.json
# hydrogen 441-condition grid (Fig. 8A) and ignition delays (Fig. 8B)
python chemkan/scripts/combustion/evaluate_hydrogen_grid.py --run-dir results/hydrogen/S2_seed0_200k \
    --out runs/eval --label main_S2_seed0_200k
python chemkan/scripts/combustion/evaluate_hydrogen_ignition.py --run-dir results/hydrogen/S2_seed0_200k \
    --out runs/eval --label main_S2_seed0_200k
# methane 441-condition grid (writes results/methane/generalization/)
python chemkan/scripts/combustion/evaluate_methane_grid.py
# trajectory plots of any checkpoint (biodiesel, hydrogen, methane)
python chemkan/scripts/combustion/plot_trajectory.py --checkpoint <checkpoint.pt> --out-dir runs/plots
```

Biodiesel runs evaluate themselves at the end of training (`metrics.json`: training solver
and a near-exact rtol 1e-10 solve); `chemkan/scripts/figures/biodiesel_runs.py` loads the
figure runs, their histories, predictions and final losses.

## Figures and analyses

```bash
python chemkan/scripts/figures/plot_all.py                  # Figs. 3-8 into results/*/figures/
python chemkan/scripts/interpretability/biodiesel.py        # into results/biodiesel/interpretability/
python chemkan/scripts/interpretability/combustion.py --system hydrogen   # or methane
```

`plot_all.py` reproduces the committed figures (PNG identical; PDF apart from its creation
date); it also rewrites `results/hydrogen/tables/fig07_per_state_mse_200k.csv`, with the same
numbers and the checkpoint's current path. `chemkan/scripts/figures/README.md` lists each figure's script and options; every
interpretability script takes `--out`.

## Data

The eight datasets in `chemkan/data/generated/` are tracked and can be regenerated; each
command in `chemkan/scripts/data_gen/README.md` reproduces its file's arrays exactly. From
`chemkan/scripts/data_gen/`:

```bash
python generate_all.py --out-dir ../../data/generated --include-fine   # biodiesel, hydrogen (+ fine grid)
python generate_methane.py --out ../../data/generated/methane.npz      # methane: see the data_gen README
```

## DeepONet baseline

Self-contained under `deeponet/` (model, preprocessing, trainer, figure-run launcher);
`deeponet/README.md` gives the commands and the attribution to the original DeepONet work.

## Adapting ChemKAN to another chemical system

Start from `chemkan/scripts/train.py`. Provide initial states of shape (cases, m + 1)
(normalized species and temperature) and target trajectories of shape (cases, m, times),
build the kinetic core with `chemkan.model.KineticCore(species_dim=m, ...)` (as
`reference_components.build_core` does) and keep the dynamics, solver, loss and optimizer
steps. A system whose temperature evolves also needs the temperature model
(`ChemKAN`, `combustion/train_combustion.py`: a Stage 1 with observed temperature, a Stage 2
with the full state and the element-conservation term); its `configure_system` shows how the
methane extension was added next to hydrogen.

## Records

`results/` holds the experiment records exactly as produced (configs, histories, metrics,
logs, checkpoints, figures, tables, the historical launch scripts and READMEs). They were
made on the development branch and name its paths; `results/PATHS.md` maps them to the
current ones. Labels recorded in run configs (for example `"experiment":
"hydrogen/author_matched"`) are kept in new runs so that the recorded runs stay resumable.

## Reference

Paper: [ChemKANs for Combustion Chemistry Modeling and Acceleration](https://arxiv.org/pdf/2504.12580).

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
