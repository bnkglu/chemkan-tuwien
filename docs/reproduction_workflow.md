# ChemKAN reproduction workflow

End-to-end steps to go from raw physical data to trained models, evaluations, and
reproduction figures/tables. Every command below was verified against the current CLIs.

The scientific flow:

```
data generation → dense H2 temperature cache → training (run dir) → evaluation → notebooks → figures/tables
```

Training scripts train models and save reproducible artifacts; they do **not** reproduce
the whole paper. The reproduction notebooks are the analysis/presentation layer — they
load trained checkpoints and compatible predictions, compute paper metrics, and export
figures/tables.

> **Sensitivity backend.** All current reproduction runs use `sensitivity = direct_autograd`
> (backprop through the Tsit5 solve). Forward Sensitivity Analysis (FSA) is **not**
> implemented; do not label these results as FSA. The run/artifact layout already leaves
> room for a future `fsa_seed0/` alongside `direct_autograd_seed0/`.

---

## Step 1 — Environment / working directory

- Python 3.11 with the pinned runtime (`cantera==3.0.0`, `numpy==1.26.4`, `torch`, a
  GitHub-pinned `torchdiffeq` exposing `tsit5`). See the repo `README.md` /
  `chemkan/README.md` for installation — not repeated here.
- Install the package once (editable): from `chemkan/`, `pip install -e .` (puts the
  `chemkan` library on the path; `scripts/` is added by the training/eval scripts).
- **Run generators from** `chemkan/scripts/data_gen`. **Run training/evaluation from**
  `chemkan/scripts`. Relative paths in the commands assume those working directories.
- Data lives under `chemkan/data/generated/`; results under `results/reproduction/`.

```bash
cd chemkan
pip install -e .            # once
```

## Step 2 — Generate canonical biodiesel data (optional if present)

```bash
cd chemkan/scripts/data_gen
python generate_biodiesel.py --out ../../data/generated/biodiesel.npz --seed 0
```

- Output: `chemkan/data/generated/biodiesel.npz` (20 train + 10 test trajectories,
  30 points over 30 s, noise levels `0/1/2/5/7/10/15 %`, isothermal `species_only`).
- Regeneration is **optional** — the repository already ships `biodiesel.npz`; regenerate
  only for full reproducibility. The 30-point grid and the existing split are a settled
  decision (`chemkan/src/chemkan/ASSUMPTIONS.md` §4b); do not change them.

### Step 2a — Add the missing 3 % noise level (Fig. 5 needs eight)

Fig. 5A uses `0/1/2/3/5/7/10/15 %`; the shipped archive carries seven of those (no 3 %).
The generator seeds every level from an independent stream
(`seed + 1000 + round(level*1000)` for train, `+ 2000` for test), so a level can be added
without disturbing any other. This appends it **in place** from the archive's own clean
trajectories — it does not re-run the ODE solver and does not touch the split:

```bash
cd chemkan/scripts/data_gen
python add_biodiesel_noise_level.py --level 0.03            # dry run: reports only
python add_biodiesel_noise_level.py --level 0.03 --apply    # writes + verifies
```

`--apply` keeps a `.npz.bak`, then re-reads the file it wrote and **fails, restoring the
backup, unless every pre-existing array is bitwise identical**.

### Step 2b — Generate the Figure-3 evaluation condition

Fig. 3 plots one explicitly published unseen condition (`TG0 = 1.94`, `ROH0 = 1.43`,
`T = 334.8 K`) that is not in the 10-case test split. It is a separate artifact; the
canonical split is not modified.

```bash
cd chemkan/scripts/data_gen
python generate_biodiesel_fig3_condition.py --dry-run       # reports only
python generate_biodiesel_fig3_condition.py                 # writes the artifact
```

- Output: `chemkan/data/generated/biodiesel_fig3_condition.npz` — the 30-point clean
  trajectory, a 601-point dense trajectory for the plotted continuous prediction, and
  deterministic noisy observations at `0/5/10/15 %` from the separate
  `seed + 3000 + round(level*1000)` stream.

## Step 3 — Generate canonical hydrogen data (optional if present)

```bash
cd chemkan/scripts/data_gen
python generate_hydrogen.py --out ../../data/generated/hydrogen.npz
```

- Output: `chemkan/data/generated/hydrogen.npz` — the canonical trajectory dataset:
  **50 saved points** over 0.6 ms, **35 training + 1 held-out** condition,
  `species_then_temperature` layout (`[Y_1..Y_9, T]`).
- This is the source of Stage-1 species targets and the train-only normalization. It is
  **distinct** from the dense temperature cache in Step 4.

The 441-condition generalization grid for Fig. 8A is a **separate, evaluation-only**
archive, `hydrogen_fine.npz` — `T0 = linspace(950, 1200, 21)` x `phi = linspace(0.5, 1.5, 21)`,
holding the 35 original training conditions, the original held-out condition and 405
further unseen ones. It is present in the working tree but **not tracked**; regenerate it
with `generate_hydrogen.py --out ../../data/generated/hydrogen_fine.npz --grid fine`
if it is missing. The 21x21 spacing is
RECONSTRUCTED from the paper's 441-condition count and figure, not tabulated there. It is
never used to fit a normalizer.

## Step 4 — Generate the 20k H2 Stage-1 temperature cache

```bash
cd chemkan/scripts/data_gen
python generate_hydrogen.py --temperature-only --n-points 20000 \
    --out ../../data/generated/hydrogen_temperature_20000.npz
```

- Contains **only** the dense external Stage-1 temperature trajectory (`t`, `train_T`
  `(20000,35,1)`, `test_T`, `train_ics`, `test_ics`, provenance). It does **not** replace
  `hydrogen.npz` and stores no dense species.
- Production resolution is **20000** points over the same 0.6 ms. Stage-1 species targets
  and output times stay at **50** points; Stage 2 does not use this provider.
- 20000 is a reproduction implementation choice, not a paper-specified value. See
  `scripts/data_gen/README.md` for the schema and the linear-interpolation caveat.

## Step 5 — Train main biodiesel ChemKAN (direct autograd, seed 0)

```bash
cd chemkan/scripts
python train_biodiesel.py \
    --epochs 10000 --seed 0 \
    --run-dir ../../results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0
```

- Produces in the run dir: `checkpoint_final.pt`, `config.json`, `run.log`, `history.csv`,
  and (transiently) `checkpoint_resume.pt` overwritten every `--checkpoint-every` (default
  500) epochs and deleted on success.
- **Smoke run:** `--epochs 100`.
- **Resume** an interrupted run: re-issue the same command with `--resume`. Resume rejects
  any change to the scientific configuration (architecture, seed, sensitivity, solver
  tolerances, learning rate, input scaling, loss/PINN, dataset, H2 Stage-1 temperature
  provider, noise) and preserves the original `config.json`; only runtime options (device,
  `--checkpoint-every`) may differ. The epoch total may **grow** but never fall below the
  already-completed epoch. Interrupting (Ctrl-C) keeps `checkpoint_resume.pt`.
- **Overwrite:** a completed run is protected; `--overwrite` starts a **genuinely fresh**
  run, first clearing all prior artifacts (old histories, stale resume checkpoint,
  predictions, metrics, config, log) so nothing from the old run leaks into the new one.

### Step 5a — Noisy biodiesel runs and the Fig.-5B clean replay

`--noise-percent` trains against the archive's stored deterministic observations at that
level; `--eval-every 1` adds two evaluation-only history columns, `test_mse_noisy` and
`test_mse_clean` (paper Eq. 22). Both default OFF, so an unflagged run reproduces the
existing clean runs bitwise, with the same four history columns.

```bash
cd chemkan/scripts
python train_biodiesel.py --noise-percent 15 --epochs 10000 --eval-every 1 --seed 0 \
    --run-dir ../../results/reproduction/chemkan/biodiesel/noise/noise15_seed0
```

The evaluation runs inside the loss function, i.e. at the parameter state **before** that
epoch's optimizer update — the same state that produced the epoch's training loss — under
`no_grad`, with the RNG state saved and restored, using the train-only normalizer and the
full test set. It takes no optimizer step and writes no parameter; training losses are
bitwise identical with and without it. Epochs that are not evaluated leave those two
columns empty rather than carrying an interpolated value.

**The Fig.-5B clean replay** is a separately labelled 0 % run whose only purpose is to
supply the 0 % panel's missing history — `B0` remains the established 0 % result for
Figs. 3 and 5A and is not replaced:

```bash
python train_biodiesel.py --epochs 10000 --eval-every 1 --seed 0 \
    --run-dir ../../results/reproduction/chemkan/biodiesel/noise/clean_replay_seed0
```

### Step 5b — DeepONet baseline (Figs. 4, 5, 6)

```bash
cd deeponet
python train_biodiesel_deeponet.py --noise-percent 15 --epochs 10000 --eval-every 1 \
    --run-dir ../results/reproduction/baselines/deeponet/biodiesel/noise/noise15_seed0
python train_biodiesel_deeponet.py --width 6 --epochs 50000 \
    --run-dir ../results/reproduction/baselines/deeponet/biodiesel/scaling/w6_seed0
```

- Same dataset, same train-only normalizer, same Eq. 18 reduction as the ChemKAN runs, and
  the same run-directory layout.
- Architecture: branch `[7, w, w, w]`, trunk `[1, w-1, w]`, Hadamard combine, head
  `Linear(w, 6)`. At `w = 8` this is the paper-described architecture and totals **340**
  parameters against the **308** reported — a documented, unexplained discrepancy. No
  architecture is chosen to match 308.
- ReLU between layers, Glorot-normal weights, zero biases, Adam `lr = 1e-3`: these are the
  bundled reference example's conventions (`deeponet/src/deeponet_dataset.py`), **not**
  ChemKAN-paper facts, and are held fixed across widths and noise levels.
- Input/output scaling is a REPRODUCTION CHOICE recorded in `biodiesel_deeponet.py`: the
  branch consumes min-max normalized `[Y0, T]`, the trunk `tau = t / t_end`, and the model
  emits normalized species. Fed raw, the initial Eq. 18 loss is ~5e7 purely from
  conditioning. The statistics are stored in the checkpoint and reconstructed at
  evaluation, never refitted.

## Step 6 — Train main hydrogen ChemKAN (dense-Cantera, direct autograd, seed 0)

```bash
cd chemkan/scripts
python train_hydrogen.py \
    --stage1-temperature-source dense-cantera --stage1-temperature-points 20000 \
    --stage1-epochs 10000 --stage2-epochs 10000 --seed 0 \
    --run-dir ../../results/reproduction/chemkan/hydrogen/main/direct_autograd_seed0
```

- Requires the 20k cache from Step 4. Produces `checkpoint_final.pt`, `config.json`,
  `run.log`, `history_stage1.csv`, `history_stage2.csv`, and a transient
  `checkpoint_resume.pt` (records Stage 1 vs Stage 2 and the epoch within the stage;
  `--resume` continues the correct stage without restarting it from zero).
- **Smoke run:** `--stage1-epochs 50 --stage2-epochs 50`.
- **Ablation (not the production method):** the original sparse 50-point provider —
  `--stage1-temperature-source training-data` (no cache needed).

## Step 7 — Evaluate trained checkpoints

Training creates the checkpoint; evaluation loads it and produces MSE / predictions /
runtime **without retraining**.

```bash
cd chemkan/scripts
# biodiesel: metrics.json + prediction artifacts (train and test)
python evaluate_biodiesel.py --run-dir ../../results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0 \
    --split test  --metrics --save-predictions
python evaluate_biodiesel.py --run-dir ../../results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0 \
    --split train --metrics --save-predictions

# hydrogen
python evaluate_hydrogen.py --run-dir ../../results/reproduction/chemkan/hydrogen/main/direct_autograd_seed0 \
    --split test  --metrics --save-predictions
python evaluate_hydrogen.py --run-dir ../../results/reproduction/chemkan/hydrogen/main/direct_autograd_seed0 \
    --split train --metrics --save-predictions
```

- `--metrics` writes/merges `RUN_DIR/metrics.json` (run_id, `<split>_mse`, parameter count,
  `evaluation_wall_time_s`, solver, and — for hydrogen — the Stage-1 temperature config).
  `evaluation_wall_time_s` is the whole-command wall time (data + checkpoint loading +
  model reconstruction + integration + metric), **not** pure model inference — the paper
  speed-up benchmark (Table I) will use a dedicated warm-up/repeated-integration timer.
- `--save-predictions` writes `RUN_DIR/predictions/<split>_predictions.npz` with full
  provenance: `run_id`, `checkpoint_sha256`, canonical-JSON `architecture`, the actual
  `u_min`/`u_max` arrays, the reference (copied ground truth), `t`, initial conditions,
  species order, and a worded metric convention. Existing artifacts are **not** overwritten
  without `--force`.
- Compatibility is enforced on load: `run_id` + `architecture` + `checkpoint_sha256` must
  all match the checkpoint, or the artifact is rejected and regenerated.

### Step 7a — The Figure-5A metric triple

`--noise-percent` scores ONE set of predicted trajectories against TWO targets: the noisy
observations (Eq. 18) and the clean underlying trajectories (Eq. 22). Only the reference
changes. All three Fig.-5A numbers come from the **final checkpoint**, never from the last
training-history row.

```bash
cd chemkan/scripts
python evaluate_biodiesel.py --run-dir <run> --split train --noise-percent 15 --metrics
python evaluate_biodiesel.py --run-dir <run> --split test  --noise-percent 15 --metrics \
    --save-predictions

cd ../../deeponet
python evaluate_biodiesel_deeponet.py --run-dir <run> --split train --noise-percent 15 --metrics
python evaluate_biodiesel_deeponet.py --run-dir <run> --split test  --noise-percent 15 --metrics
```

Written to `metrics.json` as `train_mse_noisy`, `test_mse_noisy` and `test_mse_clean`
(plus `noise_percent`). At 0 % the two test metrics coincide exactly. Omitting
`--noise-percent` keeps the previous single `<split>_mse` key, so existing runs are
unaffected — verified against `B0`: `train 6.201653e-02`, `test 8.140053e-02`.

### Step 7b — Hydrogen figure evaluations from saved checkpoints

No hydrogen training is needed for Figs. 7, 8A, 8B or Table I. All three read a completed
`checkpoint_final.pt` and write nothing back into the run directory.

```bash
cd chemkan/scripts

# Fig. 8A -- 441 per-condition MSEs, plus flags and failures, never pre-averaged
python evaluate_hydrogen_grid.py --run-dir <run> \
    --out ../../results/reproduction/chemkan/hydrogen/generalization --save-predictions

# Fig. 8B -- ignition delay on a shared dense grid for reference AND model
python evaluate_hydrogen_ignition.py --run-dir <run> \
    --out ../../results/reproduction/tables

# Table I -- local PyTorch-vs-Cantera inference benchmark
cd benchmark
python benchmark_inference.py --run-dir <run> --out ../../../results/reproduction/tables
```

- **Grid evaluation** normalizes with the canonical `hydrogen.npz` TRAIN statistics — never
  the fine archive's own, and never refitted on the 441 conditions. All conditions are
  integrated in one batch (matching how a split is evaluated); if that batch raises or
  produces a non-finite value the affected conditions are retried individually and the
  summary records it. Failures are stored as failures with their condition identifiers.
- **Ignition evaluation** puts reference and model on the *same* dense grid (601 points
  over 0-0.6 ms) and through the *same* estimator, `t[argmax(gradient(T, t))]`; the
  reference temperature is the 20k Cantera cache interpolated onto that grid. Comparing a
  50-point reference derivative with a dense model derivative is exactly what this avoids.
  The evaluated set is fixed by which REFERENCE trajectories ignite (30 of 36 — the six
  950 K cases do not ignite in the window), never by whether the model succeeds; a
  non-igniting prediction is recorded as `no_ignition_in_window` with its delay undefined.
- **The inference benchmark** is a *local* PyTorch-vs-Cantera measurement and is not
  comparable to the paper's Arrhenius.jl 2.0x. It records hardware, dtype, thread count,
  solver/tolerances, warm-up, repetitions and timing scope, and reports ignition outcome
  and peak-temperature error **beside** the timing so a fast but wrong model cannot be
  read as a speed-up.
- `check_ignition.py` remains a two-condition **gate**, not a figure evaluation. Do not use
  it to decide whether the per-condition evaluations above may run: those record outcomes
  condition by condition, including failures.

## Step 7c — Figure assembly and the comparison table

```bash
cd chemkan/scripts/diagnostics
# Fig. 4 marker positions, read from the paper PDF (600 dpi vector render)
python digitize_paper_fig4.py --pdf ../../../docs/paper/ChemKANs_*.pdf \
    --out-dir /tmp/fig4 --json ../../../results/reproduction/tables/paper_fig4_digitized.json
# Fig. 4 points + explicit fit masks from our own scaling runs
python assemble_fig4_scaling.py
```

Both write to `results/reproduction/tables/`. The digitization is validated by refitting
the four slopes the paper prints in its own figure — that validates **the extraction**, not
any model trained here.

## Step 8 — Reproduction notebooks

After training/evaluation, the reproduction notebooks (analysis layer, not training):

- `notebooks/07_biodiesel_reproduction.ipynb` — trajectories, noise, scaling, DeepONet
  comparison → paper-equivalent Figs. 3–6.
- `notebooks/08_hydrogen_reproduction.ipynb` — train/test trajectories, generalization
  grid, ignition delay, ChemNODE comparison, speedup → paper-equivalent Figs. 7–8, Table I.

Each notebook: locates the run, loads `checkpoint_final.pt`, calls the repository
`evaluate_*` functions, loads a compatible prediction artifact **or** regenerates it from
the checkpoint (never using another checkpoint's predictions), computes the paper metrics,
displays results in paper order, and saves final figures to
`results/reproduction/figures/{biodiesel,hydrogen}/` and tables to
`results/reproduction/tables/`.

**Status: Figures 3-8 and Table I are all evaluated.** Notebook 07 produces Figs. 3, 4, 5A,
5B and 6; Notebook 08 produces Figs. 7, 8A, 8B and Table I from the saved hydrogen
checkpoints with no retraining. *Evaluation completed is not the same as paper result
matched* — the per-result verdict lives in
`results/reproduction/tables/reproduction_comparison.csv`.

### Regenerating the two untracked evaluation inputs

Both are exactly reproducible from committed generators at fixed seeds and are therefore
**not tracked** (the repository keeps large or derivable `.npz` out; see `.gitignore`):

```bash
cd chemkan/scripts/data_gen
python generate_biodiesel_fig3_condition.py                 # Fig. 3's plotted condition
python generate_hydrogen.py --out ../../data/generated/hydrogen_fine.npz --grid fine  # Fig. 8A
```

Notebook 07's Figure 3 and Notebook 08's Figure 8A will not run on a fresh clone until
these exist.

---

### Reusing the historical Stage-1 checkpoint — architecture must be passed explicitly

Since 2026-09-03 the hydrogen CLI defaults to **N=4 / base-ON** (`--num-basis 4`,
`--use-base-act`). The shared Stage-1 checkpoint every existing Stage-2 diagnostic branches
from was trained under the **N=5 / base-OFF** reading. `train_hydrogen.py` refuses
`--stage1-from` across an architecture change, so any run reusing it must say so:

```bash
python scripts/train_hydrogen.py \
  --stage1-from ../results/reproduction/chemkan/hydrogen/diagnostics/stage1_seed0/checkpoint_stage1.pt \
  --num-basis 5 --no-use-base-act \
  ...
```

Omitting `--num-basis 5 --no-use-base-act` fails loudly rather than silently retraining or
loading a mismatched core. **Do not retrain Stage 1** to avoid passing these flags — the
existing checkpoint is the fixed branch point that makes the Stage-2 arms comparable to one
another.

### Direct autograd vs FSA (reminder)

The current reproduction is `direct_autograd`. FSA is a documented gap (see
`chemkan/src/chemkan/solver.py`); adding it later means a new `fsa_seed0/` run directory,
never overwriting a `direct_autograd_seed0/` run.

**The planned FSA-vs-direct-autograd comparison must reuse the same historical Stage-1
checkpoint and the same N=5/base-OFF architecture** (flags above), so that the *sensitivity
backend is the only variable*. Changing the Stage-1 checkpoint, the architecture, the seed,
the initialization or the budget at the same time would make the comparison
uninterpretable.

---

## Diagnostics — hydrogen thermodynamic pathway (not paper reproduction)

The primary hydrogen run (`main/base_off_direct_autograd_seed0`) trains to completion but
**fails the ignition gate**. Diagnostic work lives apart from the reproduction:

- **Notebook:** `chemkan/notebooks/08_hydrogen_reproduction.ipynb`, section
  *"Thermodynamic linear-path diagnosis"* (after the temperature diagnosis).
- **Scripts:** `chemkan/scripts/diagnostics/` — `_thermo_coeffs.py` (Cantera `-h_k/cp`)
  and `hydrogen_thermo_intervention.py` (coefficient intervention).
- **Diagnostic runs:** `results/reproduction/chemkan/hydrogen/diagnostics/` — never in
  `main/`, never overwriting the primary checkpoint.

**Coefficient intervention** (reads the checkpoint, writes no checkpoint):

```bash
python3 chemkan/scripts/diagnostics/hydrogen_thermo_intervention.py \
    --run-dir results/reproduction/chemkan/hydrogen/main/base_off_direct_autograd_seed0
# -> results/reproduction/tables/hydrogen_thermo_intervention.csv
```

**Controlled initialization hypothesis test** (same Stage-1 state, everything else
identical; `--thermo-init random` is the default and reproduces current behavior exactly):

```bash
# A) random init (control)
python3 chemkan/scripts/train_hydrogen.py --thermo-init random --seed 0 \
    --stage1-epochs 10000 --stage2-epochs 1000 \
    --run-dir results/reproduction/chemkan/hydrogen/diagnostics/thermo_init_random_seed0

# B) physics-seeded init (treatment)
python3 chemkan/scripts/train_hydrogen.py --thermo-init cantera --seed 0 \
    --thermo-init-temperature 1050 --thermo-init-phi 0.9 \
    --stage1-epochs 10000 --stage2-epochs 1000 \
    --run-dir results/reproduction/chemkan/hydrogen/diagnostics/thermo_init_cantera_seed0
```

Use `--stage2-epochs 100 / 500 / 1000` for a short pilot. `--thermo-init cantera` records
full provenance (mechanism, reference T/phi, `cp_mass`, coefficient vector, species order,
formula) in both `config.json` and the checkpoint.

**Status:** the intervention shows the thermodynamic linear pathway is **strongly
implicated**; it does **not** establish that random initialization is the root cause. See
the notebook's ESTABLISHED / STRONGLY IMPLICATED / NOT YET ESTABLISHED summary.
