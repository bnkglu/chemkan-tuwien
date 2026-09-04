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
  only for full reproducibility.

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
`results/reproduction/tables/`. These experiments are implemented incrementally; the
organization task only wires the workflow.

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
