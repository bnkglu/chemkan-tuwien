# Reproduction results

Trained ChemKAN / baseline runs and the figures/tables derived from them. Physical
datasets live under `chemkan/data/generated/` and are **not** duplicated here.

## Structure

```
results/reproduction/
├── chemkan/
│   ├── biodiesel/{main,noise,scaling}/<run>/
│   └── hydrogen/{main,generalization}/<run>/
├── baselines/{deeponet,chemnode}/<run>/
├── figures/{biodiesel,hydrogen}/
└── tables/
```

## One directory per run

Each real training run gets its own directory whose **name** (plus `config.json`) carries
the experiment identity — e.g. `chemkan/hydrogen/main/direct_autograd_seed0/`. A future
FSA run is `fsa_seed0/` and never overwrites `direct_autograd_seed0/`. The directory path
also defines the run's `run_id` (e.g. `chemkan/hydrogen/main/direct_autograd_seed0`), which
is stored in `config.json`, the checkpoint, and every prediction artifact.

## Standard run contents

| File | What |
|---|---|
| `checkpoint_final.pt` | authoritative trained model + metadata (`run_id`, architecture, solver, sensitivity, Stage-1 temperature for H2). Written only on success. |
| `checkpoint_resume.pt` | **temporary** resumable snapshot (model + optimizer + stage + epoch + rng). Overwritten every `--checkpoint-every` epochs; **deleted after** `checkpoint_final.pt` is written. Kept if the run crashes/interrupts, so `--resume` can continue. |
| `config.json` | human-readable mirror of the run configuration. |
| `run.log` | run-level log (start, run_id, seed, sensitivity, device, params, stage transitions, final losses, completion). |
| `history.csv` (biodiesel) | per-epoch `epoch,total_loss,mse_loss,elapsed_seconds`. |
| `history_stage1.csv` / `history_stage2.csv` (hydrogen) | per-epoch histories with `species_mse`/`state_mse` + `pinn_loss`. Flushed each epoch, so they survive interruptions; resume continues without duplicating epochs. |
| `metrics.json` | written by evaluation: `run_id`, `<split>_mse`, param count, `evaluation_wall_time_s` (whole-command wall time, **not** pure inference), solver (+ H2 Stage-1 temperature). |
| `predictions/*.npz` | optional final prediction artifacts (see below). |

`elapsed_seconds` in the history CSVs is **segment** wall time measured from the start of
the current training call, so it resets on `--resume` (not cumulative across resumes). A
completed run deletes `checkpoint_resume.pt`; an interrupted one keeps it. `--overwrite`
clears every prior artifact so a replaced run starts fresh, and `--resume` refuses to
change the run's scientific configuration.

## Prediction artifacts & compatibility

Prediction `.npz` files under `predictions/` carry full provenance: `run_id`,
`checkpoint_sha256`, canonical-JSON `architecture`, the actual `u_min`/`u_max` arrays, the
reference (copied ground truth from the canonical dataset), `t`, initial conditions,
species order, and a worded metric convention (never a bare "MSE").

**Never assume** an artifact belongs to the checkpoint under analysis. On load, the
notebook/eval helper recomputes the checkpoint's SHA-256 and compares `run_id` +
`architecture` + `checkpoint_sha256`; **all three must match** or the artifact is rejected
and regenerated from the checkpoint. Existing artifacts are not overwritten without
`--force`.

## Direct autograd vs FSA

Current runs use `sensitivity = direct_autograd` (recorded in config + checkpoint). FSA is
not implemented; do not read these as FSA results.

## What is committed

Tracked (small, final): source, docs, notebooks, canonical datasets + the
`hydrogen_temperature_20000.npz` cache, `checkpoint_final.pt` for reported runs,
`config.json`, `history*.csv`, `metrics.json`, `run.log`, final `predictions/*.npz`
(< 10 MB), tables (CSV), and report figures (PDF/PNG).

Untracked (transient): `checkpoint_resume.pt`, smoke/debug runs, intermediate epoch
checkpoints, and diagnostic dense caches (50k/100k/200k). See the root `.gitignore` for the
exact negation rules. Artifacts > 10 MB should be regenerated from the checkpoint rather
than committed.
