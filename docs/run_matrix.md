# ChemKAN reproduction — consolidated run matrix

Status: **complete.** All 26 approved runs finished and every figure evaluation
(Figures 3-8 and Table I) has been produced from them. Nothing further is scheduled for
this phase; FSA remains a separate later task.

| group | runs | state |
|---|---|---|
| ChemKAN noise 1-15 % | 7 | complete, 10,000 epochs each |
| ChemKAN clean replay (Fig.-5B history) | 1 | complete; **bitwise reproduction of `B0`**, plus an epoch-5000 snapshot |
| DeepONet noise 0-15 % | 8 | complete, 10,000 epochs each |
| ChemKAN scaling h = 2, 3, 10, 17 | 4 | complete, 5,000 epochs each (h=4 reused from the replay snapshot) |
| DeepONet scaling w = 3, 5, 6, 8, 10, 13 | 6 | complete, 50,000 epochs each |
| Hydrogen | 0 | **no retraining**; Figs. 7/8A/8B and Table I evaluated from saved checkpoints |

The 5 % ChemKAN run was interrupted once by an external process kill and **resumed** from
its epoch-7,500 snapshot; its history is contiguous over all 10,000 epochs and its
configuration was preserved by the resume guard.

**Verdicts are in `results/reproduction/tables/reproduction_comparison.csv`**, which
separates "evaluation completed" from "paper result matched" for all 20 compared results.

## B. Prerequisite data actions — approval required

| # | Action | Command | Status |
|---|---|---|---|
| B1 | Add the missing 3 % biodiesel noise realization **in place** | `chemkan/scripts/data_gen/add_biodiesel_noise_level.py --level 0.03 --apply` | **APPLIED 2026-09-05 and independently verified against the backup: exactly two arrays added (`train_states_noise03`, `test_states_noise03`), zero pre-existing arrays changed, time grid and splits unchanged.** Adds `train_states_noise03` / `test_states_noise03` from the archive's own clean trajectories using the generator's independent per-level stream (train seed 1030, test 2030). Keeps a `.npz.bak`, re-reads the file and restores the backup unless every pre-existing array is bitwise identical. `t = 0` stays exact. **This writes to the tracked, supervisor-fixed `biodiesel.npz`, so it is held for your decision.** |
| B2 | Fig.-3 evaluation condition artifact | `generate_biodiesel_fig3_condition.py` | **done** — `chemkan/data/generated/biodiesel_fig3_condition.npz` (new file; the canonical split was not touched). TG0 = 1.94, ROH0 = 1.43, T = 334.8 K; 30-point clean trajectory, 601-point dense trajectory, noisy observations at 0/5/10/15 % from a separate `seed + 3000 + …` stream. |
| B3 | 441-condition fine grid | — | **already present**: `hydrogen_fine.npz`, 21 × 21, `T0 = linspace(950,1200,21)` × `phi = linspace(0.5,1.5,21)`, 35 training / 406 unseen. It was missing from the supplied results archive but is present in this checkout, so it is reused and not regenerated; the prompt document has been corrected to say so. |

Note: all three `.npz` files above are covered by `.gitignore`'s `*.npz` rule. Only
`biodiesel.npz`, `hydrogen.npz` and `hydrogen_temperature_20000.npz` have explicit
un-ignore entries, so `biodiesel_fig3_condition.npz` and `hydrogen_fine.npz` stay local
unless you decide to track them.

---

## C. New training — biodiesel noise sweep (Figs. 3, 5A, 5B, 6)

Eight ChemKAN and eight DeepONet conditions. `B0` covers ChemKAN 0 %, so **7 + 8 = 15**
runs remain. Every run: 10,000 epochs, seed 0, `--eval-every 1`.

### C1 — ChemKAN, 7 runs

| Noise | Serves | Run directory | Architecture | Prereq |
|---|---|---|---|---|
| 1 % | 5A | `results/reproduction/chemkan/biodiesel/noise/noise01_seed0` | h=4, N=3, n_mu=2, base OFF, 156 | — |
| 2 % | 5A, 5B | `.../noise/noise02_seed0` | ″ | — |
| 3 % | 5A | `.../noise/noise03_seed0` | ″ | **B1** (one of only two B1-dependent runs) |
| 5 % | 3, 5A | `.../noise/noise05_seed0` | ″ | — |
| 7 % | 5A, 5B | `.../noise/noise07_seed0` | ″ | — |
| 10 % | 3, 5A | `.../noise/noise10_seed0` | ″ | — |
| 15 % | 3, 5A, 5B, 6 | `.../noise/noise15_seed0` | ″ | — |

```bash
cd chemkan/scripts
python train_biodiesel.py --noise-percent 15 --epochs 10000 --eval-every 1 --seed 0 \
    --run-dir ../../results/reproduction/chemkan/biodiesel/noise/noise15_seed0
```

Measured cost: 15.5 s per 200 epochs with per-epoch evaluation → **≈ 13 min per run**,
≈ 1.5 h for all seven.

### C2 — Fig.-5B clean history replay, 1 run

| Serves | Run directory | Notes |
|---|---|---|
| **Fig. 5B, 0 % panel only** | `results/reproduction/chemkan/biodiesel/noise/clean_replay_seed0` | 0 % noise, 10,000 epochs, `--eval-every 1`. `B0`'s `history.csv` has training loss only and a final checkpoint cannot reconstruct earlier test losses. **Both** curves of the 0 % panel come from this run, so they share one parameter trajectory; no history is spliced. `B0` remains the established 0 % result for Figs. 3 and 5A. The replay's final metrics are reported beside `B0`'s as a reproducibility comparison only. |

### C3 — DeepONet, 8 runs

| Noise | Run directory |
|---|---|
| 0, 1, 2, 3, 5, 7, 10, 15 % | `results/reproduction/baselines/deeponet/biodiesel/noise/noise{00,01,02,03,05,07,10,15}_seed0` |

Architecture: branch `[7,8,8,8]`, trunk `[1,7,8]`, Hadamard, head `Linear(8,6)` →
**340 measured** parameters against the paper's **308** (documented, unexplained; no
architecture is selected to match 308). Adam `lr = 1e-3`, ReLU, Glorot-normal, biased
Linear — reference-derived from `deeponet/src/deeponet_dataset.py`, not ChemKAN-paper
facts. 10,000 epochs, `--eval-every 1`. The 3 % run needs **B1** (the second and last B1-dependent run).

Measured cost: ≈ 7 s per run — negligible.

---

## D. New training — Fig. 4 width sweep: **widths not yet selected**

The prompt document requires digitizing Fig. 4 and auditing the marker positions against
feasible counts *before* fixing the widths. That digitization has not been done, so I am
**not** proposing specific width rows yet rather than inventing them.

What is settled and measured:

- ChemKAN: `N = 3` fixed, base OFF fixed, vary `hidden_dim`; `P(h) = 39h` — verified
  against `model.parameters()` at h = 4 (156). `n_mu = ceil(h/2)` is a RECONSTRUCTION
  CHOICE; the paper states no Fig. 4 rule. Candidate family to *check* against the figure:
  h = [2, 3, 4, 6, 10, 18] → P = [78, 117, 156, 234, 390, 702]. The paper's 72-vs-78
  wording inconsistency is recorded, not engineered away.
- DeepONet: family branch `[7,w,w,w]`, trunk `[1,w-1,w]`, head `Linear(w,6)`. Measured
  counts: w = 2 → 52, 3 → 85, 4 → 124, 6 → 220, **8 → 340**, 10 → 484, 12 → 652, 18 → 1300.
- Epochs: ChemKAN 5,000; DeepONet 50,000. `B0` (10,000 epochs) is **not** a 5,000-epoch
  scaling point, and no suitable 5,000-epoch snapshot exists in the archive — h = 4 must be
  trained separately for this figure.

Estimated cost once widths are fixed: ≈ 4 min per ChemKAN width, ≈ 35 s per DeepONet width.

---

## E. Evaluation only — no training

| # | Serves | Source checkpoint | Command | Output |
|---|---|---|---|---|
| E1 | Fig. 5A, 0 % point | `B0` | `evaluate_biodiesel.py --run-dir <B0> --split {train,test} --noise-percent 0 --metrics` | `metrics.json` gains `train_mse_noisy` / `test_mse_noisy` / `test_mse_clean` (identical at 0 %) |
| E2 | Fig. 3 | ChemKAN 0/5/10/15 % checkpoints | integrate `biodiesel_fig3_condition.npz` from its initial state | per-column predictions + vector PDF |
| E3 | Fig. 5A/5B/6 | all C1–C3 checkpoints | `evaluate_biodiesel*.py … --noise-percent <n> --metrics --save-predictions` | three final metrics per model/noise from the **final checkpoint**, never from a history row |
| E4 | Fig. 7 | `H0`, then `Hnorm1` | `evaluate_hydrogen.py --run-dir <run> --split {train,test} --metrics --save-predictions` | trajectories at φ=0.9/T0=1050 and φ=1.3/T0=1150 |
| E5 | Fig. 8A | `H0`, then `Hnorm1` | `evaluate_hydrogen_grid.py --run-dir <run> --out results/reproduction/chemkan/hydrogen/generalization --save-predictions` | 441 per-condition MSEs + flags + failures (CSV/JSON/npz) |
| E6 | Fig. 8B | `H0`, then `Hnorm1` | `evaluate_hydrogen_ignition.py --run-dir <run> --out results/reproduction/tables` | 30-condition ignition table on a shared 601-point grid |
| E7 | Table I | `H0` | `benchmark/benchmark_inference.py --run-dir <run> --out results/reproduction/tables` | local PyTorch-vs-Cantera timing with accuracy attached |

E5–E7 were exercised end-to-end against `H0` today, writing only to a scratch directory:

- **E5**: 441/441 conditions integrated, 0 failures, 4.3 s; MSE median `2.66`, max `3.22`.
- **E6**: the reference ignites in **30 of 36** conditions (the six 950 K cases do not, as
  the paper says), and `H0` ignites in **0** of those 30 — recorded per condition as
  `no_ignition_in_window`, never as an argmax of a flat curve.
- **E7**: ChemKAN median 1.478 s vs Cantera 0.222 s over the same task → **0.15×**, i.e.
  locally *slower*, with 0/30 ignitions and a median peak-temperature error of 1546 K
  attached to the number.

These are dry-run measurements of the tooling, not the reported results; the reported
values will be regenerated into `results/` during the hydrogen phase.

---

## F. Totals

| Category | Count |
|---|---|
| Reused, no training | 5 checkpoints (`B0`, `Bbase`, `H1`, `H0`, `Hnorm1`) |
| New ChemKAN noise runs | 7 |
| Fig.-5B clean replay (history only) | 1 |
| New DeepONet noise runs | 8 |
| Fig. 4 width sweep | pending width selection |
| Hydrogen training | **0** |
| Evaluation-only tasks | 7 groups (E1–E7) |

**Exactly two of the sixteen runs depend on B1** — ChemKAN 3 % and DeepONet 3 %. The other **fourteen are independent of it**.

Approving section C launches **16 runs, ≈ 1.8 h wall clock** (8 ChemKAN runs at ≈ 13 min each, plus ≈ 1 min for all eight DeepONet runs). No run in C reuses another
run's directory, and no reused checkpoint is counted twice.
