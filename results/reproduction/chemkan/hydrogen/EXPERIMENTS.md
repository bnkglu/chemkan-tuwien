# Hydrogen experiment ledger

All values below were read from the actual artifacts in this tree (configs, histories,
probes, metrics, checkpoints), not from prose. Paths are relative to the repo root.

Categories: **PRIMARY REPRODUCTION** · **DIAGNOSTIC / ABLATION** · **REPRODUCTION
SENSITIVITY** · **PROPOSED EXTENSION**

> Every Cantera-init and scaled-random experiment is **DIAGNOSTIC / ABLATION — NOT PAPER
> REPRODUCTION**. The paper does not specify thermo initialization; these arms exist only
> to explain our own failure mode.

### Provenance of the facts used here

| claim | status |
|---|---|
| 36 H2 initial conditions; T0 = {950,1000,1050,1100,1150,1200} K; held-out (1150 K, φ=1.3); 35 training conditions | **PAPER-EXPLICIT** |
| Fig. 7 shows the representative **training** condition **T0=1050 K, φ=0.9** | **PAPER-EXPLICIT** |
| φ grid = {0.5,0.7,0.9,1.1,1.3,1.5} | **INFERRED REPRODUCTION CHOICE** — the prose prints only five φ values while 36 cases / six points per temperature require six; **φ=0.5 is our inferred value** reconciling that. Grid is frozen (`PHI_COARSE`, `T0_COARSE`, `TEST_IC`, both `.npz` unchanged). |
| Cantera-based `θ_thermo` initialization at **any** reference state | **DIAGNOSTIC DESIGN CHOICE (ours)** — the paper specifies no thermo initialization. Neither 1050/0.9 nor 1050/0.5 is paper methodology. |
| 1050/0.9 chosen as the intended diagnostic anchor | ours, because it is the paper's Fig. 7 representative training condition — **not** an ignition threshold (the paper also reports igniting cases at T0=1000 K within 0.6 ms). |

---

## 1. PRIMARY REPRODUCTION — original hydrogen run

| field | value |
|---|---|
| run dir | `results/reproduction/chemkan/hydrogen/main/base_off_direct_autograd_seed0/` |
| question | Does the 344-parameter ChemKAN reproduce Sec. III B? |
| category | **PRIMARY REPRODUCTION** |
| Stage-1 source | trained in-run (10 000 ep, final 0.328294) |
| thermo init | default random (pre-dates the `--thermo-init` flag) |
| seed / Stage-2 epochs | 0 / 10 000 (final loss **2.127238**) |
| train / test MSE | **2.12274 / 3.15558** |
| ignition | **none** |
| training peak T | **1058.0 K** vs ref 2669.7 K |
| held-out peak T | **1159.6 K** vs ref 2761.6 K |
| interpretation | Trains to completion but never ignites; the failed baseline. |
| status | **COMPLETE — preserved unchanged as evidence** |
| artifacts | `checkpoint_final.pt`, `config.json`, `history_stage{1,2}.csv`, `metrics.json` |

## 2. Shared Stage-1 checkpoint

| field | value |
|---|---|
| run dir | `results/reproduction/chemkan/hydrogen/diagnostics/stage1_seed0/` |
| question | Provide one exact kinetic core so Stage-2 arms are controlled. |
| category | DIAGNOSTIC / ABLATION (infrastructure) |
| config | `--stage1-epochs 10000 --stage2-epochs 0`, seed 0, final Stage-1 loss **0.328294** |
| status | **COMPLETE** — `checkpoint_stage1.pt` is the shared Stage-1 source for all arms below |

## 3. DIAGNOSTIC — random-init Stage-2 control (10k)

| field | value |
|---|---|
| run dir | `.../diagnostics/thermo_init_random_stage2_10000_seed0/` |
| question | Does the failure reproduce from the shared Stage-1 with default init? |
| category | **DIAGNOSTIC / ABLATION — NOT PAPER REPRODUCTION** |
| Stage-1 source | `stage1_seed0/checkpoint_stage1.pt` · thermo init **random** · seed 0 · 10 000 ep |
| final Stage-2 loss | **2.127238** · train/test MSE **2.12274 / 3.15558** |
| ignition | **none** (`never_ignites`) |
| training / held-out peak T | 1058.0 K / 1159.6 K (ref 2669.7 / 2761.6 K) |
| probe | `thermo_linear_norm` 4.848271e-01 → **4.397495e+01**; peak T 1050.0 → 1058.0 K |
| interpretation | Reproduces the primary failure exactly ⇒ a valid control. |
| status | **COMPLETE** |

## 4. DIAGNOSTIC — Cantera-init Stage-2 (10k)

| field | value |
|---|---|
| run dir | `.../diagnostics/thermo_init_cantera_stage2_10000_seed0/` |
| question | Does physically scaled thermo init change Stage-2 behavior? |
| category | **DIAGNOSTIC / ABLATION — NOT PAPER REPRODUCTION** |
| Stage-1 source | same `checkpoint_stage1.pt` · thermo init **cantera** · seed 0 · 10 000 ep |
| final Stage-2 loss | **0.236615** · train/test MSE **0.23684 / 0.23808** |
| ignition | **yes**, from epoch 0 onward (`ignites_initially_and_persists`) |
| training | peak **2714.1 K** vs ref 2669.7 K; ignition delay **exact match (0.0 % rel.)** |
| held-out **limitation** | peak **3029.1 K** vs ref **2761.6 K** (overshoot **+267.5 K**); ignition delay **6.122e-05 s vs 7.347e-05 s → 16.7 % relative error** |
| probe | `thermo_linear_norm` 1.712274e+05 → 1.712270e+05 — **essentially unchanged**; training accepts the coefficients rather than learning them |
| **intended CLI reference** | **T = 1050 K, φ = 0.9** (paper's Fig. 7 representative *training* condition; a diagnostic anchor of ours, not paper methodology) |
| **actual coefficient computation** | **T = 1050 K**, with the initial species composition corresponding to **φ = 0.5** |
| **cause** | the historical diagnostic implementation used `data["species_TBm"][0,0]` — the *first* training composition — and **ignored `--thermo-init-phi`** in the calculation, while still passing `--thermo-init-temperature` (1050 K) to Cantera. So the temperature was as intended; only the composition was not. Note the composition is φ-dependent only, so "the first training condition" contributes **φ=0.5**; **T=950 K was never used** in the coefficient calculation. |
| verification | Recomputing at **T=1050 K with the φ=0.5 composition (float32, as the loader supplies it)** reproduces the stored historical vector **bit-for-bit** (max abs diff 0.0) and its norm **171227.4433**. The 1050/0.9 composition gives a different vector (norm 153535.3801). |
| status of fix | Future runs resolve the reference IC **exactly** from `--thermo-init-temperature/--thermo-init-phi` and fail loudly otherwise. The historical config/checkpoint were **not** modified. |
| interpretation | Thermo initialization strongly affects our implementation. Does **not** show the authors used Cantera init. |
| status | **COMPLETE** |

## 5. DIAGNOSTIC — coefficient intervention (no retraining)

| field | value |
|---|---|
| script | `chemkan/scripts/diagnostics/hydrogen_thermo_intervention.py` |
| question | Can replacing **only** `thermo.linear` restore ignition without training? |
| category | **DIAGNOSTIC / ABLATION — NOT PAPER REPRODUCTION** |
| result | Yes, at all four reference states. Training peak 2741–2892 K, held-out 2827–2970 K (ref 2669.7 / 2761.6 K); temperature-channel MSE 25.5 → 0.12–0.76. Total trajectory MSE **rises** (2.814 → 3.088–3.777). |
| artifacts | `results/reproduction/tables/hydrogen_thermo_intervention.csv`, `hydrogen_thermo_coefficients.csv`, `figures/hydrogen/hydrogen_DIAGNOSTIC_thermo_coefficients.*` |
| status | **COMPLETE** |

---

## 6. scaled-random controls (scale vs structure)

**Question:** did Cantera init succeed mainly because of its **magnitude**, or because of its
physically meaningful **signs / relative species structure**?

All five use: `--stage1-from .../stage1_seed0/checkpoint_stage1.pt`, `--seed 0`,
`--stage2-epochs 500`, `--stage2-probe --stage2-probe-epochs 0,1,2,5,10,20,50,100,200,500`.
Only `thermo.linear` differs; `kinetic.*` and `thermo.correction.*` are bit-identical
across arms (verified by test). Category for all: **DIAGNOSTIC / ABLATION — NOT PAPER
REPRODUCTION**. Status: **all five runs COMPLETE** (checkpoints + `stage2_probe.csv` present in each run dir). The §7 pre-registered classification has **not** been applied yet — the trajectory diagnostic in §6C is descriptive only.

**Control reference = T 1050 K / φ 0.5** for all five arms (Cantera target norm, descriptive
sign comparison, provenance). Rationale: the control must match the norm of the **actually
successful** historical vector, not the intended-but-unused 1050/0.9 vector.

Measured scale ladder (recomputed via the corrected resolver, not hard-coded):

| arm | ‖w‖ | factor vs default |
|---|---|---|
| default random (dir 0) | **0.484827** | ×1 |
| ×1e4 (dir 0) | 4.848271e+03 | ×1e4 |
| ×1e5 (dir 0) | 4.848271e+04 | ×1e5 |
| **exact norm match** (dir 0) | **1.712274e+05** | **×3.531722e+05** |
| Cantera reference norm @1050/φ0.5 | **171227.4433** (= historical successful norm) | — |

Sequence: **default → ×1e4 → ×1e5 → exact norm match (×3.531722e+05, dir 0)**. Norm matching
is ~3.5× the ×1e5 arm. Per-direction factors differ because ‖w_random‖ differs by direction.

### A. Fixed-direction scale response (both use **direction seed 0**)

| run dir | thermo init | dir seed | epoch-0 check |
|---|---|---|---|
| `.../diagnostics/thermo_init_scaled_random_1e4_dir0` | scaled-random ×1e4 | 0 | stable: ‖w‖ 4848.27, signs 1/9, peak 1050.0 K, **no ignition** |
| `.../diagnostics/thermo_init_scaled_random_1e5_dir0` | scaled-random ×1e5 | 0 | stable: ‖w‖ 48482.71, signs 1/9, peak 1050.0 K, **no ignition** |

> These measure scale response along **one specific random direction only**. Do not
> generalize to arbitrary directions without the norm-matched direction arms below.

### B. Exact Cantera-norm random controls (`--thermo-match-cantera-norm`)

| run dir | dir seed | ‖w‖ | sign match vs Cantera | epoch-0 check |
|---|---|---|---|---|
| `.../thermo_init_scaled_random_normmatched_dir0` | 0 | 171227.4375 (×3.531722e+05) | **1/9** | stable: peak 1050.0 K, **no ignition** |
| `.../thermo_init_scaled_random_normmatched_dir1` | 1 | 171227.4531 (×3.398490e+05) | **6/9** | stable: peak **2447.5 K**, ignites |
| `.../thermo_init_scaled_random_normmatched_dir2` | 2 | 171227.4375 (×4.800180e+05) | **4/9** | stable: peak **4499.1 K**, ignites |

Sign-match counts are **descriptive only** and were never used to accept or reject a draw.
Epoch-0 values above are the stability pre-check, not results.

---

### C. Temperature-trajectory comparison (DIAGNOSTIC plotting, no training)

Compares the actual `T(t)` produced by each initialization arm against the Cantera
reference at the two representative conditions (training 1050 K / φ 0.9, held-out
1150 K / φ 1.3), so the arms are judged on trajectory **shape and timing** and not on a
scalar loss alone. Reads checkpoints only; writes no checkpoint and no run directory.

```bash
python3 chemkan/scripts/diagnostics/plot_thermo_initialization_comparison.py
```

| artifact | path |
|---|---|
| script | `chemkan/scripts/diagnostics/plot_thermo_initialization_comparison.py` |
| main figure | `results/reproduction/chemkan/hydrogen/figures/hydrogen_DIAGNOSTIC_thermo_initialization_temperature_comparison.{png,pdf}` |
| scale-response figure | `results/reproduction/chemkan/hydrogen/figures/hydrogen_DIAGNOSTIC_thermo_initialization_scale_response.{png,pdf}` |
| table | `results/reproduction/chemkan/hydrogen/tables/hydrogen_thermo_initialization_trajectory_comparison.csv` |

Metrics use the repository ignition-delay definition (time of maximum `dT/dt`, undefined
below a 100 K rise, `hydrogen_thermo_intervention.ignition_delay`) on the dataset's own
50-point grid over [0, 0.6 ms]; the plotted model curves are additionally integrated at
601 points over the same interval for shape fidelity. Because the 50-point grid quantizes
the delay to ~12.2 µs, the CSV also carries `*_dense` columns applying the **same**
definition to the model on the 601-point grid, compared against the reference's native
50-point estimate (the Cantera reference exists only at those 50 points). No new ignition
criterion is introduced. `min_T_K*` records whether a trajectory cools before heating. These figures are an **ablation diagnostic**, not a
reproduction of paper Fig. 7.

---

## 6D. COMPLETE — long-budget persistence study (10 000 Stage-2 epochs)

**Status: COMPLETE FOR CURRENT SCOPE.** Category: **DIAGNOSTIC / ABLATION — NOT PAPER
REPRODUCTION**. Same shared Stage-1 checkpoint, global seed 0, architecture, loss, PINN,
Tsit5 tolerances, `direct_autograd` backend and Adam@2e-3 as every other arm; only the
`thermo.linear` initialization differs. Configuration identity between each arm's 500- and
10 000-epoch run is asserted programmatically in Notebook 09 §2.

| run dir (under `.../hydrogen/diagnostics/`) | initialization | train MSE | test MSE | gate train / held-out | operational ignition-flag history |
|---|---|---|---|---|---|
| `thermo_init_random_stage2_10000_seed0` | random (default control) | 2.1227 | 3.1556 | ✗ / ✗ | `never_ignites` |
| `thermo_init_cantera_stage2_10000_seed0` | Cantera physical direction | 0.2368 | 0.2381 | **✓** / ✗ | `ignites_initially_and_persists` |
| `thermo_init_scaled_random_1e5_dir0_stage2_10000` | scaled-random ×1e5, dir 0 | 3.7567 | 6.5977 | ✗ / ✗ | `ignition_emerges_and_persists` |
| `thermo_init_scaled_random_normmatched_dir0_stage2_10000` | Cantera norm, random dir 0 | 3.1108 | 5.3996 | ✗ / ✗ | `ignition_emerges_and_persists` |
| `thermo_init_scaled_random_normmatched_dir1_stage2_10000` | Cantera norm, random dir 1 | 0.2697 | **0.2071** | **✓ / ✓** | `intermittent` |
| `thermo_init_scaled_random_normmatched_dir2_stage2_10000` | Cantera norm, random dir 2 | 2.2576 | 3.9029 | ✗ / ✗ | `intermittent` |

MSE is the normalized trajectory MSE (Eq. 18) recomputed with `evaluate_hydrogen`. The gate
is |peak error| ≤ 150 K **and** |ignition-delay error| ≤ 25 %, measured against the
**genuine dense Cantera reference** (`hydrogen_temperature_20000.npz`, 20 000 points) — not
an interpolation of the 50-point data. "Operational ignition-flag history" describes when
the repo's ≥100 K guard fired over the full probe history; it is **not** a statement that a
trajectory is physically correct.

**Interpretation (concise).** Every arm improved between 500 and 10 000 epochs, but the
ranking and the ~26× spread among *equal-norm* directions survived the twenty-fold budget
increase — the direction dependence is **not** a 500-epoch transient. The best arm overall
is a **nonphysical** norm-matched random direction (`dir1`), which is the only arm passing
both representative-condition gates; Cantera init passes the training condition but
overshoots the held-out peak. Two arms (`x1e5_dir0`, `nm_dir0`) land training-condition
ignition delays inside the ±25 % gate while carrying ~26–30× worse test MSE, so **correct
ignition timing alone is not evidence of a correct model**. Across every large-scale arm
`cos(θ_final, θ_0) = 1` to ~1e-6 with relative displacement ~1e-6–1e-3, while the kinetic
core moves by 0.2–1.3 relative: the large thermodynamic vectors barely rotate and the rest
of the network adapts around them. No arm moved toward the Cantera direction.

**Full analysis:** `chemkan/notebooks/09_hydrogen_thermo_failure_analysis.ipynb` §15–§23,
including §18E (why `dir0` fails: rate-weighted misalignment, wrong-signed H2O coefficient)
and §18F (measured `dL/dθ` per block; `θ_thermo`'s gradient is 5–6 orders below the kinetic
core's, and the apparent "no rotation" is largely Adam's bounded step against ‖θ₀‖ ≈ 1.7e5).

| artifact | path |
|---|---|
| 10k trajectories vs ground truth | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_thermo_initialization_10k_temperature_comparison.{png,pdf}` |
| 500 → 10k persistence | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_thermo_initialization_500_vs_10k.{png,pdf}` |
| `‖θ_thermo‖` history | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_thermo_linear_norm_history_10k.{png,pdf}` |
| direction / displacement history | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_thermo_direction_history_10k.{png,pdf}` |
| 10k master table | `.../hydrogen/tables/hydrogen_thermo_initialization_10k_comparison.csv` |
| 500 vs 10k | `.../hydrogen/tables/hydrogen_thermo_initialization_500_vs_10k.csv` |
| coefficient evolution | `.../hydrogen/tables/hydrogen_thermo_linear_evolution_summary.csv` |
| gradient flow per block | `.../hydrogen/tables/hydrogen_thermo_gradient_flow.csv` |

**The scale/direction ablation is now COMPLETE FOR CURRENT SCOPE.** Not recommended: more
scale factors, more 10k random directions, `x1e4` at 10k, or another Cantera-init run. The
next experiment is a controlled `direct_autograd` vs `forward_sensitivity` comparison at the
**default** initialization — see Notebook 09 §23.

---

## 6E. COMPLETE — base-activation architecture sensitivity (N=4 / base-ON)

**Status: COMPLETE FOR CURRENT SCOPE.** Category: **DIAGNOSTIC / ABLATION — NOT PAPER
REPRODUCTION**. Tests the alternative 344-parameter reading of the paper's Eq. 11.

| | `num_basis` | `use_base_act` | params | status |
|---|---|---|---|---|
| Interpretation A (sections 1–6D) | 5 | OFF | 344 | INFERRED, historical count-matching reading |
| **Interpretation B (this section)** | **4** | **ON** | **344** | **INFERRED, Eq. 11-aligned reading** |

Base activation is PAPER-EXPLICIT (Eq. 11); hidden=3, n_mu=3, 344 params are PAPER-EXPLICIT;
**the grid size N is NOT stated in the paper — neither N=5 nor N=4 is quoted.** The two
readings agree block-for-block (150/135/9/50) and are indistinguishable from the count.

**Shared Stage-1 (fresh, base-ON):** `diagnostics/base_on_n4/stage1_seed0/checkpoint_stage1.pt`
(sha256 `6b25dfcf4930…`, seed 0, final loss 0.3149; base-OFF was 0.3283). All six arms below
verified by `verify_base_on_matrix.py` (33 checks) to share it and to differ only in
`thermo_init`. Every arm additionally carries an immutable `checkpoint_stage2_epoch_500.pt`
snapshot and per-epoch `nfe` / `epoch_wall_time_s` columns (new instrumentation, tested
behaviour-neutral).

| run dir (under `diagnostics/`) | init | train MSE | test MSE | gate tr/held | Stage-2 stability | wall |
|---|---|---|---|---|---|---|
| `base_on_n4/random_stage2_10000_seed0` | default random | 2.1394 | 3.1461 | ✗/✗ | stable | 8.76 h |
| `base_on_n4/cantera_stage2_10000_seed0` | Cantera init | 19.6911 | 10.0691 | ✗/✗ | DIVERGED (final/min 54x) | 2.64 h |
| `base_on_n4/scaled_random_1e5_dir0_stage2_10000` | random x1e5 dir0 | 13188.5068 | 11482.9023 | ✗/✗ | DIVERGED (final/min 2975x) | 3.56 h |
| `base_on_n4/normmatched_dir0_stage2_10000` | norm-matched dir0 | 1566.3744 | 25.8530 | ✗/✗ | DIVERGED (final/min 401x) | 2.31 h |
| `base_on_n4/normmatched_dir1_stage2_10000` | norm-matched dir1 | 0.3101 | 0.2088 | ✓/✗ | stable | 3.38 h |
| `base_on_n4/normmatched_dir2_stage2_10000` | norm-matched dir2 | 42.5627 | 41.2085 | ✗/✗ | DIVERGED (final/min 2x) | 3.02 h |

MSE = Eq. 18 (sum over 50 times). Gate = |ΔT_peak| ≤ 150 K and |Δτ| ≤ 25 % on the genuine
20 000-point Cantera grid. Stability := final loss < 2× the run's own minimum.

**Interpretation (concise).** (1) The Eq. 11 base term does **not** change the default-random
outcome — same non-igniting basin, test MSE equal to two decimals with base-OFF. (2) Scale
and direction sensitivity persist; `dir1` is again the best random direction in both readings
(test MSE ≈ 0.21 in both). (3) **New:** base-ON destabilizes Stage 2 for large `θ_thermo`
initializations — 6/6 base-OFF arms stable vs 2/6 base-ON; base-ON Cantera is *better* than
base-OFF at epoch 500 (training peak error −0.7 K) and then diverges at ~epoch 3000. (4) The
diverged losses are dominated by trace species (HO2 range 1.7e-4 → normalized error ~160);
final loss is not a usable cross-arm metric for those cells. (5) Base-ON needs ~3× the RHS
evaluations per solve; measured NFE tracks wall time, settling the §6D runtime question in
favour of solver work. (6) No base-ON arm passes both representative gates at 10k; base-OFF
`dir1` still does.

**Full analysis:** `chemkan/notebooks/09_hydrogen_thermo_failure_analysis.ipynb` §22.

| artifact | path |
|---|---|
| 10k trajectories vs ground truth | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_base_on_n4_temperature_comparison_10000.{png,pdf}` |
| 500 → 10k persistence (train / held-out) | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_base_on_n4_500_vs_10k_{training,heldout}.{png,pdf}` |
| thermo weights per arm | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_base_on_n4_thermo_weights.{png,pdf}` |
| norm + loss (both readings overlaid) | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_base_on_n4_norm_and_loss.{png,pdf}` |
| runtime + NFE | `.../hydrogen/figures/hydrogen_DIAGNOSTIC_base_on_n4_runtime_nfe.{png,pdf}` |
| full matrix (both readings × 500/10k) | `.../hydrogen/tables/hydrogen_base_on_n4_matrix.csv` (+ `_full.csv` with per-state Eq. 18 shares) |
| Stage-1 comparison | `.../hydrogen/tables/hydrogen_base_on_n4_stage1_comparison.csv` |
| matched base-OFF vs base-ON | `.../hydrogen/tables/hydrogen_base_on_n4_matched_comparison.csv` |

**Superseded table removed:** `hydrogen_thermo_failure_analysis_master.csv` (used an
interpolated 601-point reference; replaced by `hydrogen_thermo_initialization_10k_comparison.csv`
on the genuine dense grid). `plot_thermo_initialization_comparison.py` now also uses the
genuine dense reference for its `*_dense` columns.

**Next:** genuine FSA — the clearest remaining PAPER-EXPLICIT training-method difference
(not necessarily the only implementation difference; N, `θ_thermo` init and derivative
scaling are all unstated). See Notebook 09 §24.

---

## 7. Pre-registered interpretation (fixed BEFORE the runs)

Classify each arm from the **entire probe history**, not epoch 500 alone:
`never_ignites` · `ignition_emerges_and_persists` · `ignites_initially_and_persists` ·
`ignites_then_loses_ignition` · `intermittent` · `unstable_at_initialization`.

- **A.** Norm-matched random directions consistently ignite and reach the same qualitative
  low-error regime as Cantera-init ⇒ **large initialization scale is sufficient** to escape
  the flat-temperature failure; correct Cantera structure is not required for that effect.
- **B.** Norm-matched random directions consistently fail to ignite while Cantera-init
  succeeds ⇒ **magnitude alone is insufficient**; direction/sign/relative structure carries
  important information.
- **C.** Scaled-random arms ignite but stay quantitatively much worse than Cantera-init ⇒
  scale escapes the non-igniting basin, **physical structure improves the reachable
  solution**.
- **D.** Large random directions unstable at epoch 0 while Cantera-init is stable ⇒
  magnitude alone is not sufficient; **physical structure provides dynamical stability**.
- **E.** Directions 0/1/2 give qualitatively different outcomes ⇒ magnitude matters but the
  outcome is **strongly direction-dependent**; do **not** conclude scale alone is sufficient.

The ×1e4 and ×1e5 arms both use **direction 0**. If norm-matched direction 0 later behaves
qualitatively differently from directions 1 and 2, treat the direction-0 scale-response
curve as **provisional / direction-specific**.

These rules must not be reinterpreted after seeing results.

---

## 8. Current scientific status

**ESTABLISHED**
- Ordinary/default random initialization reproduces the non-igniting failure.
- Changing only the thermodynamic initialization to Cantera-scale physical coefficients
  dramatically improves Stage-2 behavior and restores ignition.
- Thermodynamic initialization strongly affects our current implementation.

**STRONGLY SUPPORTED**
- Our physical-rate Eq. 14 implementation has a severe thermodynamic parameter-scale /
  optimization issue.

**NOT ESTABLISHED**
- scale alone is sufficient;
- correct physical coefficient structure is necessary;
- the authors used Cantera initialization;
- the authors used normalized derivatives;
- FSA would or would not fix the default initialization;
- the result is independent of the `thermo.linear` random direction.

## 9. PROPOSED EXTENSION (not started)

- FSA implementation (paper's stated sensitivity method) — **not implemented**.
- Thermodynamic reparameterization (e.g. normalized-derivative Eq. 14) — **not implemented**.
- Decide only **after** analyzing the five 500-epoch arms above.
