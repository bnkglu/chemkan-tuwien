# Authors' materials

Two things the authors provided after the legacy reproduction: a written clarification of
the paper (§1) and a released biodiesel example, which §2 matches in PyTorch.

## 1. Authors' clarification

Source: authors' clarification with an email (2026-09-22). Items are
paraphrased and numbered here so the rest of the repository can cite them as "email B5".
"Released code" refers to the biodiesel example of §2 (`DENG-MIT/ChemKAN` at `d7aa5ab`).

Status: **done** = current code/data follow it; **not yet** = current code/data do not;
**legacy differs** = only the legacy runs are affected; **n.a.** = nothing to implement.

| # | clarification | released code | our repo status |
|---|---|---|---|
| G1 | Manuscript typos do not affect results. | — | n.a. |
| G2 | Inputs are min-max normalized to [0, 1] in most cases (raw temperature would saturate the tanh normalizer). | species and T min-max (lines 132-139) | done: `--input-scaling minmax`, train-only statistics |
| G3 | Neither final model uses the Swish base activation; the base term in Eq. 11 is left over from an earlier revision. | `use_base_act = false` (lines 122, 124) | biodiesel done (default off); hydrogen not yet (`train_hydrogen.py` default `use_base_act=True`; legacy `hydrogen/main/base_off_direct_autograd_seed0` used base off) |
| B1 | Biodiesel ChemKAN has 156 parameters (no base activation). | 156 | done |
| B2 | Smallest scaling model: width 2, grid 3 = 78 parameters. | n.a. | done (`scaling*/h02_seed0`, 78 parameters) |
| B3 | DeepONet 308 = branch 3→8→8→8 on (TG, ROH, T), trunk 1→7→8, head 8→6; the four always-zero initial species are dropped. | n.a. | done for new runs (`deeponet/biodiesel_deeponet.py` `BRANCH_IN = 3`, 308 parameters at w = 8); legacy DeepONet runs used the 7-input branch (340) and are still rebuilt with it; other Fig. 4 sizes: §3 |
| B4 | Half multiplication nodes (LeanKAN rule, odd counts rounded up); all-multiplication gave similar results; the paper follows Sec. 3.1.1. | layer 2 `mult_flag = true` (2 of 4 nodes) | done (`--n-mu 2`) |
| B5 | Noise is additive: N(0, 1) × species maximum × noise %, clipped at 0 (4 of 6 species start at zero). | line 98, clip line 100 | done in `biodiesel_v2.npz` (`generate_biodiesel.py --noise-mode additive_species_max`, the new default); legacy `biodiesel.npz` keeps multiplicative noise |
| B6 | The released example is a pared-down noisy case with a few minor differences from the paper; it converges in ~1e4 epochs. | 10,000 epochs (line 22) | n.a. |
| H1 | 36 initial-condition combinations; the φ list in the text omits 0.5; Fig. 8 shows all 36. | n.a. | done (`hydrogen.npz`: 35 train + 1 test) |
| H2 | Hydrogen uses no base activation, so grid size 5 is correct. | n.a. | implemented in the author-matched hydrogen script (not yet trained) |
| H3 | Training grid of 100 time steps; data from Arrhenius.jl `CVODE_BDF`, rtol = atol = 1e-14, checked against Cantera (dtmax 1e-5 s, dTmax 1 K), interpolated onto the 100 points. | n.a. | done (`hydrogen.npz`: 100 points on 0-0.6 ms; generated with Cantera, not Arrhenius.jl) |
| H4 | Dense time/temperature kept separately as a linear interpolant, sampled inside the forward pass. | n.a. | done (Stage 1 `--stage1-temperature-source dense-cantera`, 20,000 points, linear) |
| H5 | Recovering exact h_i/c_p values was neither expected nor tested. | n.a. | n.a. |
| H6 | Thermodynamic linear weights all initialized to 1e-5, so dT/dt starts near zero; random init could overpredict T and destabilize training. | n.a. | implemented in the author-matched hydrogen script (not yet trained) |
| H7 | Optional warm-up: train only the linear thermo coefficients briefly with the kinetic core frozen, then train normally. | n.a. | implemented in the author-matched hydrogen script (not yet trained) |
| H8 | The loss is not divided by the number of time steps, because all trajectories share one 100-step grid; training dynamics are unaffected. Stated for hydrogen; Eq. 18 is the same for biodiesel. | biodiesel `Flux.mse` does divide by species × times (line 166): Eq. 18 = 30 × it, see `explicit_mse_loss_check.txt` | done (`losses.trajectory_mse`: mean over states, sum over time) |
| H9 | 1e5-2e5 epochs per stage; no stopping rule; the loss flattens early but keeps decreasing slowly; higher learning rates diverged; days on a 2019 CPU with a first-order optimizer. | n.a. | implemented in the author-matched hydrogen script (not yet trained) |

## 2. Released biodiesel example match

### Scope

A controlled PyTorch match to `ChemKAN_biodiesel_example.jl` from
[DENG-MIT/ChemKAN](https://github.com/DENG-MIT/ChemKAN) at
`d7aa5abecbb595d504c86d031cd478a549c614cb`. It is neither the legacy protocol nor the
clarified protocol, and it changes no repository default. The local copy of the authors'
files matches a fresh clone of that commit byte for byte (8 files,
`julia_reference/authors_source_check.txt`). The Julia run used Julia 1.11.1 and a copy with
three API-compatibility lines changed (`julia_reference/compat.patch`).

Code: `chemkan/scripts/author_repo_match/`. Artifacts:
`results/experiments/biodiesel/author_repo_match/`.

| script | produces |
|---|---|
| `export_julia_reference.jl` | `julia_reference/*.txt`: the example's data, normalization, trained parameters `p.txt`, loss histories |
| `export_julia_init.jl` | `julia_reference/p_init.txt`: Julia's initial parameters (reruns the script to `Lux.setup`) |
| `build_dataset.py` | `data/author_repo_match_biodiesel.npz` (gitignored; rebuilt from `julia_reference/`) |
| `check_equivalence.py` | `equivalence/equivalence.json`: PyTorch model = Julia model for the same parameters |
| `train_author_repo_match.py` | the runs below; options `--data {author_repo,canonical}`, `--sensitivity {direct_autograd,fsa}`, `--solve-mode {per_trajectory,batched}`, `--init-from` |

All runs: 10,000 epochs, float64, Adam lr 1e-2, torchdiffeq Tsit5 rtol 1e-2 / atol 1e-6.
Losses are the released example's `Flux.mse` (multiply by 30 for Eq. 18). "Late" = median
over epochs 8,001-10,000. Author-data runs are compared with Julia; canonical-data runs with
legacy B0, which uses the same data and normalization; B0's late median is converted from
Eq. 18 to the `Flux.mse` convention (÷ 30) before the 0.072× / 0.073× ratios are computed.

| run | data | start | solves | gradients | wall (s) | train late | val late | ratio (train / val) |
|---|---|---|---|---|---|---|---|---|
| `julia_reference` | author | Julia seed 1234 | loop | ForwardDiff | 2,208 | 5.03e-5 | 2.03e-4 | — |
| `run_seed0` | author | seed 0 | loop | autograd | 6,702 | 7.44e-5 | 3.21e-4 | 1.48 / 1.59 × Julia |
| `run_batched_seed0` | author | seed 0 | batched | autograd | 342 | 6.70e-5 | 2.94e-4 | 1.33 / 1.45 × Julia |
| `run_juliainit` | author | Julia `p_init` | loop | autograd | 6,351 | 7.80e-5 | 3.43e-4 | 1.55 / 1.69 × Julia |
| `run_batched_juliainit` | author | Julia `p_init` | batched | autograd | 371 | 7.47e-5 | 2.86e-4 | 1.49 / 1.41 × Julia |
| `run_batched_fsa_juliainit` | author | Julia `p_init` | batched | FSA | 1,588 | 4.74e-5 | 2.41e-4 | 0.94 / 1.19 × Julia |
| `canonical_data_seed0` | canonical | seed 0 | loop | autograd | 5,821 | 8.59e-5 | 6.86e-4 | 0.072 × B0 (train) |
| `canonical_data_fsa_seed0` | canonical | seed 0 | loop | FSA | 14,389 | 8.71e-5 | 6.88e-4 | 0.073 × B0 (train) |

### Findings

One run per configuration, so there is no seed spread; read the ratios as single-run values.

- **Loop vs batched solves.** Batched solves reach the same late-loss range but follow a
  different loss trajectory, and are 17-20× faster (6,702 s → 342 s; 6,351 s → 371 s).
- **Initial weights.** Starting from Julia's exact initial parameters does not close the
  ~1.5× gap of the direct-autograd runs (1.49× batched, 1.55× loop).
- **FSA.** From Julia's initial parameters, batched FSA follows Julia's training loss
  closely: FSA/Julia = 1.0001, 0.9999 and 0.997 at epochs 500, 2,000 and 5,000. The 0.94×
  is the late median only; single late epochs are spiky. Its validation late median
  stays 1.19× Julia.
- **Julia's step control includes derivatives (source fact).** In the authors' Manifest
  versions, OrdinaryDiffEq's default error norm for ForwardDiff dual numbers includes the
  partials: DiffEqBase 7.21.2 `ext/DiffEqBaseForwardDiffExt.jl:242-249`
  (`ODE_DEFAULT_NORM = sqrt(sse(u))`) and SciMLBase 3.55.0
  `ext/SciMLBaseForwardDiffExt.jl:458` (`sse(Dual) = sse(value) + sum(sse, partials)`).
  Our FSA solve likewise controls the error of state and sensitivities; direct autograd
  controls the state only. That this explains why FSA tracks Julia is a **hypothesis**.
- **Our canonical data.** FSA and direct autograd end within ~2 % of each other. Both
  are far below legacy B0 on train (0.07×), so the released-example settings help on our
  problem, but which setting does it is not isolated.
- **Reaction 2/3 rate constants (source fact).** Released script lines 61-62 pair
  (ln A, Ea) = (19.13, 14.42) with reaction 2 (DG + ROH) and (7.93, 6.47) with reaction 3
  (MG + ROH). Paper Sec. II D 1 lists `Ea = [14.54, 6.47, 14.42]`,
  `ln(A) = [18.60, 7.93, 19.13]`, which our `generate_biodiesel.py` follows. The author
  data and our canonical data are therefore different chemical systems, and the
  author-data and canonical-data rows above are not comparable with each other.
- **Reaction-order test** (`results/experiments/biodiesel/author_repo_match/reaction_order/`).
  Two datasets built with the same recipe (`build_reaction_order_datasets.py`), trained with
  the same settings and seed (batched, direct autograd, Glorot seed 0); only the kinetics
  differ (code order vs paper order).
  - Late train paper/code = 0.83×, val = 1.14× (`code_order_seed0`, `paper_order_seed0`
    `metrics.json`): no large effect on the loss. One seed per order, and each dataset is
    normalized with its own ranges.
  - Fig. 3 case (`fig3_case_trajectories.json`, `.png`): the code-order ground truth
    matches the published Fig. 3 curves (e.g., DG ends near 0.47, MG near 0.17); paper order
    gives DG 0.29 and MG 0.32, above the 0-0.2 MG axis (visual comparison with the paper,
    not computed).
  - `code_order_seed0` vs `run_batched_seed0` (same seed and settings; recipe data vs
    Julia's exported arrays): late train 0.984×, val 0.985×. The recipe reproduces the
    Julia-data result, so the paper-order dataset built with the same recipe is a fair
    comparison.

### Released code vs paper vs email

Status: `code = email`; `code ≠ email`; `email silent, code ≠ paper`;
`email silent, paper silent`;
`email only (not in released code)`.

| item | released code (script line) | paper (section / eq.) | email | ours (canonical) | status |
|---|---|---|---|---|---|
| data normalization | min-max over all 40 trajectories incl. test and the duplicated test block; T over all ICs (132-139) | Eq. 18 text: min-max to [0, 1] | G2 (split not stated) | min-max, train-only | code = email (method) |
| layer input normalizer | layer 1 none, layer 2 `tanh_fast` (120-124) | tanh at each layer input (Sec. II C 1, after Eq. 12) | silent | tanh at every layer | email silent, code ≠ paper |
| base activation | off, 156 parameters (122, 124) | Swish base term in Eq. 11 | G3, B1 | off, 156 | code = email |
| multiplication nodes | layer 2 `mult_flag = true`, 2 of 4 (124) | n_mu = 2 (Sec. III A 1) | B4 | n_mu = 2 | code = email |
| noise | additive N(0,1) × species max × noise %, clipped at 0 (98, 100); the t = 0 observation is noised while the ODE initial condition stays clean (email silent on this) | not specified | B5 | multiplicative, clean t = 0 | code = email |
| loss scaling | `Flux.mse`, ÷ (species × times) (166) | Eq. 18: sum over time, mean over species (Sec. II C 4) | H8 | Eq. 18 | code ≠ email (factor 30, `explicit_mse_loss_check.txt`) |
| epochs | 10,000 (22) | 1e4 (Sec. III A 1) | B6 | 10,000 | code = email |
| integrated state | normalized coordinates (134-139, 147) | Eq. 13 on the thermochemical state; min-max only in the loss (Eq. 18) | silent | physical | email silent, code ≠ paper |
| time scaling | RHS ÷ 50, comment "50s" (146-147); tspan is 30 s (82) | none | silent | none | email silent, code ≠ paper |
| learning rate | Adam 1e-2 (25) | 2e-3 (Sec. II C 5) | silent | 2e-3 | email silent, code ≠ paper |
| RBF width | `exp(-((x-c)/h)^2)`, h = grid spacing (`src/utils.jl:8-13`) | Eq. 12: `exp(-r^2 / 2h^2)` | silent | Eq. 12 | email silent, code ≠ paper |
| initialization | Glorot uniform (`src/kdense_mult_add.jl:36`) | not stated | silent | `randn × 0.1` | email silent, paper silent |
| solver | `AutoTsit5(Rosenbrock23)`, rtol 1e-2, effective atol 1e-6 (35-37, 158) | Tsit5 (Sec. II C 5) | silent | Tsit5, rtol 1e-6 / atol 1e-8 | email silent, code ≠ paper |
| gradients | `ForwardDiff.gradient` through the solver (276) | forward sensitivity analysis (Sec. II C 5) | silent | direct autograd (FSA optional) | email silent, code ≠ paper |
| reaction 2/3 rates | swapped vs paper (61-62) | Sec. II D 1 | silent | paper order | email silent, code ≠ paper |
| DeepONet | none | 308 parameters (Sec. III A 3) | B3 | 340, 7-D input | email only (not in released code) |
| smallest scaling model | none | 72 and 78 (both in Sec. III A 2); 78 correct per email B2 | B2 | 78 | email only (not in released code) |

### Open items

- Which reaction order produced the published figures: the evidence points to the code
  order (Fig. 3 case, reaction-order test above); not confirmed by the authors.
- Whether the derivative-aware error control explains why FSA tracks Julia (hypothesis).
- Seed spread: one run per configuration.
- Julia on our canonical data: not run.
- Hydrogen items H1-H9: deferred to the hydrogen work.
- ~~Which loss convention the paper's reported MSE values use.~~ Answered in §4: the
  time-averaged convention (released-code `Flux.mse`, Eq. 18 / 30).
- DeepONet trunk on `biodiesel_v2`: the trained 308 trunks are exactly linear in time (§4);
  fix not yet chosen, Fig. 4/5 DeepONet runs not yet rerun.

## 3. Figure 4 sizes (neural scaling)

Fig. 4 is trained on noise-free data: ChemKAN for 5,000 epochs and DeepONet for 50,000
epochs (paper Sec. III A 2). New runs use `biodiesel_v2.npz`.

**DeepONet.** Branch [3, w, w, p] on (TG0, ROH0, T0), trunk [1, q, p], head p → 6, all
layers biased: count = w·w + w·p + q·p + 5w + 2q + 8p + 6. The sizes are explicit
architectures (`deeponet/biodiesel_deeponet.py` `FIG4_ARCHITECTURES`, each count asserted;
`train_biodiesel_deeponet.py --width w --trunk-hidden q`, with w, q, p recorded in
`config.json`):

| parameters | (w, q, p) | status |
|---|---|---|
| 78 | (3, 3, 3) | reconstructed |
| 156 | (5, 5, 5) | reconstructed |
| 249 | (7, 6, 7) | reconstructed |
| **308** | **(8, 7, 8)** | **confirmed** (email B3); the model used for Fig. 5 (Sec. III A 3) |
| 384 | (9, 9, 9) | reconstructed |
| **456** | (10, 10, 10) | size **confirmed** as the largest DeepONet (Sec. III A 2); architecture reconstructed |

Confirmed by the paper: exactly two DeepONets are larger than 308 (Sec. III A 2: "the two
largest DeepONets", "the last two testing points"). Not stated in the paper, so our choice:
the sizes other than 308 and 456, and every architecture except 308.

**ChemKAN.** 7 → H → 6 core (`train_author_repo_match.py --hidden H`), n_mu = ceil(H/2),
grid 3, 39·H parameters (asserted; H recorded in `config.json`):

| parameters | H | status |
|---|---|---|
| 78 | 2 | confirmed: width 2, grid 3 (email B2) |
| 117 | 3 | reconstructed |
| **156** | **4** | **confirmed**: the main model (paper) |
| 351 | 9 | reconstructed |
| 429 | 11 | reconstructed |

The default `--hidden 4` reproduces `run_seed0` exactly.

## 4. Figure reproduction on biodiesel_v2

The 27 runs of `chemkan/scripts/author_repo_match/run_figures.py`, all on
`biodiesel_v2.npz` with seed 0; results in
`results/experiments/biodiesel/author_repo_match/figures/`.

- **ChemKAN**: `train_author_repo_match.py --data canonical --solve-mode batched
  --sensitivity fsa` (released-example settings, §2), Fig. 5 at H = 4 (156 parameters)
  for 10,000 epochs, Fig. 4 at H = 2, 3, 4, 9, 11 for 5,000 epochs.
- **DeepONet**: `reference_final_trunk_relu`, Adam lr 1e-3; Fig. 5 at 308 parameters for
  10,000 epochs, Fig. 4 at the §3 sizes for 50,000 epochs.

Verification (`chemkan/scripts/figures/author_runs.py` `verify()`): 27 runs, 0 problems
(final checkpoint, full epoch count, finite losses, data sha256, settings as planned).

Figures: `python chemkan/scripts/figures/plot_all.py --runs figures` writes PNG + PDF to
`figures/plots/`; the legacy figures are unchanged (`--runs legacy`, the default). Notebook:
`chemkan/notebooks/07b_biodiesel_author_matched.ipynb`.

Conventions:
- **Loss**: time-averaged MSE (released-code `Flux.mse`; Eq. 18 = 30 × this).
- **Converged**: the final checkpoint (after the last Adam step), as the paper's
  "after 10⁴ epochs"; not a late-window median.
- **Fig. 4 fits**: log-log least squares, excluding the last ChemKAN point and the last
  two DeepONet points (Sec. III A 2); order = −slope.

### Paper comparison

Only numbers stated in the paper's text (`chemkan/scripts/figures/author_paper_table.py`).
One seed per run, so there is no spread; read every ratio as a single-run value.

| quantity | paper (section) | ours |
|---|---|---|
| ChemKAN scaling order, train / test | 1.0 / 0.6 (III A 2) | 1.51 / 0.464 (R² 0.91 / 0.56, 4 points) |
| ChemKAN error at the smallest size (78 parameters) | ~1e-4 (III A 2) | train 5.28e-4, test 8.87e-4 |
| DeepONet train below ChemKAN above ~200 parameters | yes (III A 2) | no: DeepONet 249-456: 4.2e-4 to 5.2e-3; ChemKAN 351, 429: 4.6e-5, 3.3e-5 |
| ChemKAN train increase, 0 → 1 % noise | 3.78e-5 (III A 3) | 4.73e-5 (noise floor 4.46e-5; Eq. 18: 1.42e-3) |
| ChemKAN train increase, 0 → 5 % noise | 9.45e-4 expected, 9.64e-4 observed (III A 3) | 1.04e-3 (noise floor 1.08e-3; Eq. 18: 3.12e-2) |
| ChemKAN noise-free test, 15 % / 0 % | ~2× (III A 3) | 1.82× |
| DeepONet noise-free test, 15 % / 0 % | ~5× (III A 3) | 0.75× |
| noise-free test at 15 %, DeepONet / ChemKAN | 4.4× (III A 3) | 7.67× |

The noise floor is the MSE between the noisy and clean training targets: the training-loss
increase a model that fits the clean trajectories exactly would show.

### Findings

- **Loss convention.** The paper's 0 → 1 % and 0 → 5 % training-loss increases are of the
  size of our time-averaged values and the noise floor, and 30× below the Eq. 18 values.
  The paper's reported MSEs are therefore in the time-averaged convention (Eq. 18 / 30),
  not Eq. 18 as written.
- **ChemKAN** follows the noise floor (training loss rises by about the noise it cannot
  fit) and its noise-free test loss stays within 2× of the 0 % value up to 15 % noise, as in
  the paper. Its scaling orders differ from the paper's (1.51 / 0.46 vs 1.0 / 0.6), from
  four points of one seed.
- **Final checkpoint vs last logged epoch.** For ChemKAN the last Adam step (lr 1e-2)
  changes the loss by ~10 % (0 % noise-free test: 2.87e-4 logged vs 3.15e-4 final), so
  single-epoch ratios are sensitive to the step at which they are read.
- **DeepONet does not reproduce.** The 308 model plateaus near 5e-3 at every noise level,
  and its Fig. 6 predictions are straight lines in time. Cause (verified on the trained
  308 runs): the trunk input is τ = t / t_end ∈ [0, 1] (our choice, commit 2825162) and the
  DeepXDE-style initialization has zero biases, so every first-layer ReLU kink sits at
  τ = 0 and the trunk is exactly linear on [0, 1]. This is separate from the final-trunk-
  ReLU fix (e2321b4). Candidate fixes: centre τ to [−1, 1], a tanh trunk, or nonzero bias
  initialization; the DeepONet rows above are therefore not a comparison with the paper's
  DeepONet.

