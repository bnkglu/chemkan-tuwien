# ChemKAN reproduction prompts

## 1. Scope, execution order and saved results

Continue the existing figure-by-figure reproduction plan in the current repository. Apply the prompts below to the existing training, evaluation and notebook infrastructure, preserving valid analysis and completed results.

Complete biodiesel Figures 3–6 first, then hydrogen Figures 7–8 and Table I using the saved hydrogen models.

**Sensitivity method:** the ChemKAN paper uses FSA (Sec. II C 5, p. 8), while the present runs use `direct_autograd`. Keep the current backend for this phase and record it in run metadata. FSA implementation, validation and comparison are a separate later task; they are not prerequisites for the present figures.

Keep the supervisor-accepted biodiesel dataset and `np.linspace(0, 30, 30)` unchanged: 30 observations including both endpoints. Keep the existing split, loss sampling and train-only normalizers.

### Exact saved-checkpoint references

Paths below are relative to the repository root in `chemkan-tuwien_with_results.zip`.

```text
B0 = results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0/checkpoint_final.pt
Bbase = results/reproduction/chemkan/biodiesel/sensitivity/base_on_matched_seed0/checkpoint_final.pt
H1 = results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/stage1_seed0/checkpoint_stage1.pt
H0 = results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/random_stage2_10000_seed0/checkpoint_final.pt
Hnorm1 = results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4/normmatched_dir1_stage2_10000/checkpoint_final.pt
```

| Saved result | Architecture and training | Use |
| --- | --- | --- |
| `B0` | h=4, N=3, n_mu=2, base OFF; 156 parameters; 10,000 epochs; 0% noise | **The established 0% result**: Fig. 3's 0% column and Fig. 5A's 0% ChemKAN point. Its `history.csv` has training loss only, so it cannot supply Fig. 5B's clean-test curve |
| `Bbase` | h=4, N=2, n_mu=2, base ON; 156 parameters; 10,000 epochs | Retain the existing biodiesel architecture comparison; it does not replace `B0` |
| `H1` | h=3, N=4, n_mu=3, base ON; completed 10,000-epoch Stage 1 | Preserve shared Stage-1 provenance; no Stage-1 retraining for these evaluations |
| `H0` | N=4/base-ON/default initialization; 344 parameters; completed 10,000-epoch Stage 2 | Primary current hydrogen reproduction attempt for Figs. 7, 8A and 8B and the local Table-I row |
| `Hnorm1` | Same architecture and Stage-1 source; norm-matched random direction 1; completed 10,000-epoch Stage 2 | Separately labeled initialization comparison using the same evaluations |

For `H0`, reuse the completed approximately 8.76-hour Stage-2 run. Evaluate its actual temperature, species and ignition behavior, including failures.

Retain the other saved initialization runs under `hydrogen/diagnostics/base_on_n4/` in the diagnostic comparison. Keep historical N=5/base-OFF results under their existing labels. Use their saved evaluations and 500-epoch snapshots where relevant; do not create new runs merely to reorganize results.

### Reuse and missing work

| Paper result | Reusable result | Remaining work |
| --- | --- | --- |
| Fig. 3, 0% | `B0` | Evaluate the exact published condition; no training |
| Fig. 3, 5/10/15% | The corresponding Fig. 5 ChemKAN runs | Train each missing noise condition once, then evaluate |
| Fig. 4 | Existing dataset and implementations | Separate width sweeps: 5,000 ChemKAN / 50,000 DeepONet epochs; the archive has no suitable 5,000-epoch `B0` snapshot |
| Fig. 5A, 0% ChemKAN | `B0` | Evaluate the saved final checkpoint with all required metrics |
| Fig. 5A, other ChemKAN points | Existing deterministic noisy data where present | Seven missing runs: 1, 2, 3, 5, 7, 10, 15% |
| Fig. 5A, DeepONet | Existing implementation/reference example | Eight missing runs: 0, 1, 2, 3, 5, 7, 10, 15% |
| Fig. 5B | Fig. 5 histories at 2, 7, 15% | Log clean-test history in new runs; the 0% panel comes entirely from the separately labeled clean replay described below |
| Fig. 6 | Fig. 5's 15% model pair | Evaluate and plot one documented training case |
| Figs. 7, 8A, 8B | `H0`; separately `Hnorm1`; the present `hydrogen_fine.npz` | Evaluations and plots |
| Table I | Saved architecture/configurations | Parameter/species counts and a separate local inference benchmark |

The existing `history.csv` beside `B0` has training losses but no historical clean-test losses. A final checkpoint can provide final test metrics, not earlier test curves. To complete the 0% Fig. 5B panel, include one separately labeled 10,000-epoch clean ChemKAN replay with per-epoch clean-test logging unless compatible intermediate checkpoints/history are found.

**Scope of the replay.** The replay exists for one purpose: to supply the missing history of Fig. 5B's 0% panel. Both curves in that panel — noisy (here clean) training MSE and noise-free test MSE — must come from the replay run, so the two plotted curves belong to one parameter trajectory; never splice a replay test curve onto `B0`'s training curve or vice versa. Everywhere else `B0` remains the established 0% result and is **not** replaced: Fig. 3's 0% column, Fig. 5A's 0% ChemKAN point and any linked 0% metric are reported from `B0`. Report the replay's final metrics beside `B0`'s as a reproducibility comparison, and record both run directories, but do not promote the replay to the reference result.

### Run matrix before training

Inspect the actual checkout and these saved configurations before launching training. Produce one consolidated run matrix with:

- figure/result served;
- status: already complete/reuse, evaluation only, or new training required;
- exact source checkpoint, where applicable;
- epochs, architecture and actual parameter count;
- noise level, seed and sensitivity backend;
- exact expected output directory.

Show the clean-history replay separately from the seven missing noisy ChemKAN runs, and mark it Fig.-5B-history-only so it is not mistaken for the 0% Fig. 3 / 5A model. Select and audit the Fig. 4 widths before including their exact rows. Do not count reused runs twice.

**Wait for approval of this run matrix before launching the training sweep.** Preparation, result inspection and missing-artifact identification can proceed first. Once approved, use the approved matrix throughout the figure prompts without asking for the same approval again.

Load saved models with their original architecture, normalizers and configuration. Generate missing evaluations from compatible checkpoints. Train only when the required trained model or historical evidence is absent.

Source references for the figure prompts are the supplied ChemKAN paper: Sec. II C 4–5 (loss/training), Sec. II D (data), Sec. III A / Figs. 3–6 (biodiesel), and Sec. III B / Figs. 7–8 / Table I (hydrogen). Keep paper statements, reconstruction choices and measured results distinguishable.

---

## 2. Biodiesel common setup before Figs. 3–6

The paper gives:

- six concentration states: TG, ROH, DG, MG, GL, R'CO₂R;
- T as an additional input;
- `T\sim U(323,343)` K;
- initial TG and ROH independently uniform on [0.5, 2];
- all four remaining concentrations initially zero;
- 20 training trajectories;
- 10 testing trajectories;
- 30-s window;
- 30 observations;
- kinetic core only.

For the base ChemKAN:

$$
7\rightarrow h\rightarrow6,\quad N=3,\quad \text{base OFF}.
$$

At `h=4`,

$$
P=(7\cdot4+4\cdot6)\cdot3 =52\cdot3 =156.
$$

Use the fixed `linspace(0,30,30)` dataset specified in Section 1 throughout both model pipelines.

Preserve the existing loss reduction and store both:

- implemented Eq. 18 reduction: average over species, sum over evaluated time points, then average over trajectories;
- conventional time-averaged MSE = Eq.18 / `N_t`, with `N_t=30` for this unchanged biodiesel objective.

Keep the same convention across model and noise comparisons. Report the choice beside comparisons with paper values.

---

## 3. Figure 3 — ChemKAN trajectory reconstruction with noise

The paper explicitly says the biodiesel ChemKAN has hidden=4, `n_mu=2`, three-point grids, and is trained for `10^4` epochs. The actual figure has columns **0%, 5%, 10%, 15%**.

The plotted unseen condition is also explicit:

$$
[TG]_0=1.94,\quad [ROH]_0=1.43,\quad DG=MG=GL=R'CO_2R=0,\quad T=334.8\text{ K}.
$$

One important point: this condition should be generated explicitly. Don't just take “test trajectory 0” from your random 10-test split and call it Fig. 3 unless it happens to have exactly those values.

### Prompt — Figure 3

```text
Reproduce ChemKAN paper Figure 3 using the shared Figure-5 runs and Section-1 reuse map.

Before changing anything, inspect the current repository and reuse the existing
biodiesel dataset, normalization convention, training implementation, and run-directory
infrastructure. Do not refactor unrelated code.

SETUP
-----
System:
- biodiesel transesterification
- six concentration states:
  TG, ROH, DG, MG, GL, R'CO2R
- temperature is isothermal and provided as an input
- kinetic ChemKAN core only; no thermodynamic superstructure

Dataset:
- 20 training cases
- 10 testing cases
- T sampled from 323–343 K
- TG0 and ROH0 sampled uniformly from 0.5–2
- DG0 = MG0 = GL0 = R'CO2R0 = 0
- use the repository's already-adopted canonical 30-point / 30-s dataset convention
  without regenerating or changing it

ChemKAN:
- AddKAN -> LeanKAN
- hidden_dim = 4
- n_mu = 2
- num_basis = 3
- use_base_act = False (the agreed parameter-count reconstruction)
- assert actual trainable parameter count == 156
- MSE only; no PINN
- Adam
- learning rate = 2e-3
- Tsit5
- sensitivity_backend = direct_autograd

Training:
- 10,000 epochs
- noise levels for Fig. 3:
    0%
    5%
    10%
    15%
- one model per noise level
- reuse B0 for 0% and share the missing 5/10/15% runs with Figure 5
- B0 is the 0% model for this figure; the Section-1 clean replay supplies Fig. 5B history
  only and is never substituted here
- launch only the missing training listed in the approved run matrix

IMPORTANT:
The paper does not provide the exact random noise realization.
Use the repository's deterministic stored noise convention and fixed seed, and mark
the noise realization as a reproduction choice.
Do not claim bitwise identity to the paper.

FIGURE-3 EVALUATION CONDITION
-----------------------------
Generate a dedicated exact evaluation condition without changing the canonical split:
TG0       = 1.94
ROH0      = 1.43
DG0       = 0
MG0       = 0
GL0       = 0
R'CO2R0   = 0
T         = 334.8 K

Do not replace this with an arbitrary test-split trajectory.

Generate and persist:
- clean ODE ground truth for this exact condition
- sparse plotted observations
- deterministic noisy observations for 0/5/10/15%
- dense plotting-time ground truth

For each noise column plot all six species:
- noisy/sparse test observations
- clean ground truth
- dense ChemKAN prediction

The ChemKAN prediction must be produced from the initial condition by ODE integration;
do not use target intermediate states.

Save:
- checkpoint/config/history/metrics for every training run
- exact Figure-3 evaluation-condition artifact
- machine-readable predictions
- vector PDF figure
- optionally PNG for convenience

Clearly label:
PAPER EXPLICIT
REPRODUCTION INFERENCE
REPRODUCTION CHOICE

Do not introduce an LR schedule, best-checkpoint selection, or other optimization
changes not stated in the paper.
```

The 0/5/10/15% ChemKAN models are shared with Fig. 5; this figure adds evaluation work only once those checkpoints exist.

---

## 4. Figure 4 — neural scaling

The paper says explicitly:

> different parameter sizes were investigated by changing the number of nodes in the hidden layers.

So:

### ChemKAN Fig. 4

Keep:

- `N = 3`
- base OFF
- same two-layer AddKAN→LeanKAN
- 0% noise
- same data
- same loss
- same optimizer

Change:

- **hidden\_dim** **`h`**.

With the chosen base-OFF interpretation:

$$
P(h)=3(7h+6h)=39h.
$$

Feasible parameter counts include:

| Hidden width `h` | ChemKAN parameters |
| --- | --- |
| 2                 | 78  |
| 3                 | 117 |
| 4                 | 156 |
| 5                 | 195 |
| 6                 | 234 |
| 8                 | 312 |
| 10                | 390 |
| 12                | 468 |
| 18                | 702 |

There is even an internal paper oddity here: one sentence says roughly **72 parameters**, while elsewhere the sparse model is described as **78**. Under the paper-stated `N=3`, 7 inputs, 6 outputs, and base-off implementation, **78 is the structurally consistent value**.

### What about `n_mu` when h changes?

The paper gives `n_mu=2` for the h=4 main model. Keep the agreed width-sweep reconstruction:

$$
n_\mu=\lceil h/2\rceil.
$$

But this exact Fig. 4 rule is **not stated in the ChemKAN paper**, so record it as an inferred/reconstruction choice.

### Epochs

This is explicit:

- ChemKAN: **5,000 epochs**
- DeepONet: **50,000 epochs**

because ChemKAN iterations are approximately an order of magnitude slower but converge in fewer epochs.

Fit ordinary least squares in log-log space:

$$
\log L = a + \Delta \log P
$$

and report `\Delta`.

Paper targets are approximately:

- ChemKAN training: `-1.0`
- DeepONet training: `-4.0`
- ChemKAN testing: `-0.6`
- DeepONet testing: `-1.4`

The paper excludes saturated points from the fitted line.

### Prompt — Figure 4

```text
Reproduce ChemKAN paper Figure 4: noise-free neural scaling of ChemKAN and DeepONet.

Include the architecture/count audit below in the Section-1 run matrix before
requesting its approval. Use that approval for the sweep.

COMMON SETUP
------------
Noise = 0%.
Use exactly the same canonical biodiesel training/test dataset, normalization,
loss convention, and seed policy used by the main biodiesel reproduction.

Compute and store BOTH:
1. literal printed Eq. 18 reduction
2. conventional time-averaged MSE = Eq18 / Nt

Never switch metric merely because one matches the paper plot better.

CHEMKAN SCALING
---------------
The paper explicitly says parameter size is varied by changing hidden-layer node count.

Therefore:
- keep num_basis = 3 FIXED
- keep use_base_act = False FIXED
- vary hidden_dim
- do NOT use grid size N as the Fig. 4 scaling variable
- AddKAN -> LeanKAN architecture remains unchanged

Parameter count must be measured from model.parameters(), not guessed.
For this architecture verify:
P = 39 * hidden_dim.

Use the already-selected multiplication-split rule:
    n_mu = ceil(hidden_dim / 2)
Mark it RECONSTRUCTION CHOICE: the ChemKAN paper does not specify the Fig. 4 rule.

Before fixing the width sweep:
1. render/digitize Fig. 4 at high resolution;
2. estimate the x locations of the ChemKAN markers;
3. compare these against feasible counts P=39h;
4. choose the closest systematic hidden-width family;
5. save the selected h values and actual counts in a manifest.

A reasonable candidate family to CHECK against the figure, not blindly adopt, is:
    h = [2, 3, 4, 6, 10, 18]
    P = [78,117,156,234,390,702]

Paper anchors include approximately 78 params and the 156-param main model.
Record the paper's 72-vs-78 wording inconsistency rather than changing architecture
to manufacture 72.

The saved 10,000-epoch B0 final checkpoint is not a 5,000-epoch scaling result.
Reuse a genuinely matching 5,000-epoch checkpoint if found; otherwise train the
approved scaling run, including h=4, separately.

Train each ChemKAN size for exactly:
    5,000 epochs
using:
    Adam
    lr = 2e-3
    Tsit5
    sensitivity_backend = direct_autograd

DEEPONET SCALING
----------------
Paper says DeepONet size is likewise changed by hidden-node counts but does not provide
a complete architecture table for every Fig. 4 marker.

Keep Eq. 4 structure:
    branch(u0) ⊙ trunk(t) -> output mapping

Do not force the parameter count to equal the paper's nominal values if doing so
contradicts the stated architecture.

Use the literal reconstructed DeepONet architecture family.
At the comparison architecture:
    branch [7,8,8,8]
    trunk  [1,7,8]
    Hadamard combination
    output Linear(8,6)
actual count = 340
paper reports = 308
record this discrepancy explicitly.

For a systematic width variable w, a transparent reconstructed family is:
    branch [7,w,w,w]
    trunk  [1,w-1,w]
    output Linear(w,6)

Measure actual count every time.
Do not call these exact paper architectures unless verified.

Before fixing widths:
1. digitize DeepONet Fig. 4 x-locations;
2. select a transparent width family spanning the same approximate range;
3. store actual parameter counts.

Train each DeepONet scaling model for:
    50,000 epochs

DeepONet optimizer details not specified by the ChemKAN paper:
retain the selected bundled reference-example choices:
    biased Linear layers
    ReLU
    Glorot-normal weight initialization
    Adam
    lr = 1e-3
Mark these as REFERENCE-DERIVED REPRODUCTION CHOICES, not ChemKAN-paper facts.
The selected example is deeponet/src/deeponet_dataset.py. Explicitly implement and
record activation placement, bias initialization and initialization of the six-output
head. Keep these choices fixed across sizes and noise levels.

SLOPE FITS
-----------
For training and testing separately fit:

    log10(loss) = intercept + Delta * log10(parameter_count)

Use ordinary least squares.

Do not automatically fit all points.
The paper explicitly fits only the pre-saturation/pre-overfitting regime.

Produce an explicit fit mask for:
- ChemKAN train
- ChemKAN test
- DeepONet train
- DeepONet test

Store:
- slope Delta
- intercept
- R^2
- standard error
- included/excluded parameter counts

Plot all points even when excluded from regression.

Paper reference slopes:
ChemKAN train ~ -1.0
DeepONet train ~ -4.0
ChemKAN test ~ -0.6
DeepONet test ~ -1.4

These are comparison targets only; never tune the implementation to force them.
```

---

## 5. Figure 5A — noise robustness

This is a **different experiment from Fig. 4**.

The paper fixes:

### ChemKAN

- 156 parameters
- hidden 4
- `n_mu=2`
- N=3
- kinetic only.

### DeepONet

Paper says 308 parameters, and describes:

- 7-d initial-state branch input;
- three branch layers, width 8;
- trunk: 1→7→8;
- final mapping from 8-dimensional representation to six outputs.

That literal biased-linear implementation is:

$$
208+78+54=340,
$$

not 308. So keep the literal 340 architecture and state the mismatch. The paper itself says 308.

Both models stop at **10,000 epochs**. The phrase “early stopping ... at 10,000 epochs” in the paper is basically a fixed training cutoff, not the usual validation-patience early stopping.

Use the eight noise levels visible in Fig. 5A (p. 13):

$$
0,\ 1,\ 2,\ 3,\ 5,\ 7,\ 10,\ 15\%.
$$

The missing archive level is 3%. Add it deterministically while preserving all existing clean trajectories and stored noise realizations.

Three metrics per model/noise level:

1. noisy **training** MSE;
2. noisy **testing** MSE;
3. **noise-free testing MSE** against the clean underlying trajectory, Eq. 22.

Eq. 22 is crucial; it is what tells you whether the model learned the hidden clean dynamics instead of just the noisy samples.

Paper references to compare—not force:

- ChemKAN training loss increase 0→1%: `3.78\times10^{-5}`;
- predicted quadratic 0→5% increase ≈ `9.45\times10^{-4}`;
- observed ≈ `9.64\times10^{-4}`;
- ChemKAN noise-free test error increases only about 2× at 15%;
- DeepONet about 5×;
- DeepONet's 15% noise-free test MSE ≈ 4.4× ChemKAN.

### Prompt — Figure 5A

```text
Reproduce Figure 5A using the fixed comparison architectures.

CHEMKAN:
- hidden_dim=4
- n_mu=2
- num_basis=3
- base OFF
- 156 parameters
- kinetic core only
- MSE only
- Adam lr=2e-3
- 10,000 epochs
- sensitivity_backend = direct_autograd

DEEPONET:
- literal Eq. 4 reconstruction
- branch [7,8,8,8]
- trunk [1,7,8]
- Hadamard combine
- Linear(8,6) output
- actual implementation count = 340
- paper nominal count = 308
- retain the architecture; do not manufacture 308
- train 10,000 epochs
- Adam lr=1e-3, ReLU, Glorot-normal initialization, biased Linear layers
- use the same selected reference-example conventions documented in Figure 4
- do not tune separately by noise level

NOISE LEVELS:
0, 1, 2, 3, 5, 7, 10, 15 percent

Use the already-persisted deterministic noisy arrays. Add the missing 3% arrays
without changing existing levels, clean data or the split.
Document the noise distribution, scale, seed and treatment of initial conditions.
Both models must use the same noisy data for each level.
Ensure the loader/trainer actually selects the requested noise targets; the current
clean-only loading path is insufficient for this sweep.

For each model/noise pair, evaluate the saved final checkpoint AFTER the 10,000th
optimizer update and compute:
1. training MSE against noisy training observations
2. testing MSE against noisy testing observations
3. noise-free testing MSE against the clean test trajectories (paper Eq. 22)

The two test metrics use identical predicted test trajectories. Only their targets
change. Predictions use initial conditions (and query times for DeepONet), without
intermediate test observations. At 0%, the two test targets and losses coincide.
Do not substitute the last pre-update training-history entry for final-checkpoint
evaluation. Keep the three metrics tied to the same checkpoint.

Use the same training-set-fitted normalizer for all three metrics.
Never fit normalization to testing data.

Save:
- 16 model/noise conditions in total: eight ChemKAN and eight DeepONet
- reuse B0 for the existing clean ChemKAN condition; the Section-1 clean replay is a
  Fig.-5B-history run and does not replace B0 at this point
- seven new noisy ChemKAN runs and eight new DeepONet runs are missing in this archive
- account separately for the clean-history replay in Section 1 (report its final metrics
  beside B0's as a comparison; B0 stays the plotted 0% value)
- preserve the source run directory for reused checkpoints; reference it in the manifest
- checkpoint
- config
- per-epoch history
- final three metrics
- predictions
- seed/noise seed

Reuse compatible ChemKAN 0/5/10/15 checkpoints for Figure 3.
Do not duplicate training.

Plot Figure 5A:
x = % noise
y = converged loss on log scale

For each model show:
- train
- noisy test
- noise-free test

Also output a CSV table containing every plotted value.

Compare numerically against paper reference observations, but do not alter the
implementation to force agreement.
```

---

## 6. Figure 5B — loss dynamics

The four panels are:

- 0%
- 2%
- 7%
- 15%.

No new models are necessary if your Fig. 5A histories contain the correct quantities.

What you need per epoch is principally:

- noisy training MSE;
- noise-free test MSE.

The paper reports interesting DeepONet behavior:

- 7%: noise-free test minimum around epoch \~5,000, then starts increasing;
- 15%: minimum around \~1,000, then strong overfitting;
- ChemKAN continues decreasing/plateauing.

### Prompt — Figure 5B

```text
Generate Figure 5B from the Figure 5 training runs.

Use the Figure-5A runs for the 2%, 7% and 15% panels. The existing B0 run lacks
clean-test history; a final checkpoint cannot reconstruct earlier test losses, so take
the 0% panel entirely from the separately labeled clean replay described in Section 1 --
BOTH its training curve and its clean-test curve. Do not mix B0's training history with
the replay's test history. B0 remains the established 0% result for Figures 3 and 5A;
compare the replay's final metrics against it and report the difference.

Required noise levels:
    0%
    2%
    7%
    15%

For BOTH ChemKAN and DeepONet, record through the full 10,000 epochs:
- noisy training MSE
- noise-free testing MSE

If the current training callback does not persist noise-free test MSE during training,
add the smallest evaluation-only callback necessary.

The callback must:
- run under no_grad
- never call optimizer.step()
- never modify parameters
- never change model mode in a way that affects training
- use the same fixed train-derived normalizer
- use the full test trajectory set
- persist its output to history CSV
- record actual evaluation epochs and whether values are before or after updates
- associate compared training/test losses with the same parameter state
- preserve the training random-number state so evaluation does not alter the run

The paper does not state the logging interval.
Record every epoch for the requested histories. If the evaluation overhead requires
a different interval, state it in the run matrix before training. Store the actual
evaluated epochs; do not fabricate intermediate history points.

Plot four panels:
0%, 2%, 7%, 15%

Each panel should contain:
- ChemKAN train
- ChemKAN noise-free test
- DeepONet train
- DeepONet noise-free test

Do not plot noisy-test history in Figure 5B.

x = epoch, 0–10000
y = loss, logarithmic scale

Paper qualitative reference only:
- 0%: ordinary convergence
- 2%: noisy train loss penalized, clean test remains relatively good
- 7%: DeepONet clean-test minimum around ~5000, then mild overfit
- 15%: DeepONet clean-test minimum around ~1000, then strong overfit
- ChemKAN: continued decrease or plateau rather than similar overfit

Do not post-process/smooth curves in a way that hides training oscillation.
If a display-only moving average is added, also plot/store raw values and label the
smoothing explicitly.
```

---

## 7. Figure 6 — 15% noise profiles

No new training.

The paper specifically uses the **15%-noise models from the previous analysis** and shows:

- noisy training observations;
- hidden clean ground truth;
- ChemKAN prediction;
- DeepONet prediction.

The exact training trajectory used for the figure is not given. So unlike Fig. 3, we cannot exactly identify the original initial condition.

### Prompt — Figure 6

```text
Reproduce Figure 6 WITHOUT training another model.

Reuse exactly:
- Figure-5 ChemKAN 15%-noise 10k checkpoint
- Figure-5 DeepONet 15%-noise 10k checkpoint

Select one deterministic TRAINING trajectory from the canonical 20-case training set.

Because the paper does not specify the exact plotted training case:
- make case index configurable
- choose one default case deterministically before inspecting prediction quality
- record:
    case_index
    TG0
    ROH0
    T
    all six initial concentrations
- label the selected trajectory as REPRODUCTION PLOTTING CHOICE

For each of the six species plot:
- noisy 15% training observations
- clean underlying ODE trajectory
- dense ChemKAN prediction
- dense DeepONet prediction

Do not retrain or tune either model for this plotting case.
Other training trajectories may be plotted separately in Notebook 07.
Do not average different initial conditions into the main Figure 6.

Persist the exact plotted arrays before rendering the figure.
```

---

## 8. Hydrogen setup and checkpoint selection

The paper describes the H₂/O₂ submechanism with nine species and 29 reactions, 35 training conditions, and the held-out condition `T0=1150 K, phi=1.3` (Sec. II D 2).

Keep the existing coarse dataset and its reconstruction:

```text
T0 = [950, 1000, 1050, 1100, 1150, 1200] K
phi = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
```

The paper's printed phi list omits 0.5 while stating 36 combinations. Keep phi=0.5 labeled as inferred from that count and the figure. The existing data do not need regeneration for model training.

Use the completed `H0` checkpoint as the primary current reproduction attempt:

```text
hidden_dim = 3
n_mu = 3
num_basis = 4
use_base_act = True
actual trainable parameter count = 344
thermodynamic initialization = saved default/random initialization
```

N=4/base-ON is the selected implementation interpretation; the paper does not specify hydrogen N or its initialization details. Historical N=5/base-OFF also gives 344 parameters and must keep its own label.

Use `Hnorm1` as the separately labeled initialization comparison. Keep the remaining saved initialization outcomes in the diagnostic table; do not present the selected comparison as proof of robustness across seeds or as a paper-specified initializer.

## 9. Hydrogen evaluation preparation from saved Stage-2 models

The paper describes two-stage training without prescribing 10,000 epochs for each stage. The saved 10,000+10,000 budgets are implementation choices; report actual completion and convergence from the histories.

```text
Prepare hydrogen evaluation from H0 and, separately, Hnorm1 as defined in Section 1.
Reuse H1 for provenance and any existing Stage-1 comparison. Do not retrain either
stage to generate the current figures, predictions or production-rate curves.

LOAD AND PRESERVE
-----------------
- Restore the complete saved model and its train-derived input/loss normalizers.
- Keep each run's saved architecture, solver settings and initialization metadata.
- Record the original Stage-1 checkpoint source and completed stage epoch counts.
- The saved training configuration uses Adam lr=2e-3, Tsit5,
  rtol=1e-6, atol=1e-8, and MSE + alpha_PINN*PINN with alpha_PINN=1e-4.
- Preserve the existing stage-specific PINN flags. The Stage-1 PINN choice and the
  20,000-point observed-temperature provider are documented implementation choices.

STAGE DISTINCTION
-----------------
- Stage 1 integrated species with observed temperature supplied externally.
- Stage 2 integrated the full predicted [Y,T] state and updated the whole network.
- Final inference integrates the full model from initial conditions only.

DERIVATIVE PREDICTIONS AFTER STAGE 2
-----------------------------------
No separate training is needed for dY/dt or dT/dt.
1. Integrate the saved Stage-2 model to obtain physical predicted states
   u_pred(t) = [Y_pred(t), T_pred(t)].
2. Construct/use ChemKANDynamics(model, input_normalizer=saved_input_normalizer).
3. In evaluation mode under no_grad, evaluate dyn(t, u_pred(t)) at each requested time.
4. The first nine outputs, in the saved species order, are dY/dt; the last is dT/dt.

The wrapper normalizes network inputs and returns physical derivatives. Do not
normalize the predicted states twice or rescale the returned rates as if they were
normalized-state derivatives. Use direct RHS evaluation for these rate curves.
Evaluation on reference states is a separate diagnostic, not the predicted trajectory.

OUTPUTS
-------
For each checkpoint save/reference:
- temperature and all species trajectories;
- normalized trajectory/species/temperature errors with the reduction stated;
- dY/dt and dT/dt arrays where used in the existing comparison;
- ignition behavior, including non-ignition and integration failures;
- checkpoint, configuration and original normalizer provenance.

Report state MSE separately from total MSE+PINN training loss.
Keep H0 as the baseline even if it fails. Hnorm1 has its own labeled evaluation set.
Use the same checkpoint within each set for Figures 7, 8A and 8B.
Do not gate all figure evaluation on two representative-condition accuracy checks:
evaluate the requested conditions and record outcomes condition by condition.

Keep Notebook 08's initialization comparison concise; retain the detailed failure and
initialization analysis in Notebook 09.
```

---

## 10. Figure 7 — H₂ trajectories

Use the published conditions:

Training:

$$
\Phi=0.9,\quad T_0=1050\text{ K}
$$

Held-out:

$$
\Phi=1.3,\quad T_0=1150\text{ K}.
$$

The paper emphasizes that these trajectories come **entirely from ChemKAN given only the initial condition**.

### Prompt — Figure 7

```text
Reproduce Figure 7 from H0, then produce a separately labeled Hnorm1 comparison.
Use one consistently identified checkpoint per figure set.

Do not train a model specifically for Figure 7.

LEFT = training condition:
    phi = 0.9
    T0 = 1050 K

RIGHT = original held-out test:
    phi = 1.3
    T0 = 1150 K

Predictions must be closed-loop integrations from the initial state only.
Do not inject reference temperatures or reference species at later times.

Plot:
A/D: temperature

B/E: species corresponding to the ChemNODE-overlap group:
     H2, O2, H2O, O, OH
     keep N2 in the modeled state but omit it from plot for clarity

C/F: additional low-concentration/reactive species:
     H
     HO2
     H2O2

Ground truth:
- canonical Cantera trajectories

Prediction:
- one integration of the final ChemKAN per condition

Use dense plotting times if desired, but do not confuse interpolation density with
additional training observations.

Verify the species display multipliers directly from the source paper figure before
hard-coding them.

Persist dense prediction arrays and the exact checkpoint/config provenance used.
```

---

## 11. Figure 8A — 441-condition generalization

This uses the **same H₂ model**. Do not retrain.

Paper says:

- 441 total initial conditions;
- 35 original training;
- 1 official held-out;
- 405 additional;
- therefore 406 total unseen during training.

The natural reconstruction is:

$$
21\times21=441,
$$

with

$$
T_0 = \mathrm{linspace}(950,1200,21)
$$

giving 12.5-K spacing and

$$
\Phi=\mathrm{linspace}(0.5,1.5,21)
$$

giving 0.05 spacing.

That exact fine grid is inferred from the figure/count, not tabulated explicitly.

### Prompt — Figure 8A

```text
Reproduce Figure 8A using H0 and, separately, Hnorm1, matching each Figure-7 set.

Do not retrain.

hydrogen_fine.npz IS PRESENT in the current checkout and is the file to use --
do not regenerate it. (It was missing from the supplied results archive, which is
what earlier drafts of this document recorded.) Verify before use that it holds:

T0 = linspace(950,1200,21)
phi = linspace(0.5,1.5,21)

=> 21 x 21 = 441 conditions.

Regenerate only if that verification fails or the file is absent:

    python generate_hydrogen.py --out ../../data/generated/hydrogen_fine.npz --grid fine

Mark this grid as RECONSTRUCTED from the paper's reported 441-condition count and
figure spacing unless a more explicit source is found.

Composition:
- 35 original training conditions
- 1 original held-out condition
- 405 additional conditions
- 406 total unseen during training

Use the original training/checkpoint normalizer.
DO NOT fit a new normalizer on the 441-condition grid.

For every condition:
1. integrate the SAME trained ChemKAN from its initial state
2. compare against corresponding Cantera ground truth
3. compute normalized trajectory MSE using the same documented Eq. 18 reduction

Produce 441 separate MSE values per checkpoint.
Do not average them into one scalar before plotting.
Record integration failures as failures with their condition identifiers; do not
omit them or substitute an invented finite MSE. Failed ignition alone does not prevent
trajectory-MSE evaluation for a successfully integrated trajectory.

Heatmap/scatter:
x = phi
y = T0
color = normalized trajectory MSE

Overlay:
- 35 training locations with paper-like cross markers
- original held-out point with a distinct marker

Persist:
- all 441 predictions or a traceable prediction archive
- a CSV with T0, phi, MSE, training/test/additional flags
- figure

Do not clip stored numerical MSE values merely to imitate the paper's color scale.
Plot display limits may be set separately.
```

---

## 12. Figure 8B — ignition delay

Again: same H₂ checkpoint.

Paper defines ignition delay as the point of **maximum temperature-rise rate**. It plots 30 igniting cases; the lowest-temperature cases do not ignite inside the 0.6-ms window.

Therefore:

- T = 1000, 1050, 1100, 1150, 1200 K;
- six φ values;
- `5\times6=30`.

### Prompt — Figure 8B

```text
Reproduce Figure 8B using H0 and, separately, Hnorm1, matching their Figure-7/8A sets.

Do not train another model.

Paper ignition definition:
    ignition delay = time of maximum dT/dt

Evaluate all 30 coarse-grid cases whose reference trajectories ignite within
0.6 ms. Do not filter this set by whether the model succeeds.

Temperatures:
    1000
    1050
    1100
    1150
    1200 K

phi:
    0.5
    0.7
    0.9
    1.1
    1.3
    1.5

The 950-K cases are excluded because they do not ignite during the studied window.

Do not estimate ignition delay from only 50 sparse plotting/training samples.

Reuse the dense reference-temperature cache hydrogen_temperature_20000.npz.
Use a persisted dense diagnostic grid, such as the existing 601-point grid over
0–0.6 ms, and sample both reference and model consistently on that grid.
Do not compare a sparse 50-point reference derivative with a dense prediction.

Use the same dense-temperature derivative estimator and peak procedure for both,
with the procedure and endpoint handling documented:
    ignition delay = time at argmax(dT/dt)
The direct model-RHS rate curves in Section 9 remain separately identified.

Assess ignition for each predicted condition using the documented diagnostic rule.
If a prediction does not ignite within the window, flag non-ignition and leave its
delay/error undefined instead of reporting the argmax of a nearly flat curve.
Record integration failures separately and retain every reference-igniting case.

Store:
T0
phi
reference ignition delay
ChemKAN ignition delay
absolute error
relative error
held-out flag
prediction status: ignited / no ignition in window / integration failed

Plot:
x = equivalence ratio
y = ignition delay

Paper:
- reference = pentagon-style marker
- ChemKAN = triangle-style marker
- separate curve/series for each initial temperature

Do not recompute a different reference definition after seeing the ChemKAN results.
```

---

## 13. Table I

Keep the paper's literature values separate from local measurements:

| Model | Networks | Parameters | Species modeled (temperature also modeled) | Reported speed-up |
| --- | --- | --- | --- | --- |
| ChemNODE | 7 | 637 | H₂, O₂, H₂O, N₂, O, OH | 2.3× |
| ChemKAN | 1 | 344 | H₂, O₂, H₂O, N₂, O, OH, H, HO₂, H₂O₂ | 2.0× |

The paper also states that extending ChemNODE to all ten thermochemical quantities would require 1,210 parameters. Use these as published comparisons; implementing ChemNODE is not part of this prompt.

For the local row, verify `H0`'s network, species and parameter counts from the saved model. Benchmark inference against Cantera over the same 36 coarse initial conditions and 0–0.6 ms window, using matched requested outputs and documented solver tolerances. Record hardware, dtype, solver/backend, warm-up, repetitions, timing scope and accuracy alongside the timing. Label this a **local PyTorch-vs-Cantera benchmark**; the paper's timing used Arrhenius.jl.

The saved training duration and generic evaluation wall-time field do not reproduce the paper's 2× inference speed-up. Until the benchmark is run, mark the local speed-up as **not yet evaluated**. If a model fails to reproduce the dynamics, retain that accuracy result beside its timing.

---

## 14. Experiment counts, notebooks and result coverage

### Training and reuse

The Fig. 5 noise comparison has **eight ChemKAN and eight DeepONet conditions**, all with a 10,000-epoch budget. With the current archive, reuse `B0` and plan **seven missing noisy ChemKAN runs plus eight DeepONet runs**. The clean ChemKAN replay for missing Fig. 5B history is one separately identified repeat, not another noise level and not a replacement for `B0` in Figs. 3 and 5A.

Fig. 3 reuses the ChemKAN 0/5/10/15% subset; Fig. 5B reuses the 0/2/7/15% histories; Fig. 6 reuses the 15% pair. Fig. 4 remains a separate width sweep with **5,000 ChemKAN / 50,000 DeepONet epochs** per selected width.

Hydrogen requires **no new model training for these evaluations**. Use `H0` throughout its Fig. 7/8A/8B set and `Hnorm1` throughout the separately labeled comparison set. Missing reference trajectories, predictions, rates, plots and timing measurements are evaluation tasks.

### Notebook handling

Do not broadly rewrite the existing notebooks.

- **Notebook 07:** preserve the notebook's resolved time-grid discussion in full — it records an accepted decision and must not be rewritten or shortened. Retain the other valid existing analysis and the existing biodiesel architecture comparison. Update/add Figs. 3–6 as persistent run artifacts become available. Correct stale noise counts, and remove leftover “time grid pending” wording, **only where such wording actually remains in the current checkout**; if the checkout is already consistent, change nothing.
- **Notebook 08:** replace stale primary N=5/base-OFF checkpoint references with `H0`. Add the Fig. 7, Fig. 8A, Fig. 8B and Table-I evaluations from saved models, fix the missing fine-grid reference and sparse/dense ignition comparison, and include the concise `Hnorm1` comparison.
- **Notebook 09:** retain detailed hydrogen failure/initialization diagnostics. Reference those results from 08 without duplicating the notebook.

Save traceable final checkpoints for new training, configurations, raw histories, metric tables and prediction arrays in the existing result structure. Reused checkpoints should remain in their source directories and be referenced by the evaluation manifest. Notebook plots must read the persisted artifacts. Keep existing figure-export conventions; additional separate PNGs are optional.

### Compare the reported findings, not only figure layouts

For every relevant paper result, include:

`paper result | our result | comparison/status | remaining difference`

Use statuses **matched**, **partially matched**, **not matched**, or **not yet evaluated**, supported by the actual measurements. Distinguish a completed evaluation from reproduced accuracy. Keep unavailable results visible with the specific reason.

| Paper part | Findings to report against the local results |
| --- | --- |
| Fig. 3 | The exact unseen initial condition; all six species; dense-trajectory smoothness and reconstruction under 0/5/10/15% noise |
| Fig. 4 | Actual parameter counts; fitted train/test slopes (ChemKAN −1.0/−0.6; DeepONet −4.0/−1.4); sparse-model accuracy, saturation and overfitting; explicit fit masks; the 72/78 and 308/340 count discrepancies |
| Fig. 5A | All three final losses; the reported 0→1% training-loss increase 3.78×10⁻⁵; predicted/observed 0→5% increases 9.45×10⁻⁴/9.64×10⁻⁴; approximately 2×/5× clean-test degradation for ChemKAN/DeepONet and the 4.4× model comparison at 15% |
| Fig. 5B | All four curves at 0/2/7/15%; clean-test minima and subsequent behavior, including the reported DeepONet minima near 5,000 epochs at 7% and 1,000 at 15% |
| Fig. 6 | Fit to clean underlying trajectories at 15% noise and whether the reported jagged DeepONet profiles occur |
| Fig. 7 | Temperature, major species and low-concentration species at both published conditions |
| Fig. 8A | All 441 per-condition errors; training/unseen-condition coverage; colder ignition-sensitive regions, including the paper's approximate 10⁻⁴ errors at the 1,000 K training conditions and 10⁻³ errors around 987.5 K |
| Fig. 8B | Ignition-delay agreement across all 30 reference-igniting conditions, with model non-ignition and failures explicitly retained |
| Table I | Network, parameter and species counts; published timings versus the separately measured local inference efficiency |

Do not tune widths, fit masks, metrics, initialization or seed after inspecting held-out outcomes to force the paper's reported numbers. Report the measured agreement and remaining discrepancies.
