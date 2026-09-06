# chemkan-tuwien

## About

Welcome to the `chemkan-tuwien` project! This repository contains the source code, materials, and documentation for interdisciplinary project at TU Wien.

**Current status:** data generation **and** the model/training stack are implemented under
`chemkan/` — KAN layers, `KineticCore` + thermodynamic superstructure, biodiesel and
two-stage hydrogen training/evaluation, Tsit5 integration via `torchdiffeq`, and a
direct-autograd sensitivity path (Forward Sensitivity Analysis is still a TODO). See
[`chemkan/README.md`](chemkan/README.md) and [`chemkan/code-overview.md`](chemkan/code-overview.md).

**Reproduction runs:** the step-by-step workflow (data generation → dense H₂ temperature
cache → training → evaluation → notebooks → figures/tables) is documented in
[`docs/reproduction_workflow.md`](docs/reproduction_workflow.md). Trained runs and their
artifacts follow the layout in [`results/reproduction/README.md`](results/reproduction/README.md)
(one directory per run: `checkpoint_final.pt`, `config.json`, `run.log`, `history*.csv`,
`metrics.json`, `predictions/`). All current runs use `sensitivity = direct_autograd`.

## Reproduction status

Nothing below is claimed as a completed paper reproduction unless it says so. No single
cause is asserted where the artifacts do not establish one.

**"Evaluation completed" is not "paper result matched."** Figures 3-8 and Table I have all
now been *evaluated* end to end from committed artifacts. Most of the paper's reported
*numbers* are still **not** reproduced. The row-by-row verdict, with an artifact path for
every claim, is
[`results/reproduction/tables/reproduction_comparison.csv`](results/reproduction/tables/reproduction_comparison.csv):
across its 20 compared results: **1 matched** (Table I's network / parameter / species
counts), **1 partially matched**, 1 partially matched for the labelled hydrogen
initialization comparison only, 2 qualitatively similar but not quantitatively matched,
1 consistent but weakly discriminating, and **14 evaluated and not matched**. Forward
Sensitivity Analysis (FSA) remains unimplemented and is a separate later task.

### Hydrogen — reproduction NOT complete

Full evidence: [`chemkan/notebooks/09_hydrogen_thermo_failure_analysis.ipynb`](chemkan/notebooks/09_hydrogen_thermo_failure_analysis.ipynb)
(§23 separates established from supported from not-established).

**Architecture labelling.** The paper's 344 parameters are matched by two inferred
readings, so results are labelled explicitly rather than by default. **N=5/base-OFF** is
the historical reading and produced §1–§21 — every thermo-init diagnostic
(`hydrogen/diagnostics/thermo_init_*`, `hydrogen/diagnostics/stage1_seed0`) and
`hydrogen/main/base_off_direct_autograd_seed0`. **N=4/base-ON** is the current script
default (since 2026-09-03) and produced only the §22 ablation
(`hydrogen/diagnostics/base_on_n4/*`). No stored result was re-run or relabelled when the
default changed. The findings below are **N=5/base-OFF** except where stated.

- **Stage 2 does not reproduce ignition** *(N=5/base-OFF)*. The default run stays within ~10 K of `T₀` and
  never raises the ignition flag, at 500 and at 10 000 epochs.
- **The failed model has a severe thermodynamic `dT/dt` deficit.** Driven with *reference*
  species rates, the *learned* `θ_thermo` still under-predicts peak `dT/dt` by ~134×/~145×.
  The deficit sits in the Eq. 14 **Linear** branch; `KAN_cor` contributes ~0.1 % of the
  Eq. 15 path at the heat-release peak.
- **Thermodynamic Linear initialization/conditioning is highly sensitive.** Three
  *equal-norm* random directions end ~26× apart in test trajectory MSE after 10 000
  epochs; scale and direction both matter, and the effect is not a short-budget transient.
- **Fixed-temperature testing shows weak low-temperature kinetic gating already after
  Stage 1.** At a frozen 950 K, isothermal Cantera is inert while the Stage-1 core already
  reacts ~6×10⁴ too strongly — Stage 2 roughly doubles this but does not create it.
- **The N=4 / base-ON interpretation does not remove the failure.** Matched at 344
  parameters, default-random test MSE is 3.16 (N=5/base-OFF) vs 3.15 (N=4/base-ON).
- **Figs. 7, 8A, 8B and Table I are now evaluated from the saved checkpoints** (no hydrogen
  retraining). The primary run `H0` (N=4/base-ON, random thermo init) ignites in **0 of the
  30** reference-igniting conditions; all 441 generalization conditions were evaluated with
  0 integration failures, median trajectory MSE 2.66. `H0` is retained as the primary
  result including its failure.
- **One separately labelled initialization (`Hnorm1`) does recover ignition** — 30/30
  reference-igniting conditions, median relative delay error 28.9 %, 441-grid median MSE
  0.254 — while also igniting in 2 conditions where the reference does not. This is **one
  initialization, not a seed study**, and it does not replace `H0` or establish a cause.
- **Table I counts match; the speed-up does not.** 1 network, 344 measured parameters,
  9 species + T, as reported. A local PyTorch-vs-Cantera benchmark measures **0.16x** (`H0`)
  and **0.50x** (`Hnorm1`) — i.e. slower than Cantera, not the paper's 2.0x against
  Arrhenius.jl. Different reference implementation, hardware and timing scope, so the two
  are reported side by side and not merged.
- **FSA remains a major paper-explicit missing method.** Forward Sensitivity Analysis is
  not implemented; all runs use direct autograd. No result here speaks to whether FSA
  would change the outcome.

These are **co-existing findings, not a ranked causal chain.** The paper leaves the grid
size `N`, the `θ_thermo` initialization, and any derivative scaling inside Eq. 14
unstated, so several explanations remain simultaneously open.

### Biodiesel — partially reproduced

Full evidence: [`chemkan/notebooks/07_biodiesel_reproduction.ipynb`](chemkan/notebooks/07_biodiesel_reproduction.ipynb).

- The **main ChemKAN implementation exists** and trains: the paper's exact 156-parameter
  architecture reconstructs the trajectories from sparse data.
- **Figs. 3, 4, 5A, 5B and 6 are evaluated.** 26 runs in total: 8 ChemKAN noise levels
  (`B0` reused at 0 %), 8 DeepONet noise levels, 4 new ChemKAN scaling widths, 6 DeepONet
  scaling widths, and one clean replay. All completed runs are preserved; nothing was
  overwritten or retrained.
- **The clean replay reproduces `B0` bitwise** — identical weights and identical training
  loss at all 10,000 epochs — adding only the per-epoch clean-test columns `B0` lacks. It
  supplies Fig. 5B's 0 % panel; `B0` remains the established 0 % result for Figs. 3 and 5A.
- **The DeepONet baseline is implemented and run** (`deeponet/biodiesel_deeponet.py` plus a
  trainer and evaluator). At the paper-described widths it measures **340** parameters
  against the **308** reported — documented, not engineered away. Its input/output scaling
  is a labelled reproduction choice.
- **The reported noise and scaling trends are not reproduced.** ChemKAN's clean-test error
  does not grow with noise (0.77x from 0 to 15 %, versus a reported ~2x), and the Fig. 4
  regressions give -0.41 / -0.38 (ChemKAN) and -0.95 / -0.62 (DeepONet) against reported
  -1.0 / -0.6 and -4.0 / -1.4. These are **descriptive all-point fits with R² <= 0.46**,
  over a different fitting scope than the paper's pre-saturation subset.
- **Two measurement limits are documented rather than worked around.** ChemKAN's
  late-training loss oscillates by 0.10-0.19 over the last 20 % of epochs (DeepONet:
  0.0003-0.0004), which exceeds the increments the paper reports for 0->1 % and 0->5 %
  noise and makes our fixed-checkpoint increments come out negative; **the cause of that
  oscillation is not established.** And no run meets our overfitting criterion, whose
  thresholds and smoothing window are our own diagnostic choices, not the paper's.
- Figure 4's widths were fixed by **digitizing the paper's own figure** to 0.11 % in
  parameter count (`docs/fig4_width_matrix.md`). Four of its five ChemKAN markers land
  exactly on `39h`; the fifth, at ~650 parameters, matches no feasible width and is
  preserved as an unexplained discrepancy.
- The absolute MSE gap to the paper's reported magnitude is **unexplained**, and its size
  depends on a time reduction the paper does not state: our literal Eq. 18 values are
  train 0.062 / test 0.081, the conventional time-averaged equivalents 2.07×10⁻³ /
  2.71×10⁻³. Both are reported in notebook 07; the paper comparison is **approximate**.

### ChemNODE

Published ChemNODE results are used as the reference baseline, following supervisor
guidance. No ChemNODE model is trained in this repository.

### Course Information
* **Course:** [194.147 Interdisciplinary Project in Data Science](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?dswid=6763&dsrid=17&semester=2026S&courseNr=194147)

### Core Reference
This project builds upon the concepts of ChemKANs (Chemistry Kolmogorov-Arnold Networks).
* **Reference:** [ChemKANs for Combustion Chemistry Modeling and Acceleration (arXiv)](https://arxiv.org/pdf/2504.12580)

**Citation:**
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