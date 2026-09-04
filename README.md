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
- **FSA remains a major paper-explicit missing method.** Forward Sensitivity Analysis is
  not implemented; all runs use direct autograd. No result here speaks to whether FSA
  would change the outcome.

These are **co-existing findings, not a ranked causal chain.** The paper leaves the grid
size `N`, the `θ_thermo` initialization, and any derivative scaling inside Eq. 14
unstated, so several explanations remain simultaneously open.

### Biodiesel — partially reproduced

Full evidence: [`chemkan/notebooks/07_biodiesel_reproduction.ipynb`](chemkan/notebooks/07_biodiesel_reproduction.ipynb).

- The **main ChemKAN implementation exists** and trains: the paper's exact 156-parameter
  architecture reconstructs the trajectories from sparse data (Fig. 3 clean column).
- **Noise / scaling reproduction remains incomplete.** Figs. 4, 5 and 6 need a ChemKAN
  scaling sweep, a DeepONet scaling sweep, and 7+7 noise runs; none have been launched.
- **The DeepONet biodiesel implementation and runs remain a major missing component.**
  `deeponet/` is a vendored copy of the upstream reference repository plus a
  parameter-count audit; a dimension/count/forward/loss/short-training smoke test now
  exists at `deeponet/biodiesel_deeponet_smoke.py`, but no scaling or noise runs exist.
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