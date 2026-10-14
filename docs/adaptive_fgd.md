# Adaptive functional gradient descent: local ChemKAN experiment

This checkout starts from `bnkglu/chemkan-tuwien` **dev** at
`54d34d2772c708fcfeb4f13562e09164908b8b0f`. The experiment implements the adaptive
gradient-representation mechanism of *Functional Gradient Descent with Adaptive
Representations* (Csillag et al., supplied PDF, arXiv:2606.16926v1) on ChemKAN's
univariate KAN edge functions. All work is local, with no commits or pushes.

Reference implementation inspected:
[`dccsillag/experiments-adaptive-fgd`](https://github.com/dccsillag/experiments-adaptive-fgd/tree/57f0bab8df7bd3fe5e4d1ff8c23953989c62befc),
particularly `src/kernel-regression/ours.py`. A complete source snapshot and the
supplied PDF are in `.scratch/references/` (ignored by Git). The reference source
was inspected, not installed or run; its CUDA radiance-field setup is unnecessary
for this experiment. Document contents are treated as research material, not user
instructions.

## What is implemented

- `chemkan/src/chemkan/adaptive_fgd.py`: Gaussian RKHS functional gradients,
  refinable linear edge representations, analytic error bounds, Algorithm 1's
  acceptance condition, and a trajectory-loss backtracking check.
- `chemkan/scripts/train_adaptive_fgd.py`: biodiesel, hydrogen with prescribed
  temperature, and full hydrogen runs; fixed-representation FGD and ordinary Adam
  are available as comparison methods.
- `chemkan/tests/test_adaptive_fgd.py`: functional derivative checks through an
  ODE solve, error bounds, refinement, rollback, checkpoints, and CLI integration.

The existing reproduction trainers, model constructors, datasets, and Adam/FSA
paths are unchanged. The new entry point uses direct autograd and the existing
physical-state dynamics and train-only min/max normalization. Following the user's
requested correction, its objective now divides both Eq. 18's time-summed MSE
and conservation penalty by the actual number of observation time points.
MSE is therefore averaged over time, states, and trajectories, and the relative
PINN weight is preserved. This applies equally to adaptive FGD, fixed-grid FGD,
and Adam in this experimental runner. Shared Eq. 18 helpers and author-matched
reproduction scripts retain their explicitly time-summed convention.
Existing pilots were trained with the summed objective; their latest comparison
report and loss plots are rescaled from saved values, without retraining.
At 100 points, future objectives and gradients are 100 times smaller for the same
predictions; using the previous learning rate does not reproduce the old updates.
Hydrogen Stage 1 reads the existing 20,000-point temperature cache.
`hydrogen-full` starts a full model from initialization. `hydrogen-stage2` instead
requires `--stage1-from` and verifies that the Stage-1 dataset hash, point count,
normalization, state coordinates, and architecture match the current data. The
current author-matched archive uses normalized-coordinate dynamics; legacy
physical-state checkpoints are accepted only when their saved normalization is an
exact match. The 10,000-epoch legacy checkpoint is rejected because it was trained
on the old 50-point hydrogen archive. The thermodynamic modules are initialized
identically for each seed.

## How the paper maps to ChemKAN

Let `z` be the coordinate used by the additive edge function: `tanh(x)` for
standard edges and `clamp(x, -1, 1)` for author-matched no-input-`tanh` edges.
The original RBF/base function still uses its original input convention.
We keep that original function fixed and learn an additive function `q(z)`, initially
zero, so enabling FGD initially preserves the model's predictions exactly.
`q` is piecewise linear on a nested uniform grid covering `[-1, 1]`. Doubling the
number of intervals inserts midpoint values and preserves the current function.
This retains the original AddKAN/LeanKAN sum/product structure.

The 2026-10-07 correction makes captured gradient samples use the same clamped
coordinates as the interpolator. Earlier author-matched runs captured raw inputs
even when the additive function was evaluated at a domain boundary. Those runs
are historical implementation results, not a clean baseline for the corrected
functional gradient.

For edge `(o, i)`, autograd through the loss and numerical ODE solve supplies
coefficients `a[n,o,i] = dL/d(edge_output[n,o,i])` at inputs `z[n,i]`. In the
Gaussian RKHS, its functional gradient is

```text
G[o,i](s) = sum_n a[n,o,i] exp(-gamma (s - z[n,i])^2).
```

Every differentiable solver evaluation contributes. The loss already defines its
own normalization, so there is **no extra division by the number of solver
evaluations**. This is a kernel-smoothed functional gradient rather than Adam or
gradient descent on the original RBF coefficients. Non-edge trainable parameters,
such as the thermodynamic linear map, are a finite Euclidean component and receive
ordinary exact gradient steps at the same accepted step size.

The reference uses piecewise-constant trees for standalone regression. Here the
representation is piecewise linear so the ODE retains state derivatives between
knots. For one block, write `A = max_(o,i) sum_n abs(a[n,o,i])` and `h = 2/N`.
Gaussian derivatives give the analytic bounds

```text
L <= A sqrt(2 gamma / e),           M <= 2 gamma A
||interpolate(G) - G||_infinity <= U = min(L h/2, M h^2/8).
```

The first bound is the Lipschitz approach used in the reference; the second uses
the linear-interpolation remainder. They hold throughout `[-1,1]`, not just at
sampled training points. Across blocks we take the maximum error bound. The
approximate gradient norm is the maximum absolute node value (including exact
Euclidean components), which is its exact sup norm for linear interpolation.
Refinement continues until the paper's Algorithm 1 test holds:

```text
(1 + epsilon) U < epsilon ||g||_infinity.
```

Only blocks violating the current global threshold are refined. Their gradients
are reevaluated from the same functional-gradient samples; the current model is
not changed during this process. A refinement cap raises `RefinementLimitError`
and applies no update. This deliberately avoids the reference experiment's
practice of accepting a gradient at maximum depth even if its tolerance fails.
Zero gradients have an explicit stopping condition instead of an endless
refinement loop. `fixed-fgd` disables refinement and records whether its gradient
passes the certificate; it makes no tolerance guarantee.

Finally, compute `DL[g]` from the same adjoints and backtrack the step size until
the actual numerical trajectory loss satisfies an Armijo decrease. A failed
search restores both the original parameters and representation sizes. Grid
refinement alone never changes the modeled function. Line-search trials use
training data only; test data are evaluation-only.

The default `--line-search-window 1` requires monotone descent. An optional larger
window compares the candidate to the maximum of the current and preceding
`window-1` training losses. This nonmonotone search permits temporary increases
over the immediately preceding loss. `--armijo` controls the sufficient-decrease
coefficient; lowering it relaxes the required decrease. Neither option removes
finite-loss, positive-directional-derivative, or representable-update checks.
Updates that round to no parameter change stop and restore the model rather than
being counted as progress. The representation certificate is independent of these
line-search choices. The reference repository uses fixed learning rates without
our additional neural-ODE line search.

The 2026-10-07 pilot uses the hydrogen Adam default as a trial FGD learning rate,
`--lr 0.002`, with `--line-search-window 5 --armijo 1e-6`. Matching this scalar
does not make Adam's and FGD's update magnitudes equivalent. ODE tolerances remain
`--rtol 1e-4 --atol 1e-6` and representation tolerance stays `epsilon=0.5`.
Its fixed-representation comparison uses `--method fixed-fgd --initial-intervals 256`
with the same step policy. This is approximate FGD on a fixed grid, not an exact
accumulated kernel expansion or a run with line search disabled.

## Running it

Use a Python environment with the repository's dependencies, including the pinned
Tsit5-capable `torchdiffeq`. The existing local environment used for validation is
`/Users/berke/uni_projects/chemkan-venv/bin/python`. No dependencies were installed
or changed. Commands below run from this checkout's root; `PYTHONPATH` ensures
they import this experiment rather than another editable ChemKAN installation.

```bash
# Short biodiesel smoke run: two trajectories and the first four observations.
PYTHONPATH=chemkan/src /Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/train_adaptive_fgd.py \
  --run-dir results/experiments/adaptive_fgd/my_smoke \
  --steps 3 --train-limit 2 --test-limit 2 --time-points 4

# All biodiesel_v2 training/test trajectories and the complete observation horizon.
PYTHONPATH=chemkan/src /Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/train_adaptive_fgd.py \
  --run-dir results/experiments/adaptive_fgd/my_biodiesel --steps 20

# Other systems: --system hydrogen-stage1 or --system hydrogen-full.
# Comparisons: --method fixed-fgd or --method adam, with a different run directory.
# Older biodiesel data: --data-file chemkan/data/generated/biodiesel_legacy.npz.

# Hydrogen Stage 2 from the compatible current 100-point Stage-1 archive.
PYTHONPATH=chemkan/src /Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/train_adaptive_fgd.py --system hydrogen-stage2 \
  --stage1-from results/experiments/hydrogen/author_matched/main/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt \
  --dtype float32 --method adaptive-fgd --steps 3 \
  --run-dir results/experiments/adaptive_fgd/my_hydrogen_stage2

# The looser ODE-tolerance trial keeps FGD epsilon fixed and changes only Tsit5.
PYTHONPATH=chemkan/src /Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/train_adaptive_fgd.py --system hydrogen-stage2 \
  --stage1-from results/experiments/hydrogen/author_matched/main/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt \
  --dtype float32 --method adaptive-fgd --steps 3 \
  --rtol 1e-4 --atol 1e-6 \
  --run-dir results/experiments/adaptive_fgd/my_hydrogen_stage2_looseode

# Give Adam approximately the same elapsed budget as the original FGD run.
PYTHONPATH=chemkan/src /Users/berke/uni_projects/chemkan-venv/bin/python \
  chemkan/scripts/train_adaptive_fgd.py --method adam --lr 0.002 \
  --steps 100000 --max-seconds 77.34517029201379 \
  --run-dir results/experiments/adaptive_fgd/my_adam_time_budget

/Users/berke/uni_projects/chemkan-venv/bin/python -m pytest chemkan/tests/test_adaptive_fgd.py -q
```

Each run requires a new directory and writes `config.json`, `diagnostics.jsonl`,
`history.csv`, `metrics.json`, and `checkpoint_final.pt`. Diagnostics include
loss before/after the update, error bounds, certification, grid sizes, solver
sample counts, accepted learning rate, and backtracking count. Configuration
records dataset hash, architecture, dtype, seed, solver, and reference commit.
Failures preserve the current checkpoint and write `failure.json` without
claiming successful completion. New checkpoints contain model state, optimizer
state (Adam moments or FGD's recent line-search losses), and update records.
`--checkpoint-every 1` saves an atomic progress checkpoint after every completed
evaluation. `--resume-from path/to/checkpoint_progress.pt` restores this state
into a new run directory. Supply matching data, optimizer, solver, dtype and
architecture settings; `--steps` is the cumulative target, not an additional
number of updates. Checkpoints lacking optimizer state or fully evaluated
records are refused. The `loss_reduction` setting must also match: old time-summed
checkpoints cannot resume into the new time-averaged objective with stale
optimizer or line-search history. Resumed timing adds saved elapsed time and excludes work
performed after the source checkpoint in the previous process.

`--max-rhs-evaluations 10000` bounds each ODE solve in the new pilot. A line-search
candidate exceeding that work budget is rejected and backtracked, with the reason
recorded in `solver_trial_failures`. A budget failure at the current model or
during evaluation stops the run and saves its state. The default 0 leaves solves
unbounded. This limit was added after a trial solve stalled during update 2 of
the first nonmonotone pilot; the bounded continuation loads its saved update 1.

`--max-seconds` optionally limits wall time, including initial train/test evaluation,
training, per-step evaluations, and logging, matching the original reported timing
scope. Setup and final artifact serialization are excluded. An in-progress update
finishes before stopping, so elapsed time can slightly exceed the budget. Metrics
also record the last fully evaluated update within the budget. `--steps` remains
an independent cap; set it high enough for a time-limited run.

Library use with an existing model:

```python
from chemkan.adaptive_fgd import AdaptiveFGD, AdaptiveFGDConfig

# Load any original ChemKAN checkpoint BEFORE wrapping the model.
optimizer = AdaptiveFGD(model, AdaptiveFGDConfig(lr=0.01, epsilon=0.5))
record = optimizer.step(lambda: loss_fn(integrate(dynamics, y0, times, solver)))
```

The closure must recompute a deterministic scalar loss, and must not call
`backward()` itself. Do not use FSA, `odeint_adjoint`, stochastic minibatch changes
between trials, or gradient checkpointing with this capture mechanism.

To reload an experimental checkpoint, reconstruct `KineticCore` (biodiesel or
hydrogen Stage 1) or `ChemKAN` (full hydrogen) from
`checkpoint['config']['architecture']`, move it to the recorded dtype/device,
call `adaptive_edges(model)` for FGD runs, and load `checkpoint['model_state']`.
For author-matched Stage 2, use its `build_model()` and `Stage2Dynamics` instead of
a plain `ChemKAN` constructor so the RBF widths, input transforms, normalized
coordinates, and `1/t_ref` scaling match the trained model.
Each edge restores its saved representation size automatically. For inference,
also restore the saved input/loss normalizers and construct the appropriate
temperature provider/dynamics. Ordinary reproduction evaluators expect the old
checkpoint schema and should not be used directly on these checkpoints.

## Scope of the numerical claim

The completed local biodiesel comparisons, configurations, checkpoints and plot
are recorded in [`results/experiments/adaptive_fgd/README.md`](../results/experiments/adaptive_fgd/README.md).
That report also contains the matched-initialization hydrogen Stage-2 trial and
its temperature-specific diagnostics.

The error certificate concerns the representation of the Gaussian functional
gradient of the **discretized, currently differentiated computation**. It does
not bound ODE discretization error, floating-point error, or generalization error.
The adaptive solver's control-flow choices and interpolation knots can introduce
nonsmooth behavior. The paper's Hilbert/Banach compatibility and PL assumptions
have not been established for the composed ChemKAN neural ODE, so this extension
does **not** inherit its global-minimum claim. Backtracking provides an observed
sufficient decrease relative to the configured training-loss reference. With a
nonmonotone window this does not mean that every step improves the current loss.

This is an adaptation of Algorithm 1 and its RKHS construction to ChemKAN, not a
reproduction of the paper's cod-RNA, wave-equation, or radiance-field benchmarks.
Grid growth increases memory and runtime; the selected limits are explicit
experimental settings, not values prescribed by either paper. No claim of better
speed or generalization than Adam follows from the short local validation runs.
