# Figure 4 width sweep — digitized marker positions and proposed matrix

**Status: approved 2026-09-06; runs launched.** The widths below are the approved set.

## 1. How the marker positions were obtained

Figure 4 is vector art in the paper PDF, so it was measured, not estimated by eye.
`chemkan/scripts/diagnostics/digitize_paper_fig4.py` renders page 11 at 600 dpi and:

- calibrates the x axis from the **log-minor tick lattice** — ticks lie at
  `x(10²) + W·log10(P/100)`, so fitting `(x(10²), W)` to all 15 detected ticks fixes the
  mapping without reading a single axis label;
- calibrates the y axis the same way, anchored on the two labelled decades (10⁻² and
  10⁻⁴), which are drawn with longer tick marks than the unlabelled ones;
- extracts marker centroids by colour, separating the marker fill from the near-identical
  trend-line fill, and recovering the ChemKAN circles by erosion because the red fit line
  runs through them.

Fit quality: **0.45 px rms** tick residual, i.e. **0.11 % in parameter count**. Both
panels return the identical decade width (972.50 px) and identical axis span
(50 → 1002 parameters), which they were not constrained to do.

**Validation of the extraction — not of any model we trained.** Refitting slopes from the
digitized points recovers the four slopes the paper prints in its own figure. This is a
check that the marker positions and both axis calibrations were read correctly: it says
nothing whatsoever about our reproduction's accuracy, because every number involved comes
from the paper's own figure. Our trained models' slopes are a separate measurement,
reported only once the runs below finish, and must never be conflated with this row:

| series | digitized | paper prints |
|---|---|---|
| ChemKAN training | **−1.04** | −1.0 |
| ChemKAN testing | **−0.59** | −0.6 |
| DeepONet training | **−4.13** | −4.0 |
| DeepONet testing | **−1.45** | −1.4 |

The paper's numbers were never used to derive the calibration, so this is a genuine check
on the extraction. Raw output: `results/reproduction/tables/paper_fig4_digitized.json`.

## 2. What the figure actually contains

**Five ChemKAN markers and six DeepONet markers** — not six and six. Confirmed both by
component analysis and by visual inspection of the rendered panel.

| | digitized parameter counts |
|---|---|
| ChemKAN | 78.0, 116.9, 155.8, 390.3, 650.3 |
| DeepONet | 78.0, 156.0, 234.0, 308.1, 456.1, 716.3 |

## 3. ChemKAN: four of five markers land exactly on `P = 39h`

With `N = 3`, base OFF, 7 inputs and 6 outputs, `P(h) = 3·(7h + 6h) = 39h`. Measured from
`model.parameters()`, **the count does not depend on `n_mu` at all** — so the
`n_mu = ceil(h/2)` rule cannot be inferred from Fig. 4, and remains a reconstruction
choice that the figure neither supports nor contradicts.

| marker | `39h` value | h | agreement |
|---|---|---|---|
| 78.0 | 78 | **2** | exact |
| 116.9 | 117 | **3** | exact |
| 155.8 | 156 | **4** | exact |
| 390.3 | 390 | **10** | exact |
| 650.3 | — | — | **no match** |

The fifth marker is the one problem. The nearest feasible widths are h=16 (624) and h=17
(663); at 0.11 % measurement precision, both are **~30 σ away**. This is a real
discrepancy, not measurement error, and it should be recorded rather than smoothed over.

Two observations, offered as observations only:

- `650 = 13·10·5` **exactly**, and `390 = 13·10·3` exactly. So the two largest markers are
  consistent with *one* width `h = 10` evaluated at a three-point and a five-point grid.
  That contradicts the paper's own statement that sizes were varied "by changing the number
  of nodes in the hidden layers", so **it is not adopted** — it is recorded because it is
  an exact arithmetic fit, not because it is the likely reading.
- The paper's stray "**72 parameters**" (vs. the structurally consistent 78) equals
  `36·2 = 3·(6h + 6h)` at h=2 — a **six**-input count. The figure's own smallest marker is
  at 78, the seven-input count. So the 72/78 inconsistency is consistent with the prose
  having counted the branch input without temperature while the figure counted with it.
  Also an observation; the architecture is not changed to produce 72.

## 4. Approved ChemKAN width matrix

`N = 3` fixed, base OFF fixed, `n_mu = ceil(h/2)` (reconstruction choice; measured to be
count-neutral), 0 % noise, seed 0, Adam lr = 2e-3, Tsit5, `direct_autograd`,
**5,000 epochs**. Counts are measured from `model.parameters()`, not assumed.

| h | n_mu | measured P | paper marker | note | run directory |
|---|---|---|---|---|---|
| 2 | 1 | 78 | 78.0 | exact | `results/reproduction/chemkan/biodiesel/scaling/h02_seed0` |
| 3 | 2 | 117 | 116.9 | exact | `.../scaling/h03_seed0` |
| 4 | 2 | 156 | 155.8 | exact — **reused, not retrained** (see below) | `.../noise/clean_replay_seed0/checkpoint_epoch_5000.pt` |
| 10 | 5 | 390 | 390.3 | exact | `.../scaling/h10_seed0` |
| 17 | 9 | 663 | 650.3 | **nearest feasible; the marker itself is unexplained** | `.../scaling/h17_seed0` |

**h = 4 is not retrained.** The clean replay saves `checkpoint_epoch_5000.pt`, the model
after exactly 5,000 optimizer steps at h=4, N=3, n_mu=2, base OFF, 0 % noise, seed 0,
lr 2e-3 — every Fig.-4 setting, and `ceil(4/2) = 2` matches its `n_mu`. The snapshot
mechanism was verified bitwise identical to an independently trained run of the same
length. One provenance wart: because the snapshot reuses the hydrogen `Stage2Snapshot`
helper, its epoch marker is stored under the key `stage2_epoch` inside a biodiesel
checkpoint. The value (5000) is correct and nothing reads that key for biodiesel; the
artifact is left as produced rather than rewritten after the fact.

**The ≈650 discrepancy is preserved, not resolved.** h=17 gives 663 against a measured
marker at 650.3 ± ~1 — a 2 % gap that no `39h` width closes (h=16 gives 624). We train
h=17 because it is the nearest feasible architecture, *not* because it reproduces the
marker. The marker remains unexplained and is reported as such alongside the paper's own
72-vs-78 and 308-vs-340 inconsistencies.

## 5. Approved DeepONet width matrix

Family: `branch [7,w,w,w]`, `trunk [1,w-1,w]`, Hadamard, head `Linear(w,6)`; 50,000
epochs, Adam lr = 1e-3, Glorot-normal, biased Linear (reference-derived), same dataset /
normalizer / loss as ChemKAN. Every count measured. Activation placement is
reference-derived: ReLU between branch layers with a linear final branch layer, ReLU after
every trunk layer including the last, and a linear head.

**Current figures use the completed `reference_final_trunk_relu` sweep.** The earlier
`legacy_final_trunk_linear` checkpoints remain at their original paths and are always
loaded with their original architecture. Their reports are archived and labelled under
`legacy_final_trunk_linear/` in the figures and tables directories. Notebook 07 also
includes a second comparison with fixed `n_mu=2` ChemKAN; both use the corrected DeepONet.

| w | measured P | nearest paper marker | offset | run directory |
|---|---|---|---|---|
| 3 | 85 | 78.0 | +9 % | `results/reproduction/baselines/deeponet/biodiesel/reference_final_trunk_relu/scaling/w03_seed0` |
| 5 | 169 | 156.0 | +8 % | `.../scaling/w05_seed0` |
| 6 | 220 | 234.0 | −6 % | `.../scaling/w06_seed0` |
| 8 | **340** | 308.1 | +10 % | `.../scaling/w08_seed0` — the comparison architecture |
| 10 | 484 | 456.1 | +6 % | `.../scaling/w10_seed0` |
| 13 | 745 | 716.3 | +4 % | `.../scaling/w13_seed0` |

### What these widths are, and are not

Both sets **reconstruct the plotted parameter range**. They are **not** exact
paper-specified architectures: the paper gives no width table for either model, and the
ChemKAN widths are recovered only because four of its five markers happen to fall on
`39h`. No width was chosen to make a count match a marker — where the family cannot reach
a marker (ChemKAN 650, DeepONet 308) the gap is reported, not engineered away. The
DeepONet offsets of ±4-10 % are a direct consequence of that rule.

## 6. Fit masks

The paper fits "prior to saturation (or in the DeepONet testing results, prior to
overfitting)". The digitized points show where those regimes end, and the same masks
reproduce the printed slopes:

| series | included | excluded |
|---|---|---|
| ChemKAN training | P ≤ 390 | 650 (loss rises) |
| ChemKAN testing | P ≤ 390 | 650 (loss rises) |
| DeepONet training | P ≤ 308 | 456, 716 |
| DeepONet testing | P ≤ 308 | 456, 716 (loss rises — overfitting) |

These are the paper's masks, inferred from its own digitized points. **Our fit masks will
be derived from our own measured curves and stated explicitly per series** — included and
excluded parameter counts, slope, intercept, R^2 and standard error — and will not be
copied from this table. Both are reported so the comparison is like-for-like.

## 7. Cost

Four new ChemKAN runs (h=2,3,10,17; h=4 is reused) + six DeepONet runs.
