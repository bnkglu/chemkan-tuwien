# Figure 4 width sweep

**Status: approved 2026-09-06; runs completed.**

The paper gives no width table for Figure 4. It says only that sizes were varied "by
changing the number of nodes in the hidden layers", and its axis spans roughly 10² to 10³
parameters. The widths below were chosen to cover that range. They are **not**
paper-specified architectures. Every parameter count is measured from
`model.parameters()`, not assumed.

## ChemKAN width matrix

`N = 3` fixed, base OFF fixed, `n_mu = ceil(h/2)` (reconstruction choice), 0 % noise,
seed 0, Adam lr = 2e-3, Tsit5, `direct_autograd`, **5,000 epochs**. With 7 inputs and
6 outputs the count is `P(h) = 3·(7h + 6h) = 39h`, independent of `n_mu`.

| h | n_mu | measured P | run directory |
|---|---|---|---|
| 2 | 1 | 78 | `results/reproduction/chemkan/biodiesel/scaling/h02_seed0` |
| 3 | 2 | 117 | `.../scaling/h03_seed0` |
| 4 | 2 | 156 | `.../noise/clean_replay_seed0/checkpoint_epoch_5000.pt` — **reused, not retrained** |
| 10 | 5 | 390 | `.../scaling/h10_seed0` |
| 17 | 9 | 663 | `.../scaling/h17_seed0` |

**h = 4 is not retrained.** The clean replay saves `checkpoint_epoch_5000.pt`, the model
after exactly 5,000 optimizer steps at h=4, N=3, n_mu=2, base OFF, 0 % noise, seed 0,
lr 2e-3 — every Fig.-4 setting, and `ceil(4/2) = 2` matches its `n_mu`. The snapshot
mechanism was verified bitwise identical to an independently trained run of the same
length. One provenance wart: because the snapshot reuses the hydrogen `Stage2Snapshot`
helper, its epoch marker is stored under the key `stage2_epoch` inside a biodiesel
checkpoint. The value (5000) is correct and nothing reads that key for biodiesel; the
artifact is left as produced rather than rewritten after the fact.

## DeepONet width matrix

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

| w | measured P | run directory |
|---|---|---|
| 3 | 85 | `results/reproduction/baselines/deeponet/biodiesel/reference_final_trunk_relu/scaling/w03_seed0` |
| 5 | 169 | `.../scaling/w05_seed0` |
| 6 | 220 | `.../scaling/w06_seed0` |
| 8 | **340** | `.../scaling/w08_seed0` — the comparison architecture |
| 10 | 484 | `.../scaling/w10_seed0` |
| 13 | 745 | `.../scaling/w13_seed0` |

## Fit masks

The paper fits "prior to saturation (or in the DeepONet testing results, prior to
overfitting)" without listing which points that includes. Our fits use **all** measured
points and are reported per series (slope, intercept, R² and standard error). The paper's
four slopes printed in its Figure 4 (Δ = −1.0, −0.6, −4.0, −1.4) are compared with ours as
printed numbers.

## Cost

Four new ChemKAN runs (h=2,3,10,17; h=4 is reused) + six DeepONet runs.
