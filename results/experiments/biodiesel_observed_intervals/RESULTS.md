# Results — observed-state interval training (biodiesel, clean)

Every number here is read from the artifacts in this directory and from
`chemkan/notebooks/11_biodiesel_observed_intervals.ipynb`, which produced them. See
`README.md` for what the experiment changes and for the commands that reproduce it.

**Scoring.** All comparisons are **full rollouts from the original initial conditions**
over the whole 30-point grid, with no intermediate observed-state reset and no test
observation supplied to the model — the same quantity the original trainer reports and
that `evaluate_biodiesel.py` writes into `metrics.json`. The observed-interval training
objective is a different quantity and is never used as evidence of improvement.

**Pairing.** Within each seed the two runs start from **identical initial parameter
tensors** with independent optimizer states; the update-0 full-rollout loss in the first
row of each run's `history.csv` is identical in both arms: `1276.5710449219` (seed 0), `718.6188354492` (seed 1),
`1109.0960693359` (seed 2).

## Headline

**Observed-interval training fits the training conditions substantially better and does
not improve held-out error. The held-out result is unanimous in the wrong direction, but
small.**

## 1. Full-rollout error at 10,000 updates

| seed | observed-interval train | test | original train | test | Δ train | Δ test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.032555 | 0.085017 | 0.062017 | 0.081401 | **−0.029462** | +0.003616 |
| 1 | 0.023246 | 0.067326 | 0.022466 | 0.063242 | +0.000780 | +0.004084 |
| 2 | 0.018138 | 0.070551 | 0.033120 | 0.055667 | **−0.014982** | +0.014884 |

Negative Δ means the experiment helped.

- **Training error:** better in **2 of 3** seeds; median ratio **0.548**. Seed 1 is
  essentially a tie (+0.00078, a 3 % relative difference).
- **Held-out error:** worse in **3 of 3** seeds; median ratio **1.065**. The seeds agree
  on the direction; the magnitude ranges from +4 % (seed 0) to +27 % (seed 2).

So the procedure buys a real gain in fitting the supervised conditions that does not
transfer to unseen initial conditions.

## 2. Do the paired seeds agree?

On training error, no: two clear improvements and one tie. On held-out error, **yes** —
all three seeds move the same way, which is the more meaningful agreement here, and it is
against the experiment. Three seeds are a small sample and this is a diagnostic choice,
not a paper requirement; the consistent sign, not the median value, is what these runs
support.

## 3. Where the error goes

Exact decomposition of the Eq. 18 loss by species (`mean_k c_k = total`), from
`tables/observed_intervals_error_breakdown.csv`.

At **training** conditions, seed 0, observed-interval is better on **all six** species.
At **held-out** conditions the picture splits, identically in all three seeds:
observed-interval is better on **TG, ROH, RCO2R** and worse on **GL** — and GL is the
largest single contributor to the held-out loss in every observed-interval run.

Seed 0, test split:

| | TG | ROH | DG | MG | GL | RCO2R | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| observed-interval | 0.00323 | 0.01142 | 0.04961 | 0.19166 | 0.24556 | 0.00862 | 0.08502 |
| original | 0.01754 | 0.04175 | 0.08522 | 0.15947 | 0.15511 | 0.02932 | 0.08140 |

Observed-interval is better on four of six species yet worse overall, because the two it
loses on (MG, GL) dominate the sum. The same shape holds for seeds 1 and 2: four of six
species better, GL worse in all three.

This is consistent with what the procedure supervises. Restarting every interval from an
observation removes any penalty for accumulated drift, and glycerol is the terminal
product whose held-out trajectory depends most on the integrated history.

## 4. Runtime

Wall time per run (`tables/observed_intervals_runtime.csv`), added up over resumed
segments: seeds 1 and 2 of the observed-interval runs were interrupted and resumed once.

| seed | observed-interval | original |
|---:|---:|---:|
| 0 | 5,493.2 s (1.53 h) | 628.1 s (0.17 h) |
| 1 | 6,768.6 s (1.88 h) | 1,072.6 s (0.30 h) |
| 2 | 6,528.6 s (1.81 h) | 1,084.6 s (0.30 h) |

The observed-interval procedure makes 29 solver calls per update instead of one. Wall time
is **not a clean benchmark**: the runs shared one machine, with up to five concurrent
processes, so it reflects scheduling as well as the method.

## 5. Answers to the questions asked

- **Do full-rollout train and held-out errors improve?** Training error: yes, clearly, in
  2 of 3 seeds (median ratio 0.548). Held-out error: **no** — worse in all three seeds
  (median ratio 1.065).
- **Do the paired seeds agree?** On the held-out direction, yes, unanimously. On training
  error, two improvements and one tie.
- **Is it faster?** No: 1.5–1.9 h per run against 0.2–0.3 h (6–9× longer).

## 6. Scope and caveats

- Observed-state resets are an additional experiment, not a procedure the
  ChemKAN paper describes. Nothing here establishes it as a correction of the
  reproduction or as the authors' undocumented method.
- Clean biodiesel only, three seeds, one architecture (156 parameters), one budget
  (10,000 updates). The seed count is a diagnostic choice, not a paper
  requirement.
- The loss-scale gap to the paper is **not** explained here. FSA is still unimplemented and
  direct autograd is not claimed equivalent to it; this experiment does not test that.
- No pre-existing file, run, checkpoint, figure, table, or document was modified.
