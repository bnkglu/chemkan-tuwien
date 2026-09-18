# PyTorch FSA vs Julia/LeanKAN FSA — 50-epoch training equivalence (2026-09-18)

Biodiesel kinetic core, **identical initial weights**, identical hyperparameters, both
trained with forward sensitivity analysis. This isolates the implementations: the
initialization draw, which differs between the two languages' RNGs, is removed as a
variable by transferring the PyTorch weights into Julia.

| | |
|---|---|
| `pytorch_fsa/` | `chemkan/scripts/train_biodiesel.py --sensitivity fsa` |
| `julia_fsa/` | `chemkan_julia/scripts/train_biodiesel.jl --init transfer` |

## Configuration (held identical)

| setting | value |
|---|---|
| initial weights | PyTorch `torch.manual_seed(0)` draw, exported to `../../reference/biodiesel/init_seed0_params.npz`, loaded in Julia with `--init transfer` |
| architecture | hidden 4, num_basis 3, n_mu 2, base activation OFF |
| edge function | paper Eq. 12 Gaussian (Julia `--convention matched`, the default) |
| optimizer | Adam, lr 2e-3 (the paper's value), full batch, 20 trajectories |
| solver | Tsit5, rtol 1e-6, atol 1e-8 |
| loss | Eq. 18, train-only min-max input scaling |
| epochs | 50 |

Precision differs by construction: the PyTorch training path is float32, the Julia path
float64. That sets the floor for every difference below.

## Result: the two implementations agree to float32 precision

Training loss per epoch, from the two `history.csv` files:

| epoch | PyTorch (float32) | Julia (float64) | relative difference |
|---|---|---|---|
| 0 | 1.2765715332e+03 | 1.2765716056e+03 | 5.7e-08 |
| 1 | 1.0993857422e+03 | 1.0993859543e+03 | 1.9e-07 |
| 10 | 1.1579838562e+02 | 1.1579843482e+02 | 4.3e-07 |
| 20 | 9.8146652222e+01 | 9.8146638141e+01 | 1.4e-07 |
| 30 | 3.7581939697e+01 | 3.7581948542e+01 | 2.4e-07 |
| 40 | 1.9845520020e+01 | 1.9845522770e+01 | 1.4e-07 |
| 49 | 1.3439208984e+01 | 1.3439211657e+01 | 2.0e-07 |

**Maximum relative difference over all 50 epochs: 5.2e-07.**

Final weights after 50 Adam steps (`julia_fsa/params_final.npz`, PyTorch-layout `pt:`
names, against `pytorch_fsa/checkpoint_final.pt`):

| tensor | shape | max abs difference | relative to max abs weight |
|---|---|---|---|
| `add.edges.w_rbf` | (4, 7, 3) | 6.71e-08 | 2.18e-07 |
| `lean.edges.w_rbf` | (6, 4, 3) | 5.23e-08 | 2.05e-07 |

The difference does not grow over the 50 steps, which is what identical gradients look
like: the error stays at the float32 round-off level instead of compounding. Since both
runs follow the same Adam trajectory from the same starting point, the FSA gradients of
the two implementations agree to the same precision.

## Timing (MacBook, 1 thread, not a controlled benchmark)

| | median s/epoch | note |
|---|---|---|
| PyTorch FSA (float32) | 0.201 | |
| Julia FSA (float64) | 0.250 | plus a 13.4 s first epoch, which is JIT compilation |

## Reproduce

From the repository root, with `chemkan_julia/` inside it:

```sh
# 1. export the PyTorch seed-0 initial weights (deterministic, no training)
~/uni_projects/chemkan-venv/bin/python chemkan_julia/python/export_reference.py \
    biodiesel --states init_seed0 --params-only

# 2. PyTorch FSA
~/uni_projects/chemkan-venv/bin/python chemkan/scripts/train_biodiesel.py \
    --sensitivity fsa --epochs 50 --seed 0 --lr 2e-3 --rtol 1e-6 --atol 1e-8 \
    --run-dir results/experiments/julia_leankan/training_equivalence/biodiesel_50ep_transfer/pytorch_fsa \
    --experiment-name julia_equivalence

# 3. Julia FSA from the same weights
julia --project=chemkan_julia -t 1 chemkan_julia/scripts/train_biodiesel.jl \
    --run-dir results/experiments/julia_leankan/training_equivalence/biodiesel_50ep_transfer/julia_fsa \
    --init transfer \
    --init-from results/experiments/julia_leankan/reference/biodiesel/init_seed0_params.npz \
    --epochs 50 --lr 2e-3 --rtol 1e-6 --atol 1e-8 --sensitivity fsa
```

Rerunning reproduced every number in this file exactly (the runs were first made outside
the repository and repeated here so the recorded paths match their location).

## Scope

50 epochs, one system (biodiesel), one initialization, one seed. It shows that the two
implementations compute the same model and the same gradients. It says nothing about
what either converges to after 10,000 epochs, and nothing about hydrogen.
