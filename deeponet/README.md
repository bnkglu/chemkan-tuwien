# DeepONet baseline (biodiesel)

The DeepONet comparison model of the ChemKAN paper's biodiesel figures (Figs. 4, 5, 6), as a
self-contained PyTorch implementation. It runs in the ChemKAN environment (see the main
README); nothing from the original DeepONet repository is needed.

| File | Contents |
|---|---|
| `deeponet_model.py` | the network: branch [TG0, ROH0, T0] → 8 → 8 → 8, trunk t → 7 → 8, Hadamard product, linear head to the 6 species (308 parameters at width 8, the paper's count); the Fig. 4 sizes `FIG4_ARCHITECTURES` (78 … 456 parameters) |
| `deeponet_data.py` | input/output preprocessing (raw time in the trunk, author-global normalized states), initialization and activation options, and the metrics |
| `train_deeponet.py` | trainer: one run per call, full-batch Adam; writes `config.json`, `history.csv`, `metrics.json` and `checkpoint_final.pt` (among others) to `--run-dir` |
| `run_figure_runs.sh` | the 14 figure runs of `results/biodiesel/deeponet_figure_runs/` (8 noise levels × 10,000 epochs for Fig. 5, 6 sizes × 50,000 epochs for Fig. 4) |

## Reproducing the figure runs

```bash
bash deeponet/run_figure_runs.sh             # every run whose checkpoint_final.pt is missing
DRY_RUN=1 bash deeponet/run_figure_runs.sh   # print the 14 commands only
```

The settings of these runs: tanh activations, Adam lr 1e-2, truncated Glorot-normal weights
with zero biases (`--init Aprime_tf_truncated_glorot_zero_bias`), author-global normalized
states, mean-MSE objective, seed 0, `biodiesel_v2.npz`. The paper states neither the
activation nor the learning rate; `results/biodiesel/notes/deeponet_tanh_R1_summary.txt` and
`deeponet_tanh_R1_lr1e-2_summary.txt` record the 10-seed comparison behind this choice (final
training MSE, median over seeds: ReLU 2.4e-3, tanh at lr 1e-3 8.9e-5, tanh at lr 1e-2 3.3e-5).

A single run, e.g. the 308-parameter model at 5 % noise:

```bash
python deeponet/train_deeponet.py --state-mode author_global_normalized_states --objective mean_mse \
    --init Aprime_tf_truncated_glorot_zero_bias --lr 1e-2 --activation tanh --branch-mode normalized \
    --report-train-only-norm --seed 0 --noise-percent 5 --epochs 10000 --run-dir runs/deeponet/noise05
```

`python deeponet/train_deeponet.py --help` lists every option. The trainer has no resume;
an existing run directory is refused unless `--overwrite` is given.

## Evaluation

Each run's `metrics.json` holds the final train and test losses (section
`figure_normalization`: the same normalization and time-averaged MSE as the ChemKAN runs).
The figure scripts read them and predict trajectories from `checkpoint_final.pt` through
`chemkan/scripts/figures/biodiesel_runs.py` (`final_losses`, `converged`, `deeponet_predict`).

## Attribution

DeepONet is due to L. Lu, P. Jin, G. Pang, Z. Zhang and G. E. Karniadakis, *Learning
nonlinear operators via DeepONet based on the universal approximation theorem of operators*,
Nature Machine Intelligence 3, 218–229 (2021), with reference code at
https://github.com/lululxvi/deeponet and https://github.com/lululxvi/deepxde. The
reference-derived conventions used here (biased layers, activation placement, Glorot-normal
initialization, Adam lr 1e-3 as the reference default) are cited in the docstrings of
`deeponet_model.py` and `deeponet_data.py` with the DeepXDE 0.11.2 source lines they follow.
This folder contains no code from those repositories; the original repository's own README
and license (CC BY-NC-SA 4.0) are not included, since they apply to that code.
