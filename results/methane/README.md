# Author-matched methane, main run (seed 0)

**Methane is the project proposal's extension; the paper has no methane results.**

- **Data:** `chemkan/data/generated/methane.npz`. GRI-Mech 3.0 in Cantera, 52 species + T, 36 constant-pressure conditions (T0 1400–1650 K × φ 0.5–1.5), held-out 1550 K / φ 1.3, 100 points over 5 ms.
- **Trainer:** `train_author_matched_hydrogen.py --system methane`, with the hydrogen author-matched settings:
  - Adam lr 2e-3;
  - Tsit5 rtol 1e-6 / atol 1e-8, batched, float32;
  - direct autograd, no warm-up, PINN α 1e-4;
  - t_ref = 5 ms.

| dir | what | epochs |
|---|---|---|
| `S1_seed0/` | Stage 1. JupyterHub to 28,000 epochs, continued on the Mac (`host_transfer.json`), stopped at ~54,800; 5k archives up to 50k | 0 → 54,790 |
| `S2_seed0/` | Stage 2 from the **epoch-50,000 Stage-1 archive**, extended in place (`--resume --epochs`) to 100k, 150k and 200k | 0 → 200,000 |
| `figures/`, `plots/`, `plots_interim/` | figures from notebook 15 and trajectory plots of earlier snapshots | |
| `generalization/` | 441-condition grid (Fig. 8A analogue), see its README | |

Scripts: `../run_stage2_from_50k.sh`, `../extend_stage2_to_150k.sh`, `../extend_stage2_to_200k.sh`.

## Selected model: epoch 180,000

The final 200k model sits on a training-loss spike, so the reported model is the best
5,000-epoch snapshot by training loss: `S2_seed0/archive/stage2_S2_seed0_epoch180000.pt`. The
held-out condition is not used for the choice (notebook 15, "Snapshot selection").

Exact losses of the saved weights. Time-averaged MSE; "training" is the trainer's loss, i.e.
the mean over the 35 training conditions:

| model | training | held-out (1550 K, φ 1.3) |
|---|---|---|
| **epoch 180,000 (selected)** | **1.220e-3** (history row 180,001: 1.2198e-3) | **1.163e-3** |
| epoch 200,000 (final, `stage2_final.pt`) | 2.885e-3 | 1.274e-3 |

**Not converged:** the 1,000-epoch median of the training loss was still falling at 200k:
2.01e-3 at 100k, 1.71e-3 at 185k, 1.43e-3 at 200k.

The trainer logs epoch N's loss at the parameters *before* step N. The checkpoint saved at N
holds the weights *after* it, so the saved model's loss is history row N+1.

## Stage-2 history

`S2_seed0/stage2_history.csv` holds all 200,000 rows with the columns `epoch, loss, pinn,
mse_time_avg, elapsed_seconds`, identical to the training output in those columns.
- **On the development branch** the file also has `mse`, `kin_out_max_raw`, `kin_out_max_scaled`,
  `w_norm` and `nfe` (`PATHS.md`).
- **Not in either branch:** the training output's 52 per-species thermo-weight columns
  `w_<species>` (the full file is 236 MB, above GitHub's 100 MB limit).
- **The weights themselves** are in every checkpoint.

**Checkpoints:**
- committed: `stage2_final.pt` and all 40 snapshots in `S2_seed0/archive/` (5k–200k);
- not committed: `stage2_working.pt`, a resume file.
