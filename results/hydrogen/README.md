# Author-matched hydrogen, main run (seed 0)

Historical settings: `run_main.sh` (t_ref 6e-4 s, lr 2e-3, Tsit5 rtol 1e-6 / atol 1e-8, batched, float32,
direct autograd, no warm-up).

| dir | what | epochs |
|---|---|---|
| `S1_seed0/` | Stage 1; best loss near epoch 60k, diverged after; archive at 100k | 100,000 |
| `S2_seed0/` | Stage 2 from the epoch-100k Stage-1 archive | 0 → 100,000 |
| `S2_seed0_150k/` | `S2_seed0` continued (`--resume --epochs 150000`) | → 150,000 |
| `S2_seed0_200k/` | `S2_seed0_150k` continued | → 200,000 |

**Stage-2 history.** The full per-epoch history is `S2_seed0_200k/stage2_history.csv`
(epochs 1-200,000). The `S2_seed0` and `S2_seed0_150k` histories are byte-identical to its
first 100,000 and 150,000 rows, so only the 200k file is committed. In this repository the
history files keep the columns `epoch, loss, pinn, mse_time_avg, elapsed_seconds` (all rows);
the complete files, with the diagnostic and thermo-weight columns, are on the development
branch (`PATHS.md`).

**Checkpoints.** Each run's `stage2_final.pt`; the 5,000-epoch snapshots once each
(`S2_seed0/archive`: 20k-95k, `S2_seed0_150k/archive`: 105k-145k,
`S2_seed0_200k/archive`: 155k-195k). The copies of earlier snapshots inside the extension
directories are not committed.

`eval/S2_seed0.json`, `figures/`, `plots/`, `tables/`: evaluations of the Stage-2 checkpoints.

For a new training run, use `bash chemkan/scripts/combustion/run_hydrogen.sh`
from the repository root (defaults to `runs/hydrogen/`). The launch scripts in this
result folder retain historical paths; see [`../PATHS.md`](../PATHS.md).
