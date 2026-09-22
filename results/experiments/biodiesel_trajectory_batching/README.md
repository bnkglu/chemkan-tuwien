# Biodiesel: trajectory batching

Full-trajectory biodiesel training with mini-batches of training trajectories instead of one
full batch of all 20. Everything else follows the default biodiesel run (Adam lr 2e-3, default
initialization, clean data, seed 0, 10,000 epochs). One epoch is one pass over the 20
training trajectories. Analysis: [`chemkan/notebooks/13_biodiesel_trajectory_batching.ipynb`](../../../chemkan/notebooks/13_biodiesel_trajectory_batching.ipynb).

| directory | batch size | batches / epoch | epochs | optimizer steps |
|---|---|---|---|---|
| `full_rollout_bs1_normal_clean_seed0/` | 1 | 20 | 10,000 | 200,000 |
| `full_rollout_bs5_normal_clean_seed0/` | 5 | 4 | 10,000 | 40,000 |
| `timing_probe_bs1_100ep/` | 1 | 20 | 100 | 2,000 |

The full-batch (B = 20) reference is `results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0`.
`timing_probe_bs1_100ep/` is a short run used only for the runtime projection in
`timing_probe_bs1.json`.

- `tables/`: `batching_comparison.csv`, `batching_final_comparison.csv`,
  `batching_bs5_comparison.csv`, `batching_late_window.json`, `batching_runtime.csv`.
- `figures/`: training and held-out loss curves, per-species errors, species profiles and the
  Fig. 5B-layout comparison at 0 % noise.

Each run directory has the standard contents (`checkpoint_final.pt`, `config.json`,
`history.csv`, `run.log`; the batch-size-5 run also keeps `checkpoint_epoch_2500.pt`, the
matched-steps point of `batching_bs5_comparison.csv`). `config.json` records the batching
settings under `batching`.
