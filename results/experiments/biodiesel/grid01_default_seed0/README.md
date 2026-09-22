# Biodiesel: additive-layer RBF grid on [0, 1] (`grid01_default_seed0`)

One biodiesel ChemKAN run (seed 0, 10,000 epochs, 156 parameters) that differs from the
default biodiesel run only in the **additive (first) layer's RBF grid**. The inputs to that
layer are min-max scaled states, which lie in [0, 1], while the default grid spans [-1, 1].
This run places the additive-layer centers on the input range instead:

| layer | default run | this run |
|---|---|---|
| additive (first) | centers −1, 0, 1 | centers 0, 0.5, 1, width h = 0.5 |
| LeanKAN (second) | centers −1, 0, 1, h = 1.0 | unchanged |

Everything else follows the default biodiesel run: base activation off, Adam lr 2e-3,
default initialization, full batch, clean data, Tsit5 rtol 1e-6 / atol 1e-8.

## Provenance

- **Grid values:** `config.json` does not record the modified grid. The values above are read
  from the saved RBF centers in `checkpoint_final.pt`.
- **Evaluation:** the run is evaluated in
  [`../../paper_alignment_audit_20260910/grid01_email_check.json`](../../paper_alignment_audit_20260910/grid01_email_check.json),
  which identifies it by checkpoint SHA-256 (`4f957289…`, matching this directory's
  `checkpoint_final.pt`). The same file evaluates the default run
  `results/reproduction/legacy/biodiesel/chemkan/main/direct_autograd_seed0` (`06d4a34c…`) as the
  baseline. It reports final and late-window (epochs 8,000–9,999) train/test losses for both.
- **Where it is used:** [`docs/reproduction_summary.md`](../../../../docs/reproduction_summary.md)
  cites that evaluation as "additive-grid".
- **Original location:** the run was trained on 2026-09-10 at
  `results/experiments/grid01_default_seed0/` and moved here on 2026-09-22. Its `config.json`
  (`run_id` `biodiesel/grid01_default_seed0`) and `run.log` keep the original path as
  written at training time.
