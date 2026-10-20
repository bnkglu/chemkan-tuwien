# Result locations: original → current

The results were produced under `results/experiments/` on the development branch and moved
here with `git mv`, without changing their content. The run records (configs, metrics,
histories, logs, checkpoints, READMEs, reports and the historical launch scripts) still
name the **original** paths and script names; this table maps them to the current ones.
The only files under `results/` whose content was changed are the inference benchmark's code
(`benchmark.py`, `make_report.py`, `julia/export_inputs.py`), updated to the current paths;
its inputs and outputs are unchanged.

## Result folders

| Original | Current |
|---|---|
| `results/experiments/biodiesel/author_repo_match/README.md` | `results/biodiesel/README.md` |
| `results/experiments/biodiesel/author_repo_match/figures/fig04/`, `fig05/` (ChemKAN runs), `run_log_*.txt` | `results/biodiesel/chemkan_figure_runs/` |
| `results/experiments/biodiesel/author_repo_match/figures/plots/` | `results/biodiesel/figures/` |
| `results/experiments/biodiesel/author_repo_match/figures_tanh_r1_lr1e-2/` | `results/biodiesel/deeponet_figure_runs/` |
| `results/experiments/biodiesel/author_repo_match/equivalence/` | `results/biodiesel/equivalence/` |
| `results/experiments/biodiesel/author_repo_match/reaction_order/` | `results/biodiesel/reaction_order/` |
| `results/experiments/biodiesel/author_repo_match/julia_reference/`, `julia_data/` | `results/biodiesel/released_example/julia_reference/`, `julia_data/` |
| `results/experiments/biodiesel/author_repo_match/data/` (generated, not tracked) | `results/biodiesel/released_example/data/` |
| `results/experiments/biodiesel/interpretability/` | `results/biodiesel/interpretability/` |
| `results/experiments/biodiesel/deeponet_diagnostics/tanh_R1/summary.txt` | `results/biodiesel/notes/deeponet_tanh_R1_summary.txt` |
| `results/experiments/biodiesel/deeponet_diagnostics/tanh_R1_lr1e-2/summary.txt` | `results/biodiesel/notes/deeponet_tanh_R1_lr1e-2_summary.txt` |
| `results/experiments/biodiesel/tolerance_sweep_diagnostic/REPORT.md` | `results/biodiesel/notes/tolerance_sweep_REPORT.md` |
| `results/experiments/hydrogen/author_matched/main/` | `results/hydrogen/` |
| `results/experiments/hydrogen/author_matched/smoke/` | `results/hydrogen/settings_screening/` |
| `results/experiments/hydrogen/interpretability/` | `results/hydrogen/interpretability/` |
| `results/experiments/methane/author_matched/main/` | `results/methane/` |
| `results/experiments/methane/author_matched/*.sh` (launch scripts) | `results/methane/*.sh` |
| `results/experiments/methane/interpretability/` | `results/methane/interpretability/` |
| `results/experiments/inference_benchmark/` | `results/inference_benchmark/` |

## Scripts named in the records

| Original | Current |
|---|---|
| `chemkan/scripts/author_repo_match/train_author_repo_match.py` | `chemkan/scripts/train_biodiesel.py` |
| `chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py` | `chemkan/scripts/combustion/train_combustion.py` (entry points `train_hydrogen.py`, `train_methane.py`; `--system` becomes the choice of entry point) |
| `chemkan/scripts/author_repo_match/_author_match.py` | `chemkan/scripts/reference_components.py` |
| `chemkan/scripts/author_repo_match/run_figures.py` | `chemkan/scripts/biodiesel/run_figure_runs.py` |
| `chemkan/scripts/author_repo_match/build_dataset.py` | `chemkan/scripts/biodiesel/build_released_dataset.py` (writes `released_example_biodiesel.npz`, formerly `author_repo_match_biodiesel.npz`) |
| `chemkan/scripts/author_repo_match/build_reaction_order_datasets.py`, `check_equivalence.py` | `chemkan/scripts/biodiesel/` (same names) |
| `chemkan/scripts/author_repo_match/export_julia_{init,noisy_data,reference}.jl` | `chemkan/scripts/biodiesel/julia/export_{init,noisy_data,reference}.jl` |
| `chemkan/scripts/author_repo_match/evaluate_author_matched_hydrogen.py` | `chemkan/scripts/combustion/evaluate_checkpoint.py` |
| `chemkan/scripts/author_repo_match/plot_trajectory.py` | `chemkan/scripts/combustion/plot_trajectory.py` |
| `chemkan/scripts/author_repo_match/smoke_check_solve_mode_dtype.py` | `chemkan/scripts/combustion/check_solve_mode_dtype.py` |
| `chemkan/scripts/evaluate_hydrogen.py` | `chemkan/scripts/combustion/predictor.py` |
| `chemkan/scripts/evaluate_hydrogen_grid.py`, `evaluate_hydrogen_ignition.py` | `chemkan/scripts/combustion/` (same names) |
| `results/experiments/methane/author_matched/main/generalization/evaluate_methane_grid.py` | `chemkan/scripts/combustion/evaluate_methane_grid.py` |
| `results/experiments/{biodiesel,hydrogen,methane}/interpretability/interpretability.py` | `chemkan/scripts/interpretability/biodiesel.py`, `combustion.py --system {hydrogen,methane}` |
| `chemkan/scripts/data_gen/extensions/generate_methane.py`, `methane_ignition_window.py` | `chemkan/scripts/data_gen/` (same names) |
| `chemkan/scripts/figures/author_runs.py`, `author_paper_table.py` | `chemkan/scripts/figures/biodiesel_runs.py`, `paper_table.py` |
| `deeponet/biodiesel_deeponet.py` | `deeponet/deeponet_model.py` |
| `deeponet/biodiesel_deeponet_repro.py` | `deeponet/deeponet_data.py` |
| `deeponet/train_biodiesel_deeponet_repro.py` | `deeponet/train_deeponet.py` |
| `deeponet/reproduce_diagnostics.sh` (target `figure_runs_lr1e-2`) | `deeponet/run_figure_runs.sh` |
| notebooks `07b_biodiesel_author_matched`, `08b_hydrogen_author_matched`, `15_methane_author_matched` | `chemkan/notebooks/biodiesel_reproduction`, `hydrogen_reproduction`, `methane_extension` |

## Labels kept in new run records

New runs still record the labels the historical runs carry, because resuming compares them:
`"experiment": "<system>/author_matched"` (hydrogen, methane), `"experiment":
"biodiesel/author_repo_match"` (biodiesel), and the DeepONet trainer's `"path": "reproduction
(biodiesel_deeponet_repro.py)"`. They are identifiers, not paths. The biodiesel trainer's data
choice keeps its value `--data author_repo` (the released example's arrays).

## Historical launch scripts

`results/hydrogen/run_main.sh`, `resume_main.sh`, `results/hydrogen/settings_screening/run_chain.sh`,
`results/methane/run_main.sh`, `run_stage2_from_50k.sh`, `extend_stage2_to_{150k,200k}.sh` and
`results/inference_benchmark/run_all.sh`, `run_methane.sh` are kept unchanged as the record of
how the runs were launched (original paths). The working launchers are
`chemkan/scripts/combustion/run_hydrogen.sh`, `run_methane.sh`,
`chemkan/scripts/benchmark/run_inference_benchmark.sh`, `chemkan/scripts/biodiesel/run_figure_runs.py`
and `deeponet/run_figure_runs.sh`.
