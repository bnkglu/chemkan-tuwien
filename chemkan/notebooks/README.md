# Notebooks

Run from this folder with the project environment (`jupyter lab` or `jupyter notebook`).
The stored outputs are the record of the executions that produced the results; they were
not re-executed when the repository was reorganized, so file paths printed in some outputs
are the original ones (`results/PATHS.md` maps them).

## Reproduction notebooks

| Notebook | Contents |
|---|---|
| `biodiesel_reproduction.ipynb` | biodiesel Figs. 3–6 and the paper-vs-ours table from the figure runs in `results/biodiesel/`, with the DeepONet baseline |
| `hydrogen_reproduction.ipynb` | hydrogen Stage 1 and Stage 2 (100k, 150k, 200k): loss curves, evaluation, Figs. 7, 8A (441-condition grid) and 8B (ignition delay) |
| `methane_extension.ipynb` | the methane extension (project proposal): Stage-2 training history, selection of the 180k snapshot, error per condition, trajectories, temperature of all 36 conditions, ignition delay (the 441-condition grid is `scripts/combustion/evaluate_methane_grid.py`) |

Their code cells use only the current scripts and paths (checked statically). None of them
trains. They read the committed results; `hydrogen_reproduction.ipynb` also re-runs the
hydrogen evaluations and writes their tables and figures to `results/hydrogen/` again.

## Explanation notebooks (`extra/`)

Step-by-step explanations of KAN and ChemKAN on top of the library in `chemkan/src/chemkan`.
They show the library's own building blocks (`KineticCore`, `KineticDynamics`,
`chemkan.training`), not the exact configuration of the final trainers, which follows the
authors' released example (`scripts/train_biodiesel.py`, `scripts/combustion/train_combustion.py`;
the shortest complete example is `scripts/train.py`).

| Notebook | State |
|---|---|
| `01_implement_kan.ipynb` | runs; its stored outputs were computed on an earlier biodiesel dataset, the code now loads `biodiesel_v2.npz`, so a re-run prints different numbers |
| `02_einsum_edge_functions_paper_aligned.ipynb` | **does not run** on the current library (`TypeError: 'Tensor' object is not callable`); needs updating |
| `03_chemkan_from_scratch_paper_aligned.ipynb` | runs; stored outputs from the earlier biodiesel dataset, as for 01 |
| `04_chemkan_actual_solution_walkthrough.ipynb` | **does not run** (`NameError: edge0`), and its Part VII reviews an earlier biodiesel trainer that had the same file name as the current `scripts/train_biodiesel.py`; needs rewriting against the current trainer |

`chemkan/scripts/data_gen/data_generation_walkthrough.ipynb` (data generation step by step)
has cells that use names defined in no earlier cell (`t_linspace`, `SPECIES`, …), so it
does not run top to bottom as stored.
