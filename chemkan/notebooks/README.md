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

| Notebook | Contents |
|---|---|
| `01_implement_kan.ipynb` | a guided workbook: one learnable RBF function, the AddKAN and LeanKAN layers, and a small KAN-ODE (milestones 1–4) |
| `02_einsum_edge_functions_paper_aligned.ipynb` | the one `einsum` that evaluates every KAN edge function, mapped to the paper's indices and re-derived by loops, broadcasting and `matmul` on three biodiesel trajectories |
| `03_chemkan_from_scratch_paper_aligned.ipynb` | a ChemKAN built inline step by step (RBF, edges, layers, the 156-parameter core, scaling, dynamics) and trained as a neural ODE on biodiesel |
| `04_chemkan_actual_solution_walkthrough.ipynb` | the library's kinetic path checked step by step (normalization, core, dynamics, solver, loss), and how `scripts/train.py` chains the same pieces |

The four were re-executed top to bottom on `biodiesel_v2.npz` with the current library.

`chemkan/scripts/data_gen/data_generation_walkthrough.ipynb` (data generation step by step)
does not run top to bottom as stored: section 1e configures `generate_biodiesel.generate()`
with an earlier argument set (no `ic_source`), and sections 1c, 1e and 1g describe the
earlier recipe (sampled initial conditions, multiplicative noise) rather than the one that
produced `biodiesel_v2.npz`; the cells after 1e that use its dataset fail with it. The other
sections run.
