# Biodiesel reproduction: selected final results

The public release contains selected, seed-0, author-matched ChemKAN figure runs and the
corresponding project-specific DeepONet baseline on the canonical
[`biodiesel_v2.npz`](../../chemkan/data/generated/biodiesel_v2.npz). Earlier experimental
paths and full historical diagnostics are retained on the development (`dev`) branch; see
[`results/PATHS.md`](../PATHS.md) for the development-to-release location mapping.

## Runs and outputs

| Figure group | Method | Model/configuration | Epochs | Output directory |
|---|---|---|---:|---|
| Fig. 5 (also provides Figs. 3 and 6 models) | ChemKAN | 156 parameters, hidden width 4; FSA, RHS divisor 50; noise 0, 1, 2, 3, 5, 7, 10, 15% | 10,000 each | `chemkan_figure_runs/fig05/` |
| Fig. 5 | DeepONet | 308 parameters; project-specific tanh comparison; same noise levels | 10,000 each | `deeponet_figure_runs/fig05/` |
| Fig. 4 | ChemKAN | Hidden widths 2, 3, 4, 9 and 11; noise-free | 5,000 each | `chemkan_figure_runs/fig04/` |
| Fig. 4 | DeepONet | 78, 156, 249, 308, 384 and 456 parameters; noise-free | 50,000 each | `deeponet_figure_runs/fig04/` |

The figure scripts read selected checkpoints from these folders. See the main
[`README.md`](../../README.md) and [`docs/reproduction_summary.md`](../../docs/reproduction_summary.md)
for the conventions, paper comparisons, and limitations. `figures/` contains the generated
plots; `equivalence/`, `released_example/`, `reaction_order/`, `interpretability/`, and
`notes/` contain supporting numerical evidence and analyses.

## Reproduce the commands

From the **repository root**, after installing the environment:

```bash
python chemkan/scripts/biodiesel/run_figure_runs.py --list
python chemkan/scripts/biodiesel/run_figure_runs.py --dry-run
python chemkan/scripts/biodiesel/run_figure_runs.py --only fig05_chemkan_noise05
DRY_RUN=1 bash deeponet/run_figure_runs.sh
```

**To train all runs safely without overwriting archived results**, choose an independent output root:

```bash
export BIODIESEL_RESULTS=runs/biodiesel_rerun
python chemkan/scripts/biodiesel/run_figure_runs.py
bash deeponet/run_figure_runs.sh
python chemkan/scripts/figures/plot_all.py --only fig03 fig04 fig05a fig05b fig06
```

Both launchers skip existing final checkpoints. The ChemKAN launcher can resume compatible
interrupted runs; the DeepONet trainer has no resume support. They produce trained models
in the selected result root; training a full sweep is
computationally expensive. The example commands above are documentation of the
verified entry points, **not** a claim that reruns across machines are bit-identical.

For standalone ChemKAN training, see `chemkan/scripts/train_biodiesel.py`. The small
researcher-oriented kinetic-only example is `chemkan/scripts/train.py`. The complete
DeepONet source copied for historical reference in `deeponet/upstream_reference/` is **not**
the training implementation used to generate these figure runs.
