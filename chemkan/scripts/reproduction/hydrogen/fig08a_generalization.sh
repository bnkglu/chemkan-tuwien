#!/bin/bash
# Figure 8A - 441-condition generalization map. Evaluation only; no training.
#
# Normalization is the canonical hydrogen.npz TRAIN statistics - never the fine grid's
# own, never refit on the evaluation grid. Every condition gets its own MSE; nothing is
# averaged before plotting. Integration failures are recorded as failures with their
# condition identifiers, never dropped or replaced by an invented finite value.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_checkpoints.sh"
require_python

"$(dirname "${BASH_SOURCE[0]}")/00_data.sh" ${DRY_RUN:+--dry-run}
say "Figure 8A - checkpoints (no hydrogen training)"
require_hydrogen_checkpoints

_grid() {
  run "441-condition evaluation for $1" -- \
    bash -c "cd '$REPO/chemkan/scripts' && '$PY' evaluate_hydrogen_grid.py \
      --run-dir '$2' --out '$RESULTS/chemkan/hydrogen/generalization' --save-predictions --force"
}
for_each_hydrogen_set _grid

render_notebook 08_hydrogen_reproduction.ipynb
say "Output"
info "${FIGURES#"$REPO"/}/hydrogen/fig08a_hydrogen_generalization_441.pdf"
info "${RESULTS#"$REPO"/}/chemkan/hydrogen/generalization/*_generalization_441.csv"
verdict "Fig. 8A"
