#!/bin/bash
# Figure 8B - ignition delay. Evaluation only; no training.
#
# Reference and model are put on the SAME dense 601-point grid over 0-0.6 ms and through
# the SAME argmax-dT/dt estimator, so a coarse 50-point reference derivative is never
# compared against a dense model derivative. The evaluated set is fixed by which REFERENCE
# trajectories ignite (30 of 36; the six 950 K cases do not ignite in the window), never by
# whether the model succeeds. A non-igniting prediction is recorded as
# no_ignition_in_window with its delay left undefined.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_checkpoints.sh"
require_python

say "Figure 8B - checkpoints (no hydrogen training)"
require_hydrogen_checkpoints

_ign() {
  run "ignition-delay evaluation for $1" -- \
    bash -c "cd '$REPO/chemkan/scripts' && '$PY' evaluate_hydrogen_ignition.py \
      --run-dir '$2' --out '$TABLES' --force"
}
for_each_hydrogen_set _ign

render_notebook 08_hydrogen_reproduction.ipynb
say "Output"
info "${FIGURES#"$REPO"/}/hydrogen/fig08b_hydrogen_ignition_delay.pdf"
info "${TABLES#"$REPO"/}/hydrogen_ignition_delay_*.csv  (per-condition status)"
verdict "Fig. 8B"
