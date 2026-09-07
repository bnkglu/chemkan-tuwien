#!/bin/bash
# Figure 7 - trajectories at the two published conditions (phi=0.9/T0=1050 K training,
# phi=1.3/T0=1150 K held out). Evaluation only; no training.
#
# Predictions are closed-loop integrations from the initial state alone: no reference
# temperature or species is injected at later times.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_checkpoints.sh"
require_python

say "Figure 7 - checkpoints (no hydrogen training)"
require_hydrogen_checkpoints

_eval() {
  run "evaluate $1 (metrics + prediction artifacts, both splits)" -- \
    bash -c "cd '$REPO/chemkan/scripts' &&
      '$PY' evaluate_hydrogen.py --run-dir '$2' --split train --metrics --save-predictions >/dev/null 2>&1 || true;
      '$PY' evaluate_hydrogen.py --run-dir '$2' --split test  --metrics --save-predictions >/dev/null 2>&1 || true;
      '$PY' evaluate_hydrogen.py --run-dir '$2' --split test  --metrics >/dev/null"
}
say "Split evaluation"
for_each_hydrogen_set _eval

render_notebook 08_hydrogen_reproduction.ipynb
say "Output"
info "${FIGURES#"$REPO"/}/hydrogen/fig07_hydrogen_trajectories_H0.pdf"
info "${FIGURES#"$REPO"/}/hydrogen/fig07_hydrogen_trajectories_Hnorm1.pdf"
verdict "Fig. 7"
