#!/bin/bash
# Figure 5B - loss dynamics at 0/2/7/15%.
#
# Needs per-epoch noise-free test history, which every run above logs via an
# evaluation-only callback: it runs inside the loss function (so each logged test value
# belongs to the parameter state that produced that epoch's training loss), under no_grad,
# with the RNG saved and restored, taking no optimizer step.
#
# The 0% panel comes ENTIRELY from the clean replay - both curves, one parameter
# trajectory. B0's own history has no test column. The replay reproduces B0 bitwise, so
# the panel plots B0's trajectory rather than a substitute, and B0 remains the established
# 0% result for Figures 3 and 5A.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_runs.sh"
require_python

DRY_RUN="$DRY_RUN" bash "$(dirname "${BASH_SOURCE[0]}")/00_data.sh"
clean_replay_run
say "The 2/7/15% panels (shared with Figure 5A)"
for pct in 02 07 15; do
  train_run "$CKB/noise/noise${pct}_seed0" "$REPO/chemkan/scripts" \
    "$PY" train_biodiesel.py --noise-percent "$((10#$pct))" --epochs 10000 --eval-every 1 --seed 0
  reference_deeponet_run noise "noise${pct}_seed0"
done
reference_deeponet_run noise noise00_seed0

render_notebook 07_biodiesel_reproduction.ipynb
say "Output"
info "${FIGURES#"$REPO"/}/biodiesel/fig05b_biodiesel_loss_dynamics.pdf"
info "${TABLES#"$REPO"/}/biodiesel_fig5b_overfit_assessment.json"
info "No run meets OUR overfitting criterion. Its thresholds (test rise >10%, further"
info "smoothed training fall >5%), the 201-epoch window and the 90%-of-budget cutoff are"
info "our diagnostic choices, not the paper's; the JSON stores the measurements so the"
info "criterion can be re-applied at other thresholds."
verdict "Fig. 5B"
