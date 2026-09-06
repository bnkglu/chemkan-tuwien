#!/bin/bash
# Figure 6 - 15%-noise profiles. NO new training: reuses the Figure-5 15% ChemKAN and
# DeepONet checkpoints on one training trajectory.
#
# The paper does not identify which training case it plots, so case index 0 is a
# documented REPRODUCTION PLOTTING CHOICE, fixed before inspecting any prediction.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_runs.sh"
require_python

"$(dirname "${BASH_SOURCE[0]}")/00_data.sh" ${DRY_RUN:+--dry-run}
say "Figure 6 prerequisites: the 15% pair (shared with Figure 5)"
train_run "$CKB/noise/noise15_seed0" "$REPO/chemkan/scripts" \
  "$PY" train_biodiesel.py --noise-percent 15 --epochs 10000 --eval-every 1 --seed 0
train_run "$DOB/noise/noise15_seed0" "$REPO/deeponet" \
  "$PY" train_biodiesel_deeponet.py --noise-percent 15 --epochs 10000 --eval-every 1 --seed 0

render_notebook 07_biodiesel_reproduction.ipynb
say "Output"
info "${FIGURES#"$REPO"/}/biodiesel/fig06_biodiesel_15pct_profiles.pdf"
verdict "Fig. 6"
