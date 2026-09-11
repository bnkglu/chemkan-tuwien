#!/bin/bash
# Figure 3 - ChemKAN trajectory reconstruction at 0/5/10/15% noise, on the paper's
# explicitly published unseen condition (TG0=1.94, ROH0=1.43, T=334.8 K).
#
# B0 supplies the 0% column and is never retrained. Predictions are closed-loop
# integrations from the initial condition on a dense 601-point grid.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_runs.sh"
require_python

DRY_RUN="$DRY_RUN" bash "$(dirname "${BASH_SOURCE[0]}")/00_data.sh"

say "Figure 3 prerequisites: the 5/10/15% ChemKAN runs (shared with Figure 5)"
for pct in 05 10 15; do
  train_run "$CKB/noise/noise${pct}_seed0" "$REPO/chemkan/scripts" \
    "$PY" train_biodiesel.py --noise-percent "$((10#$pct))" \
      --epochs 10000 --eval-every 1 --seed 0
done

figure_script fig03_biodiesel_trajectories.py
say "Output"
info "${FIGURES#"$REPO"/}/biodiesel/fig03_biodiesel_noise_columns.pdf"
verdict "Fig. 3"
