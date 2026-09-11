#!/bin/bash
# Figure 6 - 15%-noise profiles. NO new training: reuses the Figure-5 15% ChemKAN and
# DeepONet checkpoints.
#
# Plotted at Figure 3's published condition (TG0=1.94, ROH0=1.43, T=334.8 K), using that
# artifact's clean truth and stored 15% noise realization, so Figures 3 and 6 are directly
# comparable. The paper plots an unidentified TRAINING trajectory; this condition is
# UNSEEN by the models. Reusing Figure 3's temperature here is a REPRODUCTION PLOTTING
# CHOICE and has not been independently confirmed against the paper's Figure 6.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_runs.sh"
require_python

DRY_RUN="$DRY_RUN" bash "$(dirname "${BASH_SOURCE[0]}")/00_data.sh"
say "Figure 6 prerequisites: the 15% pair (shared with Figure 5)"
train_run "$CKB/noise/noise15_seed0" "$REPO/chemkan/scripts" \
  "$PY" train_biodiesel.py --noise-percent 15 --epochs 10000 --eval-every 1 --seed 0
reference_deeponet_run noise noise15_seed0

figure_script fig06_biodiesel_profiles.py
say "Output"
info "${FIGURES#"$REPO"/}/biodiesel/fig06_biodiesel_15pct_profiles.pdf"
verdict "Fig. 6"
