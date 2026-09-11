#!/bin/bash
# Figure 4 - neural scaling. The widths span the parameter range of the paper's Figure 4
# axis, 10^2 to 10^3 parameters (docs/fig4_width_matrix.md):
#   ChemKAN  h = 2, 3, 4, 10, 17  ->  78, 117, 156, 390, 663 measured parameters, 5,000 epochs
#   DeepONet w = 3, 5, 6, 8, 10, 13 -> 85, 169, 220, 340, 484, 745, 50,000 epochs
#
# They cover the plotted parameter range; the paper gives no width table, so they are
# not paper-specified architectures.
#
# h=4 is NOT trained here: it is the clean replay's checkpoint_epoch_5000.pt, the model
# after exactly 5,000 optimizer steps under every Figure-4 setting.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_runs.sh"
require_python

clean_replay_run                       # provides the h=4 point via its epoch-5000 snapshot

say "ChemKAN scaling widths (N=3, base OFF, n_mu=ceil(h/2), 5,000 epochs)"
for spec in "2 1" "3 2" "10 5" "17 9"; do
  set -- $spec
  train_run "$CKB/scaling/h$(printf %02d "$1")_seed0" "$REPO/chemkan/scripts" \
    "$PY" train_biodiesel.py --hidden-dim "$1" --n-mu "$2" --num-basis 3 \
      --epochs 5000 --seed 0 --experiment-name scaling
done

say "Corrected DeepONet scaling widths"
for w in 3 5 6 8 10 13; do
  reference_deeponet_run scaling "w$(printf %02d "$w")_seed0"
done

run "validate completed sweeps and render both Figure-4 comparisons" -- \
  "$PY" "$REPO/chemkan/scripts/diagnostics/refresh_biodiesel_reports.py"

say "Output"
info "${FIGURES#"$REPO"/}/biodiesel/fig04_biodiesel_neural_scaling{,_nmu2}.pdf"
info "${TABLES#"$REPO"/}/biodiesel_fig4_points.csv   (per-point late-epoch oscillation bands)"
info "${TABLES#"$REPO"/}/biodiesel_fig4_fits.csv     (descriptive all-point regressions)"
info "Fits describe all measured points; the paper fits a pre-saturation subset."
verdict "Fig. 4"
