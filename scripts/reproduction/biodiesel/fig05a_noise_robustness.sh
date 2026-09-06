#!/bin/bash
# Figure 5A - noise robustness across 0/1/2/3/5/7/10/15% for both models.
#
# Three metrics per model and level, ALL from the final checkpoint (never the last
# training-history row - the checkpoint is the state after the 10,000th update, that row
# is the state before it). The two test metrics use the SAME predicted trajectories and
# differ only in target: noisy observations (Eq. 18) vs clean trajectories (Eq. 22).
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_runs.sh"
require_python

"$(dirname "${BASH_SOURCE[0]}")/00_data.sh" ${DRY_RUN:+--dry-run}
noise_runs

say "Final-checkpoint evaluation"
evaluate_noise_levels chemkan  00 01 02 03 05 07 10 15
evaluate_noise_levels deeponet 00 01 02 03 05 07 10 15

render_notebook 07_biodiesel_reproduction.ipynb
say "Output"
info "${FIGURES#"$REPO"/}/biodiesel/fig05a_biodiesel_noise_robustness.pdf"
info "${TABLES#"$REPO"/}/biodiesel_fig5a_metrics.csv"
verdict "Fig. 5A"
