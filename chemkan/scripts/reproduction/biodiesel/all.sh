#!/bin/bash
# Every biodiesel figure, in paper order. Each step is idempotent, so shared runs are
# trained once and reused by the later figures.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
FLAGS=(); [ "$DRY_RUN" = "1" ] && FLAGS+=(--dry-run); [ "$NO_RENDER" = "1" ] && FLAGS+=(--no-render)
for f in fig03_trajectories fig05a_noise_robustness fig05b_loss_dynamics \
         fig06_15pct_profiles fig04_neural_scaling; do
  "$D/$f.sh" "${FLAGS[@]}"
done
# Each figure was drawn by its own script above. The notebook adds interpretation, not
# figures, so re-executing it is opt-in.
maybe_render_notebook 07_biodiesel_reproduction.ipynb
