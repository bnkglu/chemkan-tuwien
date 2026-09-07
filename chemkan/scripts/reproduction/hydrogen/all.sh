#!/bin/bash
# Every hydrogen result, in paper order. Evaluation only - no hydrogen training anywhere.
# Table I is last because its timing wants an otherwise idle machine.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
FLAGS=(); [ "$DRY_RUN" = "1" ] && FLAGS+=(--dry-run); [ "$NO_RENDER" = "1" ] && FLAGS+=(--no-render)
for f in fig07_trajectories fig08a_generalization fig08b_ignition_delay table1_efficiency; do
  "$D/$f.sh" "${FLAGS[@]}" --no-render
done
render_notebook 08_hydrogen_reproduction.ipynb
