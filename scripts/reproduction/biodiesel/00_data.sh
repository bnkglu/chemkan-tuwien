#!/bin/bash
# Biodiesel data prerequisites. Idempotent and safe to re-run.
#
#   1. the 3% noise level, appended IN PLACE to biodiesel.npz from its own clean
#      trajectories using the generator's independent per-level seeding rule
#   2. Figure 3's explicitly published unseen condition, as a separate artifact
#
# Neither step regenerates the canonical trajectories, the 30-point grid or the 20/10
# split. The 3% step keeps a backup and restores it unless every pre-existing array is
# bitwise identical after the write.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
require_python

say "Biodiesel data prerequisites"

# Path passed as an argument: this script must work from any working directory.
if "$PY" - "$REPO/chemkan/data/generated/biodiesel.npz" <<'PYEOF'
import sys, numpy as np
sys.exit(0 if "train_states_noise03" in
         np.load(sys.argv[1], allow_pickle=False).files else 1)
PYEOF
then
  info "skip   3% noise level (already present in biodiesel.npz)"
else
  run "append the 3% noise level (verified bitwise against a backup)" -- \
    bash -c "cd '$REPO/chemkan/scripts/data_gen' && '$PY' add_biodiesel_noise_level.py --level 0.03 --apply"
fi

FIG3_NPZ="$REPO/chemkan/data/generated/biodiesel_fig3_condition.npz"
if [ -f "$FIG3_NPZ" ]; then
  info "skip   Figure-3 condition artifact (already present)"
else
  run "generate the Figure-3 condition (TG0=1.94, ROH0=1.43, T=334.8 K)" -- \
    bash -c "cd '$REPO/chemkan/scripts/data_gen' && '$PY' generate_biodiesel_fig3_condition.py"
fi

info "note: biodiesel_fig3_condition.npz is untracked by design - exactly reproducible"
info "      from the committed generator at a fixed seed."
