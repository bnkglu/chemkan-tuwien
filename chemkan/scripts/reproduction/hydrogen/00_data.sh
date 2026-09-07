#!/bin/bash
# Hydrogen data prerequisites: the 441-condition fine grid used by Figure 8A.
#
# Evaluation-only data. It is untracked by design - exactly reproducible from the
# committed generator - and is NEVER used to fit a normalizer.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
require_python

say "Hydrogen data prerequisites"
FINE="$REPO/chemkan/data/generated/hydrogen_fine.npz"
if [ -f "$FINE" ]; then
  info "skip   hydrogen_fine.npz (already present)"
else
  run "generate the 21x21 fine grid (T0 950-1200 K, phi 0.5-1.5)" -- \
    bash -c "cd '$REPO/chemkan/scripts/data_gen' && '$PY' generate_hydrogen.py \
      --out ../../data/generated/hydrogen_fine.npz --grid fine"
fi
info "The 21x21 spacing is RECONSTRUCTED from the paper's 441-condition count and figure;"
info "the paper does not tabulate it."
