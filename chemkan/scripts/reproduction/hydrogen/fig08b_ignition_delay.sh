#!/bin/bash
# Figure 8B - ignition delay. Evaluation only; no training.
#
# Reference and model are put on the SAME dense 601-point grid over 0-0.6 ms and through
# the SAME argmax-dT/dt estimator, so a coarse 50-point reference derivative is never
# compared against a dense model derivative. The evaluated set is fixed by which REFERENCE
# trajectories ignite (30 of 36; the six 950 K cases do not ignite in the window), never by
# whether the model succeeds: it is the paper's own 30-condition set (T0 = 1000-1200 K).
# The delay is the paper's definition throughout -- argmax dT/dt, with no minimum-rise
# requirement -- and is reported for every evaluated prediction. No ignited/not verdict is
# applied anywhere; each row carries temperature rise and peak dT/dt for BOTH reference and
# prediction, so a flat curve with a confident delay cannot be read as an ignition.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_checkpoints.sh"
require_python

say "Figure 8B - checkpoints (no hydrogen training)"
require_hydrogen_checkpoints

_ign() {
  run "ignition-delay evaluation for $1" -- \
    bash -c "cd '$REPO/chemkan/scripts' && '$PY' evaluate_hydrogen_ignition.py \
      --run-dir '$2' --out '$TABLES' --force"
}
for_each_hydrogen_set _ign

figure_script fig08_hydrogen_ignition.py
say "Output"
info "${FIGURES#"$REPO"/}/hydrogen/fig08b_hydrogen_ignition_delay.pdf"
info "${TABLES#"$REPO"/}/hydrogen_ignition_delay_*.csv  (delay + rise + peak dT/dt)"
verdict "Fig. 8B"
