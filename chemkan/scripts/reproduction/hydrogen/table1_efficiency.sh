#!/bin/bash
# Table I - networks, parameters, species, and a LOCAL efficiency measurement.
# Evaluation only; no training.
#
# This is a local PyTorch-vs-Cantera benchmark over the same task (36 conditions, 50
# output times, 0-0.6 ms). It is NOT the paper's Arrhenius.jl measurement: different
# reference implementation, hardware and timing scope. The two are reported side by side
# and never merged. Accuracy is measured in the same run and reported with the timing,
# because a model that does not reproduce the dynamics is fast for the wrong reason.
#
# Run this on an OTHERWISE IDLE machine - a loaded CPU makes the timing meaningless.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_checkpoints.sh"
require_python

say "Table I - checkpoints (no hydrogen training)"
require_hydrogen_checkpoints
warn "timing is only meaningful on an idle machine; close other heavy work first"

_bench() {
  run "inference benchmark for $1 (2 warm-up + 5 repetitions)" -- \
    bash -c "cd '$REPO/chemkan/scripts/benchmark' && '$PY' benchmark_inference.py \
      --run-dir '$2' --out '$TABLES' --warmup 2 --reps 5 --cantera-tolerance-sweep --force"
}
for_each_hydrogen_set _bench

# Table I is assembled in notebook 08 and has no figure script, so this step still
# executes the notebook. Every hydrogen FIGURE is drawn by its own script.
render_notebook 08_hydrogen_reproduction.ipynb
say "Output"
info "${TABLES#"$REPO"/}/hydrogen_efficiency.csv"
info "${TABLES#"$REPO"/}/hydrogen_inference_benchmark_*.json"
verdict "Table I"
