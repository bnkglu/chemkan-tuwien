#!/bin/bash
# The biodiesel runs shared by Figures 3, 5A, 5B and 6. Sourced by those scripts so that
# running several of them never trains the same model twice.
#
# B0 is REUSED for ChemKAN 0% and is never retrained. The clean replay is a separate
# 0% run whose only purpose is Figure 5B's missing history; it reproduces B0 bitwise.

noise_runs() {
  local levels=(01 02 03 05 07 10 15)
  say "ChemKAN noise runs (10,000 epochs each; B0 covers 0%)"
  info "reuse  B0 for 0%: ${CKB#"$REPO"/}/main/direct_autograd_seed0"
  for pct in "${levels[@]}"; do
    train_run "$CKB/noise/noise${pct}_seed0" "$REPO/chemkan/scripts" \
      "$PY" train_biodiesel.py --noise-percent "$((10#$pct))" \
        --epochs 10000 --eval-every 1 --seed 0
  done

  say "Corrected DeepONet noise runs used by the current figures"
  for pct in 00 "${levels[@]}"; do
    reference_deeponet_run noise "noise${pct}_seed0"
  done
}

reference_deeponet_run() {
  local checkpoint="$DOB_REF/$1/$2/checkpoint_final.pt"
  if [ ! -f "$checkpoint" ]; then
    warn "Missing corrected checkpoint: $checkpoint"
    warn "Use biodiesel/deeponet_reference.sh to prepare the corrected checkpoints."
    exit 1
  fi
  info "reuse corrected $1/$2 (current figure source)"
}

clean_replay_run() {
  say "Figure-5B clean replay (0% noise, with the epoch-5000 snapshot Figure 4 reuses)"
  train_run "$CKB/noise/clean_replay_seed0" "$REPO/chemkan/scripts" \
    "$PY" train_biodiesel.py --epochs 10000 --eval-every 1 --seed 0 --snapshot-epochs 5000
}

# evaluate_noise_levels <model> <levels...>
evaluate_noise_levels() {
  local model="$1"; shift
  for pct in "$@"; do
    local n=$((10#$pct)) dir script workdir
    if [ "$model" = chemkan ]; then
      dir="$CKB/noise/noise${pct}_seed0"; script=evaluate_biodiesel.py
      workdir="$REPO/chemkan/scripts"
      [ "$n" -eq 0 ] && dir="$CKB/main/direct_autograd_seed0"
    else
      # Evaluate the runs the notebook actually plots (reference_final_trunk_relu).
      dir="$DOB/noise/noise${pct}_seed0"; script=evaluate_biodiesel_deeponet.py
      workdir="$REPO/deeponet"
    fi
    run "evaluate $model @ ${n}% (three metrics, all from the FINAL checkpoint)" -- \
      bash -c "cd '$workdir' && '$PY' $script --run-dir '$dir' --split train --noise-percent $n --metrics >/dev/null &&
               '$PY' $script --run-dir '$dir' --split test  --noise-percent $n --metrics >/dev/null"
  done
}
