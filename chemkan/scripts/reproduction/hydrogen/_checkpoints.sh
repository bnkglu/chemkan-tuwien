#!/bin/bash
# Hydrogen checkpoint guard, shared by every hydrogen figure script.
#
# THERE IS NO HYDROGEN TRAINING IN THIS DIRECTORY. Figures 7, 8A, 8B and Table I are all
# evaluated from completed checkpoints. If a checkpoint is missing these scripts fail and
# point at the workflow doc rather than silently launching a multi-hour two-stage run.
#
#   H0     - the PRIMARY reproduction (N=4/base-ON, 344 parameters, random thermo init).
#            It completed 10,000 Stage-2 epochs and FAILS to ignite. It is retained as
#            the primary result including its failure.
#   Hnorm1 - a SEPARATELY LABELLED initialization comparison that does ignite. One
#            initialization, not a seed study. It does not replace H0.

require_hydrogen_checkpoints() {
  local how="see docs/reproduction_workflow.md Step 6 (train_hydrogen.py); NOT run by these scripts"
  require_file "$H0_DIR/checkpoint_final.pt" "$how"
  require_file "$HNORM1_DIR/checkpoint_final.pt" "$how"
  info "H0     (primary, fails to ignite): ${H0_DIR#"$REPO"/}"
  info "Hnorm1 (labelled comparison)     : ${HNORM1_DIR#"$REPO"/}"
}

# for_each_hydrogen_set <function-name>  -> calls fn <label> <dir> <stem>
for_each_hydrogen_set() {
  "$1" "H0"     "$H0_DIR"     "random_stage2_10000_seed0"
  "$1" "Hnorm1" "$HNORM1_DIR" "normmatched_dir1_stage2_10000"
}
