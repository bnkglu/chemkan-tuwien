#!/bin/bash
# Start H0-FSA and Hnorm1-FSA as soon as the shared FSA Stage-1 run has completed.
# Detached orchestration helper (run with nohup from the repository root). It launches
# nothing unless H_STAGE1-FSA finished with checkpoint_stage1.pt and checkpoint_final.pt.
set -u
cd "$(dirname "$0")/../../.."
PY="${CHEMKAN_PYTHON:-$HOME/uni_projects/chemkan-venv/bin/python}"
S1=results/experiments/fsa/hydrogen/stage1_fsa_seed0
LOG=results/experiments/fsa/chain_stage2.log
echo "$(date -u +%FT%TZ) waiting for $S1" >> "$LOG"
while pgrep -f "train_hydrogen.py .*stage1_fsa_seed0" > /dev/null; do sleep 60; done
if [ ! -f "$S1/checkpoint_stage1.pt" ] || [ ! -f "$S1/checkpoint_final.pt" ]; then
  echo "$(date -u +%FT%TZ) Stage-1 process gone without final checkpoints; NOT launching Stage 2" >> "$LOG"
  exit 1
fi
echo "$(date -u +%FT%TZ) Stage 1 complete; launching H0-FSA and Hnorm1-FSA" >> "$LOG"
nohup "$PY" chemkan/scripts/fsa/fsa_runs.py launch H0-FSA > results/experiments/fsa/launch_H0-FSA.log 2>&1 &
sleep 5
nohup "$PY" chemkan/scripts/fsa/fsa_runs.py launch Hnorm1-FSA > results/experiments/fsa/launch_Hnorm1-FSA.log 2>&1 &
echo "$(date -u +%FT%TZ) launched" >> "$LOG"
