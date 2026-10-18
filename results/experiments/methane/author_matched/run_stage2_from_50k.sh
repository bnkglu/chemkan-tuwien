#!/usr/bin/env bash
# Methane main run, shortened: Stage 1 was stopped at ~54.8k epochs and Stage 2 starts from
# its permanent epoch-50,000 archive for 50,000 epochs (instead of 100k + 100k in
# run_main.sh). Same settings otherwise. Rerunnable: resumes Stage 2 from its checkpoint.
#
#   PYTHON=... nohup results/experiments/methane/author_matched/run_stage2_from_50k.sh >> methane_s2_nohup.log 2>&1 &
set -e
cd "$(dirname "$0")/../../../.."
P=${PYTHON:-$(command -v python)}
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
X=chemkan/scripts/author_repo_match/plot_trajectory.py
M=results/experiments/methane/author_matched/main
A=$M/S1_seed0/archive/stage1_S1_seed0_epoch50000.pt
C="--system methane --seed 0 --solve-mode batched --dtype float32 --sensitivity direct_autograd --threads 8"
echo "START $(date)  python $P"
$P $X --checkpoint $A --out-dir $M/plots
if [ ! -f $M/S2_seed0/stage2_final.pt ]; then
  if [ -f $M/S2_seed0/stage2_working.pt ]; then
    $P $T stage2 --run-dir $M/S2_seed0 --resume --epochs 50000 --archive-every 5000 $C
  else
    $P $T stage2 --run-dir $M/S2_seed0 --stage1-archive $A --epochs 50000 --warmup-epochs 0 --archive-every 5000 $C
  fi
fi
$P $X --checkpoint $M/S2_seed0/stage2_final.pt --out-dir $M/plots
echo "DONE $(date)"
