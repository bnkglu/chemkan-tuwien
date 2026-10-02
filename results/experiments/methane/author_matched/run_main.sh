#!/usr/bin/env bash
# Main author-matched methane run: the hydrogen main run's settings with --system methane
# (t_ref = the 5 ms methane window, set by --system), no warm-up, batched, float32, direct
# autograd, 8 threads. Stage 1 100k -> trajectory plots -> Stage 2 100k from the Stage-1
# archive -> trajectory plots.
#
# Run from anywhere; rerun after an interruption: each stage resumes from its last
# checkpoint (every 500 epochs) and a finished stage is skipped.
#
#   nohup results/experiments/methane/author_matched/run_main.sh >> methane_main_nohup.log 2>&1 &
set -e
cd "$(dirname "$0")/../../../.."
P=${PYTHON:-$(command -v python)}
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
X=chemkan/scripts/author_repo_match/plot_trajectory.py
M=results/experiments/methane/author_matched/main
C="--system methane --seed 0 --solve-mode batched --dtype float32 --sensitivity direct_autograd --threads 8"
echo "START $(date)  python $P"

if [ ! -f $M/S1_seed0/stage1_final.pt ]; then
  R=""; [ -f $M/S1_seed0/stage1_working.pt ] && R="--resume"
  $P $T stage1 --run-dir $M/S1_seed0 $R --epochs 100000 --archive-every 5000 $C
fi
$P $X --checkpoint $M/S1_seed0/stage1_final.pt --out-dir $M/plots

if [ ! -f $M/S2_seed0/stage2_final.pt ]; then
  if [ -f $M/S2_seed0/stage2_working.pt ]; then
    $P $T stage2 --run-dir $M/S2_seed0 --resume --epochs 100000 --archive-every 5000 $C
  else
    $P $T stage2 --run-dir $M/S2_seed0 \
      --stage1-archive $M/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt \
      --epochs 100000 --warmup-epochs 0 --archive-every 5000 $C
  fi
fi
$P $X --checkpoint $M/S2_seed0/stage2_final.pt --out-dir $M/plots
echo "DONE $(date)"
