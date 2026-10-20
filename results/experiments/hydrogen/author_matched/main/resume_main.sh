#!/bin/sh
# Resume the main author-matched hydrogen run (same settings as run_main.sh).
# Continues Stage 1 from S1_seed0/stage1_working.pt; then evaluates it, runs Stage 2
# (resuming it too if S2_seed0 already has a working checkpoint), and evaluates Stage 2.
# Safe to rerun after any interruption. Run from the repository root.
set -e
P=~/uni_projects/chemkan-venv/bin/python
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
E=chemkan/scripts/author_repo_match/evaluate_author_matched_hydrogen.py
M=results/experiments/hydrogen/author_matched/main
C="--seed 0 --t-ref 6e-4 --solve-mode batched --dtype float32 --sensitivity direct_autograd"
mkdir -p $M/eval
echo "RESUME $(date)"
if [ ! -f $M/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt ]; then
    $P $T stage1 --run-dir $M/S1_seed0 --epochs 100000 $C --resume
fi
[ -f $M/eval/S1_seed0.json ] || $P $E --checkpoint $M/S1_seed0/stage1_final.pt --out $M/eval/S1_seed0.json
if [ -f $M/S2_seed0/stage2_working.pt ]; then
    $P $T stage2 --run-dir $M/S2_seed0 --epochs 100000 --warmup-epochs 0 $C --resume
else
    $P $T stage2 --run-dir $M/S2_seed0 --stage1-archive $M/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt --epochs 100000 --warmup-epochs 0 $C
fi
$P $E --checkpoint $M/S2_seed0/stage2_final.pt --out $M/eval/S2_seed0.json
echo "DONE $(date)"
