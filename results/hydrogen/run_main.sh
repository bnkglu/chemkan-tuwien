#!/bin/sh
# Main author-matched hydrogen run: settings of smoke arm A (selection.json):
# t_ref 6e-4 s, no warm-up, batched, float32, direct autograd, default threads,
# default checkpoint interval (500). Stage 1 -> eval -> Stage 2 -> eval.
set -e
P=~/uni_projects/chemkan-venv/bin/python
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
E=chemkan/scripts/author_repo_match/evaluate_author_matched_hydrogen.py
M=results/experiments/hydrogen/author_matched/main
C="--seed 0 --t-ref 6e-4 --solve-mode batched --dtype float32 --sensitivity direct_autograd"
mkdir -p $M/eval
echo "START $(date)"
$P $T stage1 --run-dir $M/S1_seed0 --epochs 100000 $C
$P $E --checkpoint $M/S1_seed0/stage1_final.pt --out $M/eval/S1_seed0.json
$P $T stage2 --run-dir $M/S2_seed0 --stage1-archive $M/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt --epochs 100000 --warmup-epochs 0 $C
$P $E --checkpoint $M/S2_seed0/stage2_final.pt --out $M/eval/S2_seed0.json
echo "DONE $(date)"
