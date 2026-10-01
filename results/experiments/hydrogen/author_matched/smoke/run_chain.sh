#!/bin/sh
# Smoke queue: Stage 1 (S1a t_ref 6e-4, S1b t_ref 1), Stage 2 A-D, evaluation. One at a time.
# Solve mode / dtype chosen from solve_mode_dtype.json: batched, float32.
set -e
P=~/uni_projects/chemkan-venv/bin/python
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
E=chemkan/scripts/author_repo_match/evaluate_author_matched_hydrogen.py
O=results/experiments/hydrogen/author_matched/smoke
C="--solve-mode batched --dtype float32 --sensitivity direct_autograd"   # torch default threads
echo "START $(date)"
$P $T stage1 --run-dir $O/S1a --epochs 2000 --seed 0 --t-ref 6e-4 $C
$P $T stage1 --run-dir $O/S1b --epochs 2000 --seed 0 --t-ref 1 $C
$P $T stage2 --run-dir $O/A --stage1-archive $O/S1a/archive/stage1_S1a_epoch2000.pt --epochs 2000 --warmup-epochs 0 --seed 0 --t-ref 6e-4 $C
$P $T stage2 --run-dir $O/B --stage1-archive $O/S1a/archive/stage1_S1a_epoch2000.pt --epochs 2000 --warmup-epochs 1000 --seed 0 --t-ref 6e-4 $C
$P $T stage2 --run-dir $O/C --stage1-archive $O/S1b/archive/stage1_S1b_epoch2000.pt --epochs 2000 --warmup-epochs 0 --seed 0 --t-ref 1 $C
$P $T stage2 --run-dir $O/D --stage1-archive $O/S1b/archive/stage1_S1b_epoch2000.pt --epochs 2000 --warmup-epochs 1000 --seed 0 --t-ref 1 $C
mkdir -p $O/eval
for r in S1a S1b; do $P $E --checkpoint $O/$r/stage1_final.pt --out $O/eval/$r.json; done
for r in A B C D; do $P $E --checkpoint $O/$r/stage2_final.pt --out $O/eval/$r.json; done
echo "DONE $(date)"
