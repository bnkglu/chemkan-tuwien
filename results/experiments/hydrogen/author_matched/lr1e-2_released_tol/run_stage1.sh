#!/bin/sh
# Author-matched hydrogen Stage 1 with the released biodiesel example's learning rate and
# solver tolerances: Adam lr 1e-2 (Flux.Adam(1f-2)), rtol 1e-2 / atol 1e-6 (ODEProblem
# reltol=1e-2, solve abstol=1e-6). Everything else as main/ (smoke arm A): t_ref 6e-4 s,
# batched, float32, direct autograd, seed 0. Permanent archive every 5,000 epochs.
# Stage 1 -> eval only; Stage 2 is started separately from a chosen archive.
# Rerunnable: resumes from S1_seed0/stage1_working.pt. Run from the repository root.
set -e
P=~/uni_projects/chemkan-venv/bin/python
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
E=chemkan/scripts/author_repo_match/evaluate_author_matched_hydrogen.py
M=results/experiments/hydrogen/author_matched/lr1e-2_released_tol
C="--seed 0 --t-ref 6e-4 --solve-mode batched --dtype float32 --sensitivity direct_autograd --lr 1e-2 --rtol 1e-2 --atol 1e-6 --archive-every 5000"
mkdir -p $M/eval
echo "START $(date)"
if [ -f $M/S1_seed0/stage1_working.pt ]; then R=--resume; else R=; fi
$P $T stage1 --run-dir $M/S1_seed0 --epochs 100000 $C $R
$P $E --checkpoint $M/S1_seed0/stage1_final.pt --out $M/eval/S1_seed0.json
echo "DONE $(date)"
