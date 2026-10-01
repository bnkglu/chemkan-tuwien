#!/bin/sh
# Author-matched hydrogen Stage 1 with the RHS divided by 50, as the released biodiesel code
# does literally (du ./ 50), instead of by the hydrogen time window (t_ref 6e-4 s).
# Time stays in seconds. Otherwise the main/ settings: lr 2e-3, rtol 1e-6, atol 1e-8,
# batched, float32, direct autograd, seed 0. 10,000 epochs, permanent archive every 1,000.
# Rerunnable (resumes). Run from the repository root.
set -e
P=~/uni_projects/chemkan-venv/bin/python
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
M=results/experiments/hydrogen/author_matched/t_ref50
if [ -f $M/S1_seed0/stage1_working.pt ]; then R=--resume; else R=; fi
echo "START $(date)"
$P $T stage1 --run-dir $M/S1_seed0 --seed 0 --t-ref 50 --solve-mode batched --dtype float32 --sensitivity direct_autograd --epochs 10000 --archive-every 1000 $R
echo "DONE $(date)"
