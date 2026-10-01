#!/bin/sh
# Separates the two changes of lr1e-2_released_tol/ (which collapsed at epoch ~3,045):
#   tol_released_lr2e-3/  released tolerances (rtol 1e-2, atol 1e-6), paper lr 2e-3
#   lr1e-2_tol_default/   released lr 1e-2, default tolerances (rtol 1e-6, atol 1e-8)
# Everything else as main/: t_ref 6e-4 s, batched, float32, direct autograd, seed 0.
# 10,000 Stage-1 epochs each, run one after the other; permanent archive every 1,000.
# Rerunnable: each run resumes from its working checkpoint. Run from the repository root.
set -e
P=~/uni_projects/chemkan-venv/bin/python
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
M=results/experiments/hydrogen/author_matched/ablation_lr_tol
C="--seed 0 --t-ref 6e-4 --solve-mode batched --dtype float32 --sensitivity direct_autograd --epochs 10000 --archive-every 1000"
run() {  # $1 = run dir, rest = lr/tolerance flags
    d=$M/$1; shift
    [ -f $d/stage1_final.pt ] && { echo "skip $d (done)"; return; }
    if [ -f $d/stage1_working.pt ]; then R=--resume; else R=; fi
    echo "START $d $(date)"
    $P $T stage1 --run-dir $d $C "$@" $R
    echo "END $d $(date)"
}
run tol_released_lr2e-3 --lr 2e-3 --rtol 1e-2 --atol 1e-6
run lr1e-2_tol_default  --lr 1e-2 --rtol 1e-6 --atol 1e-8
echo "DONE $(date)"
