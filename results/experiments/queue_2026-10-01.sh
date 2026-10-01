#!/bin/sh
# Sequential queue (one training at a time), run from the repository root:
#  1. wait for hydrogen t_ref50 Stage 1 (10k) to finish
#  2. biodiesel: the best author-matched setting (run_batched_fsa_juliainit: author_repo data,
#     Julia initial weights, batched FSA, lr 1e-2, rtol 1e-2 / atol 1e-6, 10,000 epochs,
#     4 threads), but the RHS divided by the data time window (30 s) instead of 50
#  3. resume hydrogen main Stage 2 (main/resume_main.sh: Stage 1 and its eval are done,
#     so it resumes S2_seed0 from stage2_working.pt, epoch 2,500, up to 100,000)
P=~/uni_projects/chemkan-venv/bin/python
echo "QUEUE START $(date)"
while pgrep -f "t_ref50/run_stage1.sh" >/dev/null; do sleep 30; done
echo "t_ref50 finished $(date)"
B=results/experiments/biodiesel/author_repo_match/run_batched_fsa_juliainit_window30
if [ ! -f $B/checkpoint_final.pt ]; then
    if [ -f $B/checkpoint_resume.pt ]; then R=--resume; else R=; fi
    echo "biodiesel START $(date)"
    $P chemkan/scripts/author_repo_match/train_author_repo_match.py --run-dir $B \
        --solve-mode batched --sensitivity fsa --threads 4 --epochs 10000 \
        --init-from results/experiments/biodiesel/author_repo_match/julia_reference/p_init.txt $R \
        > $B.console.log 2>&1
    echo "biodiesel END rc=$? $(date)"
fi
echo "hydrogen Stage 2 resume START $(date)"
sh results/experiments/hydrogen/author_matched/main/resume_main.sh >> results/experiments/hydrogen/author_matched/main/chain.log 2>&1
echo "hydrogen END rc=$? $(date)"
