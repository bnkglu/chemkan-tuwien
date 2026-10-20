#!/usr/bin/env bash
# The methane main run of results/methane/ (the project proposal's extension), stage by stage:
#   Stage 1, 100k epochs (archive every 5k)  ->  Stage 2 from the 50k Stage-1 archive, 200k
#   epochs (archive every 5k), with trajectory plots after each stage.
# Settings as recorded in results/methane/*/config.json: seed 0, t_ref 5e-3 s (the methane
# window), batched solve, float32, direct autograd; Adam lr 2e-3, Tsit5 rtol 1e-6 / atol 1e-8
# (the trainer defaults). The selected methane model is the Stage-2 archive of epoch 180,000
# (notebook methane_extension; the 200k checkpoint sits on a loss spike).
#
#   bash chemkan/scripts/combustion/run_methane.sh              # into runs/methane/
#   OUT=<dir> PYTHON=<python> THREADS=8 bash chemkan/scripts/combustion/run_methane.sh
#
# Rerun after an interruption: a finished stage is skipped, an interrupted one resumed. The
# recorded Stage 2 was resumed at several epochs (run.log) and extended in steps of 50k
# (results/methane/run_stage2_from_50k.sh, extend_stage2_to_*.sh, kept unchanged); one
# uninterrupted --epochs 200000 run is not claimed to be bit-identical to it.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=${PYTHON:-python}
T=chemkan/scripts/train_methane.py
X=chemkan/scripts/combustion/plot_trajectory.py
M=${OUT:-runs/methane}
C="--seed 0 --solve-mode batched --dtype float32 --sensitivity direct_autograd --threads ${THREADS:-8}"

if [ ! -f "$M/S1_seed0/stage1_final.pt" ]; then
    R=""; [ -f "$M/S1_seed0/stage1_working.pt" ] && R="--resume"
    $P $T stage1 --run-dir "$M/S1_seed0" $R --epochs 100000 --archive-every 5000 $C
    $P $X --checkpoint "$M/S1_seed0/stage1_final.pt" --out-dir "$M/plots"
fi
if [ ! -f "$M/S2_seed0/stage2_final.pt" ]; then
    if [ -f "$M/S2_seed0/stage2_working.pt" ]; then
        $P $T stage2 --run-dir "$M/S2_seed0" --resume --epochs 200000 --warmup-epochs 0 \
            --archive-every 5000 $C
    else
        $P $T stage2 --run-dir "$M/S2_seed0" --epochs 200000 --warmup-epochs 0 --archive-every 5000 \
            --stage1-archive "$M/S1_seed0/archive/stage1_S1_seed0_epoch50000.pt" $C
    fi
    $P $X --checkpoint "$M/S2_seed0/stage2_final.pt" --out-dir "$M/plots"
fi
echo "done: $M"
