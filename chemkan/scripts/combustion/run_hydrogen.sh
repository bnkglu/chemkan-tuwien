#!/usr/bin/env bash
# The hydrogen main run of results/hydrogen/, stage by stage:
#   Stage 1, 100k epochs  ->  Stage 2, 100k epochs from the 100k Stage-1 archive  ->
#   copies of Stage 2 continued to 150k and to 200k epochs (results/hydrogen/S2_seed0_150k,
#   S2_seed0_200k), each stage evaluated with combustion/evaluate_checkpoint.py.
# Settings as recorded in results/hydrogen/*/config.json: seed 0, t_ref 6e-4 s, batched solve,
# float32, direct autograd; Adam lr 2e-3, Tsit5 rtol 1e-6 / atol 1e-8 (the trainer defaults).
#
#   bash chemkan/scripts/combustion/run_hydrogen.sh             # into runs/hydrogen/
#   OUT=<dir> PYTHON=<python> bash chemkan/scripts/combustion/run_hydrogen.sh
#
# Rerun after an interruption: a stage with an evaluation is done and skipped, a stage with a
# working checkpoint is resumed. The recorded runs were themselves interrupted and resumed
# several times (see their run.log); an uninterrupted rerun is not claimed to be bit-identical.
# The original launchers are kept unchanged in results/hydrogen/ (run_main.sh, resume_main.sh).
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=${PYTHON:-python}
T=chemkan/scripts/train_hydrogen.py
E=chemkan/scripts/combustion/evaluate_checkpoint.py
M=${OUT:-runs/hydrogen}
C="--seed 0 --t-ref 6e-4 --solve-mode batched --dtype float32 --sensitivity direct_autograd"
mkdir -p "$M/eval"

resume_flag() { [ -f "$1" ] && echo "--resume" || true; }    # resume_flag <working checkpoint>

if [ ! -f "$M/eval/S1_seed0.json" ]; then
    $P $T stage1 --run-dir "$M/S1_seed0" --epochs 100000 $(resume_flag "$M/S1_seed0/stage1_working.pt") $C
    $P $E --checkpoint "$M/S1_seed0/stage1_final.pt" --out "$M/eval/S1_seed0.json"
fi
if [ ! -f "$M/eval/S2_seed0.json" ]; then
    if [ -f "$M/S2_seed0/stage2_working.pt" ]; then
        $P $T stage2 --run-dir "$M/S2_seed0" --resume --epochs 100000 --warmup-epochs 0 $C
    else
        $P $T stage2 --run-dir "$M/S2_seed0" --epochs 100000 --warmup-epochs 0 $C \
            --stage1-archive "$M/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt"
    fi
    $P $E --checkpoint "$M/S2_seed0/stage2_final.pt" --out "$M/eval/S2_seed0.json"
fi
for N in 150 200; do                         # continue a copy of the previous Stage 2
    SRC=$([ $N = 150 ] && echo "$M/S2_seed0" || echo "$M/S2_seed0_150k")
    DST="$M/S2_seed0_${N}k"
    [ -f "$M/eval/S2_seed0_${N}k.json" ] && continue
    [ -d "$DST" ] || cp -Rp "$SRC" "$DST"
    $P $T stage2 --run-dir "$DST" --resume --epochs ${N}000 --warmup-epochs 0 $C
    $P $E --checkpoint "$DST/stage2_final.pt" --out "$M/eval/S2_seed0_${N}k.json"
done
echo "done: $M"
