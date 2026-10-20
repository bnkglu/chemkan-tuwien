#!/usr/bin/env bash
# DeepONet figure runs of the biodiesel figures (results/experiments/biodiesel/author_repo_match/
# figures_tanh_r1_lr1e-2/): the reference DeepONet settings (Lu/DeepXDE truncated Glorot normal,
# zero biases, raw t, author-global normalized states, mean-MSE objective) with tanh and Adam
# lr 1e-2, seed 0, biodiesel_v2. Fig. 5: 8 noise levels, 10,000 epochs; Fig. 4: 6 sizes, 50,000
# epochs. The commands are those of the figure_runs target that produced the runs; the ChemKAN
# figure runs are launched by chemkan/scripts/author_repo_match/run_figures.py.
#
#   bash deeponet/run_figure_runs.sh            # run every missing run, one at a time
#   DRY_RUN=1 bash deeponet/run_figure_runs.sh  # print the commands only
#
# Run from anywhere; PYTHON selects the interpreter (default: python). A run whose directory
# already has checkpoint_final.pt is skipped, so an interrupted batch can be restarted.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
T=deeponet/train_biodiesel_deeponet_repro.py
OUT=results/experiments/biodiesel/author_repo_match/figures_tanh_r1_lr1e-2
BASE=(--state-mode author_global_normalized_states --objective mean_mse
      --init Aprime_tf_truncated_glorot_zero_bias --lr 1e-2 --activation tanh
      --branch-mode normalized --seed 0 --report-train-only-norm)

run() {                       # run <run_dir> <trainer args...>
    local dir=$1; shift
    if [[ ${DRY_RUN:-0} == 1 ]]; then echo "+ $PY $T ${BASE[*]} $* --run-dir $dir"; return; fi
    if [[ -f $dir/checkpoint_final.pt ]]; then echo "skip (done): $dir"; return; fi
    echo "+ $PY $T ${BASE[*]} $* --run-dir $dir"
    "$PY" "$T" "${BASE[@]}" "$@" --run-dir "$dir"
}

for p in 0 1 2 3 5 7 10 15; do
    run $OUT/fig05/deeponet_noise$(printf %02d $p)_seed0 --noise-percent $p --epochs 10000
done
for spec in "78 3 3" "156 5 5" "249 7 6" "308 8 7" "384 9 9" "456 10 10"; do
    set -- $spec
    run $OUT/fig04/deeponet_p$(printf %03d $1)_seed0 --width $2 --trunk-hidden $3 --epochs 50000
done
