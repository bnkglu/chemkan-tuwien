#!/usr/bin/env bash
# Reproduce the kept biodiesel DeepONet diagnostics of 2026-10-01 (results/experiments/biodiesel/
# deeponet_diagnostics/: the ReLU reference, the ReLU 30-seed test, R1 + tanh at lr 1e-3 and 1e-2)
# and the tanh DeepONet figure runs, one training at a time.
#
#   bash deeponet/reproduce_diagnostics.sh list             # available targets
#   bash deeponet/reproduce_diagnostics.sh <target> [...]   # run targets in the given order
#   bash deeponet/reproduce_diagnostics.sh all              # every reproducible target
#   DRY_RUN=1 bash deeponet/reproduce_diagnostics.sh <target>   # print the commands only
#
# Run from the repository root. PYTHON selects the interpreter (default: python). A run whose
# run directory already has checkpoint_final.pt is skipped, so an interrupted target can be
# restarted. Targets that reuse earlier runs' initial tensors (--init-from) or results list
# their prerequisites; run those first. All runs: biodiesel_v2, 0 % noise unless stated,
# author-global normalized states, mean-MSE objective (averaged over time points), raw time t.
#
# The other diagnostics (init / lr / time-scaling / placement / reachability screens) were removed
# from the repository; their run files are in deeponet_diagnostics_backup_2026-10-02.zip (outside
# the repository). fig5_0pct_10k/summary.txt and analyze10k.py cover arms 1-4, of which arms 2-3
# are only in that backup.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
T=deeponet/train_biodiesel_deeponet_repro.py
D=results/experiments/biodiesel/deeponet_diagnostics
F=results/experiments/biodiesel/author_repo_match
BASE=(--state-mode author_global_normalized_states --objective mean_mse)
R1=(--init Aprime_tf_truncated_glorot_zero_bias)     # Lu/DeepXDE truncated Glorot normal, zero bias
TORCH=(--init B_torch_default)                       # PyTorch nn.Linear default

run() {                       # run <run_dir> <trainer args...>
    local dir=$1; shift
    if [[ ${DRY_RUN:-0} == 1 ]]; then echo "+ $PY $T ${BASE[*]} $* --run-dir $dir"; return; fi
    if [[ -f $dir/checkpoint_final.pt ]]; then echo "skip (done): $dir"; return; fi
    echo "+ $PY $T ${BASE[*]} $* --run-dir $dir"
    "$PY" "$T" "${BASE[@]}" "$@" --run-dir "$dir"
}
step() {                      # step <command...>: analysis / non-training step
    echo "+ $*"
    [[ ${DRY_RUN:-0} == 1 ]] || "$@"
}
r1_init() {                   # the R1 initial tensors of seed $1 (arm1 seeds 0-4, Stage 4 R1 5-29)
    if (( $1 < 5 )); then echo "$D/fig5_0pct_10k/arm1_refinit_lr1e-3/seed$1/checkpoint_init.pt"
    else echo "$D/relu_reachability/stage4_tail/R1/seed$1/checkpoint_init.pt"; fi
}

# ---------------------------------------------------------------------------- targets
relu_reference() {      # ReLU, reference init (R1) lr 1e-3 and PyTorch-default init lr 2e-3, 10k, seeds 0-4
    for s in {0..4}; do
        run $D/fig5_0pct_10k/arm1_refinit_lr1e-3/seed$s "${R1[@]}" --lr 1e-3 --epochs 10000 --seed $s
        run $D/fig5_0pct_10k/arm4_torchinit_lr2e-3/seed$s "${TORCH[@]}" --lr 2e-3 --epochs 10000 --seed $s
    done
}
relu_30_seeds() {       # ReLU 30-seed test at 10k, seeds 5-29 (seeds 0-4 = relu_reference) + report
    for s in {5..29}; do
        run $D/relu_reachability/stage4_tail/R1/seed$s "${R1[@]}" --lr 1e-3 --branch-mode normalized --epochs 10000 --seed $s
    done
    for s in {5..29}; do
        run $D/relu_reachability/stage4_tail/R5/seed$s "${TORCH[@]}" --lr 2e-3 --branch-mode normalized --epochs 10000 --seed $s
    done
    (cd $D/relu_reachability && step "$PY" stage4_report.py)
}
tanh_r1() {             # R1 + tanh, lr 1e-3, seeds 0-9, paired with the R1 initial tensors
    for s in {0..9}; do
        run $D/tanh_R1/seed$s "${R1[@]}" --lr 1e-3 --activation tanh --branch-mode normalized \
            --epochs 10000 --seed $s --init-from "$(r1_init $s)"
    done
    run $D/tanh_R1_no_initfrom/seed0 "${R1[@]}" --lr 1e-3 --activation tanh --branch-mode normalized \
        --epochs 10000 --seed 0
}
tanh_r1_lr1e-2() {      # main DeepONet: R1 + tanh, lr 1e-2, seeds 0-9, paired with the R1 initial tensors
    for s in {0..9}; do
        run $D/tanh_R1_lr1e-2/seed$s "${R1[@]}" --lr 1e-2 --activation tanh --branch-mode normalized \
            --epochs 10000 --seed $s --init-from "$(r1_init $s)" --report-train-only-norm
    done
}
figure_runs() {         # 07b DeepONet figure runs (seed 0): $1 = lr, $2 = output folder
    local lr=$1 out=$2
    for p in 0 1 2 3 5 7 10 15; do
        run $out/fig05/deeponet_noise$(printf %02d $p)_seed0 "${R1[@]}" --lr $lr --activation tanh \
            --branch-mode normalized --seed 0 --report-train-only-norm --noise-percent $p --epochs 10000
    done
    for spec in "78 3 3" "156 5 5" "249 7 6" "308 8 7" "384 9 9" "456 10 10"; do
        set -- $spec
        run $out/fig04/deeponet_p$(printf %03d $1)_seed0 "${R1[@]}" --lr $lr --activation tanh \
            --branch-mode normalized --seed 0 --report-train-only-norm --width $2 --trunk-hidden $3 --epochs 50000
    done
}
figure_runs_lr1e-2() { # 07b DeepONet figure runs: R1 + tanh, lr 1e-2 (the 07b default), seed 0
    figure_runs 1e-2 $F/figures_tanh_r1_lr1e-2; }
figure_runs_lr1e-3() { # the same figure runs with lr 1e-3 (author_runs variant "tanh_r1")
    figure_runs 1e-3 $F/figures_tanh_r1; }

TARGETS=(relu_reference relu_30_seeds tanh_r1 tanh_r1_lr1e-2 figure_runs_lr1e-2 figure_runs_lr1e-3)

usage() {
    echo "usage: bash $0 list | all | <target> [...]"
    echo "targets (in dependency order):"
    for t in "${TARGETS[@]}"; do
        printf '  %-22s %s\n' "$t" "$(grep -m1 "^${t}() {" "$0" | sed 's/^[^#]*# *//')"
    done
}

[[ $# -gt 0 ]] || { usage; exit 1; }
case $1 in
    list) usage ;;
    all) for t in "${TARGETS[@]}"; do echo "=== $t"; "$t"; done ;;
    *) for t in "$@"; do
           [[ " ${TARGETS[*]} " == *" $t "* ]] || { echo "unknown target: $t"; usage; exit 1; }
           echo "=== $t"; "$t"
       done ;;
esac
