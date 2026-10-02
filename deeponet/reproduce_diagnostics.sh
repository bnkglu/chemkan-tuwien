#!/usr/bin/env bash
# Reproduce the biodiesel DeepONet diagnostics of 2026-10-01 (results/experiments/biodiesel/
# deeponet_diagnostics/) and the tanh DeepONet figure runs, one training at a time.
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
# Not reproducible with the current trainer (its t/t_end option was removed): the historical
# time-scaling arms fig5_0pct_10k/arm5 (ReLU, t/30) and arm7 (tanh, t/30); their configs,
# metrics and summaries are committed.

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
r5_init() {                   # the PyTorch-default initial tensors (arm4 seeds 0-4, Stage 4 R5 5-29)
    if (( $1 < 5 )); then echo "$D/fig5_0pct_10k/arm4_torchinit_lr2e-3/seed$1/checkpoint_init.pt"
    else echo "$D/relu_reachability/stage4_tail/R5/seed$1/checkpoint_init.pt"; fi
}

# ---------------------------------------------------------------------------- targets
first_diagnostics() {   # ReLU, Glorot normal + zero bias, lr 1e-3, 2.5k: raw vs normalized states
    run $D/A_raw_states_mean_mse_seed0 --state-mode raw_states --epochs 2500 --seed 0
    run $D/B_author_global_normalized_states_mean_mse_seed0 --epochs 2500 --seed 0
    run $D/C_author_global_normalized_states_eq18_obs_seed0 --objective eq18_obs --epochs 2500 --seed 0
}
seed_sweep() {          # ReLU, Glorot normal + zero bias, lr 1e-3, 2.5k, seeds 0-9
    for s in {0..9}; do run $D/seed_sweep/normalized_mean_mse_seed$s --epochs 2500 --seed $s; done
}
init_ablation() {       # four initializers, ReLU, lr 1e-3, 2.5k, seeds 0-9 (arm A = seed_sweep)
    for arm in Aprime_tf_truncated_glorot_zero_bias C_xavier_normal_uniform_bias B_lux_default B_torch_default; do
        for s in {0..9}; do run $D/init_ablation/$arm/seed$s --init $arm --epochs 2500 --seed $s; done
    done
}
relu_10k() {            # arms 1-4: R1 / PyTorch-default init x lr 1e-3 / 2e-3, ReLU, 10k, seeds 0-4
    for s in {0..4}; do
        run $D/fig5_0pct_10k/arm1_refinit_lr1e-3/seed$s "${R1[@]}" --lr 1e-3 --epochs 10000 --seed $s
        run $D/fig5_0pct_10k/arm2_refinit_lr2e-3/seed$s "${R1[@]}" --lr 2e-3 --epochs 10000 --seed $s
        run $D/fig5_0pct_10k/arm3_torchinit_lr1e-3/seed$s "${TORCH[@]}" --lr 1e-3 --epochs 10000 --seed $s
        run $D/fig5_0pct_10k/arm4_torchinit_lr2e-3/seed$s "${TORCH[@]}" --lr 2e-3 --epochs 10000 --seed $s
    done
    step "$PY" $D/fig5_0pct_10k/analyze10k.py
}
tanh_torchinit() {      # arm6: tanh, PyTorch-default init, lr 2e-3, 10k, seeds 0-4 (needs relu_10k)
    for s in {0..4}; do
        run $D/fig5_0pct_10k/arm6_tanh_torchinit_lr2e-3_raw_t/seed$s "${TORCH[@]}" --lr 2e-3 \
            --activation tanh --epochs 10000 --seed $s --init-from "$(r5_init $s)"
    done
}
lbfgs() {               # L-BFGS continuation of the arm-4 final checkpoints (needs relu_10k)
    step "$PY" $D/fig5_0pct_10k/lbfgs_from_arm4.py
}
reachability_stage1() { # branch-input modes, PyTorch-default init, lr 2e-3, 2.5k (needs relu_10k)
    for mode in normalized raw raw_species_norm_T; do
        for s in {0..4}; do
            run $D/relu_reachability/stage1_branch/$mode/seed$s "${TORCH[@]}" --lr 2e-3 \
                --branch-mode $mode --epochs 2500 --seed $s --init-from "$(r5_init $s)"
        done
    done
    (cd $D/relu_reachability && step "$PY" stage1_report.py)
}
reachability_stage2() { # shared-knot capacity fit (no training) -> stage2_knots/knots.json
    (cd $D/relu_reachability/stage2_knots && step "$PY" fit_shared_knots.py)
}
reachability_stage3() { # 10-seed screen at 2.5k (needs init_ablation, reachability_stage1)
    for s in {0..9}; do
        run $D/relu_reachability/stage3_screen/R2_refinit_lr2e-3/seed$s "${R1[@]}" --lr 2e-3 --branch-mode normalized --epochs 2500 --seed $s
        run $D/relu_reachability/stage3_screen/R3_glorot_uniform_lr1e-3/seed$s --init R3_glorot_uniform_zero_bias --lr 1e-3 --branch-mode normalized --epochs 2500 --seed $s
        run $D/relu_reachability/stage3_screen/R4_glorot_uniform_lr2e-3/seed$s --init R3_glorot_uniform_zero_bias --lr 2e-3 --branch-mode normalized --epochs 2500 --seed $s
    done
    for s in {5..9}; do
        run $D/relu_reachability/stage3_screen/R5_torchinit_lr2e-3/seed$s "${TORCH[@]}" --lr 2e-3 --branch-mode normalized --epochs 2500 --seed $s
    done
    (cd $D/relu_reachability && step "$PY" stage3_report.py)
}
reachability_stage4() { # 30-seed tails at 10k, seeds 5-29 (seeds 0-4 = arm1 / arm4; needs relu_10k)
    for s in {5..29}; do
        run $D/relu_reachability/stage4_tail/R1/seed$s "${R1[@]}" --lr 1e-3 --branch-mode normalized --epochs 10000 --seed $s
    done
    for s in {5..29}; do
        run $D/relu_reachability/stage4_tail/R5/seed$s "${TORCH[@]}" --lr 2e-3 --branch-mode normalized --epochs 10000 --seed $s
    done
    (cd $D/relu_reachability && step "$PY" stage4_report.py)
}
reachability_stage5() { # planted K = 3 / 4 kinks, DIAGNOSTIC (needs reachability_stage2)
    for k in 3 4; do
        for s in {0..4}; do
            run $D/relu_reachability/stage5_planted/P$k/seed$s "${R1[@]}" --lr 2e-3 --branch-mode normalized \
                --epochs 10000 --seed $s --plant-knots $D/relu_reachability/stage2_knots/knots.json --plant-k $k
        done
    done
    (cd $D/relu_reachability && step "$PY" stage5_report.py)
}
final_constructive() {  # F1 factorized spline, F2A/F2B/F3 frozen-hinge networks (needs stage2)
    (cd $D/relu_reachability/final_constructive && step "$PY" final_constructive.py f1 \
        && step "$PY" final_constructive.py neural)
}
activation_placement() { # placements B/C/D under R1 + B under PyTorch init (needs relu_10k, stage4)
    for p in B C D; do
        for s in {0..9}; do
            run $D/activation_placement/R1/$p/seed$s "${R1[@]}" --lr 1e-3 --branch-mode normalized \
                --epochs 10000 --seed $s --placement $p --init-from "$(r1_init $s)"
        done
    done
    for s in {0..9}; do
        run $D/activation_placement/R5_control/B/seed$s "${TORCH[@]}" --lr 2e-3 --branch-mode normalized \
            --epochs 10000 --seed $s --placement B --init-from "$(r5_init $s)"
    done
    step "$PY" $D/activation_placement/analyze_placement.py     # needs relu_10k, reachability_stage4
}
tanh_r1() {             # R1 + tanh, lr 1e-3, seeds 0-9, paired R1 inits (needs relu_10k, stage4)
    for s in {0..9}; do
        run $D/tanh_R1/seed$s "${R1[@]}" --lr 1e-3 --activation tanh --branch-mode normalized \
            --epochs 10000 --seed $s --init-from "$(r1_init $s)"
    done
    run $D/tanh_R1_no_initfrom/seed0 "${R1[@]}" --lr 1e-3 --activation tanh --branch-mode normalized \
        --epochs 10000 --seed 0
}
tanh_r1_lr1e-2() {      # main DeepONet: R1 + tanh, lr 1e-2, seeds 0-9 (needs relu_10k, stage4)
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

TARGETS=(first_diagnostics seed_sweep init_ablation relu_10k tanh_torchinit lbfgs
         reachability_stage1 reachability_stage2 reachability_stage3 reachability_stage4
         reachability_stage5 final_constructive activation_placement tanh_r1 tanh_r1_lr1e-2
         figure_runs_lr1e-2 figure_runs_lr1e-3)

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
