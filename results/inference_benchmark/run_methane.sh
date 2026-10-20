#!/bin/bash
# Methane timing runs (run from the repository root, nothing else training): the selected
# 180k snapshot in Python and Julia, the Julia references, and the final 200k model in Python
# for comparison; then the report.
set -e
B=results/experiments/inference_benchmark
P=${PYTHON:-$HOME/uni_projects/chemkan-venv/bin/python}
J="julia --project=$B/julia -t 1"
echo "START $(date)"
$P $B/julia/make_arrhenius_mech.py gri30.yaml
$P $B/julia/export_inputs.py methane
echo "== python methane (180k) $(date)"
$P $B/benchmark.py --system methane > $B/run_methane.log 2>&1
echo "== julia chemkan methane $(date)"
$J $B/julia/chemkan_julia.jl methane > $B/julia/run_chemkan_methane.log 2>&1
echo "== julia reference methane $(date)"
$J $B/julia/reference_julia.jl methane > $B/julia/run_reference_methane.log 2>&1
echo "== python methane final 200k $(date)"
$P $B/benchmark.py --system methane --out-dir $B/methane_final200k \
   --checkpoint results/experiments/methane/author_matched/main/S2_seed0/stage2_final.pt > $B/run_methane_final200k.log 2>&1
$P $B/make_report.py
echo "DONE $(date)"
