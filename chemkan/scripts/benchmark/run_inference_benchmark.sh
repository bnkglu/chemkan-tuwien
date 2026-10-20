#!/usr/bin/env bash
# The inference benchmark of results/inference_benchmark/ (ChemKAN vs Cantera / Arrhenius.jl
# timing for hydrogen, biodiesel and methane), one timing run after the other. The benchmark
# is self-contained in that folder: its Python and Julia code, the pinned Julia environment
# (julia/Project.toml, julia/Manifest.toml), inputs and outputs. Running it OVERWRITES the
# folder's results with new timings, which are machine-dependent by nature.
#
#   bash chemkan/scripts/benchmark/run_inference_benchmark.sh
#   PYTHON=<python> bash chemkan/scripts/benchmark/run_inference_benchmark.sh
#
# Needs Julia (instantiate once: julia --project=results/inference_benchmark/julia -e
# 'using Pkg; Pkg.instantiate()'). Run on an otherwise idle machine. The original launchers
# are kept unchanged in that folder (run_all.sh, run_methane.sh).
set -euo pipefail
cd "$(dirname "$0")/../../.."
B=results/inference_benchmark
P=${PYTHON:-python}
J="julia --project=$B/julia -t 1"
echo "START $(date)"
$P $B/julia/make_arrhenius_mech.py h2o2.yaml
$P $B/julia/make_arrhenius_mech.py gri30.yaml
$P $B/julia/export_inputs.py hydrogen biodiesel methane
for s in hydrogen biodiesel methane; do
    echo "== python $s $(date)"
    $P $B/benchmark.py --system $s > $B/run_$s.log 2>&1
    echo "== julia chemkan $s $(date)"
    $J $B/julia/chemkan_julia.jl $s > $B/julia/run_chemkan_$s.log 2>&1
    echo "== julia reference $s $(date)"
    $J $B/julia/reference_julia.jl $s > $B/julia/run_reference_$s.log 2>&1
done
echo "== python methane, final 200k checkpoint $(date)"
$P $B/benchmark.py --system methane --out-dir $B/methane_final200k \
    --checkpoint results/methane/S2_seed0/stage2_final.pt > $B/run_methane_final200k.log 2>&1
$P $B/make_report.py
echo "DONE $(date)"
