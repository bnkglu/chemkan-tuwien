#!/bin/bash
# All timing runs of this benchmark, one after the other on an otherwise unloaded training
# slot. Run from the repository root. Python (PyTorch, Cantera, SciPy) and Julia
# (ChemKAN port, Arrhenius.jl, released-example kinetics), then the report.
set -e
B=results/experiments/inference_benchmark
P=${PYTHON:-$HOME/uni_projects/chemkan-venv/bin/python}
J="julia --project=$B/julia -t 1"
echo "START $(date)"
$P $B/julia/make_arrhenius_mech.py h2o2.yaml
$P $B/julia/export_inputs.py hydrogen biodiesel
for s in hydrogen biodiesel; do
  echo "== python $s $(date)"
  $P $B/benchmark.py --system $s > $B/run_$s.log 2>&1
  echo "== julia chemkan $s $(date)"
  $J $B/julia/chemkan_julia.jl $s > $B/julia/run_chemkan_$s.log 2>&1
  echo "== julia reference $s $(date)"
  $J $B/julia/reference_julia.jl $s > $B/julia/run_reference_$s.log 2>&1
done
$P $B/make_report.py
echo "DONE $(date)"
