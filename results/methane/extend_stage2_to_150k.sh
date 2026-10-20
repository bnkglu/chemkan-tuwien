#!/usr/bin/env bash
# Waits for the methane Stage-2 run (S2_seed0, target 100k) to finish. If its loss has not
# reached the 1e-4 range (median of the last 200 epochs >= 1e-3, time-averaged MSE), keeps
# the 100k model as archive/stage2_S2_seed0_epoch100000.pt, plots it, and extends the same
# run to 150,000 epochs (same settings). Run from the repository root.
set -u
cd "$(dirname "$0")/../../../.."
P=${PYTHON:-$HOME/uni_projects/chemkan-venv/bin/python}
T=chemkan/scripts/author_repo_match/train_author_matched_hydrogen.py
X=chemkan/scripts/author_repo_match/plot_trajectory.py
M=results/experiments/methane/author_matched/main
D=$M/S2_seed0
PID=${1:?pid of the running Stage-2 trainer}
echo "WAIT for pid $PID  $(date)"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "trainer exited $(date)"
# iCloud may have offloaded files: fetch what the next steps read
for f in $D/stage2_final.pt $D/stage2_working.pt $D/config.json chemkan/data/generated/methane.npz chemkan/data/generated/methane_temperature_20000.npz; do brctl download "$f" 2>/dev/null; done
find chemkan/src chemkan/scripts -name '*.py' -flags +dataless -exec brctl download {} \; 2>/dev/null
for i in $(seq 1 120); do
  n=$( { ls -lO $D/stage2_final.pt $D/stage2_working.pt $D/config.json chemkan/data/generated/methane*.npz; find chemkan/src chemkan/scripts -name '*.py' -flags +dataless; } 2>/dev/null | grep -c dataless)
  [ "$n" = 0 ] && break; sleep 10
done
E=$($P -c "import torch;print(torch.load('$D/stage2_final.pt',weights_only=False)['phase_epochs_completed'])") || { echo "no readable stage2_final.pt; stop"; exit 1; }
[ "$E" = 100000 ] || { echo "final checkpoint at $E, not 100000 (crashed?); stop"; exit 1; }
MED=$($P -c "
import csv,statistics as s
v=[float(r['mse_time_avg']) for r in csv.DictReader(open('$D/stage2_history.csv'))]
print(f'{s.median(v[-200:]):.4e}')")
echo "epoch 100000: median of last 200 epochs $MED"
if $P -c "import sys;sys.exit(0 if float('$MED')<1e-3 else 1)"; then echo "reached the 1e-4 range; not extending"; exit 0; fi
cp -n $D/stage2_final.pt $D/archive/stage2_S2_seed0_epoch100000.pt
$P $X --checkpoint $D/archive/stage2_S2_seed0_epoch100000.pt --out-dir $M/plots
echo "EXTEND to 150000 $(date)"
$P $T stage2 --run-dir $D --resume --epochs 150000 --archive-every 5000 --system methane --seed 0 \
  --solve-mode batched --dtype float32 --sensitivity direct_autograd --threads 8
echo "DONE $(date)"
