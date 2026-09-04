#!/bin/bash
# Progress of the base-ON (N=4) hydrogen matrix. Read-only; safe to run any time.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
B="$R/results/reproduction/chemkan/hydrogen/diagnostics/base_on_n4"
ARMS=(stage1_seed0 random_stage2_10000_seed0 cantera_stage2_10000_seed0 \
      scaled_random_1e5_dir0_stage2_10000 normmatched_dir0_stage2_10000 \
      normmatched_dir1_stage2_10000 normmatched_dir2_stage2_10000)

printf '%-38s %-9s %8s %12s %10s\n' RUN STAGE EPOCHS LOSS STATUS
for a in "${ARMS[@]}"; do
  d="$B/$a"
  if [ ! -d "$d" ]; then printf '%-38s %-9s %8s %12s %10s\n' "$a" - - - queued; continue; fi
  if   [ -s "$d/history_stage2.csv" ] && [ "$a" != stage1_seed0 ]; then h="$d/history_stage2.csv"; st=stage2
  elif [ -s "$d/history_stage1.csv" ]; then h="$d/history_stage1.csv"; st=stage1
  else printf '%-38s %-9s %8s %12s %10s\n' "$a" starting - - running; continue; fi
  n=$(( $(wc -l < "$h") - 1 ))
  loss=$(tail -1 "$h" | cut -d, -f2 | cut -c1-11)
  if [ -f "$d/checkpoint_final.pt" ]; then s=DONE; else s=running; fi
  printf '%-38s %-9s %8s %12s %10s\n' "$a" "$st" "$n" "$loss" "$s"
done
echo
echo "active process:"; pgrep -fl train_hydrogen.py | sed 's/.*--run-dir /  -> /' | head -2
