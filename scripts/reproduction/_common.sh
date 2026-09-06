#!/bin/bash
# Shared helpers for the per-figure reproduction scripts. Sourced, not executed.
#
# Guarantees every caller inherits:
#   * IDEMPOTENT   - a run with checkpoint_final.pt is skipped, never re-trained and never
#                    overwritten; a run with only checkpoint_resume.pt is resumed. No
#                    script here ever passes --overwrite.
#   * REPRODUCIBLE - fixed seeds, and the same CLI the workflow doc documents.
#   * HONEST       - producing a figure is not the same as matching the paper. Every
#                    script ends by pointing at the row-by-step verdict table.
#
# Environment:
#   CHEMKAN_PYTHON   interpreter to use (default: python3). Must have torch, a tsit5-capable
#                    torchdiffeq, and cantera. See README / docs/reproduction_workflow.md.
#   DRY_RUN=1        print what would run, change nothing (same as --dry-run).
#   NO_RENDER=1      skip notebook execution (same as --no-render).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${CHEMKAN_PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
NO_RENDER="${NO_RENDER:-0}"

RESULTS="$REPO/results/reproduction"
CKB="$RESULTS/chemkan/biodiesel"
DOB="$RESULTS/baselines/deeponet/biodiesel"
HYD="$RESULTS/chemkan/hydrogen/diagnostics/base_on_n4"
TABLES="$RESULTS/tables"
FIGURES="$RESULTS/figures"

# The two hydrogen checkpoints, by their documented labels.
H0_DIR="$HYD/random_stage2_10000_seed0"           # primary reproduction (fails to ignite)
HNORM1_DIR="$HYD/normmatched_dir1_stage2_10000"   # labelled initialization comparison

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=1 ;;
    --no-render) NO_RENDER=1 ;;
    -h|--help)   SHOW_HELP=1 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   !! %s\n' "$*" >&2; }

run() {                      # run <description> -- <command...>
  local desc="$1"; shift
  [ "$1" = "--" ] && shift
  if [ "$DRY_RUN" = "1" ]; then
    printf '   [dry-run] %s\n              %s\n' "$desc" "$*"
    return 0
  fi
  info "$desc"
  "$@"
}

require_python() {
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  if ! "$PY" -c "import torch" >/dev/null 2>&1; then
    warn "'$PY' cannot import torch."
    warn "Set CHEMKAN_PYTHON to the project interpreter, e.g."
    warn "    CHEMKAN_PYTHON=~/uni_projects/chemkan-venv/bin/python $0"
    exit 1
  fi
}

require_file() {             # require_file <path> <how to produce it>
  if [ ! -f "$1" ]; then
    warn "missing: $1"
    warn "produce it with: $2"
    exit 1
  fi
}

# ---------------------------------------------------------------- training helpers
# train_run <run_dir> <workdir> <command...>
#   Skips a completed run, resumes an interrupted one, otherwise starts fresh.
#   Never overwrites: a completed run is a result, not a cache.
train_run() {
  local dir="$1"; shift
  local workdir="$1"; shift
  local name; name="$(basename "$dir")"
  if [ -f "$dir/checkpoint_final.pt" ]; then
    info "skip   $name (already complete - not re-trained, not overwritten)"
    return 0
  fi
  local extra=()
  if [ -f "$dir/checkpoint_resume.pt" ]; then
    extra=(--resume)
    info "resume $name from its last snapshot"
  else
    info "train  $name"
  fi
  if [ "$DRY_RUN" = "1" ]; then
    printf '   [dry-run] (cd %s && %s --run-dir %s %s)\n' \
      "${workdir#"$REPO"/}" "$*" "${dir#"$REPO"/}" "${extra[*]-}"
    return 0
  fi
  ( cd "$workdir" && "$@" --run-dir "$dir" ${extra[@]+"${extra[@]}"} )
}

# ------------------------------------------------------------- rendering + closing
render_notebook() {          # render_notebook <notebook basename>
  if [ "$NO_RENDER" = "1" ]; then
    info "skipping notebook render (--no-render); figures on disk are unchanged"
    return 0
  fi
  run "render $1 (executes it in place; this is where the plot is produced)" -- \
    "$PY" -m jupyter nbconvert --to notebook --inplace --execute \
      --ExecutePreprocessor.timeout=2400 "$REPO/chemkan/notebooks/$1"
}

verdict() {                  # verdict <figure label as it appears in the table>
  local key="$1"
  local csv="$TABLES/reproduction_comparison.csv"
  say "Verdict for $key"
  if [ ! -f "$csv" ]; then
    warn "no comparison table yet at ${csv#"$REPO"/}"
    return 0
  fi
  "$PY" - "$csv" "$key" <<'PYEOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["figure"] == sys.argv[2]]
if not rows:
    print(f"   (no rows for {sys.argv[2]})"); raise SystemExit
for r in rows:
    print(f"   [{r['status']}]")
    print(f"      paper : {r['paper_result']} = {r['paper_value']}")
    print(f"      ours  : {r['our_result']}")
    print(f"      note  : {r['remaining_difference']}")
    print(f"      artifact: {r['artifact_path']}\n")
print("   'evaluation completed' is NOT 'paper result matched' - see the status field.")
PYEOF
}
