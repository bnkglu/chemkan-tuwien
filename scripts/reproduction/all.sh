#!/bin/bash
# The whole reproduction: biodiesel Figures 3-6, then hydrogen Figures 7-8 and Table I.
# Idempotent - completed runs are skipped, never overwritten.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
FLAGS=(); [ "$DRY_RUN" = "1" ] && FLAGS+=(--dry-run); [ "$NO_RENDER" = "1" ] && FLAGS+=(--no-render)
"$D/biodiesel/all.sh" "${FLAGS[@]}"
"$D/hydrogen/all.sh"  "${FLAGS[@]}"
say "Full comparison"
info "${TABLES#"$REPO"/}/reproduction_comparison.csv - Figures 3-8 and Table I,"
info "with 'evaluation completed' kept distinct from 'paper result matched'."
