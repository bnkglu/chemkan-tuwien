#!/bin/bash
# Figure-4 comparison data: fixed n_mu=2 at every ChemKAN width.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${CHEMKAN_PYTHON:-python3}" "$SCRIPT_DIR/_training.py" nmu2 "$@"
