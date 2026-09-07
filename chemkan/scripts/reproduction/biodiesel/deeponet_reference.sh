#!/bin/bash
# Train the corrected biodiesel DeepONet only; preserve legacy runs and figures.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${CHEMKAN_PYTHON:-python3}" "$SCRIPT_DIR/_training.py" deeponet "$@"
