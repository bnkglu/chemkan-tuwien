"""Deprecated entry point -- Figure 4 now lives in scripts/figures/fig04_biodiesel_scaling.py.

The implementation moved there unchanged so that every paper figure is produced by one
readable script. This shim is kept because ``refresh_biodiesel_reports.py`` and the
reproduction shell wrappers invoke this path; it forwards the same command line
(``--n-mu``, ``--deeponet-version``) to the single implementation.

Prefer calling the figure script directly:

    python chemkan/scripts/figures/fig04_biodiesel_scaling.py --n-mu 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

from fig04_biodiesel_scaling import main

if __name__ == "__main__":
    main()
