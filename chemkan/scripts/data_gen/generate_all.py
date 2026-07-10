"""
Regenerate the ChemKAN reproduction datasets from scratch.

By default this runs only the two reproduction datasets (biodiesel and
hydrogen). The hydrogen fine-grid generalization set (hydrogen_fine.npz) is
optional; pass --include-fine to also generate it. Methane is an optional
extension and is not part of the original ChemKAN reproduction, so it is not
run here -- generate it separately with
`python extensions/generate_methane.py` if needed.

This just runs the generators in turn and prints the output path and runtime
for each.

    python generate_all.py --out-dir ../../data/generated
    python generate_all.py --out-dir ../../data/generated --include-fine
    python generate_all.py --out-dir ../../data/generated --skip biodiesel
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SEED = 0

# Default reproduction datasets: biodiesel + hydrogen.
JOBS = {
    "biodiesel": ["generate_biodiesel.py", "--seed", str(SEED)],
    "hydrogen": ["generate_hydrogen.py", "--grid", "coarse"],
}

# Optional hydrogen fine grid (Fig. 8A generalization set); only run with --include-fine.
FINE_JOB = ("hydrogen-fine", ["generate_hydrogen.py", "--grid", "fine"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/generated"))
    p.add_argument("--skip", action="append", default=[], choices=list(JOBS))
    p.add_argument("--include-fine", action="store_true",
                   help="also generate the optional hydrogen_fine.npz")
    cfg = p.parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = dict(JOBS)
    if cfg.include_fine:
        jobs[FINE_JOB[0]] = FINE_JOB[1]

    here = Path(__file__).parent

    for name, cmd in jobs.items():
        if name in cfg.skip:
            print(f"[skip] {name}")
            continue
        out = cfg.out_dir / f"{name.replace('-', '_')}.npz"
        print(f"\n[{name}] -> {out}", flush=True)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(here / cmd[0]), "--out", str(out), *cmd[1:]],
                       check=True)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s  ({out})")


if __name__ == "__main__":
    raise SystemExit(main())
