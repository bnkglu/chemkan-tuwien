"""Regenerate every final ChemKAN paper figure.

Biodiesel Figs. 3, 4, 5A, 5B, 6 from the biodiesel_v2 figure runs (time-averaged MSE) into
``results/biodiesel/figures/``; hydrogen Figs. 7, 8A,
8B from the main run's 200k Stage-2 checkpoint into
``results/hydrogen/figures/`` (notebook hydrogen_reproduction's arguments).
Calls each figure's own function; there is no separate implementation here. Every figure
reads finished checkpoints, histories and tables -- nothing trains.

    python chemkan/scripts/figures/plot_all.py
    python chemkan/scripts/figures/plot_all.py --only fig07 fig08a
"""

from __future__ import annotations

import argparse
import traceback

import fig03_biodiesel_trajectories as fig03
import fig04_biodiesel_scaling as fig04
import fig05_biodiesel_loss as fig05b
import fig05_biodiesel_noise as fig05a
import fig06_biodiesel_profiles as fig06
import fig07_hydrogen_trajectories as fig07
import fig08_hydrogen_generalization as fig08a
import fig08_hydrogen_ignition as fig08b
from common import FIGURES_HYDROGEN, TABLES_HYDROGEN, use_headless_backend

FIGURES = {
    "fig03": fig03.make_author_figure,
    "fig04": fig04.make_author_figure,
    "fig05a": fig05a.make_author_figure,
    "fig05b": fig05b.make_author_figure,
    "fig06": fig06.make_author_figure,
    "fig07": lambda: fig07.make_figure(output_dir=FIGURES_HYDROGEN,
                                       table_path=TABLES_HYDROGEN / "fig07_per_state_mse_200k.csv",
                                       time_averaged=True),
    "fig08a": lambda: fig08a.make_figure(output_dir=FIGURES_HYDROGEN / "fig08a_200k",
                                         time_averaged=True),
    "fig08b": lambda: fig08b.make_figure(
        output_path=FIGURES_HYDROGEN / "fig08b_hydrogen_ignition_delay_200k"),
}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES),
                   help="regenerate a subset (default: all)")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a figure fails, and report at the end")
    args = p.parse_args()
    failures = []
    for name in args.only:
        print(f"--- {name} ---", flush=True)
        try:
            FIGURES[name]()
        except Exception as error:
            if not args.keep_going:
                raise
            failures.append((name, error))
            traceback.print_exc()
    if failures:
        print(f"\n{len(failures)} figure(s) failed: "
              f"{', '.join(name for name, _ in failures)}")
        raise SystemExit(1)
    print(f"\nregenerated {len(args.only)} figure(s)")


if __name__ == "__main__":
    main()
