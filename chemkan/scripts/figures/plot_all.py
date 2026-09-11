"""Regenerate every ChemKAN paper figure into the reproduction results directory.

Calls each figure's own ``make_figure``; there is no separate implementation here. Every
figure reads finished checkpoints, histories and tables -- nothing trains.

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
from common import (
    FIGURES_BIODIESEL,
    FIGURES_HYDROGEN,
    TABLES,
    loss_reduction,
    use_headless_backend,
)

# Figures that carry a loss value accept --time-averaged; the flag divides the DISPLAYED
# loss by N_t (30 biodiesel, 50 hydrogen) and writes a "_time_averaged" companion file.
# Figures 6 and 8B show no loss, so they have no such variant.
TIME_AVERAGED = False


def _suffix():
    return loss_reduction(0, TIME_AVERAGED)[2]


def _fig03():
    fig03.make_figure(
        output_path=f"{FIGURES_BIODIESEL / 'fig03_biodiesel_noise_columns'}{_suffix()}",
        time_averaged=TIME_AVERAGED)
    fig03.make_clean_column_figure(
        output_path=f"{FIGURES_BIODIESEL / 'fig03_biodiesel_clean_column'}{_suffix()}",
        time_averaged=TIME_AVERAGED)


def _fig04():
    fig04.make_figure(n_mu="scaled", time_averaged=TIME_AVERAGED)
    fig04.make_figure(n_mu="2", time_averaged=TIME_AVERAGED)


def _fig06():
    fig06.make_figure(output_path=FIGURES_BIODIESEL / "fig06_biodiesel_15pct_profiles",
                      metrics_path=TABLES / "biodiesel_fig6_profile_metrics.json")
    fig06.make_case0_figure(
        output_path=FIGURES_BIODIESEL / "fig06_biodiesel_15pct_profiles_case0",
        metrics_path=TABLES / "biodiesel_fig6_profile_metrics_case0.json")


def _fig07():
    """Both Figure-7 views; the companion reuses the main view's evaluation."""
    _, results = fig07.make_figure(
        output_dir=FIGURES_HYDROGEN,
        table_path=TABLES / "fig07_hydrogen_per_state_mse.csv",
        time_averaged=TIME_AVERAGED)
    if not TIME_AVERAGED:      # the companion shows mass fractions, not loss
        fig07.make_true_scale_figure(output_dir=FIGURES_HYDROGEN, precomputed=results)


FIGURES = {
    "fig03": _fig03,
    "fig04": _fig04,
    "fig05a": lambda: fig05a.make_figure(
        output_path=f"{FIGURES_BIODIESEL / 'fig05a_biodiesel_noise_robustness'}{_suffix()}",
        time_averaged=TIME_AVERAGED),
    "fig05b": lambda: fig05b.make_figure(
        output_path=f"{FIGURES_BIODIESEL / 'fig05b_biodiesel_loss_dynamics'}{_suffix()}",
        time_averaged=TIME_AVERAGED),
    "fig06": _fig06,
    "fig07": _fig07,
    "fig08a": lambda: fig08a.make_figure(output_dir=FIGURES_HYDROGEN,
                                         time_averaged=TIME_AVERAGED),
    "fig08b": lambda: fig08b.make_figure(
        output_path=FIGURES_HYDROGEN / "fig08b_hydrogen_ignition_delay"),
}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES),
                   help="regenerate a subset (default: all)")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a figure fails, and report at the end")
    p.add_argument("--time-averaged", action="store_true",
                   help="also-available display convention: divide the plotted loss by "
                        "N_t and write '_time_averaged' companions (figures 3, 4, 5A, 5B, "
                        "7, 8A; 6 and 8B show no loss)")
    args = p.parse_args()
    global TIME_AVERAGED
    TIME_AVERAGED = args.time_averaged
    if TIME_AVERAGED:
        args.only = [n for n in args.only if n not in ("fig06", "fig08b")]

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
    print(f"\nregenerated {len(args.only)} figure group(s)")


if __name__ == "__main__":
    main()
