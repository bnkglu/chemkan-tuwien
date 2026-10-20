"""Figure 8B -- hydrogen ignition delay.

Ignition delay against equivalence ratio at several initial temperatures, comparing the
Cantera reference with the ChemKAN prediction. Default table: the main run's 200k Stage-2
checkpoint (notebook 08b's arguments); ``--table`` plots others instead. The model's
marker is absent wherever it fails to ignite.

Reads the precomputed ignition-delay tables; it neither trains nor re-integrates.

    python chemkan/scripts/figures/fig08_hydrogen_ignition.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES_HYDROGEN,
    HYDROGEN_LABEL,
    TABLES_HYDROGEN,
    require_file,
    save_figure,
    use_headless_backend,
)

DEFAULT_TABLES = {HYDROGEN_LABEL: TABLES_HYDROGEN / f"hydrogen_ignition_delay_{HYDROGEN_LABEL}.csv"}


def load_delays(csv_path):
    """One ignition-delay table. Rows are the reference-igniting conditions only."""
    with open(require_file(csv_path, "ignition-delay table"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    return rows


def summarize(rows):
    """Delay error and the neutral diagnostics, over every evaluated condition.

    No ignition threshold is applied anywhere: the delay is argmax dT/dt for each of the
    paper's 30 conditions. The temperature rise and peak dT/dt are reported beside it, so
    a flat trajectory posting a small delay error cannot be read as an ignition.
    """
    relative = [abs(float(r["relative_error"])) for r in rows if r["relative_error"]]
    rise_model = [float(r["chemkan_temperature_rise_K"]) for r in rows
                  if r["chemkan_temperature_rise_K"]]
    rise_ref = [float(r["reference_temperature_rise_K"]) for r in rows
                if r["reference_temperature_rise_K"]]
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {"conditions": len(rows),
            "statuses": statuses,
            "median_model_temperature_rise_K": float(np.median(rise_model)) if rise_model else None,
            "median_reference_temperature_rise_K": float(np.median(rise_ref)) if rise_ref else None,
            "median_relative_error": float(np.median(relative)) if relative else None,
            "max_relative_error": float(max(relative)) if relative else None}


def plot_figure(tables):
    fig, axes = plt.subplots(1, len(tables), figsize=(7 * len(tables), 4.6), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (label, rows) in zip(axes, tables.items()):
        temperatures = sorted({float(r["T0_K"]) for r in rows})
        colours = plt.cm.viridis(np.linspace(0, .9, len(temperatures)))
        rise_model = np.median([float(r["chemkan_temperature_rise_K"]) for r in rows
                                if r["chemkan_temperature_rise_K"]])
        rise_ref = np.median([float(r["reference_temperature_rise_K"]) for r in rows
                              if r["reference_temperature_rise_K"]])
        for colour, T0 in zip(colours, temperatures):
            at_T0 = [r for r in rows if float(r["T0_K"]) == T0]
            ax.plot([float(r["phi"]) for r in at_T0],
                    [float(r["reference_delay_s"]) * 1e3 for r in at_T0],
                    "p-", color=colour, ms=8, label=f"ref {T0:.0f} K")
            # Every evaluated condition is plotted: the delay is argmax dT/dt with no
            # threshold deciding whether it "counts". The panel subtitle carries the
            # temperature rise, which is what separates an ignition from a flat curve.
            drawn = [r for r in at_T0 if r["chemkan_delay_s"]]
            if drawn:
                ax.plot([float(r["phi"]) for r in drawn],
                        [float(r["chemkan_delay_s"]) * 1e3 for r in drawn],
                        "^--", color=colour, ms=8, mfc="none", label=f"ChemKAN {T0:.0f} K")
        ax.set_xlabel("equivalence ratio")
        ax.grid(alpha=.3)
        ax.set_title(f"{label}\nmedian temperature rise: model {rise_model:.0f} K vs "
                     f"reference {rise_ref:.0f} K", fontsize=9)
    axes[0].set_ylabel("ignition delay [ms]")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Fig. 8B - pentagon = Cantera reference, open triangle = ChemKAN\n"
                 "delay = argmax dT/dt (paper Sec. III B) for every evaluated condition; "
                 "read it with the temperature rise in each panel title", fontsize=10)
    fig.tight_layout()
    return fig


def make_figure(table_paths=None, output_path=None, *, show=False):
    """Paper Figure 8B. Returns ``(fig, results)`` with a summary per model.

    ``table_paths`` (label -> ignition-delay CSV) replaces the default models.
    """
    paths = dict(table_paths) if table_paths else dict(DEFAULT_TABLES)
    tables = {label: load_delays(path) for label, path in paths.items()}
    fig = plot_figure(tables)
    save_figure(fig, output_path, dpi=200)
    if show:
        plt.show()
    return fig, {label: summarize(rows) for label, rows in tables.items()}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=FIGURES_HYDROGEN / "fig08b_hydrogen_ignition_delay_200k")
    p.add_argument("--table", action="append", default=[], metavar="LABEL=CSV",
                   help="plot this ignition-delay table instead of the default one "
                        "(repeatable; from evaluate_hydrogen_ignition.py)")
    args = p.parse_args()
    tables = {k: Path(v) for k, v in (t.split("=", 1) for t in args.table)}
    _, results = make_figure(table_paths=tables, output_path=args.output)
    for label, s in results.items():
        print(f"{label}: {s['conditions']} evaluated conditions (the paper's set) | "
              f"statuses {s['statuses']}")
        if s["median_relative_error"] is not None:
            # Always printed together: a small delay error on a flat trajectory is not
            # an ignition, and the rise is what says so.
            print(f"    |relative delay error|: median {s['median_relative_error']:.1%}  "
                  f"max {s['max_relative_error']:.1%}")
            print(f"    median temperature rise: model "
                  f"{s['median_model_temperature_rise_K']:.0f} K vs reference "
                  f"{s['median_reference_temperature_rise_K']:.0f} K")
    print(f"wrote {args.output}.pdf/.png")


if __name__ == "__main__":
    main()
