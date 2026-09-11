"""Reproduce ChemKAN Figure 8B -- hydrogen ignition delay.

Ignition delay against equivalence ratio at several initial temperatures, comparing the
Cantera reference with the ChemKAN prediction, for two Stage-2 initializations. The model's
marker is absent wherever it fails to ignite.

Reads the precomputed ignition-delay tables; it neither trains nor re-integrates.

    python chemkan/scripts/figures/fig08_hydrogen_ignition.py
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES_HYDROGEN, TABLES, require_file, save_figure, use_headless_backend

DEFAULT_STEMS = {
    "H0 (primary, random init)": "random_stage2_10000_seed0",
    "Hnorm1 (labelled init comparison)": "normmatched_dir1_stage2_10000",
}


def load_delays(csv_path):
    """One ignition-delay table. Rows are the reference-igniting conditions only."""
    with open(require_file(csv_path, "ignition-delay table"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    return rows


def summarize(rows):
    """Ignition counts, status tally and the relative delay error where the model ignites."""
    relative = [abs(float(r["relative_error"])) for r in rows if r["relative_error"]]
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {"conditions": len(rows),
            "ignited": sum(1 for r in rows if r["status"].startswith("ignited")),
            "statuses": statuses,
            "median_relative_error": float(np.median(relative)) if relative else None,
            "max_relative_error": float(max(relative)) if relative else None}


def plot_figure(tables):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (label, rows) in zip(axes, tables.items()):
        temperatures = sorted({float(r["T0_K"]) for r in rows})
        colours = plt.cm.viridis(np.linspace(0, .9, len(temperatures)))
        n_ignited = sum(1 for r in rows if r["status"].startswith("ignited"))
        for colour, T0 in zip(colours, temperatures):
            at_T0 = [r for r in rows if float(r["T0_K"]) == T0]
            ax.plot([float(r["phi"]) for r in at_T0],
                    [float(r["reference_delay_s"]) * 1e3 for r in at_T0],
                    "p-", color=colour, ms=8, label=f"ref {T0:.0f} K")
            ignited = [r for r in at_T0 if r["chemkan_delay_s"]]
            if ignited:
                ax.plot([float(r["phi"]) for r in ignited],
                        [float(r["chemkan_delay_s"]) * 1e3 for r in ignited],
                        "^--", color=colour, ms=8, mfc="none", label=f"ChemKAN {T0:.0f} K")
        ax.set_xlabel("equivalence ratio")
        ax.grid(alpha=.3)
        ax.set_title(f"{label}\nmodel ignites in {n_ignited}/{len(rows)} "
                     f"reference-igniting cases", fontsize=9)
    axes[0].set_ylabel("ignition delay [ms]")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Fig. 8B - pentagon = Cantera reference, open triangle = ChemKAN "
                 "(absent where the model does not ignite)")
    fig.tight_layout()
    return fig


def make_figure(table_paths=None, output_path=None, *, show=False):
    """Paper Figure 8B. Returns ``(fig, results)`` with a summary per initialization."""
    paths = {label: TABLES / f"hydrogen_ignition_delay_{stem}.csv"
             for label, stem in DEFAULT_STEMS.items()}
    paths.update(table_paths or {})
    tables = {label: load_delays(path) for label, path in paths.items()}
    fig = plot_figure(tables)
    save_figure(fig, output_path, dpi=200)
    if show:
        plt.show()
    return fig, {label: summarize(rows) for label, rows in tables.items()}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=FIGURES_HYDROGEN / "fig08b_hydrogen_ignition_delay")
    args = p.parse_args()
    _, results = make_figure(output_path=args.output)
    for label, s in results.items():
        print(f"{label}: {s['conditions']} reference-igniting conditions | "
              f"statuses {s['statuses']}")
        if s["median_relative_error"] is not None:
            print(f"    |relative delay error|: median {s['median_relative_error']:.1%}  "
                  f"max {s['max_relative_error']:.1%}")
    print(f"wrote {args.output}.pdf/.png")


if __name__ == "__main__":
    main()
