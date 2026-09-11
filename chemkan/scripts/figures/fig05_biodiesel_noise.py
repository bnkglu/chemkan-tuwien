"""Reproduce ChemKAN Figure 5A -- biodiesel noise robustness.

Converged loss versus observation-noise level for ChemKAN and DeepONet, read from the
final checkpoints' recorded metrics. Three curves per model: the training loss and the
test loss against the noisy observations (Eq. 18), and the test loss against the clean
trajectories (Eq. 22).

Reads the metrics table only -- it neither trains nor re-evaluates checkpoints. Rebuild
that table with chemkan/scripts/diagnostics/refresh_biodiesel_reports.py if it is stale.

    python chemkan/scripts/figures/fig05_biodiesel_noise.py
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from common import (
    DATA,
    DEEPONET_VERSION,
    FIGURES_BIODIESEL,
    TABLES,
    loss_reduction,
    require_file,
    save_figure,
    use_headless_backend,
)
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

# Shared with Figure 5B, which imports them from here so the two panels of one paper
# figure cannot drift apart. Model -> metric -> colour.
FIG5_COLORS = {
    "ChemKAN": {"train_mse_noisy": "#D2BD45", "test_mse_noisy": "#4786B8",
                "test_mse_clean": "#D92F35"},
    "DeepONet": {"train_mse_noisy": "#A76540", "test_mse_noisy": "#EF8426",
                 "test_mse_clean": "#55AD5B"},
}
FIG5_LINES = {"train_mse_noisy": (0, (6, 3)),
              "test_mse_noisy": (0, (1, 1.2)), "test_mse_clean": "-"}
FIG5_LABELS = {"train_mse_noisy": "Train", "test_mse_noisy": "Noisy test",
               "test_mse_clean": "Noise-free test"}
METRICS = ("train_mse_noisy", "test_mse_noisy", "test_mse_clean")

def observation_times(npz_name):
    """Number of saved observation times N_t, read from the dataset (never hard-coded)."""
    import numpy as np
    with np.load(DATA / npz_name, allow_pickle=True) as archive:
        return len(archive["t"])


def load_metrics(metrics_path):
    """Rows of the Figure-5A metrics table, plus the noise levels actually plotted."""
    with open(require_file(metrics_path, "Figure-5A metrics table"), newline="") as f:
        rows = list(csv.DictReader(f))
    levels = sorted({int(r["noise_percent"]) for r in rows if r["role"] == "plotted"})
    return rows, levels


def validate_metrics(rows, deeponet_version):
    """The DeepONet rows must all come from the architecture this reproduction selected."""
    versions = {r["architecture_version"] for r in rows if r["model"] == "DeepONet"}
    if versions != {deeponet_version}:
        raise ValueError(f"DeepONet rows carry architecture versions {versions}, "
                         f"expected only {deeponet_version!r}")


def series(rows, levels, model, key):
    """One metric across the plotted noise levels, in level order."""
    plotted = [r for r in rows if r["role"] == "plotted" and r["model"] == model]
    return [float(next(r for r in plotted if int(r["noise_percent"]) == p)[key])
            for p in levels]


def plot_figure(rows, levels, divisor=1.0, note=""):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for model, marker in (("ChemKAN", "o"), ("DeepONet", "^")):
        for key in METRICS:
            ax.plot(levels, [v / divisor for v in series(rows, levels, model, key)],
                    color=FIG5_COLORS[model][key], ls=FIG5_LINES[key], lw=2.2,
                    marker=marker, ms=6, label=f"{model}: {FIG5_LABELS[key]}")
    ax.set_xlabel("Observation noise [%]", fontsize=12)
    ax.set_ylabel(f"Converged loss\nNormalized trajectory MSE{note}", fontsize=12)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(
        base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
    ax.set_xticks(levels)
    ax.set_xlim(min(levels) - 0.25, max(levels) + 0.25)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_title("Fig. 5A - noise robustness (corrected DeepONet, final checkpoints)",
                 fontsize=13)
    # Matplotlib fills columns in order: ChemKAN on the left, DeepONet on the right.
    ax.legend(ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              handlelength=3.5, columnspacing=2.2, framealpha=1)
    ax.grid(axis="y", which="major", alpha=0.25)
    fig.tight_layout()
    return fig


def make_figure(metrics_path=None, output_path=None, *,
                deeponet_version=DEEPONET_VERSION, time_averaged=False, show=False):
    """Paper Figure 5A. Returns ``(fig, results)`` with the plotted series per model."""
    metrics_path = metrics_path or TABLES / "biodiesel_fig5a_metrics.csv"
    rows, levels = load_metrics(metrics_path)
    validate_metrics(rows, deeponet_version)
    n_times = observation_times("biodiesel.npz")
    divisor, note, _ = loss_reduction(n_times, time_averaged)
    results = {"noise_levels": levels, "reduction": note, "n_times": n_times,
               **{model: {key: series(rows, levels, model, key) for key in METRICS}
                  for model in ("ChemKAN", "DeepONet")}}
    fig = plot_figure(rows, levels, divisor,
                      "" if divisor == 1.0 else f"\n{note}")
    save_figure(fig, output_path)
    if show:
        plt.show()
    return fig, results


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default=None, help="biodiesel_fig5a_metrics.csv")
    p.add_argument("--output",
                   default=FIGURES_BIODIESEL / "fig05a_biodiesel_noise_robustness")
    p.add_argument("--time-averaged", action="store_true",
                   help="divide the plotted loss by N_t (display convention only)")
    args = p.parse_args()
    _, _, suffix = loss_reduction(0, args.time_averaged)
    _, results = make_figure(metrics_path=args.metrics,
                             output_path=f"{args.output}{suffix}",
                             time_averaged=args.time_averaged)
    print(f"loss reported as {results['reduction']}")
    levels = results["noise_levels"]
    print(f"{'noise':>5} " + " ".join(f"{m[:2]} {FIG5_LABELS[k][:5]:>11}"
                                      for m in ("ChemKAN", "DeepONet") for k in METRICS))
    for i, pct in enumerate(levels):
        print(f"{pct:4d}% " + " ".join(f"{results[m][k][i]:14.4e}"
                                       for m in ("ChemKAN", "DeepONet") for k in METRICS))
    print(f"wrote {args.output}{suffix}.pdf/.png")


if __name__ == "__main__":
    main()
