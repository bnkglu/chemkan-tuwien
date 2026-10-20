"""Figure 5A -- biodiesel noise robustness.

Converged loss versus observation-noise level for ChemKAN and DeepONet: the final
checkpoints' training loss, test loss against the noisy observations, and test loss
against the clean trajectories (time-averaged). Never trains.

    python chemkan/scripts/figures/fig05_biodiesel_noise.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES_AUTHOR,
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
    ax.set_ylabel(f"Converged loss{note}" if note
                  else "Converged loss\nNormalized trajectory MSE", fontsize=12)
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


# --------------------------------------------------------------------------- biodiesel_v2 runs
# Data from the biodiesel_v2 figure runs, drawn by the unchanged plot_figure above.

AUTHOR_NOISE = (0, 1, 2, 3, 5, 7, 10, 15)


def author_noise_rows():
    """Converged (final-checkpoint, after 10,000 epochs) train / noisy-test /
    noise-free-test loss of the Fig. 5 runs, time-averaged, as plot_figure ``rows``."""
    import author_runs as ar
    runs = ar.by_name()
    rows = []
    for model, key in (("ChemKAN", "chemkan"), ("DeepONet", "deeponet")):
        for p in AUTHOR_NOISE:
            c = ar.converged(runs[f"fig05_{key}_noise{p:02d}"])
            rows.append({"role": "plotted", "model": model, "noise_percent": p,
                         "train_mse_noisy": c["train"], "test_mse_noisy": c["test_noisy"],
                         "test_mse_clean": c["test_clean"]})
    return rows, list(AUTHOR_NOISE)


def make_author_figure(output_path=None, *, show=False):
    """Fig. 5A from the biodiesel_v2 runs, drawn by ``plot_figure``."""
    import author_runs as ar
    rows, levels = author_noise_rows()
    fig = plot_figure(rows, levels, 1.0, f"\n{ar.CONVENTION_SHORT}")
    fig.axes[0].set_title("Fig. 5A - noise robustness (biodiesel_v2 runs; final checkpoints "
                          f"after 10,000 epochs; {ar.DEEPONET_LABEL[ar.DEEPONET_VARIANT]})",
                          fontsize=13)
    save_figure(fig, output_path or FIGURES_AUTHOR / "fig05a_noise")
    if show:
        plt.show()
    return fig, rows


def main():
    use_headless_backend()
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_author_figure()


if __name__ == "__main__":
    main()
