"""Figure 5B -- biodiesel training and noise-free test loss dynamics.

Per-epoch training loss (against the noisy observations) and noise-free test loss at
0/2/7/15 % noise for ChemKAN and DeepONet, read from the Fig. 5 runs' histories
(time-averaged). Neither trains nor evaluates checkpoints.

    python chemkan/scripts/figures/fig05_biodiesel_loss.py
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

# One palette for both halves of paper Figure 5, defined with Figure 5A.
from fig05_biodiesel_noise import (
    FIG5_COLORS,
    FIG5_LABELS,
    FIG5_LINES,
)
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

NOISE_LEVELS = (0, 2, 7, 15)

def plot_figure(panels, divisor=1.0, note=""):
    """One panel per noise level: the raw per-epoch training and noise-free test loss."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for ax, pct in zip(axes.flat, NOISE_LEVELS):
        for name in ("ChemKAN", "DeepONet"):
            e, train, test = panels[pct][name]
            train, test = train / divisor, test / divisor
            for key, values in (("train_mse_noisy", train), ("test_mse_clean", test)):
                ax.plot(e, values, color=FIG5_COLORS[name][key], ls=FIG5_LINES[key], lw=0.8,
                        label=f"{name}: {FIG5_LABELS[key]}")
            i = int(np.argmin(test))
            ax.plot(e[i], test[i], "v", color=FIG5_COLORS[name]["test_mse_clean"],
                    ms=7, mec="k", mew=0.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
        ax.set_xlabel("Training epoch", fontsize=11)
        ax.set_ylabel(note or "Loss\nNormalized trajectory MSE", fontsize=11)
        ax.set_xticks([0, 5000, 10000])
        ax.set_xlim(0, 10000)
        ax.tick_params(axis="both", which="major", labelsize=10, labelleft=True)
        ax.set_title(f"{pct}% noise", fontsize=13, fontweight="bold")
        ax.grid(axis="y", which="major", alpha=0.25)
    # Extend the shared axis to the reference's 10^-4 scale without changing any value.
    axes[0, 0].set_ylim(bottom=1e-4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.04),
               fontsize=10, handlelength=3.5, columnspacing=2.2, framealpha=1)
    fig.suptitle("Fig. 5B - training and noise-free test loss (corrected DeepONet)",
                 fontsize=14, y=0.985)
    fig.text(0.5, 0.012, "Triangles: noise-free test minima.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96), h_pad=2.0, w_pad=2.0)
    return fig


# --------------------------------------------------------------------------- biodiesel_v2 runs
# Data from the biodiesel_v2 figure runs, drawn by the unchanged plot_figure above.

def make_author_figure(output_path=None, *, show=False):
    """Fig. 5B from the biodiesel_v2 runs, drawn by ``plot_figure``."""
    import biodiesel_runs as ar
    runs = ar.by_name()
    panels = {}
    for pct in NOISE_LEVELS:
        panels[pct] = {}
        for name, key in (("ChemKAN", "chemkan"), ("DeepONet", "deeponet")):
            h = ar.history(runs[f"fig05_{key}_noise{pct:02d}"])
            panels[pct][name] = (h["epoch"], h["train"], h["test_clean"])
    fig = plot_figure(panels, 1.0, "Loss\ntime-averaged MSE")
    fig._suptitle.set_text(f"Fig. 5B - training and noise-free test loss (biodiesel_v2 runs; "
                           f"{ar.DEEPONET_LABEL[ar.DEEPONET_VARIANT]}; {ar.CONVENTION_SHORT})")
    # plot_figure fixes the bottom at 1e-4 (Eq. 18 scale); time-averaged values reach lower.
    low = min(float(v.min()) for pct in panels for e, tr, te in panels[pct].values() for v in (tr, te))
    fig.axes[0].set_ylim(bottom=10 ** np.floor(np.log10(low)))
    save_figure(fig, output_path or FIGURES_AUTHOR / "fig05b_loss_histories")
    if show:
        plt.show()
    return fig


def main():
    use_headless_backend()
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_author_figure()


if __name__ == "__main__":
    main()
