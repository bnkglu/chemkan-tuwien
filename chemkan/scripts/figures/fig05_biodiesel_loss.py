"""Reproduce ChemKAN Figure 5B -- biodiesel training and noise-free test loss dynamics.

Per-epoch training loss (against the noisy observations) and noise-free test loss
(Eq. 22) at 0/2/7/15 % noise, for ChemKAN and DeepONet. Reads run histories only; it
neither trains nor evaluates checkpoints.

The 0 % ChemKAN panel comes from the clean replay, which reproduces the main clean run
bitwise and additionally carries the per-epoch test column the main run lacks.

    python chemkan/scripts/figures/fig05_biodiesel_loss.py
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from common import (
    CHEMKAN_BIODIESEL,
    DATA,
    DEEPONET_BIODIESEL,
    DEEPONET_VERSION,
    FIGURES_BIODIESEL,
    ROOT,
    loss_reduction,
    require_file,
    save_figure,
    use_headless_backend,
)

# One palette for both halves of paper Figure 5, defined with Figure 5A.
from fig05_biodiesel_noise import FIG5_COLORS, FIG5_LABELS, FIG5_LINES
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

NOISE_LEVELS = (0, 2, 7, 15)
SMOOTHING_EPOCHS = 101      # display-only moving average; the raw values are what is stored

def observation_times(npz_name):
    """Number of saved observation times N_t, read from the dataset (never hard-coded)."""
    import numpy as np
    with np.load(DATA / npz_name, allow_pickle=True) as archive:
        return len(archive["t"])


def default_chemkan_run(noise_percent: int):
    if noise_percent == 0:
        return CHEMKAN_BIODIESEL / "noise/clean_replay_seed0"
    return CHEMKAN_BIODIESEL / f"noise/noise{noise_percent:02d}_seed0"


def default_deeponet_run(noise_percent: int, version=DEEPONET_VERSION):
    return DEEPONET_BIODIESEL / version / f"noise/noise{noise_percent:02d}_seed0"


def load_history(run_dir):
    """(epoch, training loss, noise-free test loss) from a run's history.csv."""
    path = require_file(run_dir / "history.csv", "run history")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"epoch", "total_loss", "test_mse_clean"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}; the run needs --eval-every")
    return (np.array([float(r["epoch"]) for r in rows]),
            np.array([float(r["total_loss"]) for r in rows]),
            np.array([float(r["test_mse_clean"]) for r in rows]))


def late_span(epochs, values):
    """Peak-to-peak of the RAW values over the last 20 % of epochs."""
    late = epochs >= 0.8 * epochs.max()
    return float(np.ptp(values[late]))


def plot_figure(panels, smoothing=SMOOTHING_EPOCHS, divisor=1.0, note=""):
    """One panel per noise level; raw values faint, moving average bold."""
    smooth = lambda y: np.convolve(y, np.ones(smoothing) / smoothing, mode="valid")
    off = (smoothing - 1) // 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for ax, pct in zip(axes.flat, NOISE_LEVELS):
        for name in ("ChemKAN", "DeepONet"):
            e, train, test = panels[pct][name]
            train, test = train / divisor, test / divisor
            for key, values in (("train_mse_noisy", train), ("test_mse_clean", test)):
                color = FIG5_COLORS[name][key]
                ax.plot(e, values, color=color, ls=FIG5_LINES[key], lw=0.6, alpha=0.20)
                ax.plot(e[off:len(e) - off], smooth(values), color=color,
                        ls=FIG5_LINES[key], lw=1.8, label=f"{name}: {FIG5_LABELS[key]}")
            i = int(np.argmin(test))
            ax.plot(e[i], test[i], "v", color=FIG5_COLORS[name]["test_mse_clean"],
                    ms=7, mec="k", mew=0.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
        ax.set_xlabel("Training epoch", fontsize=11)
        ax.set_ylabel(f"Loss\nNormalized trajectory MSE{note}", fontsize=11)
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
    fig.text(0.5, 0.012, f"Faint curves: raw per-epoch loss; bold curves: {smoothing}-epoch "
             "moving average; triangles: raw noise-free test minima.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96), h_pad=2.0, w_pad=2.0)
    return fig


def make_figure(chemkan_runs=None, deeponet_runs=None, output_path=None, *,
                time_averaged=False, show=False):
    """Paper Figure 5B. Returns ``(fig, results)``.

    ``results["runs"][(model, noise)]`` holds each run's late oscillation spans; the
    reduction metadata sits beside it rather than inside the per-run mapping.

    ``chemkan_runs`` / ``deeponet_runs`` map a noise percentage to a run directory, so a
    notebook can plot other runs; anything omitted falls back to the reproduction runs.
    """
    runs_ck = {p: default_chemkan_run(p) for p in NOISE_LEVELS}
    runs_ck.update(chemkan_runs or {})
    runs_do = {p: default_deeponet_run(p) for p in NOISE_LEVELS}
    runs_do.update(deeponet_runs or {})

    panels, runs = {}, {}
    for pct in NOISE_LEVELS:
        panels[pct] = {"ChemKAN": load_history(runs_ck[pct]),
                       "DeepONet": load_history(runs_do[pct])}
        for name, (e, train, test) in panels[pct].items():
            runs[(name, pct)] = {"train_late_span": late_span(e, train),
                                    "test_late_span": late_span(e, test),
                                    "test_min_epoch": int(e[int(np.argmin(test))]),
                                    "test_min": float(test.min())}
    n_times = observation_times("biodiesel.npz")
    divisor, note, _ = loss_reduction(n_times, time_averaged)
    results = {"runs": runs, "reduction": note}
    fig = plot_figure(panels, divisor=divisor,
                      note="" if divisor == 1.0 else f"\n{note}")
    save_figure(fig, output_path)
    if show:
        plt.show()
    return fig, results


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed-intervals", action="store_true",
                   help="plot the available clean interval-trained ChemKAN against "
                        "the clean DeepONet, using full-trajectory losses")
    p.add_argument("--output", default=FIGURES_BIODIESEL / "fig05b_biodiesel_loss_dynamics")
    p.add_argument("--time-averaged", action="store_true",
                   help="divide the plotted loss by N_t (display convention only)")
    args = p.parse_args()
    if args.observed_intervals:
        make_interval_clean_figure(time_averaged=args.time_averaged)
        return
    _, _, suffix = loss_reduction(0, args.time_averaged)
    _, results = make_figure(output_path=f"{args.output}{suffix}",
                             time_averaged=args.time_averaged)
    print("late-training oscillation span (last 20% of epochs, RAW):")
    print(f"{'noise':>5} {'model':9s} {'train span':>12s} {'clean-test span':>16s}")
    for (name, pct), r in sorted(results["runs"].items(),
                                 key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{pct:4d}% {name:9s} {r['train_late_span']:12.4f} "
              f"{r['test_late_span']:16.4f}")
    print(f"wrote {args.output}{suffix}.pdf/.png")


def make_interval_clean_figure(*, time_averaged=False, show=False):
    """Available 0%-noise counterpart of Figure 5B; interval objective is excluded.

    The training procedure differs, but ChemKAN is evaluated by its complete
    trajectories in both curves. Sparse recorded evaluations are plotted raw.
    """
    exp = ROOT / "results/experiments/biodiesel_observed_intervals"
    with (exp / "seed0/history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns = ("epoch", "full_rollout_train_mse", "full_rollout_test_mse_clean")
    values = np.array([[float(row[key]) if row[key] else np.nan for key in columns]
                       for row in rows])
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values) or values[-1, 0] != 10000:
        raise ValueError("Missing final interval-model rollout evaluations")
    data = {"ChemKAN": tuple(values.T), "DeepONet": load_history(default_deeponet_run(0))}
    divisor, note, suffix = loss_reduction(observation_times("biodiesel.npz"), time_averaged)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    exported = []
    for model, (epochs, train, test) in data.items():
        for key, loss in (("train_mse_noisy", train), ("test_mse_clean", test)):
            valid = np.isfinite(loss)
            ax.semilogy(epochs[valid], loss[valid] / divisor,
                        color=FIG5_COLORS[model][key], ls=FIG5_LINES[key], lw=1.3,
                        label=f"{model}: {FIG5_LABELS[key]}")
            exported.extend({"model": model, "metric": key, "epoch": int(e),
                             "loss_timesummed": float(v), "display_divisor": divisor}
                            for e, v in zip(epochs[valid], loss[valid]))
    ax.set_xlabel("Optimizer updates / training epochs (one update per epoch)")
    ax.set_ylabel("Normalized trajectory loss" + (f"\n{note}" if divisor != 1 else " (time-summed)"))
    ax.set_xlim(0, 10000)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Figure 5B | 0% noise\n"
                 "Interval-trained ChemKAN, seed 0; reference DeepONet\n"
                 "ChemKAN curves use full rollouts; local interval loss is separate", fontsize=11)
    save_figure(fig, exp / f"figures/fig5b{suffix}", dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
    with (exp / "tables/fig5b_clean_trajectory_losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(f"Wrote {exp / f'figures/fig5b{suffix}'}.pdf/.png (clean panel only).")
    return fig


if __name__ == "__main__":
    main()
