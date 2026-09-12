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
from pathlib import Path

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
    relative_to_root,
    require_file,
    save_figure,
    use_headless_backend,
)

# One palette for both halves of paper Figure 5, defined with Figure 5A.
from fig05_biodiesel_noise import (
    FIG5_COLORS,
    FIG5_LABELS,
    FIG5_LINES,
    INTERVAL_MODEL_COLORS,
    INTERVAL_MODEL_LABELS,
    load_interval_clean_final_rows,
)
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

NOISE_LEVELS = (0, 2, 7, 15)

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
                      note="" if divisor == 1.0 else note.replace(": ", ":\n", 1))
    save_figure(fig, output_path)
    if show:
        plt.show()
    return fig, results


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed-intervals", action="store_true",
                   help="plot the clean interval objective and the seed-0 full-rollout "
                        "comparison with original ChemKAN and reference DeepONet")
    p.add_argument("--output", default=FIGURES_BIODIESEL / "fig05b_biodiesel_loss_dynamics")
    p.add_argument("--time-averaged", action="store_true",
                   help="write a _time_averaged companion (derived diagnostic: Eq. 18 / N_t)")
    args = p.parse_args()
    if args.observed_intervals:
        _, _, suffix = loss_reduction(0, args.time_averaged)
        exp = ROOT / "results/experiments/biodiesel_observed_intervals"
        make_interval_clean_figure(output_path=exp / f"figures/fig5b{suffix}",
                                   time_averaged=args.time_averaged)
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


def make_interval_clean_figure(*, output_path=None, time_averaged=False, show=False):
    """Clean Figure 5B: actual interval objective and seed-0 rollout evaluation.

    Returns ``(fig, results)`` for notebook use. Curves come from saved histories;
    separate markers show the verified final-checkpoint metrics at 10,000 epochs.
    no training, checkpoint evaluation, smoothing or resampling is performed.
    ``time_averaged`` only divides the full-rollout panel by its observation count;
    the local training objective retains its own recorded endpoint-sum convention.
    With ``output_path=None``, neither figures nor companion tables are written.
    """
    exp = ROOT / "results/experiments/biodiesel_observed_intervals"
    n_times = observation_times("biodiesel.npz")
    divisor, note, suffix = loss_reduction(n_times, time_averaged)
    final_precision = 8 if time_averaged else 6
    interval_convention = (
        f"sum over {n_times - 1} restarted interval endpoints; mean over species and "
        "training trajectories; train-only min-max normalization"
    )
    rollout_convention = (
        f"sum over {n_times} observation times; mean over species and trajectories; "
        "train-only min-max normalization; complete trajectory from original initial conditions"
    )

    def read_columns(path, columns):
        with require_file(path, "recorded training history").open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(columns) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} lacks {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty history: {path}")
        return {column: np.array([float(row[column]) if row[column] else np.nan
                                  for row in rows]) for column in columns}

    local, rollout, local_rows, rollout_rows = {}, {}, [], []
    for seed in (0, 1, 2):
        path = exp / f"seed{seed}/history.csv"
        history = read_columns(path, ("epoch", "total_loss"))
        valid = np.isfinite(history["epoch"]) & np.isfinite(history["total_loss"])
        epochs, loss = history["epoch"][valid], history["total_loss"][valid]
        if not len(epochs) or epochs[-1] != 10000:
            raise ValueError(f"Missing final interval objective in {path}")
        local[seed] = {"epoch": epochs, "total_loss": loss,
                       "source_path": relative_to_root(path)}
        local_rows.extend({"model": "observed_interval", "seed": seed,
                           "metric": "total_loss", "epoch": int(epoch),
                           "loss_endpoint_sum": float(value), "display_divisor": 1.0,
                           "metric_convention": interval_convention,
                           "source_path": relative_to_root(path)}
                          for epoch, value in zip(epochs, loss))

    sources = {
        "original": (default_chemkan_run(0) / "history.csv",
                     {"train": "total_loss", "test": "test_mse_clean"}),
        "observed_interval": (exp / "seed0/history.csv",
                              {"train": "full_rollout_train_mse",
                               "test": "full_rollout_test_mse_clean"}),
        "deeponet": (default_deeponet_run(0) / "history.csv",
                     {"train": "total_loss", "test": "test_mse_clean"}),
    }
    for model, (path, columns) in sources.items():
        history = read_columns(path, ("epoch", *columns.values()))
        rollout[model] = {"seed": 0, "source_path": relative_to_root(path)}
        for split, column in columns.items():
            valid = np.isfinite(history["epoch"]) & np.isfinite(history[column])
            epochs, loss = history["epoch"][valid], history[column][valid]
            if not len(epochs):
                raise ValueError(f"No recorded {column} values in {path}")
            if model == "observed_interval" and epochs[-1] != 10000:
                raise ValueError(f"Missing final interval-model {split} rollout evaluation")
            rollout[model][split] = {"epoch": epochs, "loss_timesummed": loss,
                                     "source_column": column}
            rollout_rows.extend({"model": model, "seed": 0, "split": split,
                                 "metric": column, "epoch": int(epoch),
                                 "loss_timesummed": float(value),
                                 "display_divisor": divisor,
                                 "metric_convention": rollout_convention,
                                 "source_path": relative_to_root(path)}
                                for epoch, value in zip(epochs, loss))

    for row in rollout_rows:
        row["record_type"] = "history"
    final_rows = [row for row in load_interval_clean_final_rows() if row["seed"] == 0]
    for row in final_rows:
        model = row["method"]
        split = "train" if row["split"] == "train" else "test"
        loss = row["full_rollout_loss"]
        rollout[model].setdefault("final_checkpoint", {})[split] = loss
        rollout_rows.append({"model": model, "seed": 0, "split": split,
                             "metric": "final_full_rollout_loss", "epoch": row["optimizer_updates"],
                             "loss_timesummed": loss, "display_divisor": divisor,
                             "metric_convention": rollout_convention,
                             "source_path": row["source_path"], "record_type": "final_checkpoint"})

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.3),
                             gridspec_kw={"height_ratios": [4, 1.2]})
    local_ax, rollout_ax = axes[0]
    local_table_ax, rollout_table_ax = axes[1]
    for seed, style in zip((0, 1, 2), ("-", "--", ":")):
        history = local[seed]
        local_ax.semilogy(history["epoch"], history["total_loss"],
                          color=INTERVAL_MODEL_COLORS["observed_interval"],
                          ls=style, lw=1.05, alpha=0.85, label=f"Seed {seed}")
    local_ax.set_title("A. Actual interval training objective\n"
                       "Observed-interval ChemKAN: all three seeds", fontsize=11)
    local_ax.set_ylabel(f"Normalized endpoint loss\n(sum over {n_times - 1} restarted intervals)")
    local_ax.legend(frameon=False, fontsize=9)

    for model, history in rollout.items():
        for split, style in (("train", "--"), ("test", "-")):
            values = history[split]
            rollout_ax.semilogy(
                values["epoch"], values["loss_timesummed"] / divisor,
                color=INTERVAL_MODEL_COLORS[model], ls=style, lw=1.05,
                label=f"{INTERVAL_MODEL_LABELS[model]}: "
                      f"{'training' if split == 'train' else 'held-out'}",
            )
            rollout_ax.scatter(10000, history["final_checkpoint"][split] / divisor,
                               color=INTERVAL_MODEL_COLORS[model],
                               marker="s" if split == "train" else "o", s=38,
                               edgecolors="black", linewidths=0.6, zorder=5)
    rollout_ax.set_title("B. Complete-trajectory evaluation\n"
                         "Training and clean held-out trajectories: seed 0", fontsize=11)
    rollout_ax.set_ylabel(
        "Normalized full-rollout loss\n"
        + (f"(summed over {n_times} observation times)" if divisor == 1
           else f"(time-averaged diagnostic: summed loss / {n_times})")
    )
    rollout_ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for ax in (local_ax, rollout_ax):
        ax.set_xlabel("Training epoch")
        ax.set_xlim(0, 10000)
        ax.set_xticks([0, 5000, 10000])
        ax.grid(axis="y", which="major", alpha=0.25)
    rollout_ax.set_xlim(0, 10300)

    def final_table(ax, title, headers, values, widths):
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, pad=6)
        table = ax.table(cellText=values, colLabels=headers, colWidths=widths,
                         cellLoc="center", bbox=[0, 0.02, 1, 0.93])
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("0.8")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor("#eef2f5")
                cell.get_text().set_weight("bold")
        return table

    final_table(local_table_ax, "Final observed-interval training objective | 10,000 epochs",
                ["Interval-trained ChemKAN", "Endpoint loss"],
                [[f"Seed {seed}", f"{local[seed]['total_loss'][-1]:.9f}"]
                 for seed in (0, 1, 2)], [0.55, 0.45])
    full_table = final_table(
        rollout_table_ax, ("Final full-rollout losses" + (f" / {n_times}" if time_averaged else "")
                           + " | seed 0, 10,000 epochs"),
        ["Model", "Training", "Held-out"],
        [[INTERVAL_MODEL_LABELS[model],
          f"{history['final_checkpoint']['train'] / divisor:.{final_precision}f}",
          f"{history['final_checkpoint']['test'] / divisor:.{final_precision}f}"]
         for model, history in rollout.items()], [0.56, 0.22, 0.22])
    for index, model in enumerate(rollout, start=1):
        full_table[index, 0].get_text().set_color(INTERVAL_MODEL_COLORS[model])

    fig.suptitle("Observed-interval experiment, Fig. 5B layout | 0% observation noise" + (
        f"\nFull-rollout loss / N_t (N_t = {n_times}); interval objective unchanged"
        if time_averaged else ""), fontsize=14)
    fig.text(0.5, 0.015,
             "Right: dashed = training; solid = held-out. Final checkpoints at epoch 10,000: "
             "squares = training, circles = held-out.\n"
             "Tables show rounded final values; CSVs retain full precision. "
             "Left and right use different prediction procedures.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), w_pad=2.0, h_pad=2.0)

    written = save_figure(fig, output_path, dpi=180)
    if output_path is not None:
        table_dir = Path(output_path).parent.parent / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in ((f"fig5b_clean_trajectory_losses{suffix}.csv", rollout_rows),
                           (f"fig5b_interval_objective_losses{suffix}.csv", local_rows)):
            path = table_dir / name
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)
    if show:
        plt.show()
    results = {
        "interval_objective_histories": local,
        "full_rollout_histories": rollout,
        "interval_objective_rows": local_rows,
        "full_rollout_rows": rollout_rows,
        "noise_percent": 0,
        "n_times": n_times,
        "interval_objective_convention": interval_convention,
        "full_rollout_convention": rollout_convention,
        "full_rollout_reduction": note,
        "full_rollout_display_divisor": divisor,
        "history_note": "Raw histories retained; separate final-checkpoint markers agree with Figure 5A.",
        "written": written,
    }
    return fig, results


if __name__ == "__main__":
    main()
