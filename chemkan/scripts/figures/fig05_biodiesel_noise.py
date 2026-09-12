"""Reproduce ChemKAN Figure 5A -- biodiesel noise robustness.

Converged loss versus observation-noise level for ChemKAN and DeepONet, read from the
final checkpoints' recorded metrics. Three curves per model: the training loss and the
test loss against the noisy observations (Eq. 18), and the test loss against the clean
trajectories (Eq. 22).

Reads the metrics table only -- it neither trains nor re-evaluates checkpoints. Rebuild
that table with chemkan/scripts/diagnostics/refresh_biodiesel_reports.py if it is stale.

    python chemkan/scripts/figures/fig05_biodiesel_noise.py

The optional observed-interval mode plots final clean-data results for the three
paired ChemKAN seeds and one reference DeepONet seed, with no noise-sweep claim:

    python chemkan/scripts/figures/fig05_biodiesel_noise.py --observed-intervals
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    DATA,
    DEEPONET_VERSION,
    FIGURES_BIODIESEL,
    ROOT,
    TABLES,
    load_checkpoint,
    loss_reduction,
    relative_to_root,
    require_file,
    save_figure,
    use_headless_backend,
)
from matplotlib.lines import Line2D
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

# The clean observed-interval comparison uses the same method colours in Figures 5A/B.
# Keep the standard reproduction's FIG5_* styling above unchanged.
INTERVAL_MODEL_COLORS = {
    "original": "#0072B2",
    "observed_interval": "#D55E00",
    "deeponet": "#009E73",
}
INTERVAL_MODEL_LABELS = {
    "original": "Original ChemKAN",
    "observed_interval": "Observed-interval ChemKAN",
    "deeponet": "Reference DeepONet (seed 0)",
}
INTERVAL_EXPERIMENT = ROOT / "results/experiments/biodiesel_observed_intervals"
INTERVAL_METRIC_CONVENTION = (
    "normalized full-trajectory MSE: mean over species, sum over observation times, "
    "mean over trajectories; train-only min-max normalization"
)

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
                      "" if divisor == 1.0 else "\n" + note.replace(": ", ":\n", 1))
    save_figure(fig, output_path)
    if show:
        plt.show()
    return fig, results


def _clean_final_metadata(run_dir, seed):
    """Validate the saved final checkpoint budget, rather than a history row index."""
    config_path = require_file(run_dir / "config.json", "run configuration")
    with config_path.open() as f:
        config = json.load(f)
    checkpoint_path = run_dir / "checkpoint_final.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    training = checkpoint["training"]
    if config["seed"] != seed or training["seed"] != seed:
        raise ValueError(f"Seed mismatch in {run_dir}")
    if config["epochs"] != 10000 or training["epochs"] != 10000:
        raise ValueError(f"Expected a final 10,000-update checkpoint: {run_dir}")
    updates_per_epoch = config.get("updates_per_epoch", training.get("batches_per_epoch", 1))
    updates = training.get("total_optimizer_steps", training["epochs"] * updates_per_epoch)
    if updates != 10000 or updates_per_epoch != 1:
        raise ValueError(f"Expected one optimizer step per epoch: {run_dir}")
    noise = config.get("noise")
    if training.get("noise_percent") not in (None, 0) or (
            noise is not None and noise.get("percent") != 0):
        raise ValueError(f"Expected a clean-data run: {run_dir}")
    return checkpoint, {
        "optimizer_updates": updates,
        "checkpoint_source": relative_to_root(checkpoint_path),
        "config_source": relative_to_root(config_path),
        "update_count_source": "final checkpoint training metadata; one update per epoch",
    }


def load_interval_clean_final_rows():
    """Verified final full-rollout values; never substitute pre-update history losses."""
    comparison_path = INTERVAL_EXPERIMENT / "tables/observed_intervals_final_comparison.csv"
    with require_file(comparison_path, "paired final comparison").open(newline="") as f:
        paired = list(csv.DictReader(f))
    if len(paired) != 3 or {int(row["seed"]) for row in paired} != {0, 1, 2}:
        raise ValueError("The final comparison must contain exactly paired seeds 0, 1, 2")
    rows = []

    def add_rows(method, seed, values, source_path, metadata):
        for split, loss in values.items():
            loss = float(loss)
            if not np.isfinite(loss) or loss <= 0:
                raise ValueError(f"Invalid final loss: {method}, seed {seed}, {split}: {loss}")
            rows.append({"method": method, "model_label": INTERVAL_MODEL_LABELS[method],
                         "seed": seed, "split": split, "noise_percent": 0,
                         "full_rollout_loss": loss,
                         "metric_convention": INTERVAL_METRIC_CONVENTION,
                         "source_path": relative_to_root(source_path), **metadata})

    for row in sorted(paired, key=lambda row: int(row["seed"])):
        seed = int(row["seed"])
        original = (ROOT / "results/reproduction/chemkan/biodiesel/noise/clean_replay_seed0"
                    if seed == 0 else INTERVAL_EXPERIMENT / f"baseline_original_seed{seed}")
        for method, prefix, run_dir in (
                ("original", "orig", original),
                ("observed_interval", "obs", INTERVAL_EXPERIMENT / f"seed{seed}")):
            checkpoint, metadata = _clean_final_metadata(run_dir, seed)
            values = {"train": row[f"{prefix}_train"], "test_clean": row[f"{prefix}_test"]}
            # Interval checkpoints also record final evaluation values. Allow float32
            # roundoff between the in-training evaluation and the verified table.
            if method == "observed_interval":
                final = checkpoint["final_metrics"]
                for split, key in (("train", "full_rollout_train_mse"),
                                   ("test_clean", "full_rollout_test_mse_clean")):
                    if not np.isclose(float(values[split]), float(final[key]), rtol=2e-6, atol=1e-9):
                        raise ValueError(f"Final comparison differs from {run_dir}: {key}")
            add_rows(method, seed, values, comparison_path, metadata)

    canonical_path = TABLES / "biodiesel_fig5a_metrics.csv"
    canonical, _ = load_metrics(canonical_path)
    validate_metrics(canonical, DEEPONET_VERSION)
    reference = [row for row in canonical if row["model"] == "DeepONet"
                 and row["role"] == "plotted" and int(row["noise_percent"]) == 0]
    if len(reference) != 1:
        raise ValueError("Expected exactly one clean DeepONet reference row")
    reference = reference[0]
    reference_dir = ROOT / reference["run_dir"]
    checkpoint, metadata = _clean_final_metadata(reference_dir, 0)
    if checkpoint["architecture"]["architecture_version"] != DEEPONET_VERSION:
        raise ValueError("The reference checkpoint uses a different DeepONet architecture")
    metrics_path = reference_dir / "metrics.json"
    with require_file(metrics_path, "DeepONet final metrics").open() as f:
        metrics = json.load(f)
    for key in ("train_mse_noisy", "test_mse_clean", "test_mse_noisy"):
        if not np.isclose(float(reference[key]), float(metrics[key]), rtol=1e-12, atol=0):
            raise ValueError(f"Canonical DeepONet table differs from final metrics: {key}")
    if float(reference["test_mse_noisy"]) != float(reference["test_mse_clean"]):
        raise ValueError("Clean and noisy test metrics must coincide at 0% noise")
    metadata["metrics_source"] = relative_to_root(metrics_path)
    add_rows("deeponet", 0, {"train": reference["train_mse_noisy"],
                             "test_clean": reference["test_mse_clean"]},
             canonical_path, metadata)
    return rows


def make_interval_clean_figure(*, output_path=None, time_averaged=False, show=False):
    """Clean Figure 5A: final paired ChemKAN seeds and one DeepONet reference.

    Returns ``(fig, results)``. The companion CSV records raw and plotted losses with
    checkpoint/source provenance. With ``output_path=None`` no files are written.
    """
    rows = load_interval_clean_final_rows()
    n_times = observation_times("biodiesel.npz")
    divisor, note, suffix = loss_reduction(n_times, time_averaged)
    label_precision = 8 if time_averaged else 6
    for row in rows:
        row["n_observation_times"] = n_times
        row["plotted_loss"] = row["full_rollout_loss"] / divisor
        row["plotted_reduction"] = note

    markers = {0: "o", 1: "s", 2: "^"}
    positions = {"original": 0, "observed_interval": 1, "deeponet": 2}
    fig, axes = plt.subplots(1, 2, figsize=(13 if time_averaged else 12, 5.8))
    for ax, split, title in zip(axes, ("train", "test_clean"),
                                ("Training trajectories", "Clean held-out trajectories")):
        selected = [row for row in rows if row["split"] == split]
        for seed in markers:
            pair = [next(row for row in selected if row["method"] == method
                         and row["seed"] == seed)
                    for method in ("original", "observed_interval")]
            ax.plot([positions[row["method"]] for row in pair],
                    [row["plotted_loss"] for row in pair], color="0.65", lw=1.1, zorder=1)
        for row in selected:
            method, seed = row["method"], row["seed"]
            x = positions[method]
            ax.scatter(x, row["plotted_loss"], color=INTERVAL_MODEL_COLORS[method],
                       marker=markers[seed] if method != "deeponet" else "D",
                       s=70, edgecolors="white", linewidths=0.6, zorder=3)
            label_offset = 0
            if time_averaged and split == "test_clean":
                peers = sorted((r for r in selected if r["method"] == method),
                               key=lambda r: r["plotted_loss"])
                label_offset = 12 * (peers.index(row) - (len(peers) - 1) / 2)
            ax.annotate(f"{row['plotted_loss']:.{label_precision}f}", (x, row["plotted_loss"]),
                        xytext=(10, label_offset), textcoords="offset points", ha="left",
                        va="center", fontsize=9, color="0.2",
                        arrowprops=({"arrowstyle": "-", "color": "0.65", "lw": 0.6,
                                     "shrinkA": 2, "shrinkB": 5} if label_offset else None),
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8,
                              "alpha": 0.9})
        ax.set_title(title, fontsize=12)
        ax.set_xticks(list(positions.values()),
                      ["Original\nChemKAN", "Observed-interval\nChemKAN", "Reference\nDeepONet (seed 0)"])
        ax.set_xlim(-0.4, 2.6 if time_averaged else 2.45)
        ax.set_xlabel("Method (all at 0% observation noise)")
        ax.set_yscale("log")
        ax.margins(y=0.18)
        if time_averaged:
            ax.set_ylim(1e-4, max(row["plotted_loss"] for row in selected) * 1.4)
        ax.set_ylabel("Full-rollout loss\n" + (
            f"Normalized trajectory MSE / N_t (N_t = {n_times})" if time_averaged
            else "Normalized trajectory MSE (time sum)"))
        ax.yaxis.set_major_locator(LogLocator(
            base=10, subs=(1, 2, 5) if time_averaged else (1, 2, 3, 4, 5, 6, 7, 8, 9)))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(
            base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
        ax.grid(axis="y", which="major", alpha=0.2)
    fig.suptitle("Observed-interval experiment, Fig. 5A layout | 0% noise: "
                 "final full-rollout losses at 10,000 epochs" + (
        f"\nTime-averaged diagnostic: loss / N_t (N_t = {n_times})" if time_averaged else ""),
        fontsize=13)
    handles = [Line2D([], [], color="0.3", ls="none", marker=marker, ms=7,
                       label=f"ChemKAN seed {seed}") for seed, marker in markers.items()]
    handles.append(Line2D([], [], color=INTERVAL_MODEL_COLORS["deeponet"], ls="none",
                          marker="D", ms=7, label="DeepONet seed 0"))
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               frameon=False, fontsize=9)
    fig.text(0.5, 0.01, "Grey lines join paired ChemKAN seeds. "
             f"Labels show final losses to {'eight' if time_averaged else 'six'} decimal places; "
             "source CSV retains full precision.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))

    written = save_figure(fig, output_path)
    csv_path = None
    if output_path is not None:
        figure_dir = Path(output_path).parent
        table_root = figure_dir.parent if figure_dir.name == "figures" else figure_dir
        csv_path = table_root / "tables" / f"fig5a_clean_final_losses{suffix}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    results = {"noise_levels": [0], "optimizer_updates": 10000, "rows": rows,
               "metric_convention": INTERVAL_METRIC_CONVENTION, "reduction": note,
               "display_divisor": divisor,
               "n_times": n_times, "figure_paths": written, "csv_path": csv_path}
    if show:
        plt.show()
    return fig, results


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default=None, help="biodiesel_fig5a_metrics.csv")
    p.add_argument("--output",
                   default=None)
    p.add_argument("--observed-intervals", action="store_true",
                   help="final clean paired ChemKAN comparison with a seed-0 DeepONet reference")
    p.add_argument("--time-averaged", action="store_true",
                   help="write a _time_averaged companion (derived diagnostic: Eq. 18 / N_t)")
    args = p.parse_args()
    if args.observed_intervals and args.metrics is not None:
        p.error("--metrics belongs to the noise-sweep mode; the interval mode uses verified final tables")
    output = args.output or (INTERVAL_EXPERIMENT / "figures/fig5a" if args.observed_intervals
                             else FIGURES_BIODIESEL / "fig05a_biodiesel_noise_robustness")
    _, _, suffix = loss_reduction(0, args.time_averaged)
    if args.observed_intervals:
        _, results = make_interval_clean_figure(output_path=f"{output}{suffix}",
                                                time_averaged=args.time_averaged)
        print(f"loss reported as {results['reduction']}")
        print(f"wrote {output}{suffix}.pdf/.png and {results['csv_path']}")
        return
    _, results = make_figure(metrics_path=args.metrics,
                             output_path=f"{output}{suffix}",
                             time_averaged=args.time_averaged)
    print(f"loss reported as {results['reduction']}")
    levels = results["noise_levels"]
    print(f"{'noise':>5} " + " ".join(f"{m[:2]} {FIG5_LABELS[k][:5]:>11}"
                                      for m in ("ChemKAN", "DeepONet") for k in METRICS))
    for i, pct in enumerate(levels):
        print(f"{pct:4d}% " + " ".join(f"{results[m][k][i]:14.4e}"
                                       for m in ("ChemKAN", "DeepONet") for k in METRICS))
    print(f"wrote {output}{suffix}.pdf/.png")


if __name__ == "__main__":
    main()
