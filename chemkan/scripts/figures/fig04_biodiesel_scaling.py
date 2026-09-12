"""Reproduce ChemKAN Figure 4 -- biodiesel neural scaling.

Loss versus trainable-parameter count for ChemKAN and DeepONet at several widths, with a
log-log regression through every measured point. Evaluates finished scaling checkpoints;
it never trains.

Moved unchanged from chemkan/scripts/diagnostics/assemble_fig4_scaling.py, which remains
the validated implementation this file is derived from. Only the structure changed: the
same evaluation, the same fit, the same plot.

    python chemkan/scripts/figures/fig04_biodiesel_scaling.py
    python chemkan/scripts/figures/fig04_biodiesel_scaling.py --n-mu 2
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    CHEMKAN_BIODIESEL,
    DATA,
    DEEPONET_BIODIESEL,
    FIGURES_BIODIESEL,
    ROOT,
    TABLES,
    add_repo_paths,
    loss_reduction,
    relative_to_root,
    require_file,
    save_figure,
    use_headless_backend,
)
from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, LogLocator, StrMethodFormatter

add_repo_paths()

import evaluate_biodiesel_deeponet as don
from evaluate_biodiesel import evaluate_biodiesel

CHEMKAN_SCALING = CHEMKAN_BIODIESEL / "scaling"
CLEAN_REPLAY = CHEMKAN_BIODIESEL / "noise/clean_replay_seed0"
DEEPONET_WIDTHS = (3, 5, 6, 8, 10, 13)
CHEMKAN_EPOCHS = 5000            # Figure-4 budget for every ChemKAN scaling point
DEEPONET_EPOCHS = 50000

# Reported paper slopes are comparison values; the plotted lines keep OUR measured fits.
PAPER_SLOPES = {
    "train_loss": {"ChemKAN": -1.0, "DeepONet": -4.0},
    "test_loss": {"ChemKAN": -0.6, "DeepONet": -1.4},
}
# EXPLICIT MASK: ALL points are included. The paper fits "prior to saturation"; our curves
# have no identifiable saturation regime to cut at -- they are NON-MONOTONIC in parameter
# count. A minimum-based mask degenerates (it left the ChemKAN training fit with 2 points
# and an undefined standard error), so it is not used. Every point is fitted, and the poor
# R^2 is reported rather than improved by dropping points after the fact.
MASK_RULE = ("all measured points, with no post-hoc exclusions; descriptive regression; "
             "fitting scope differs from the paper's pre-saturation subset")

def observation_times(npz_name):
    """Number of saved observation times N_t, read from the dataset (never hard-coded)."""
    with np.load(DATA / npz_name, allow_pickle=True) as archive:
        return len(archive["t"])


def late_band(run_dir, upto):
    """(min, max, median) of the RAW training loss over the last 20% of epochs up to `upto`.

    The ChemKAN training loss oscillates strongly, so a single fixed-checkpoint value is one
    draw from this band. The median is a robust secondary estimate; it is reported ALONGSIDE
    the final-checkpoint value, never in place of it. Only the training loss has a per-epoch
    record here -- the scaling runs were launched without in-training test evaluation.
    """
    with open(run_dir / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    losses = np.array([float(r["total_loss"]) for r in rows
                       if int(float(r["epoch"])) < upto])
    late = losses[int(.8 * len(losses)):]
    return float(late.min()), float(late.max()), float(np.median(late))


def chemkan_checkpoints(n_mu):
    """(width, checkpoint, reused) per ChemKAN scaling point.

    The h=4 point reuses the clean replay's 5,000-epoch snapshot rather than repeating an
    identical run; ``--n-mu 2`` reads its own manifest.
    """
    if n_mu == "2":
        manifest = json.loads(
            (CHEMKAN_SCALING.parent / "scaling_nmu2/manifest_scaling_seed0.json").read_text())
        return [(j["width"], ROOT / j["checkpoint"], "scaling_nmu2" not in j["checkpoint"])
                for j in manifest["jobs"]]
    spec = [(h, CHEMKAN_SCALING / f"h{h:02d}_seed0" / "checkpoint_final.pt", False)
            for h in (2, 3, 10, 17)]
    spec.append((4, CLEAN_REPLAY / "checkpoint_epoch_5000.pt", True))
    return spec


def evaluate_points(n_mu, deeponet_version):
    """Evaluate every scaling checkpoint. Returns one row per measured point."""
    points = []
    for h, ckpt_path, reused in sorted(chemkan_checkpoints(n_mu)):
        require_file(ckpt_path, f"ChemKAN h={h} checkpoint")
        train = evaluate_biodiesel(str(ckpt_path), "train", "cpu")
        test = evaluate_biodiesel(str(ckpt_path), "test", "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expected_nmu = 2 if n_mu == "2" else (h + 1) // 2
        if ckpt["architecture"]["n_mu"] != expected_nmu:
            raise ValueError(f"h={h}: checkpoint n_mu {ckpt['architecture']['n_mu']} "
                             f"!= expected {expected_nmu}")
        if ckpt.get("stage2_epoch", ckpt["training"]["epochs"]) != CHEMKAN_EPOCHS:
            raise ValueError(f"h={h}: checkpoint is not a {CHEMKAN_EPOCHS}-epoch result")
        lo, hi, median = late_band(ckpt_path.parent, CHEMKAN_EPOCHS)
        points.append({"model": "ChemKAN", "width": h, "parameters": train["n_params"],
                           "train_loss": train["mse"], "test_loss": test["mse"],
                           "train_loss_late_median": median, "train_late_min": lo,
                           "train_late_max": hi, "source": relative_to_root(ckpt_path),
                           "reused": reused, "architecture_version": "", "n_mu": expected_nmu})

    deeponet_scaling = DEEPONET_BIODIESEL / deeponet_version / "scaling"
    for w in DEEPONET_WIDTHS:
        run_dir = deeponet_scaling / f"w{w:02d}_seed0"
        ckpt_path = require_file(run_dir / "checkpoint_final.pt", f"DeepONet w={w} checkpoint")
        train = don.evaluate(str(ckpt_path), "train", "cpu")
        test = don.evaluate(str(ckpt_path), "test", "cpu")
        if train["model"].architecture_version != deeponet_version:
            raise ValueError(f"w={w}: DeepONet architecture "
                             f"{train['model'].architecture_version!r} != {deeponet_version!r}")
        if train["ckpt"]["training"]["epochs"] != DEEPONET_EPOCHS:
            raise ValueError(f"w={w}: DeepONet run is not {DEEPONET_EPOCHS} epochs")
        lo, hi, median = late_band(run_dir, DEEPONET_EPOCHS)
        points.append({"model": "DeepONet", "width": w, "parameters": train["n_params"],
                           "train_loss": train["mse"], "test_loss": test["mse"],
                           "train_loss_late_median": median, "train_late_min": lo,
                           "train_late_max": hi, "source": relative_to_root(ckpt_path),
                           "reused": False, "architecture_version": deeponet_version, "n_mu": ""})
    return points


def fit_line(parameters, losses):
    """Least-squares log-log fit -> (slope, intercept, R^2, standard error of the slope)."""
    x, y = np.log10(parameters), np.log10(losses)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(((y - (slope * x + intercept)) ** 2).sum())
    r_squared = 1 - residual / float(((y - y.mean()) ** 2).sum())
    std_error = (float(np.sqrt(residual / (n - 2) / ((x - x.mean()) ** 2).sum()))
                 if n > 2 else float("nan"))
    return float(slope), float(intercept), float(r_squared), std_error


def fit_scaling(points):
    """One descriptive regression per model and metric, over ALL points (see MASK_RULE)."""
    fits = []
    for model in ("ChemKAN", "DeepONet"):
        subset = sorted([p for p in points if p["model"] == model],
                        key=lambda r: r["parameters"])
        if len(subset) < 3:
            continue
        for metric in ("train_loss", "test_loss", "train_loss_late_median"):
            if metric == "train_loss_late_median" and any(r.get(metric) is None for r in subset):
                continue
            P = np.array([r["parameters"] for r in subset])
            L = np.array([r[metric] for r in subset])
            slope, intercept, r_squared, std_error = fit_line(P, L)
            fits.append({"model": model, "metric": metric, "slope": slope, "intercept": intercept,
                             "r_squared": r_squared, "std_error": std_error, "n_included": len(P),
                             "included": ",".join(str(int(v)) for v in P),
                             "excluded": "(none)", "mask_rule": MASK_RULE})
    return fits


def plot_figure(points, fits, n_mu, deeponet_version, divisor=1.0, note=""):
    parameter_counts = sorted({int(p["parameters"]) for p in points})
    loss_values = [p[metric] / divisor for p in points
                   for metric in ("train_loss", "test_loss")]
    loss_limits = (10 ** np.floor(np.log10(min(loss_values))),
                   10 ** np.ceil(np.log10(max(loss_values))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, metric, panel, title in ((axes[0], "train_loss", "A", "Training MSE"),
                                     (axes[1], "test_loss", "B", "Testing MSE")):
        slope_rows = []
        for model, colour in (("ChemKAN", "crimson"), ("DeepONet", "seagreen")):
            subset = sorted([p for p in points if p["model"] == model],
                            key=lambda r: r["parameters"])
            if not subset:
                continue
            P = np.array([r["parameters"] for r in subset])
            L = np.array([r[metric] for r in subset]) / divisor
            ax.plot(P, L, "o", color=colour, ms=8, label=f"{model}: our results")
            fit = next((f for f in fits if f["model"] == model and f["metric"] == metric), None)
            if fit:
                included = np.array([int(v) for v in fit["included"].split(",")])
                xs = np.linspace(np.log10(included.min()), np.log10(included.max()), 20)
                ax.plot(10 ** xs,
                        10 ** (fit["slope"] * xs + fit["intercept"]) / divisor, "-",
                        color=colour, lw=1.6, label=f"{model}: fit to our results")
                slope_rows.append([model, f"{fit['slope']:.2f}",
                                   f"{PAPER_SLOPES[metric][model]:.1f}"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of trainable parameters", fontsize=11, labelpad=8)
        ax.set_xlim(70, 1000)
        ax.set_ylim(loss_limits)
        # Decade labels match the y-axis notation; minor ticks identify every measured size.
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
        ax.xaxis.set_minor_locator(FixedLocator(parameter_counts))
        ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
        ax.tick_params(axis="both", which="major", labelsize=10, labelleft=True)
        ax.tick_params(axis="x", which="major", pad=29, length=6)
        ax.tick_params(axis="x", which="minor", labelsize=8, labelrotation=60, pad=3, length=3)
        ax.set_title(f"({panel}) {title}", fontsize=13)
        ax.grid(alpha=.25, which="both")
        if slope_rows:
            comparison = ax.table(cellText=slope_rows,
                                  colLabels=["Model", r"Our $\Delta$", r"Paper $\Delta$"],
                                  cellLoc="center", colWidths=[0.42, 0.29, 0.29],
                                  bbox=(0.03, 0.025, 0.59, 0.21), zorder=5)
            comparison.auto_set_font_size(False)
            comparison.set_fontsize(10)
            for (row, _col), cell in comparison.get_celld().items():
                cell.set_edgecolor("0.8")
                cell.set_linewidth(0.6)
                cell.set_facecolor("0.95" if row == 0 else "white")
                if row == 0:
                    cell.get_text().set_fontweight("bold")
                else:
                    cell.get_text().set_color(
                        {"ChemKAN": "crimson", "DeepONet": "seagreen"}[slope_rows[row - 1][0]])
    axes[0].set_ylabel(f"Loss{note}" if note else "Loss (Eq. 18)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               fontsize=9, handlelength=3, columnspacing=2.5, framealpha=1)
    ck_label = "fixed n_mu=2" if n_mu == "2" else "n_mu=ceil(h/2)"
    fig.suptitle(f"Fig. 4 - neural scaling: ChemKAN {ck_label}", fontsize=15, y=0.99)
    fig.text(0.5, 0.94,
             f"DeepONet: {deeponet_version}. Circles: measured results; lines: fitted trends.",
             ha="center", fontsize=10)
    fig.text(0.5, 0.012, r"$\Delta$ is the slope of log(loss) vs log(parameters). "
             "Our fits use all runs; the paper reports fits before saturation.",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.20, 1, 0.91))
    return fig


def make_figure(n_mu="scaled", deeponet_version="reference", output_path=None,
                points_path=None, fits_path=None, *, time_averaged=False, show=False):
    """Paper Figure 4. Returns ``(fig, results)`` with the measured points and the fits."""
    version = ("reference_final_trunk_relu" if deeponet_version == "reference"
               else "legacy_final_trunk_linear")
    suffix = ("_nmu2" if n_mu == "2" else "") + ("_legacy" if deeponet_version == "legacy" else "")
    points = evaluate_points(n_mu, version)
    fits = fit_scaling(points)

    if points_path is None:
        points_path = TABLES / f"biodiesel_fig4_points{suffix}.csv"
    if fits_path is None:
        fits_path = TABLES / f"biodiesel_fig4_fits{suffix}.csv"
    for path, rows in ((points_path, points), (fits_path, fits)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    n_times = observation_times("biodiesel.npz")
    divisor, note, ta_suffix = loss_reduction(n_times, time_averaged)
    fig = plot_figure(points, fits, n_mu, version, divisor,
                      "" if divisor == 1.0 else f"\n{note}")
    if output_path is None:
        output_path = (FIGURES_BIODIESEL
                       / f"fig04_biodiesel_neural_scaling{suffix}{ta_suffix}")
    save_figure(fig, output_path)
    if show:
        plt.show()
    return fig, {"points": points, "fits": fits, "suffix": suffix, "reduction": note}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-mu", choices=("scaled", "2"), default="scaled")
    p.add_argument("--deeponet-version", choices=("reference", "legacy"), default="reference")
    p.add_argument("--output", default=None, help="output path without extension")
    p.add_argument("--time-averaged", action="store_true",
                   help="write a _time_averaged companion (derived diagnostic: Eq. 18 / N_t)")
    args = p.parse_args()
    _, results = make_figure(n_mu=args.n_mu, deeponet_version=args.deeponet_version,
                             output_path=args.output,
                             time_averaged=args.time_averaged)
    print(f"loss reported as {results['reduction']}")
    points = results["points"]
    print(f"{'model':9s} {'w':>3s} {'params':>7s} {'train(final)':>13s} {'test(final)':>12s} "
          f"{'train late median':>18s} {'late band':>22s}")
    for r in sorted(points, key=lambda r: (r["model"], r["parameters"])):
        band = f"[{r['train_late_min']:.2e},{r['train_late_max']:.2e}]"
        print(f"{r['model']:9s} {r['width']:3d} {r['parameters']:7d} {r['train_loss']:13.4e} "
              f"{r['test_loss']:12.4e} {r['train_loss_late_median']:18.4e} {band:>22s}")
    for f in results["fits"]:
        print(f"fit {f['model']:9s} {f['metric']:24s} slope {f['slope']:+.3f} "
              f"R^2 {f['r_squared']:.3f} (n={f['n_included']})")


if __name__ == "__main__":
    main()
