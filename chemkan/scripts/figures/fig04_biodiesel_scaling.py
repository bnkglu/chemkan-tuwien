"""Figure 4 -- biodiesel neural scaling.

Final-checkpoint train and noise-free test loss (time-averaged) versus trainable-parameter
count for the noise-free ChemKAN and DeepONet Fig. 4 runs, with log-log fits that exclude
the last ChemKAN and the last two DeepONet points (paper Sec. III A 2). Reads the runs'
recorded metrics; never trains.

    python chemkan/scripts/figures/fig04_biodiesel_scaling.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES_AUTHOR,
    add_repo_paths,
    relative_to_root,
    save_figure,
    use_headless_backend,
)
from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, LogLocator, StrMethodFormatter

add_repo_paths()



# Reported paper slopes are comparison values; the plotted lines keep OUR measured fits.
PAPER_SLOPES = {
    "train_loss": {"ChemKAN": -1.0, "DeepONet": -4.0},
    "test_loss": {"ChemKAN": -0.6, "DeepONet": -1.4},
}

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


# --------------------------------------------------------------------------- biodiesel_v2 runs
# Data from the biodiesel_v2 figure runs, drawn by the unchanged plot_figure above.

AUTHOR_FIT_RULE = ("paper Sec. III A 2: the last ChemKAN point and the last two DeepONet "
                   "points are excluded (saturation)")
AUTHOR_FIT_EXCLUDE = {"ChemKAN": 1, "DeepONet": 2}


def author_scaling_data() -> dict:
    """Final-checkpoint train / noise-free test loss (time-averaged) of the noise-free Fig. 4
    runs as plot_figure ``points``, and ``fits`` (fit_line) with the paper's exclusions."""
    import biodiesel_runs as ar
    points = []
    for model, key in (("ChemKAN", "chemkan"), ("DeepONet", "deeponet")):
        for run in (r for r in ar.plan() if r["name"].startswith(f"fig04_{key}")):
            cfg = ar.config(run)
            params = cfg["architecture"]["parameter_count"] if key == "chemkan" else cfg["parameter_count"]
            loss = ar.final_losses(run)
            points.append({"model": model, "parameters": int(params),
                           "train_loss": loss["train"], "test_loss": loss["test_clean"],
                           "source": relative_to_root(run["dir"] / "checkpoint_final.pt")})
    fits = []
    for model, exclude in AUTHOR_FIT_EXCLUDE.items():
        subset = sorted((p for p in points if p["model"] == model), key=lambda r: r["parameters"])
        used, dropped = subset[:-exclude], subset[-exclude:]
        for metric in ("train_loss", "test_loss"):
            P = np.array([r["parameters"] for r in used])
            slope, intercept, r_squared, std_error = fit_line(P, np.array([r[metric] for r in used]))
            fits.append({"model": model, "metric": metric, "slope": slope, "intercept": intercept,
                         "r_squared": r_squared, "std_error": std_error, "n_included": len(P),
                         "included": ",".join(str(int(v)) for v in P),
                         "excluded": ",".join(str(r["parameters"]) for r in dropped),
                         "mask_rule": AUTHOR_FIT_RULE})
    return {"points": points, "fits": fits}


def make_author_figure(output_path=None, *, data=None, show=False):
    """Fig. 4 from the biodiesel_v2 runs, drawn by ``plot_figure``."""
    import biodiesel_runs as ar
    data = data or author_scaling_data()
    fig = plot_figure(data["points"], data["fits"], "scaled",
                      f"{ar.DEEPONET_LABEL[ar.DEEPONET_VARIANT]}, biodiesel_v2 runs", 1.0,
                      f"\n{ar.CONVENTION_SHORT}")
    # Time-averaged points sit lower than Eq. 18 ones, so the Delta tables move to
    # empty corners: (A) upper right, (B) lower right. Table has no public setter for this.
    for ax, bbox in zip(fig.axes[:2], ((0.40, 0.76, 0.59, 0.21), (0.40, 0.03, 0.59, 0.21))):
        for table in ax.tables:
            table._bbox = bbox
    for text in fig.texts:          # plot_figure's footnote describes an all-points fit
        if text.get_text().startswith(r"$\Delta$ is the slope"):
            text.set_text(r"$\Delta$ is the slope of log(loss) vs log(parameters). "
                          "Fits exclude the last ChemKAN and the last two "
                          "DeepONet points (paper Sec. III A 2).")
    save_figure(fig, output_path or FIGURES_AUTHOR / "fig04_scaling")
    if show:
        plt.show()
    return fig, data


def main():
    use_headless_backend()
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_author_figure()


if __name__ == "__main__":
    main()
