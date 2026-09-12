"""Reproduce ChemKAN Figure 8A -- hydrogen generalization over a 441-condition grid.

Eq. 18 loss for every (initial temperature, equivalence ratio) pair on a 21x21 grid, for
two Stage-2 initializations. Reads the precomputed generalization tables; it neither
trains nor re-integrates.

Two views are produced: the paper's colour range (0-10 x 10^-4) and a full-range
supplementary view, because our losses run far above the paper's scale and would otherwise
saturate to a single colour.

    python chemkan/scripts/figures/fig08_hydrogen_generalization.py
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from common import (
    CHEMKAN_HYDROGEN,
    DATA,
    FIGURES_HYDROGEN,
    loss_reduction,
    require_file,
    save_figure,
    use_headless_backend,
)
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

GENERALIZATION = CHEMKAN_HYDROGEN / "generalization"
DEFAULT_STEMS = {
    "H0 (primary, random init)": "random_stage2_10000_seed0",
    "Hnorm1 (labelled init comparison)": "normmatched_dir1_stage2_10000",
}
# The paper's Figure-8A colour range. Ours exceed it, hence the second view.
PAPER_MSE_MAX = 10 * 1e-4
GRID_CONDITIONS = 441
TRAINING_CONDITIONS = 35
HELD_OUT_CONDITIONS = 1
GRID_SIDE = 21


def observation_times(npz_name="hydrogen.npz"):
    """Number of saved observation times N_t, read from the dataset (never hard-coded)."""
    with np.load(DATA / npz_name, allow_pickle=True) as archive:
        return len(archive["t"])


def load_grid(csv_path):
    """One 441-row generalization table -> the grid matrix plus the raw per-point arrays."""
    with open(require_file(csv_path, "generalization table"), newline="") as f:
        rows = list(csv.DictReader(f))
    temperature = np.array([float(r["T0_K"]) for r in rows])
    phi = np.array([float(r["phi"]) for r in rows])
    mse = np.array([float(r["trajectory_mse"]) for r in rows])
    ok = np.array([r["status"] == "ok" for r in rows])
    training = np.array([r["is_training"] == "True" for r in rows])
    held_out = np.array([r["is_held_out"] == "True" for r in rows])

    if len(rows) != GRID_CONDITIONS:
        raise ValueError(f"{csv_path}: {len(rows)} rows, expected {GRID_CONDITIONS}")
    if training.sum() != TRAINING_CONDITIONS or held_out.sum() != HELD_OUT_CONDITIONS:
        raise ValueError(f"{csv_path}: {training.sum()} training / {held_out.sum()} held-out "
                         f"conditions, expected {TRAINING_CONDITIONS}/{HELD_OUT_CONDITIONS}")
    if not (np.isfinite(mse[ok]).all() and (mse[ok] >= 0).all()):
        raise ValueError(f"{csv_path}: non-finite or negative loss among successful runs")

    temps, phis = np.unique(temperature), np.unique(phi)
    if len(temps) != GRID_SIDE or len(phis) != GRID_SIDE:
        raise ValueError(f"{csv_path}: grid is {len(temps)}x{len(phis)}, "
                         f"expected {GRID_SIDE}x{GRID_SIDE}")
    matrix = np.full((len(temps), len(phis)), np.nan)
    for T0, ph, value, valid in zip(temperature, phi, mse, ok):
        matrix[np.searchsorted(temps, T0), np.searchsorted(phis, ph)] = value if valid else np.nan
    return {"temps": temps, "phis": phis, "matrix": matrix, "temperature": temperature,
            "phi": phi, "mse": mse, "ok": ok, "training": training, "held_out": held_out}


def plot_figure(grids, vmax, paper_scale, divisor=1.0, note=""):
    """Two side-by-side maps sharing one colour scale."""
    norm = Normalize(vmin=0, vmax=vmax, clip=False)
    cmap = plt.get_cmap("Reds").copy()            # lower MSE = lighter, as in the paper
    cmap.set_over(cmap(1.0))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (label, g) in zip(axes, grids.items()):
        im = ax.pcolormesh(g["phis"], g["temps"], g["matrix"] / divisor,
                           shading="nearest",
                           cmap=cmap, norm=norm, rasterized=True)
        ax.scatter(g["phi"][g["training"]], g["temperature"][g["training"]], marker="x",
                   s=30, c="navy", lw=.8, label=f"{TRAINING_CONDITIONS} training conditions")
        ax.scatter(g["phi"][g["held_out"]], g["temperature"][g["held_out"]], marker="o",
                   s=55, c="teal", edgecolors="0.2", lw=.8, label="held-out condition")
        if (~g["ok"]).any():
            ax.scatter(g["phi"][~g["ok"]], g["temperature"][~g["ok"]], marker="X", s=45,
                       c="black", label="integration failed")
        ax.set_xticks([.5, .7, .9, 1.1, 1.3, 1.5])
        ax.set_yticks([950, 1000, 1050, 1100, 1150, 1200])
        ax.set_xlim(.475, 1.525)
        ax.set_ylim(943.75, 1206.25)
        ax.set_xlabel("Equivalence ratio")
        ax.set_ylabel("Initial temperature [K]")
        ax.tick_params(direction="in", top=True, right=True, labelleft=True)
        ok, mse = g["ok"], g["mse"] / divisor
        detail = (f"{(mse[ok] > vmax).sum()}/{ok.sum()} errors above colour limit; "
                  if paper_scale else "")
        ax.set_title(f"{label}\n{detail}median {np.median(mse[ok]):.3e}", fontsize=9)
    bar = fig.colorbar(im, ax=axes, fraction=.035, pad=.025,
                       extend="max" if paper_scale else "neither")
    bar.set_label(f"MSE — normalized trajectory loss{note}" if note
                  else "MSE — normalized trajectory loss (Eq. 18)")
    if paper_scale:
        bar.set_ticks(np.linspace(0, PAPER_MSE_MAX, 11))
        bar.formatter = FuncFormatter(lambda value, pos: f"{value / 1e-4:g}")
        bar.update_ticks()
        bar.ax.set_title(r"$\times 10^{-4}$", fontsize=10, pad=10)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.17), ncol=2, fontsize=8,
                   frameon=False)
    fig.suptitle(r"Figure 8A — paper-sized range: 0–10 $\times10^{-4}$; lower MSE = lighter"
                 if paper_scale else
                 f"Figure 8A — full measured range: 0–{vmax:g} (no $\\times10^{{-4}}$ factor); "
                 "lower MSE = lighter",
                 fontsize=13)
    return fig


def make_figure(table_paths=None, output_dir=None, *, time_averaged=False,
                show=False):
    """Paper Figure 8A. Returns ``(figs, results)`` keyed by 'paper' and 'full_range'."""
    paths = {label: GENERALIZATION / f"{stem}_generalization_441.csv"
             for label, stem in DEFAULT_STEMS.items()}
    paths.update(table_paths or {})
    grids = {label: load_grid(path) for label, path in paths.items()}

    n_times = observation_times("hydrogen_fine.npz")      # the grid the 441 losses sum over
    divisor, note, ta_suffix = loss_reduction(n_times, time_averaged)
    maximum = max(float(np.nanmax(g["matrix"])) for g in grids.values()) / divisor
    ceiling = MaxNLocator(nbins=6).tick_values(0, maximum)[-1]

    scale_rows = [{"model": label, "minimum": float(g["mse"][g["ok"]].min()),
                   "median": float(np.median(g["mse"][g["ok"]])),
                   "maximum": float(g["mse"][g["ok"]].max()),
                   "above_paper_scale": int((g["mse"][g["ok"]] > PAPER_MSE_MAX).sum()),
                   "evaluated": int(g["ok"].sum()), "paper_scale_max": PAPER_MSE_MAX}
                  for label, g in grids.items()]

    figs = {}
    for key, paper_scale, vmax, suffix in (("paper", True, PAPER_MSE_MAX, ""),
                                           ("full_range", False, ceiling, "_full_range")):
        fig = plot_figure(grids, vmax, paper_scale, divisor,
                          "" if divisor == 1.0 else f"\n{note}")
        figs[key] = fig
        if output_dir is not None:
            save_figure(fig,
                        f"{output_dir}/fig08a_hydrogen_generalization_441"
                        f"{suffix}{ta_suffix}", dpi=200)
    if show:
        plt.show()
    return figs, {"scale_rows": scale_rows, "grids": grids, "reduction": note,
                  "n_times": n_times}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=FIGURES_HYDROGEN)
    p.add_argument("--time-averaged", action="store_true",
                   help="write a _time_averaged companion (derived diagnostic: Eq. 18 / N_t)")
    args = p.parse_args()
    _, results = make_figure(output_dir=args.output_dir,
                             time_averaged=args.time_averaged)
    print(f"loss reported as {results['reduction']}")
    print(f"{'model':36} {'min':>11} {'median':>11} {'max':>11} {'>paper':>7} {'n':>5}")
    for r in results["scale_rows"]:
        print(f"{r['model']:36} {r['minimum']:11.3e} {r['median']:11.3e} "
              f"{r['maximum']:11.3e} {r['above_paper_scale']:7d} {r['evaluated']:5d}")


if __name__ == "__main__":
    main()
