"""Reproduce ChemKAN Figure 7 -- hydrogen trajectories at the two published conditions.

Temperature and nine species predicted from the initial state alone, at the paper's
training condition (T0 = 1050 K, phi = 0.9) and its held-out condition (T0 = 1150 K,
phi = 1.3), for two Stage-2 initializations. Loads finished checkpoints only.

Two figures are produced together, one per initialization, because they share one set of
y-axis limits per row -- computed across both models and both conditions so the panels are
directly comparable.

    python chemkan/scripts/figures/fig07_hydrogen_trajectories.py
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    CHEMKAN_HYDROGEN,
    DATA,
    FIGURES_HYDROGEN,
    TABLES,
    add_repo_paths,
    load_checkpoint,
    loss_reduction,
    relative_to_root,
    require_file,
    save_figure,
    use_headless_backend,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

add_repo_paths()

from _data import load_input_scaling
from evaluate_hydrogen import build_chemkan, solver_from_ckpt

from chemkan.dynamics import ChemKANDynamics
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import integrate

# Panel rows: temperature, major species, minor species.
GROUPS = [["T"], ["H2", "O2", "H2O", "O", "OH"], ["H", "HO2", "H2O2"]]
CONDITIONS = [(1050.0, 0.9, "training"), (1150.0, 1.3, "held-out")]
DENSE_POINTS = 601

# DISPLAY ONLY. Each species is plotted multiplied by 10**power so that curves spanning
# five decades share one axis. The multiplier travels with the legend entry. Losses are
# always computed on the UNSCALED states.
DISPLAY_POWER = {"T": 0, "H2": 2, "O2": 1, "H2O": 1, "O": 2, "OH": 2,
                 "H": 3, "HO2": 4, "H2O2": 6}
TEX = {"H2": r"H_2", "O2": r"O_2", "H2O": r"H_2O", "O": "O", "OH": "OH",
       "H": "H", "HO2": r"HO_2", "H2O2": r"H_2O_2"}
# Paper y-ranges per row; widened only where our data exceeds them.
PAPER_ROW_LIMITS = [(1000., 3000.), (0., 4.), (0., 7.)]
# Symlog linear band for the true-scale companion: some predicted mass fractions
# go negative at the held-out condition, and a log axis would drop them silently.
LINTHRESH = 1e-6

BASE_ON_N4 = CHEMKAN_HYDROGEN / "diagnostics/base_on_n4"
DEFAULT_RUNS = {
    "H0 (primary, random init)": BASE_ON_N4 / "random_stage2_10000_seed0",
    "Hnorm1 (labelled init comparison)": BASE_ON_N4 / "normmatched_dir1_stage2_10000",
}
TAGS = {"H0 (primary, random init)": "H0",
        "Hnorm1 (labelled init comparison)": "Hnorm1"}


def load_reference(data_path=None):
    """Cantera reference trajectories, species names and the Eq. 18 normalizer."""
    npz = np.load(require_file(data_path or DATA / "hydrogen.npz", "hydrogen dataset"),
                  allow_pickle=True)
    species = [str(s) for s in npz["species"]]
    return {"t": npz["t"], "ics": npz["ics"], "states": npz["states"], "species": species,
            "normalizer": MinMaxNormalizer(torch.as_tensor(npz["u_min"]),
                                           torch.as_tensor(npz["u_max"]))}


def evaluate_condition(run_dir, reference, T0, phi, t_dense):
    """Integrate one condition from its initial state; return curves and per-state losses."""
    ckpt = load_checkpoint(require_file(run_dir / "checkpoint_final.pt", "checkpoint"))
    model = build_chemkan(ckpt, len(reference["species"]), "cpu")
    dynamics = ChemKANDynamics(model, input_normalizer=load_input_scaling(ckpt, "cpu"))
    solver = solver_from_ckpt(ckpt)

    matches = np.flatnonzero(np.isclose(reference["ics"][:, 0], T0)
                             & np.isclose(reference["ics"][:, 1], phi))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one dataset condition at T0={T0}, phi={phi}; "
                         f"found {len(matches)}")
    ref = reference["states"][int(matches[0])]
    u0 = torch.as_tensor(ref[0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dense = integrate(dynamics, u0, torch.as_tensor(t_dense, dtype=torch.float32),
                          solver)[:, 0].numpy()
        at_obs = integrate(dynamics, u0,
                           torch.as_tensor(reference["t"], dtype=torch.float32), solver)[:, 0]
        error = (reference["normalizer"].normalize(at_obs)
                 - reference["normalizer"].normalize(torch.as_tensor(ref, dtype=torch.float32)))
        losses = error.square().sum(0).numpy()
    if not (np.isfinite(dense).all() and np.isfinite(losses).all()):
        raise ValueError(f"non-finite prediction at T0={T0}, phi={phi}")
    return {"ref": ref, "pred": dense, "losses": losses}


def shared_row_limits(cache, species):
    """One y-range per row, spanning both models and both conditions (display-scaled)."""
    limits = []
    for row, names in enumerate(GROUPS):
        lo, hi = PAPER_ROW_LIMITS[row]
        for item in cache.values():
            for sp in names:
                k = len(species) if sp == "T" else species.index(sp)
                for arr in (item["ref"], item["pred"]):
                    values = arr[:, k] * 10. ** DISPLAY_POWER[sp]
                    lo = min(lo, float(values.min()))
                    hi = max(hi, float(values.max()))
        paper_lo, paper_hi = PAPER_ROW_LIMITS[row]
        pad = .035 * (hi - lo)
        limits.append((lo - pad if lo < paper_lo else lo,
                       hi + pad if hi > paper_hi else hi))
    return limits


def plot_model(label, cache, species, t_obs, t_dense, row_limits, divisor=1.0, note=""):
    """Three rows (temperature, major, minor) x two conditions for one initialization."""
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_of = {sp: colours[j % len(colours)] for names in GROUPS
                 for j, sp in enumerate(names)}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.8))
    for col, (T0, phi, kind) in enumerate(CONDITIONS):
        item = cache[label, T0, phi]
        ref, pred, losses = item["ref"], item["pred"], item["losses"]
        for row, names in enumerate(GROUPS):
            ax = axes[row, col]
            indices = [len(species) if sp == "T" else species.index(sp) for sp in names]
            for sp, k in zip(names, indices):
                scale, colour = 10. ** DISPLAY_POWER[sp], colour_of[sp]
                ax.plot(t_dense * 1e3, pred[:, k] * scale, color=colour, lw=1.6,
                        label=rf"${TEX[sp]}\times10^{{{DISPLAY_POWER[sp]}}}$" if row else None)
                ax.plot(t_obs * 1e3, ref[:, k] * scale, "o", color=colour, ms=3.5,
                        mec="0.3", mew=.35, alpha=.65)
            ax.set_xlim(-.01, .615)
            ax.set_ylim(*row_limits[row])
            ax.set_xticks(np.arange(0, .61, .1))
            ax.tick_params(direction="in", top=True, right=True, length=5)
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("T [K]" if row == 0
                          else "Displayed Y = mass fraction × legend factor")
            if row == 0 and row_limits[row] == (1000., 3000.):
                ax.set_yticks(np.arange(1000, 3001, 500))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=7 if row == 2 else 5))
            letter = "ABC"[row] if col == 0 else "DEF"[row]
            ax.set_title(f"({letter})", loc="left", fontsize=13)
            ax.text(.97, .94,
                    f"Normalized loss = {losses[indices].mean() / divisor:.3e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
            if row == 0:
                ax.set_title(f"{kind}: $T_0$={T0:g} K, $\\phi$={phi:g}\n"
                             f"10-state loss = {losses.mean() / divisor:.3e}", fontsize=9)
            else:
                ax.legend(loc="upper left", fontsize=10, ncol=2, framealpha=.9,
                          title="Species × display multiplier", title_fontsize=9,
                          borderpad=.35, handlelength=1.3, columnspacing=1.1,
                          labelspacing=.3)
    fig.suptitle(f"Figure 7 — {label}{note}", fontsize=14, y=.995)
    fig.legend(handles=[Line2D([], [], color="0.35", marker="o", ls="none", ms=4,
                               label="Cantera reference (50 observation times)"),
                        Line2D([], [], color="0.35", lw=1.6,
                               label="ChemKAN (601-point curve)")],
               loc="lower center", bbox_to_anchor=(.5, .055), ncol=2, frameon=False)
    fig.text(.5, .035, r"Read the legend: $H_2O_2\times10^6$ means plotted 1 = actual mass "
             r"fraction $10^{-6}$; plotted 5 = $5\times10^{-6}$.",
             ha="center", fontsize=10, color="0.15")
    fig.text(.5, .006, "Display choices: O₂ ×10¹ and H ×10³ (paper annotations differ); "
             "losses use unscaled states.", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, .10, 1, .975), h_pad=1.3)
    return fig


def evaluate_all(runs, reference, t_dense):
    """Evaluate every (model, condition) pair once; both Figure-7 views share the result."""
    species = reference["species"]
    cache, rows = {}, []
    for label, run_dir in runs.items():
        for T0, phi, kind in CONDITIONS:
            item = evaluate_condition(run_dir, reference, T0, phi, t_dense)
            cache[label, T0, phi] = item
            losses = item["losses"]
            rows.append({"model": label, "condition": f"{T0:g}/{phi:g}", "role": kind,
                         "checkpoint": relative_to_root(run_dir / "checkpoint_final.pt"),
                         "case_Eq18_MSE": float(losses.mean()), "T_MSE": float(losses[-1]),
                         **{f"{sp}_MSE": float(losses[k]) for k, sp in enumerate(species)}})
    return cache, rows


def plot_model_true_scale(label, cache, species, t_obs, t_dense, row_limits):
    """The same predictions with NO display multipliers, on a symlog species axis."""
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colour_of = {sp: colours[j % len(colours)] for names in GROUPS
                 for j, sp in enumerate(names)}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.8))
    for col, (T0, phi, kind) in enumerate(CONDITIONS):
        item = cache[label, T0, phi]
        ref, pred, losses = item["ref"], item["pred"], item["losses"]
        for row, names in enumerate(GROUPS):
            ax = axes[row, col]
            indices = [len(species) if sp == "T" else species.index(sp) for sp in names]
            for sp, k in zip(names, indices):
                ax.plot(t_dense * 1e3, pred[:, k], color=colour_of[sp], lw=1.6,
                        label=rf"${TEX[sp]}$" if row else None)
                ax.plot(t_obs * 1e3, ref[:, k], "o", color=colour_of[sp], ms=3.5,
                        mec="0.3", mew=.35, alpha=.65)
            ax.set_xlim(-.01, .615)
            ax.set_xticks(np.arange(0, .61, .1))
            ax.tick_params(direction="in", top=True, right=True, length=5)
            ax.set_xlabel("Time [ms]")
            if row == 0:
                ax.set_ylabel("T [K]")
                ax.set_ylim(*row_limits[0])
                ax.set_yticks(np.arange(1000, 3001, 500))
                ax.set_title(f"{kind}: $T_0$={T0:g} K, $\\phi$={phi:g}\n"
                             f"10-state loss = {losses.mean():.3e}", fontsize=9)
            else:
                ax.set_ylabel("Y (mass fraction, true scale)")
                ax.set_yscale("symlog", linthresh=LINTHRESH)
                ax.axhspan(-LINTHRESH, LINTHRESH, color="0.9", zorder=0)
                lo = min(0., min(pred[:, k].min() for k in indices)) * 1.6 - LINTHRESH
                hi = max(max(pred[:, k].max() for k in indices),
                         max(ref[:, k].max() for k in indices)) * 1.6
                ax.set_ylim(lo, hi)
            letter = "ABC"[row] if col == 0 else "DEF"[row]
            ax.set_title(f"({letter})", loc="left", fontsize=13)
            ax.text(.97, .06 if row else .94,
                    f"Normalized loss = {losses[indices].mean():.3e}",
                    transform=ax.transAxes, ha="right", va="bottom" if row else "top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
            if row:
                ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=.8,
                          borderpad=.35, handlelength=1.3, columnspacing=1.1,
                          labelspacing=.3)
    fig.suptitle(f"Figure 7 \u2014 {label} (true mass fraction, no display multipliers)",
                 fontsize=14, y=.995)
    fig.legend(handles=[Line2D([], [], color="0.35", marker="o", ls="none", ms=4,
                               label="Cantera reference (50 observation times)"),
                        Line2D([], [], color="0.35", lw=1.6,
                               label="ChemKAN (601-point curve)"),
                        Patch(facecolor="0.9",
                              label=f"linear band |Y| < {LINTHRESH:g} (symlog)")],
               loc="lower center", bbox_to_anchor=(.5, .015), ncol=3, frameon=False)
    fig.text(.5, .006, "Species at true mass fraction; losses are computed on the "
             "train-min-max-normalized states.", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, .065, 1, .975), h_pad=1.3)
    return fig


def make_true_scale_figure(run_dirs=None, data_path=None, output_dir=None,
                           precomputed=None, *, show=False):
    """Figure-7 companion on the true mass-fraction scale. Returns ``(figs, results)``.

    Pass ``precomputed`` (the ``results`` from ``make_figure``) to reuse its evaluation
    instead of integrating everything again.
    """
    if precomputed is not None:
        cache, reference = precomputed["cache"], precomputed["reference"]
        row_limits, t_dense = precomputed["row_limits"], precomputed["t_dense"]
    else:
        runs = dict(DEFAULT_RUNS)
        runs.update(run_dirs or {})
        reference = load_reference(data_path)
        t_dense = np.linspace(float(reference["t"][0]), float(reference["t"][-1]),
                              DENSE_POINTS)
        cache, _ = evaluate_all(runs, reference, t_dense)
        row_limits = shared_row_limits(cache, reference["species"])

    species = reference["species"]
    negatives = {}
    for (label, T0, phi), item in cache.items():
        bad = [sp for k, sp in enumerate(species) if item["pred"][:, k].min() < 0]
        if bad:
            negatives[label, T0, phi] = bad

    figs = {}
    for label in {key[0] for key in cache}:
        fig = plot_model_true_scale(label, cache, species, reference["t"], t_dense,
                                    row_limits)
        figs[label] = fig
        if output_dir is not None:
            save_figure(
                fig,
                f"{output_dir}/fig07_hydrogen_trajectories_{TAGS[label]}_true_scale",
                dpi=200)
    if show:
        plt.show()
    return figs, {"negatives": negatives}


def make_figure(run_dirs=None, data_path=None, output_dir=None, table_path=None, *,
                time_averaged=False, show=False):
    """Paper Figure 7. Returns ``(figs, results)``.

    ``figs`` maps each initialization label to its figure -- both are produced together
    because they share y-axis limits. ``results`` is one row per model and condition with
    the per-state Eq. 18 losses.
    """
    runs = dict(DEFAULT_RUNS)
    runs.update(run_dirs or {})
    reference = load_reference(data_path)
    species = reference["species"]
    t_dense = np.linspace(float(reference["t"][0]), float(reference["t"][-1]), DENSE_POINTS)

    cache, rows = evaluate_all(runs, reference, t_dense)
    row_limits = shared_row_limits(cache, species)
    divisor, note, ta_suffix = loss_reduction(len(reference["t"]), time_averaged)
    figs = {}
    for label in runs:
        fig = plot_model(label, cache, species, reference["t"], t_dense, row_limits,
                         divisor, "" if divisor == 1.0 else f"  —  {note}")
        figs[label] = fig
        if output_dir is not None:
            save_figure(fig,
                        f"{output_dir}/fig07_hydrogen_trajectories_{TAGS[label]}"
                        f"{ta_suffix}", dpi=200)
    if table_path is not None:
        with open(table_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if show:
        plt.show()
    return figs, {"rows": rows, "row_limits": row_limits, "cache": cache,
                  "reference": reference, "t_dense": t_dense, "reduction": note}


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=FIGURES_HYDROGEN)
    p.add_argument("--table", default=TABLES / "fig07_hydrogen_per_state_mse.csv")
    p.add_argument("--time-averaged", action="store_true",
                   help="write a _time_averaged companion (derived diagnostic: Eq. 18 / N_t)")
    args = p.parse_args()
    _, results = make_figure(output_dir=args.output_dir, table_path=args.table,
                             time_averaged=args.time_averaged)
    print(f"loss reported as {results['reduction']}")
    _, companion = make_true_scale_figure(output_dir=args.output_dir, precomputed=results)
    for r in results["rows"]:
        print(f"{r['model']:36} {r['condition']:>10} {r['role']:>9}  "
              f"10-state {r['case_Eq18_MSE']:.4e}  T {r['T_MSE']:.4e}")
    print("Shared Figure-7 limits (temperature, major species, minor species):",
          results["row_limits"])
    print("Predicted mass fractions that go negative "
          "(visible only on the symlog companion):")
    for (label, T0, phi), bad in sorted(companion["negatives"].items()):
        print(f"  {label:38s} T0={T0:.0f} K phi={phi}: {', '.join(bad)}")
    if not companion["negatives"]:
        print("  none")


if __name__ == "__main__":
    main()
