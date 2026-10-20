"""Figure 3 -- biodiesel trajectory reconstruction under noise.

The ChemKAN Fig. 5 runs at 0/5/10/15 % observation noise predict the six species from the
initial condition alone at the paper's Fig. 3 condition (test row 24 of biodiesel_v2).
Each panel reports the time-averaged per-species loss against the clean trajectory and
against the noisy test observations. Reads finished checkpoints; never trains.

    python chemkan/scripts/figures/fig03_biodiesel_trajectories.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES_AUTHOR,
    add_repo_paths,
    save_figure,
    use_headless_backend,
)

add_repo_paths()



# The paper trains one model per noise level and shows them side by side.
NOISE_LEVELS = (0, 5, 10, 15)


def plot_figure(condition, results, note):
    """Six species (rows) x four noise levels (columns)."""
    species, t_dense, t_obs = condition["species"], condition["t_dense"], condition["t"]
    levels = results["levels"]
    fig, axes = plt.subplots(6, 4, figsize=(14, 14), sharex=True)
    for col, pct in enumerate(NOISE_LEVELS):
        r = levels[pct]
        for row, sp in enumerate(species):
            ax = axes[row, col]
            ax.plot(t_dense, condition["states_dense"][:, row], color="0.35", lw=1.6,
                    label="clean truth")
            ax.plot(t_dense, r["dense"][:, row], color="crimson", lw=1.6, ls="--",
                    label="ChemKAN")
            ax.scatter(t_obs, condition["noisy"][pct][:, row], s=12, color="steelblue",
                       zorder=3, label="observations")
            ax.text(0.97, 0.97 if row < 2 else 0.03,
                    f"Loss (clean): {r['loss_clean'][row]:.3f}\n"
                    f"Loss (obs): {r['loss_obs'][row]:.3f}",
                    transform=ax.transAxes, ha="right",
                    va="top" if row < 2 else "bottom", fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
            if col == 0:
                ax.set_ylabel(sp)
            if row == 0:
                ax.set_title(f"{pct}% noise")
            if row == 5:
                ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    y0 = condition["y0"]
    fig.suptitle(f"Fig. 3 - TG0={y0[0]}, ROH0={y0[1]}, T={condition['T']} K "
                 f"(paper-explicit unseen condition){note}", y=0.995)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- biodiesel_v2 runs
# Data from the biodiesel_v2 figure runs, drawn by the unchanged plot_figure above.

def author_trajectory_data():
    """Test row 24 (the paper's Fig. 3 condition) in plot_figure's format: the condition
    and results["levels"][pct] with the dense ChemKAN prediction and per-species
    time-averaged losses against the clean truth and the noisy test observations."""
    import biodiesel_runs as ar
    case = ar.fig3_case(NOISE_LEVELS)
    clean = ar.dataset()["test_states"][ar.FIG3_TEST_INDEX]
    condition = {"t_dense": case["t_dense"], "t": case["t"], "states_dense": case["truth"],
                 "states": clean, "species": case["species"],
                 "y0": np.round(case["y0"], 3), "T": round(case["T"], 2),
                 "noisy": case["observations"]}
    runs = ar.by_name()
    results = {"levels": {}, "reduction": ar.CONVENTION, "n_times": len(case["t"])}
    for pct in NOISE_LEVELS:
        run = runs[f"fig05_chemkan_noise{pct:02d}"]
        src = ar.config(run)["data_source"]
        rng = np.array(src["species_max"]) - np.array(src["species_min"])
        at_obs = ar.chemkan_predict(run, case["y0"], case["T"], case["t"])
        results["levels"][pct] = {
            "dense": ar.chemkan_predict(run, case["y0"], case["T"], case["t_dense"]),
            "loss_clean": (((at_obs - clean) / rng) ** 2).mean(axis=0),
            "loss_obs": (((at_obs - condition["noisy"][pct]) / rng) ** 2).mean(axis=0),
            "checkpoint": str(run["dir"] / "checkpoint_final.pt")}
    return condition, results


def make_author_figure(output_path=None, *, show=False):
    """Fig. 3 from the biodiesel_v2 runs, drawn by ``plot_figure``."""
    import biodiesel_runs as ar
    condition, results = author_trajectory_data()
    fig = plot_figure(condition, results,
                      f"\nbiodiesel_v2 runs (test row 24); loss: {ar.CONVENTION_SHORT}; "
                      f"{ar.EVAL_SOLVER_NOTE}")
    # plot_figure prints 3 decimals; time-averaged per-species losses need exponents.
    for col, pct in enumerate(NOISE_LEVELS):
        r = results["levels"][pct]
        for row in range(len(condition["species"])):
            fig.axes[row * len(NOISE_LEVELS) + col].texts[0].set_text(
                f"Loss (clean): {r['loss_clean'][row]:.2e}\nLoss (obs): {r['loss_obs'][row]:.2e}")
    save_figure(fig, output_path or FIGURES_AUTHOR / "fig03_trajectories")
    if show:
        plt.show()
    return fig, results


def main():
    use_headless_backend()
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_author_figure()


if __name__ == "__main__":
    main()
