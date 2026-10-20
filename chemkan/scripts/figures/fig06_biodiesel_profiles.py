"""Figure 6 -- biodiesel species profiles under 15 % observation noise.

ChemKAN and DeepONet, both trained on the same 15 %-noise observations, predict all six
species from the initial condition alone at the paper's Fig. 3 condition (test row 24 of
biodiesel_v2). Loads finished checkpoints; never trains.

    python chemkan/scripts/figures/fig06_biodiesel_profiles.py
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



NOISE_PERCENT = 15


def plot_profiles(times, species, truth, noisy, pred_chemkan, pred_deeponet, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for k, sp in enumerate(species):
        ax = axes[k // 3, k % 3]
        ax.scatter(times, noisy[:, k], s=18, color="steelblue", zorder=3,
                   label=f"{NOISE_PERCENT}% noisy observations")
        ax.plot(times, truth[:, k], color="0.35", lw=1.8, label="Ground truth")
        ax.plot(times, pred_chemkan[:, k], color="crimson", lw=1.6, ls="--", label="ChemKAN")
        ax.plot(times, pred_deeponet[:, k], color="seagreen", lw=1.6, ls="-.", label="DeepONet")
        ax.set_title(sp)
        ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- biodiesel_v2 runs
# Data from the biodiesel_v2 figure runs, drawn by the unchanged plot_profiles above.

def author_profile_data() -> dict:
    """Test row 24 at NOISE_PERCENT: clean truth, noisy test observations and both models'
    predictions on the observation grid (as plot_profiles expects), plus each model's
    time-averaged noise-free MSE on this trajectory."""
    import author_runs as ar
    case = ar.fig3_case((NOISE_PERCENT,))
    runs = ar.by_name()
    clean = ar.dataset()["test_states"][ar.FIG3_TEST_INDEX]
    src = ar.config(runs[f"fig05_chemkan_noise{NOISE_PERCENT:02d}"])["data_source"]
    rng = np.array(src["species_max"]) - np.array(src["species_min"])
    out = {"case": case, "truth": clean, "noisy": case["observations"][NOISE_PERCENT],
           "pred": {}, "mse": {}}
    for model, fn in (("chemkan", ar.chemkan_predict), ("deeponet", ar.deeponet_predict)):
        pred = fn(runs[f"fig05_{model}_noise{NOISE_PERCENT:02d}"], case["y0"], case["T"], case["t"])
        out["pred"][model] = pred
        out["mse"][model] = float(np.mean(((pred - clean) / rng) ** 2))
    return out


def make_author_figure(output_path=None, *, data=None, show=False):
    """Fig. 6 from the biodiesel_v2 runs, drawn by ``plot_profiles``."""
    import author_runs as ar
    data = data or author_profile_data()
    case = data["case"]
    fig = plot_profiles(case["t"], case["species"], data["truth"], data["noisy"],
                        data["pred"]["chemkan"], data["pred"]["deeponet"],
                        f"Fig. 6 - {NOISE_PERCENT}% noise, test row 24 (TG0={case['y0'][0]:.3f}, "
                        f"ROH0={case['y0'][1]:.3f}, T={case['T']:.2f} K), biodiesel_v2 runs\n"
                        f"noise-free {ar.CONVENTION_SHORT}: ChemKAN "
                        f"{data['mse']['chemkan']:.2e}, {ar.DEEPONET_LABEL[ar.DEEPONET_VARIANT]} "
                        f"{data['mse']['deeponet']:.2e}\n{ar.EVAL_SOLVER_NOTE}")
    save_figure(fig, output_path or FIGURES_AUTHOR / "fig06_profiles_15pct")
    if show:
        plt.show()
    return fig, data


def main():
    use_headless_backend()
    argparse.ArgumentParser(description=__doc__).parse_args()
    make_author_figure()


if __name__ == "__main__":
    main()
