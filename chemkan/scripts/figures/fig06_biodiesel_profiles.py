"""Reproduce ChemKAN Figure 6 -- biodiesel species profiles under 15 % observation noise.

ChemKAN and DeepONet, both trained on the same 15 %-noise observations, predicting all six
species from the initial condition alone at the paper's Figure-3 condition. Loads finished
checkpoints only; it never trains.

A companion view repeats the comparison at test case 0, the condition the Figure-3 clean
column uses. The two are different trajectories, so their losses are companion readings
rather than a before/after comparison.

    python chemkan/scripts/figures/fig06_biodiesel_profiles.py
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    CHEMKAN_BIODIESEL,
    DATA,
    DEEPONET_BIODIESEL,
    DEEPONET_VERSION,
    FIGURES_BIODIESEL,
    TABLES,
    add_repo_paths,
    load_checkpoint,
    require_file,
    save_figure,
    use_headless_backend,
)

add_repo_paths()

import biodiesel_deeponet as bdon
from _data import load_biodiesel, load_input_scaling
from evaluate_biodiesel import build_kinetic_core, integrate_biodiesel, solver_from_ckpt
from evaluate_biodiesel_deeponet import build_model

from chemkan.losses import trajectory_mse
from chemkan.normalization import MinMaxNormalizer

NOISE_PERCENT = 15


def default_chemkan_run():
    return CHEMKAN_BIODIESEL / f"noise/noise{NOISE_PERCENT:02d}_seed0"


def default_deeponet_run(version=DEEPONET_VERSION):
    return DEEPONET_BIODIESEL / version / f"noise/noise{NOISE_PERCENT:02d}_seed0"


def predict_chemkan(run_dir, y0, temperature, times, n_species):
    ckpt = load_checkpoint(require_file(run_dir / "checkpoint_final.pt", "ChemKAN checkpoint"))
    core = build_kinetic_core(ckpt, n_species, "cpu")
    return integrate_biodiesel(core, load_input_scaling(ckpt, "cpu"), solver_from_ckpt(ckpt),
                               y0, temperature, times, "cpu")[:, 0, :].numpy()


def predict_deeponet(run_dir, y0, temperature, times, n_species, expected_version):
    """DeepONet predicts in normalized space; inverse-transform for a physical plot."""
    ckpt = load_checkpoint(require_file(run_dir / "checkpoint_final.pt", "DeepONet checkpoint"))
    model = build_model(ckpt, "cpu")
    if model.architecture_version != expected_version:
        raise ValueError(f"DeepONet checkpoint is {model.architecture_version!r}, "
                         f"expected {expected_version!r}")
    normalization = ckpt["normalization"]
    full = MinMaxNormalizer(normalization["u_min"], normalization["u_max"])
    loss_norm = full.subset(slice(0, n_species))
    branch, tau = bdon.prepare_inputs({"Y0": y0, "T_const": temperature, "t": times},
                                      full, float(normalization["t_end_s"]))
    with torch.no_grad():
        return loss_norm.denormalize(model(branch, tau))[:, 0, :].numpy(), loss_norm


def clean_mse(prediction, truth, loss_norm):
    """Eq. 22: the prediction scored against the CLEAN trajectory, in normalized space."""
    pred_n = loss_norm.normalize(torch.as_tensor(prediction).unsqueeze(1))
    truth_n = loss_norm.normalize(torch.as_tensor(truth).unsqueeze(1))
    return float(trajectory_mse(pred_n, truth_n))


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


def make_figure(chemkan_run=None, deeponet_run=None, condition_path=None,
                output_path=None, metrics_path=None, *,
                deeponet_version=DEEPONET_VERSION, show=False):
    """Paper Figure 6, at the Figure-3 condition. Returns ``(fig, results)``."""
    condition_path = condition_path or DATA / "biodiesel_fig3_condition.npz"
    with np.load(require_file(condition_path, "Figure-3 condition"), allow_pickle=True) as art:
        t_plot = art["t"]
        times = torch.as_tensor(t_plot, dtype=torch.float32)
        species = [str(s) for s in art["species"]]
        truth, noisy = art["states"], art[f"states_noise{NOISE_PERCENT:02d}"]
        y0 = torch.as_tensor(art["y0"], dtype=torch.float32).unsqueeze(0)
        temperature = torch.tensor(float(art["T"]), dtype=torch.float32).reshape(1)

    m = len(species)
    pred_ck = predict_chemkan(chemkan_run or default_chemkan_run(), y0, temperature, times, m)
    pred_do, loss_norm = predict_deeponet(deeponet_run or default_deeponet_run(),
                                          y0, temperature, times, m, deeponet_version)
    if pred_ck.shape != truth.shape or pred_do.shape != truth.shape:
        raise ValueError(f"prediction shapes {pred_ck.shape}/{pred_do.shape} "
                         f"do not match the reference {truth.shape}")

    results = {"ChemKAN": clean_mse(pred_ck, truth, loss_norm),
               "DeepONet": clean_mse(pred_do, truth, loss_norm)}
    fig = plot_profiles(
        t_plot, species, truth, noisy, pred_ck, pred_do,
        f"Fig. 6 - {NOISE_PERCENT}% noise, corrected DeepONet; same condition as Fig. 3 "
        f"(TG0={float(y0[0, 0]):.2f}, ROH0={float(y0[0, 1]):.2f}, "
        f"T={float(temperature):.1f} K)")
    save_figure(fig, output_path)

    if metrics_path is not None:
        with open(metrics_path, "w") as f:
            json.dump({"initial_state": y0[0].tolist(), "temperature_K": float(temperature),
                       "deeponet_version": deeponet_version, "clean_mse": results,
                       "condition": "Figure 3 dedicated unseen condition",
                       "observation_times": len(times)}, f, indent=2)
    if show:
        plt.show()
    return fig, results


def make_case0_figure(chemkan_run=None, deeponet_run=None, case=0, output_path=None,
                      metrics_path=None, *, deeponet_version=DEEPONET_VERSION, show=False):
    """The same comparison at a held-out test trajectory (the Figure-3 clean column's case).

    A different trajectory from ``make_figure``'s condition, so the two sets of losses are
    companion readings, not a before/after comparison.
    """
    data = load_biodiesel(split="test", noise_percent=NOISE_PERCENT)
    species = [str(s) for s in data["species"]]
    m = len(species)
    times = data["t"]
    truth = data["species_TBm"][:, case].numpy()        # clean reference (Eq. 22)
    noisy = data["targets_TBm"][:, case].numpy()        # the stored noise realization
    y0 = data["Y0"][case:case + 1]                      # the initial state is always clean
    temperature = data["T_const"][case:case + 1]

    pred_ck = predict_chemkan(chemkan_run or default_chemkan_run(), y0, temperature, times, m)
    pred_do, loss_norm = predict_deeponet(deeponet_run or default_deeponet_run(),
                                          y0, temperature, times, m, deeponet_version)
    results = {"ChemKAN": clean_mse(pred_ck, truth, loss_norm),
               "DeepONet": clean_mse(pred_do, truth, loss_norm)}
    fig = plot_profiles(
        times.numpy(), species, truth, noisy, pred_ck, pred_do,
        f"Fig. 6 companion - {NOISE_PERCENT}% noise at the Figure-3 clean-column "
        f"condition (test case {case}: TG0={float(y0[0, 0]):.2f}, "
        f"ROH0={float(y0[0, 1]):.2f}, T={float(temperature[0]):.1f} K)")
    save_figure(fig, output_path)
    if metrics_path is not None:
        with open(metrics_path, "w") as f:
            json.dump({"initial_state": y0[0].tolist(),
                       "temperature_K": float(temperature[0]),
                       "deeponet_version": deeponet_version, "clean_mse": results,
                       "condition": f"test split case {case}",
                       "observation_times": len(times)}, f, indent=2)
    if show:
        plt.show()
    return fig, results


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=FIGURES_BIODIESEL / "fig06_biodiesel_15pct_profiles")
    p.add_argument("--metrics", default=TABLES / "biodiesel_fig6_profile_metrics.json")
    p.add_argument("--case0-output",
                   default=FIGURES_BIODIESEL / "fig06_biodiesel_15pct_profiles_case0")
    p.add_argument("--case0-metrics",
                   default=TABLES / "biodiesel_fig6_profile_metrics_case0.json")
    p.add_argument("--skip-case0", action="store_true")
    args = p.parse_args()

    _, results = make_figure(output_path=args.output, metrics_path=args.metrics)
    for name, value in results.items():
        print(f"{name}: clean-trajectory Eq. 18 MSE at the Fig.-3 condition = {value:.6g}")
    print(f"wrote {args.output}.pdf/.png")
    if not args.skip_case0:
        _, case0 = make_case0_figure(output_path=args.case0_output,
                                     metrics_path=args.case0_metrics)
        for name, value in case0.items():
            print(f"{name}: clean-trajectory Eq. 18 MSE at test case 0 = {value:.6g}")
        print(f"wrote {args.case0_output}.pdf/.png")


if __name__ == "__main__":
    main()
