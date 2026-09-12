"""Reproduce ChemKAN Figure 3 -- biodiesel trajectory reconstruction under noise.

Six species are predicted from the initial condition alone at the paper's explicit unseen
condition, for models trained at 0/5/10/15 % observation noise. Each panel reports the
Eq. 18 loss against the clean trajectory and against the noisy observations.

Loads no training code and never trains: it reads finished checkpoints only.

    python chemkan/scripts/figures/fig03_biodiesel_trajectories.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    CHEMKAN_BIODIESEL,
    DATA,
    FIGURES_BIODIESEL,
    add_repo_paths,
    load_checkpoint,
    loss_reduction,
    require_file,
    save_figure,
    use_headless_backend,
)

add_repo_paths()

from _data import load_biodiesel, load_input_scaling
from evaluate_biodiesel import (
    build_kinetic_core,
    integrate_biodiesel,
    solver_from_ckpt,
)

from chemkan.normalization import MinMaxNormalizer

# The paper trains one model per noise level and shows them side by side.
NOISE_LEVELS = (0, 5, 10, 15)
DENSE_POINTS = 400          # display-only resolution for the single-column variant


def default_run_dir(noise_percent: int):
    """Where each noise level's ChemKAN run lives. 0 % is the main clean run."""
    if noise_percent == 0:
        return CHEMKAN_BIODIESEL / "main/direct_autograd_seed0"
    return CHEMKAN_BIODIESEL / f"noise/noise{noise_percent:02d}_seed0"


def load_condition(condition_path):
    """The paper-explicit unseen condition, with its clean and per-noise-level states."""
    artifact = np.load(require_file(condition_path, "Figure-3 condition"),
                       allow_pickle=True)
    return {
        "t_dense": artifact["t_dense"], "t": artifact["t"],
        "states_dense": artifact["states_dense"], "states": artifact["states"],
        "species": [str(s) for s in artifact["species"]],
        "y0": artifact["y0"], "T": float(artifact["T"]),
        "noisy": {pct: artifact[f"states_noise{pct:02d}"] for pct in NOISE_LEVELS},
    }


def predict(checkpoint_path, y0, temperature, times, n_species):
    """Integrate a trained kinetic core from ``y0`` over ``times`` -> (T, n_species)."""
    ckpt = load_checkpoint(checkpoint_path)
    core = build_kinetic_core(ckpt, n_species, "cpu")
    return integrate_biodiesel(
        core, load_input_scaling(ckpt, "cpu"), solver_from_ckpt(ckpt),
        torch.as_tensor(y0, dtype=torch.float32).unsqueeze(0),
        torch.tensor([temperature]),
        torch.as_tensor(times, dtype=torch.float32), "cpu")[:, 0, :]


def per_species_losses(prediction, reference, normalizer):
    """Eq. 18 per species: squared error on normalized states, summed over time."""
    error = normalizer.normalize(prediction) - normalizer.normalize(
        torch.as_tensor(reference, dtype=torch.float32))
    return (error ** 2).sum(dim=0).numpy()


def plot_clean_condition_column(condition, dense_prediction, training_label, *, full_rollout_loss=None):
    """The paper's six-row clean column, with the training procedure identified.

    ``dense_prediction`` must be a complete prediction from the initial condition;
    observed-reset endpoint predictions do not belong in this figure.
    """
    fig, axes = plt.subplots(6, 1, figsize=(5.6, 12), layout="constrained", sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(condition["t_dense"], condition["states_dense"][:, k],
                color=".25", lw=1.6, label="Clean reference")
        ax.plot(condition["t_dense"], dense_prediction[:, k], color="crimson",
                ls="--", lw=1.6, label="ChemKAN prediction")
        ax.scatter(condition["t"], condition["states"][:, k], s=17,
                   color="steelblue", zorder=3, label="Clean observations")
        ax.set_ylabel(condition["species"][k])
        ax.grid(alpha=.15)
        ax.set_xlim(float(condition["t"][0]), float(condition["t"][-1]))
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, fontsize=9,
               frameon=False)
    y0 = condition["y0"]
    loss_label = ("" if full_rollout_loss is None else
                  f"\nFull-rollout loss = {full_rollout_loss:.9f}\n"
                  f"Normalized error summed over {len(condition['t'])} observation times")
    fig.suptitle("Figure 3 | 0% training noise\n"
                 f"{training_label}\n"
                 f"TG = {y0[0]:.2f}, ROH = {y0[1]:.2f}; other species = 0\n"
                 f"T = {float(condition['T']):.1f} K | Complete rollout{loss_label}",
                 fontsize=10)
    return fig


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


def make_figure(run_dirs=None, condition_path=None, output_path=None, *,
                time_averaged=False, show=False):
    """Paper Figure 3. Returns ``(fig, results)``.

    ``results["levels"][pct]`` holds each noise level's losses; the reduction metadata
    sits beside it so the per-level data can be iterated without tripping over it.

    ``run_dirs`` maps a noise percentage to a run directory, so a notebook can point the
    figure at other runs; anything omitted falls back to the committed reproduction run.
    """
    condition_path = condition_path or DATA / "biodiesel_fig3_condition.npz"
    condition = load_condition(condition_path)
    runs = {pct: default_run_dir(pct) for pct in NOISE_LEVELS}
    runs.update(run_dirs or {})

    # Train-only statistics, exactly as the Eq. 18 loss uses.
    train = load_biodiesel("train")
    normalizer = MinMaxNormalizer(train["u_min"], train["u_max"])
    n_species = len(condition["species"])
    if condition["states"].shape[1] != n_species:
        raise ValueError(f"condition has {condition['states'].shape[1]} species columns "
                         f"but {n_species} species names")

    divisor, note, _ = loss_reduction(len(condition["t"]), time_averaged)
    results = {"levels": {}, "reduction": note, "n_times": len(condition["t"])}
    for pct in NOISE_LEVELS:
        checkpoint = require_file(runs[pct] / "checkpoint_final.pt", f"{pct}% checkpoint")
        dense = predict(checkpoint, condition["y0"], condition["T"],
                        condition["t_dense"], n_species).numpy()
        at_obs = predict(checkpoint, condition["y0"], condition["T"],
                         condition["t"], n_species)
        results["levels"][pct] = {
            "dense": dense,
            "loss_clean": per_species_losses(at_obs, condition["states"],
                                             normalizer) / divisor,
            "loss_obs": per_species_losses(at_obs, condition["noisy"][pct],
                                           normalizer) / divisor,
            "checkpoint": str(checkpoint),
        }

    fig = plot_figure(condition, results,
                      "" if divisor == 1.0 else f"\nloss reported as {note}")
    save_figure(fig, output_path)
    if show:
        plt.show()
    return fig, results


def make_clean_column_figure(run_dir=None, case=0, output_path=None, *,
                             time_averaged=False, show=False):
    """The earlier single-column view: the clean model at one held-out test trajectory.

    Superseded by ``make_figure`` (which uses the paper's own condition) but still part of
    the committed record, so its plotting logic lives here rather than in a notebook. The
    dense curve is display only; the reported MSE is scored on the 30 observation times.
    """
    run_dir = run_dir or CHEMKAN_BIODIESEL / "main/direct_autograd_seed0"
    ckpt = load_checkpoint(require_file(run_dir / "checkpoint_final.pt", "checkpoint"))
    data = load_biodiesel("test")
    species = [str(s) for s in data["species"]]
    m = len(species)
    core = build_kinetic_core(ckpt, m, "cpu")
    input_norm, solver = load_input_scaling(ckpt, "cpu"), solver_from_ckpt(ckpt)
    n_params = sum(p.numel() for p in core.parameters())

    t = data["t"]
    t_dense = np.linspace(float(t[0]), float(t[-1]), DENSE_POINTS)
    y0, temperature = data["species_TBm"][0][case:case + 1], data["T_const"][case:case + 1]
    dense = integrate_biodiesel(core, input_norm, solver, y0, temperature,
                                torch.as_tensor(t_dense, dtype=torch.get_default_dtype())
                                )[:, 0].detach().numpy()
    at_obs = integrate_biodiesel(core, input_norm, solver, y0, temperature, t)[:, 0].detach()
    truth = data["species_TBm"][:, case]

    normalizer = MinMaxNormalizer(data["u_min"][:m], data["u_max"][:m])
    divisor, note, _ = loss_reduction(len(t), time_averaged)
    per_sp = per_species_losses(at_obs, truth.numpy(), normalizer) / divisor
    case_mse = float(per_sp.mean())

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for k, sp in enumerate(species):
        a = ax[k // 3, k % 3]
        a.plot(t, truth[:, k].numpy(), "o", ms=4, color="k", label="observations (30 pts)")
        a.plot(t_dense, dense[:, k], "-", lw=1.6, color="tab:red", label="ChemKAN (dense)")
        a.set_ylabel(sp)
        a.set_title(f"{sp}   MSE = {per_sp[k]:.4f}", fontsize=9)
        if k >= 3:
            a.set_xlabel("t [s]")
        if k == 0:
            a.legend(fontsize=7)
    fig.suptitle(f"Sec. III A.1 / Fig. 3 - clean-data column only   "
                 f"({n_params} parameters, test case {case}, "
                 f"T = {float(temperature[0]):.1f} K, Eq. 18 MSE = {case_mse:.4f})"
                 f"{'' if divisor == 1.0 else chr(10) + 'loss reported as ' + note}\n"
                 f"NOT the paper's Fig. 3 condition; 5/10/15% columns not yet trained",
                 fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=300)
    if show:
        plt.show()
    return fig, {"per_species_mse": per_sp, "case_mse": case_mse, "case": case,
                 "t_dense": t_dense, "reduction": note}


def make_interval_figure(output_path=None, *, show=False):
    """Verified clean interval-model Figure 3, shared by notebook 11 and the CLI."""
    from plot_biodiesel_interval_fig3 import make_figure as make_interval
    return make_interval(output_path=output_path, show=show)


def main():
    use_headless_backend()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed-intervals", action="store_true",
                   help="draw the clean interval-trained seed-0 model and endpoint "
                        "diagnostic into the observed-interval experiment directory")
    p.add_argument("--condition", default=None, help="Figure-3 condition .npz")
    p.add_argument("--output", default=FIGURES_BIODIESEL / "fig03_biodiesel_noise_columns",
                   help="output path without extension (pdf and png are written)")
    p.add_argument("--clean-column-output",
                   default=FIGURES_BIODIESEL / "fig03_biodiesel_clean_column")
    p.add_argument("--skip-clean-column", action="store_true")
    p.add_argument("--time-averaged", action="store_true",
                   help="write a _time_averaged companion (derived diagnostic: Eq. 18 / N_t)")
    args = p.parse_args()
    if args.observed_intervals:
        from plot_biodiesel_interval_fig3 import main as make_interval_fig3
        make_interval_fig3()
        return
    _, _, suffix = loss_reduction(0, args.time_averaged)
    args.output = f"{args.output}{suffix}"
    args.clean_column_output = f"{args.clean_column_output}{suffix}"

    _, results = make_figure(condition_path=args.condition, output_path=args.output,
                             time_averaged=args.time_averaged)
    print(f"loss reported as {results['reduction']}")
    for pct in NOISE_LEVELS:
        r = results["levels"][pct]
        print(f"{pct:2d}% noise: Eq.18 vs clean {r['loss_clean'].mean():.6f} | "
              f"vs observations {r['loss_obs'].mean():.6f}")
    print(f"wrote {args.output}.pdf/.png")
    if not args.skip_clean_column:
        _, clean = make_clean_column_figure(output_path=args.clean_column_output,
                                           time_averaged=args.time_averaged)
        print(f"clean column: test case {clean['case']} Eq.18 MSE = {clean['case_mse']:.4f}")
        print(f"wrote {args.clean_column_output}.pdf/.png")


if __name__ == "__main__":
    main()
