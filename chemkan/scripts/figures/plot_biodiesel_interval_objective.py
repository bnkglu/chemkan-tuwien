"""Plot the observed-interval objective from existing clean biodiesel runs.

Run from any directory with the project's Python environment. No training occurs.
Both final training methods are scored using the SAME observed-interval procedure.
Figures and source tables go into biodiesel_observed_intervals/{figures,tables}.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "chemkan/src"))
sys.path.insert(0, str(ROOT / "chemkan/scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chemkan-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from _data import load_biodiesel, load_input_scaling
from chemkan.dynamics import KineticDynamics
from chemkan.losses import trajectory_mse
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import integrate
from chemkan.temperature import ConstantTemperature
from evaluate_biodiesel import build_kinetic_core, solver_from_ckpt
from train_biodiesel_observed_intervals import accumulate_interval_gradients

EXP = ROOT / "results/experiments/biodiesel_observed_intervals"
SEEDS = (0, 1, 2)
METHODS = ("original", "observed_interval")
METHOD_LABELS = ("Full-trajectory training", "Observed-interval training")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(method, seed):
    if method == "observed_interval":
        directory = EXP / f"seed{seed}"
    elif seed == 0:
        directory = ROOT / "results/reproduction/chemkan/biodiesel/main/direct_autograd_seed0"
    else:
        directory = EXP / f"baseline_original_seed{seed}"
    return directory / "checkpoint_final.pt"


def restore_grid_widths(core):
    """Recover h from stored uniform centers, independent of current grid defaults.

    These legacy checkpoints save centers but not h (a Python float). The RBF
    implementation sets h equal to center spacing. Loading centers alone after
    a default grid change would therefore reconstruct the wrong model.
    """
    grids = {}
    for name in ("add", "lean"):
        edges = getattr(core, name).edges
        centers = edges.centers.detach()
        spacing = centers.diff()
        if not bool(torch.all(spacing > 0)):
            raise ValueError(f"{name}: RBF centers must be increasing")
        torch.testing.assert_close(spacing, spacing[0].expand_as(spacing))
        edges.h = float((centers[-1] - centers[0]) / (centers.numel() - 1))
        grids[name] = {"centers": centers.tolist(), "h": edges.h}
    return grids


@torch.no_grad()
def evaluate_checkpoint(path, data, loss_norm):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if list(ckpt["data"]["species"]) != list(data["species"]):
        raise ValueError("Checkpoint and data species order differ")
    core = build_kinetic_core(ckpt, len(data["species"]), "cpu")
    grids = restore_grid_widths(core)
    solver = solver_from_ckpt(ckpt)
    dynamics = KineticDynamics(core, ConstantTemperature(data["T_const"]),
                              input_normalizer=load_input_scaling(ckpt, "cpu"))
    obs, t = data["species_TBm"], data["t"]
    obs_norm = loss_norm.normalize(obs)
    endpoints = torch.stack([
        integrate(dynamics, obs[j], t[j:j + 2], solver)[-1]
        for j in range(len(t) - 1)
    ])
    squared_error = (loss_norm.normalize(endpoints) - obs_norm[1:]).square()
    objective = trajectory_mse(loss_norm.normalize(endpoints), obs_norm[1:])
    # Independently compare the decomposition with the trainer's actual routine.
    trainer_objective = accumulate_interval_gradients(
        dynamics, obs, obs_norm, t, loss_norm, solver, backward=False)
    torch.testing.assert_close(objective, trainer_objective, rtol=1e-5, atol=1e-9)
    full = integrate(dynamics, obs[0], t, solver)
    full_loss = trajectory_mse(loss_norm.normalize(full), obs_norm)
    if not bool(torch.isfinite(squared_error).all() and torch.isfinite(full).all()):
        raise ValueError("Nonfinite predictions")
    return {
        "objective": float(objective), "full_loss": float(full_loss),
        "sq": squared_error.numpy(), "endpoints": endpoints.numpy(),
        "full": full.numpy(), "grids": grids, "solver": ckpt["solver"],
        "checkpoint": str(path.relative_to(ROOT)),
    }


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(EXP / "figures" / f"{name}.{extension}", dpi=180,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def convergence(histories):
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for seed, color in zip(SEEDS, COLORS):
        epochs = np.array([int(row["epoch"]) for row in histories[seed]])
        loss = np.array([float(row["total_loss"]) for row in histories[seed]])
        ax.semilogy(epochs, loss, color=color, lw=1, label=f"Seed {seed}")
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Observed-interval training objective")
    ax.grid(alpha=.18, which="major")
    ax.legend(frameon=False)
    fig.suptitle("Observed-interval objective convergence\n"
                 "Sum over 29 endpoints; mean over 6 species and 20 training conditions",
                 fontsize=12)
    save(fig, "interval_objective_convergence")


def species_bars(evaluations, species):
    fig, ax = plt.subplots(figsize=(10, 4.9), layout="constrained")
    x, width = np.arange(len(species)), .34
    for i, (method, label, color) in enumerate(zip(METHODS, METHOD_LABELS, COLORS)):
        values = np.stack([evaluations[method, seed]["sq"].sum(0).mean(0)
                           for seed in SEEDS])
        positions = x + (i - .5) * width
        ax.bar(positions, values.mean(0), width, color=color, alpha=.8, label=label)
        for seed in SEEDS:
            ax.scatter(positions + (seed - 1) * .065, values[seed], s=23,
                       marker=("o", "s", "^")[seed], color="black", zorder=3)
    ax.set_xticks(x, species)
    ax.set_yscale("log")
    ax.set_ylabel("Squared normalized endpoint error\n(sum over intervals; mean over training conditions)")
    ax.set_title("Interval error by species at 10,000 updates\n"
                 "Both training methods evaluated with observed-state resets")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.supxlabel("Bars: mean of 3 seeds; dots: individual seeds. "
                  "Mean across the 6 species equals the interval objective.", fontsize=9)
    save(fig, "interval_objective_per_species")


def time_heatmap(evaluations, species, t):
    maps = [np.stack([evaluations[method, seed]["sq"].mean(1) for seed in SEEDS])
            .mean(0).T for method in METHODS]
    norm = LogNorm(vmin=min(a.min() for a in maps), vmax=max(a.max() for a in maps))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", sharey=True)
    ticks = np.array([0, 4, 9, 14, 19, 24, len(t) - 2])
    for ax, values, label in zip(axes, maps, METHOD_LABELS):
        im = ax.imshow(values, aspect="auto", cmap="magma", norm=norm,
                       interpolation="nearest")
        ax.set_title(label)
        ax.set_yticks(np.arange(len(species)), species)
        ax.set_xticks(ticks, [f"{t[j + 1]:.1f}" for j in ticks])
        ax.set_xlabel("Interval endpoint time (s)")
    fig.colorbar(im, ax=axes, label="Squared normalized endpoint error\n"
                 "(mean over 20 training conditions and 3 seeds)", shrink=.85)
    fig.suptitle("Biodiesel diagnostic: interval errors at 10,000 updates\n"
                 "Each column starts from its observed state; both panels share one color scale",
                 fontsize=12)
    save(fig, "interval_objective_time_species")


def endpoint_profiles(result, data):
    # Fixed first condition and seed 0, chosen without ranking prediction errors.
    case = 0
    t, truth = data["t"].numpy(), data["species_TBm"].numpy()[:, case]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained", sharex=True)
    for k, ax in enumerate(axes.flat):
        ax.plot(t, result["full"][:, case, k], color=COLORS[0], lw=1.7,
                label="Complete rollout from initial condition")
        ax.scatter(t, truth[:, k], facecolors="none", edgecolors="#555555", s=27,
                   linewidths=.8, label="Observed concentrations", zorder=3)
        ax.scatter(t[1:], result["endpoints"][:, case, k], color=COLORS[1],
                   marker="x", s=25, linewidths=1.1,
                   label="Independent interval endpoint predictions", zorder=4)
        ax.set_title(data["species"][k])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration (archive units)")
        ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9,
               frameon=False)
    temp = float(data["T_const"].reshape(-1)[case])
    fig.suptitle(f"Observed-interval trained model: seed 0, training condition 0, T = {temp:.2f} K\n"
                 "Orange crosses each use a fresh observed starting state; they are not a continuous trajectory",
                 fontsize=11)
    save(fig, "interval_objective_endpoint_profiles")


def main():
    data = load_biodiesel(split="train", noise_percent=None)
    loss_norm = MinMaxNormalizer(data["u_min"], data["u_max"])
    histories = {seed: read_csv(EXP / f"seed{seed}/history.csv") for seed in SEEDS}
    recorded = {int(row["seed"]): row for row in read_csv(
        EXP / "tables/observed_intervals_final_comparison.csv")}
    evaluations, summary, cells = {}, [], []
    for method in METHODS:
        for seed in SEEDS:
            result = evaluate_checkpoint(checkpoint_path(method, seed), data, loss_norm)
            expected_rollout = float(recorded[seed][
                "obs_train" if method == "observed_interval" else "orig_train"])
            np.testing.assert_allclose(result["full_loss"], expected_rollout,
                                       rtol=1e-5, atol=1e-9)
            expected_objective = None
            if method == "observed_interval":
                if int(histories[seed][-1]["epoch"]) != 10000:
                    raise ValueError("Expected completed 10,000-update interval run")
                expected_objective = float(histories[seed][-1]["total_loss"])
                np.testing.assert_allclose(result["objective"], expected_objective,
                                           rtol=1e-5, atol=1e-9)
            evaluations[method, seed] = result
            per_cell = result["sq"].mean(1)
            np.testing.assert_allclose(per_cell.sum(0).mean(), result["objective"],
                                       rtol=1e-5, atol=1e-9)
            summary.append({"method": method, "seed": seed, "updates": 10000,
                            "interval_objective": result["objective"],
                            "recorded_interval_objective": expected_objective,
                            "full_rollout_train_loss": result["full_loss"],
                            "recorded_full_rollout_train_loss": expected_rollout})
            for j, row in enumerate(per_cell):
                for k, value in enumerate(row):
                    cells.append({"method": method, "seed": seed, "interval": j,
                                  "t_start_s": float(data["t"][j]),
                                  "t_end_s": float(data["t"][j + 1]),
                                  "species": data["species"][k],
                                  "mean_normalized_squared_endpoint_error": float(value)})
            print(f"Verified {method}, seed {seed}: interval={result['objective']:.9g}; "
                  f"full rollout={result['full_loss']:.9g}", flush=True)

    for directory in ("figures", "tables"):
        (EXP / directory).mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    convergence(histories)
    species_bars(evaluations, data["species"])
    time_heatmap(evaluations, data["species"], data["t"].numpy())
    endpoint_profiles(evaluations["observed_interval", 0], data)
    write_csv(EXP / "tables/interval_objective_summary.csv", summary)
    write_csv(EXP / "tables/interval_objective_error_by_time_species.csv", cells)
    print("Saved 4 figures as PDF and PNG and 2 CSV tables.")


if __name__ == "__main__":
    main()
