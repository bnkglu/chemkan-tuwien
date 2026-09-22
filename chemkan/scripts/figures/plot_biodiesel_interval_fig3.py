"""Plot the interval-trained models at the paper's Figure 3 condition.

Writes the endpoint diagnostic for the clean seed-0 model, its verified prediction and
metrics tables, and the Figure-3 layout with one interval-trained model per noise column
(0 / 5 / 10 / 15 %). Uses the existing reference archive and final checkpoints; never
trains a model.
"""

from __future__ import annotations

import json
from pathlib import Path

# Reuse the verified legacy-checkpoint reconstruction and headless plot setup.
from plot_biodiesel_interval_objective import (
    COLORS, EXP, ROOT, checkpoint_path, evaluate_checkpoint, interval_style, read_csv,
    restore_grid_widths, save, write_csv,
)

import matplotlib.pyplot as plt
import numpy as np
import torch

from _data import DATA_DIR, load_biodiesel, load_input_scaling
from chemkan.normalization import MinMaxNormalizer
from evaluate_biodiesel import build_kinetic_core, integrate_biodiesel, solver_from_ckpt
from fig03_biodiesel_trajectories import (
    NOISE_LEVELS, make_figure as make_fig3_columns, plot_clean_condition_column,
)
from common import save_figure, use_headless_backend

CONDITION_TEXT = "TG = 1.94, ROH = 1.43, DG = MG = GL = RCO2R = 0; T = 334.8 K"


def load_condition():
    path = DATA_DIR / "biodiesel_fig3_condition.npz"
    with np.load(path, allow_pickle=True) as archive:
        condition = {key: archive[key].copy() for key in
                     ("t", "t_dense", "states", "states_dense", "y0", "T")}
        condition["species"] = [str(s) for s in archive["species"]]
        np.testing.assert_array_equal(archive["states_noise00"], archive["states"])
    np.testing.assert_allclose(condition["y0"], [1.94, 1.43, 0, 0, 0, 0],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(condition["T"], 334.8, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(condition["states"][0], condition["y0"])
    for split in ("train", "test"):
        split_data = load_biodiesel(split)
        same_y0 = np.isclose(split_data["Y0"].numpy(), condition["y0"],
                             rtol=0, atol=1e-6).all(axis=1)
        same_temp = np.isclose(split_data["T_const"].numpy().reshape(-1),
                              float(condition["T"]), rtol=0, atol=1e-3)
        if np.any(same_y0 & same_temp):
            raise ValueError(f"Expected Figure 3 condition to be separate from {split}")
    data = {
        "t": torch.as_tensor(condition["t"], dtype=torch.float32),
        "species_TBm": torch.as_tensor(condition["states"], dtype=torch.float32)[:, None],
        "T_const": torch.tensor([float(condition["T"])], dtype=torch.float32),
        "species": condition["species"],
    }
    return condition, data, path


def dense_rollout(path, condition, recorded_sparse):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    core = build_kinetic_core(ckpt, len(condition["species"]), "cpu")
    restore_grid_widths(core)
    # Include every observation time exactly; check the dense plotting solve agrees
    # with the separate observation-grid solve used to report the loss.
    observation_times = condition["t"].astype(np.float32)
    dense_times = condition["t_dense"].astype(np.float32)
    combined = np.unique(np.concatenate((observation_times, dense_times)))
    prediction = integrate_biodiesel(
        core, load_input_scaling(ckpt, "cpu"), solver_from_ckpt(ckpt),
        torch.as_tensor(condition["y0"], dtype=torch.float32)[None],
        torch.tensor([float(condition["T"])]), torch.from_numpy(combined))[:, 0].numpy()
    np.testing.assert_allclose(prediction[np.searchsorted(combined, observation_times)],
                               recorded_sparse[:, 0], rtol=1e-5, atol=1e-7)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite dense trajectory")
    return prediction[np.searchsorted(combined, dense_times)]


def plot_endpoint_comparison(condition, result, dense):
    with interval_style():
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", sharex=True)
        for k, ax in enumerate(axes.flat):
            ax.plot(condition["t_dense"], condition["states_dense"][:, k], color=".35",
                    lw=1.2, label="Clean reference trajectory")
            ax.plot(condition["t_dense"], dense[:, k], color=COLORS[0], lw=1.7,
                    label="Complete rollout from initial condition")
            ax.scatter(condition["t"], condition["states"][:, k], facecolors="none",
                       edgecolors=".35", s=27, linewidths=.8, zorder=3,
                       label="Clean observations (30 points)")
            ax.scatter(condition["t"][1:], result["endpoints"][:, 0, k],
                       color=COLORS[1], marker="x", s=25, linewidths=1.1, zorder=4,
                       label="Independent interval endpoint predictions")
            ax.set_title(condition["species"][k])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Concentration (archive units)")
            ax.grid(alpha=.15)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9,
                   frameon=False)
        fig.suptitle("Figure 3 endpoint diagnostic | Clean interval-trained model, seed 0\n"
                     f"{CONDITION_TEXT}\n"
                     "Orange endpoints use fresh observed states; the blue rollout uses only the initial condition",
                     fontsize=11)
        save(fig, "fig3_endpoint_diagnostic")


def make_figure(output_path=None, *, show=False, include_endpoint_diagnostic=False,
                tables_dir=None):
    """Return the verified clean-column figure and metrics; save only when requested.

    The standard Figure 3 entry point shares this reconstruction, including the legacy
    RBF widths and the observation-grid consistency checks. ``output_path`` saves the
    single-column figure; ``tables_dir`` writes the prediction and metrics tables there
    instead of beside ``output_path``, so the tables can be produced without the figure.
    """
    path = checkpoint_path("observed_interval", 0)
    train = load_biodiesel("train")
    loss_norm = MinMaxNormalizer(train["u_min"], train["u_max"])
    # Validate checkpoint reconstruction on the known training result first.
    verified = evaluate_checkpoint(path, train, loss_norm)
    last_row = read_csv(EXP / "seed0/history.csv")[-1]
    if int(last_row["epoch"]) != 10000:
        raise ValueError("Expected the final 10,000-update checkpoint")
    np.testing.assert_allclose(verified["objective"], float(last_row["total_loss"]),
                               rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(verified["full_loss"], float(last_row["full_rollout_train_mse"]),
                               rtol=1e-5, atol=1e-9)
    condition, data, condition_path = load_condition()
    result = evaluate_checkpoint(path, data, loss_norm)
    dense = dense_rollout(path, condition, result["full"])
    if include_endpoint_diagnostic:
        plot_endpoint_comparison(condition, result, dense)
    with interval_style():
        fig = plot_clean_condition_column(
            condition, dense, "Observed-interval training, seed 0; 10,000 epochs",
            full_rollout_loss=result["full_loss"])
        save_figure(fig, output_path, dpi=180)

    sparse_rows = []
    for j, time in enumerate(condition["t"]):
        for k, species in enumerate(condition["species"]):
            sparse_rows.append({"time_s": float(time), "species": species,
                                "clean_reference": float(condition["states"][j, k]),
                                "full_rollout": float(result["full"][j, 0, k]),
                                "independent_endpoint": (float(result["endpoints"][j - 1, 0, k])
                                                         if j else None)})
    dense_rows = [
        {"time_s": float(time), "species": species,
         "clean_reference": float(condition["states_dense"][j, k]),
         "full_rollout": float(dense[j, k])}
        for j, time in enumerate(condition["t_dense"])
        for k, species in enumerate(condition["species"])]
    full_sq = (loss_norm.normalize(torch.from_numpy(result["full"])) -
               loss_norm.normalize(data["species_TBm"])).square()
    metadata = {
        "checkpoint": result["checkpoint"],
        "reference_archive": str(condition_path.relative_to(ROOT)),
        "condition": "Paper Figure 3; distinct from canonical train and test conditions",
        "y0": condition["y0"].tolist(), "temperature_K": float(condition["T"]),
        "training_method": "observed_interval", "training_noise_percent": 0,
        "seed": 0, "optimizer_updates": 10000,
        "n_observation_times": len(condition["t"]),
        "n_dense_display_times": len(condition["t_dense"]),
        "full_rollout_loss_timesummed": result["full_loss"],
        "conditional_interval_loss_timesummed": result["objective"],
        "per_species_full_rollout_error_timesummed": dict(zip(
            condition["species"], full_sq.sum(0).mean(0).tolist())),
        "per_species_interval_error_timesummed": dict(zip(
            condition["species"], result["sq"].sum(0).mean(0).tolist())),
        "loss_reduction": "Mean over 6 species; sum over 30 saved times for full "
                          "rollout (initial error zero), or 29 interval endpoints. "
                          "One condition. Train-only species min/max normalization.",
        "interpretation": "The endpoint diagnostic supplies intermediate observations "
                          "at the unseen condition. Only the full rollout is a prediction "
                          "from initial conditions comparable in evaluation procedure "
                          "to Figure 3. This is an interval-trained model, not a claim "
                          "that the paper used interval training.",
        "reference_origin": "Existing synthetic trajectory from the repository's "
                            "mechanistic biodiesel ODE.",
        "solver": result["solver"], "grids": result["grids"],
    }
    tables = (Path(tables_dir) if tables_dir is not None else
              Path(output_path).parent.parent / "tables" if output_path is not None else None)
    if tables is not None:
        tables.mkdir(parents=True, exist_ok=True)
        write_csv(tables / "interval_fig3_seed0_predictions.csv", sparse_rows)
        write_csv(tables / "interval_fig3_seed0_dense.csv", dense_rows)
        (tables / "interval_fig3_seed0_metrics.json").write_text(
            json.dumps(metadata, indent=2) + "\n")
    if show:
        plt.show()
    return fig, metadata


def interval_run_dirs():
    """One interval-trained seed-0 run per Figure-3 noise column; 0 % is the clean run."""
    return {pct: (EXP / "seed0" if pct == 0 else EXP / f"noise/noise{pct:02d}_seed0")
            for pct in NOISE_LEVELS}


def make_noise_columns(run_dirs=None, *, output_path=None, show=False):
    """Observed-interval Figure 3 (one interval model per noise column) in INTERVAL_STYLE.

    Wraps the shared Figure-3 function, which also draws the reproduction's default-styled
    Figure 3; it saves inside itself, so the style covers drawing and saving.
    """
    with interval_style():
        return make_fig3_columns(run_dirs=run_dirs or interval_run_dirs(),
                                 output_path=output_path, show=show)


def main():
    use_headless_backend()
    # Verified clean seed-0 reconstruction: endpoint diagnostic and the metrics tables.
    # Its single-column figure is not saved; the noise-column figure below supersedes it.
    fig, metrics = make_figure(include_endpoint_diagnostic=True, tables_dir=EXP / "tables")
    plt.close(fig)
    print("Figure 3 condition, seed 0: conditional interval loss = "
          f"{metrics['conditional_interval_loss_timesummed']:.9g}; full-rollout loss = "
          f"{metrics['full_rollout_loss_timesummed']:.9g}")

    # The paper's layout: six species by four training-noise columns, one model each.
    columns, results = make_noise_columns(output_path=EXP / "figures/fig3_noise_columns")
    plt.close(columns)
    for pct in NOISE_LEVELS:
        r = results["levels"][pct]
        print(f"{pct:2d}% noise: Eq.18 vs clean {r['loss_clean'].mean():.6f} | "
              f"vs its observations {r['loss_obs'].mean():.6f}")
    print("Saved the endpoint diagnostic, the noise-column figure, and the seed-0 tables.")


if __name__ == "__main__":
    main()
