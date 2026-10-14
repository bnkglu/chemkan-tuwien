"""Evaluate saved Hydrogen Stage 2 FGD pilots and write a comparison report.

Run with PYTHONPATH=chemkan/src. This script only evaluates saved models; it never
constructs an optimizer or performs a training update. The final checkpoint from
each run is used, including the restored checkpoint from an interrupted/failed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS, assert_species_order
from _data import DATA_DIR, load_hydrogen
from author_repo_match.train_author_matched_hydrogen import (
    Stage2Dynamics,
    build_model,
    configure_system,
)
from train_adaptive_fgd import integrate_with_budget

from chemkan.adaptive_fgd import adaptive_edges
from chemkan.losses import element_conservation_loss, trajectory_mse
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def first_available(*values):
    return next((value for value in values if value is not None), None)


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_records(run_dir: Path, checkpoint: dict) -> tuple[list[dict], list[str]]:
    notes = []
    records = checkpoint.get("records")
    if records is None:
        records = []
        lines = (run_dir / "diagnostics.jsonl").read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                notes.append("Ignored an incomplete final diagnostics line.")
    completed = int(checkpoint.get("completed_steps", len(records)))
    if len(records) < completed:
        raise ValueError(f"{run_dir}: fewer records than checkpoint completed_steps")
    if len(records) > completed:
        notes.append("Diagnostics after the saved checkpoint were excluded.")
    return records[:completed], notes


@torch.no_grad()
def evaluate_split(model, normalizer, data, count, nt, t_ref, solver, pinn, dtype, max_evaluations):
    physical_target = data["full_TBm1"][:nt, :count].to(dtype=dtype)
    target = normalizer.normalize(physical_target)
    times = data["t"][:nt].to(dtype=dtype)
    prediction = integrate_with_budget(
        Stage2Dynamics(model, t_ref), target[0], times, solver, max_evaluations
    )
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("nonfinite model trajectory")
    physical_prediction = normalizer.denormalize(prediction)
    squared = (prediction - target).square()
    state_count = target.shape[-1]
    # Both contributions retain Eq.18's division by all ten output states.
    species_mse = squared[..., :-1].sum(dim=(0, 2)).mean() / state_count
    temperature_mse = squared[..., -1].sum(dim=0).mean() / state_count
    mse = trajectory_mse(prediction, target)
    raw_pinn = element_conservation_loss(
        physical_prediction[..., :-1],
        ELEMENT_COUNTS.to(dtype=dtype),
        ATOMIC_WEIGHTS.to(dtype=dtype),
        MOLAR_WEIGHTS.to(dtype=dtype),
    )
    weighted_pinn = raw_pinn * (float(pinn["alpha"]) if pinn["enabled"] else 0.0)
    pred_temperature = physical_prediction[..., -1]
    reference_temperature = physical_target[..., -1]
    pred_rise = pred_temperature.max(dim=0).values - pred_temperature[0]
    reference_rise = reference_temperature.max(dim=0).values - reference_temperature[0]
    summary = {
        "loss": float(mse + weighted_pinn),
        "trajectory_mse": float(mse),
        "species_mse_contribution": float(species_mse),
        "temperature_mse_contribution": float(temperature_mse),
        "raw_element_conservation_penalty": float(raw_pinn),
        "weighted_element_conservation_penalty": float(weighted_pinn),
        "temperature_mae_K": float((pred_temperature - reference_temperature).abs().mean()),
        "median_predicted_max_temperature_rise_K": float(pred_rise.median()),
        "median_reference_max_temperature_rise_K": float(reference_rise.median()),
        "predicted_max_temperature_rise_K_per_condition": pred_rise.tolist(),
        "reference_max_temperature_rise_K_per_condition": reference_rise.tolist(),
        "trajectories": int(count),
        "time_points": int(nt),
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in summary.values()):
        raise ValueError("nonfinite evaluation metric")
    curves = {
        "time_microseconds": (times * 1e6).tolist(),
        "predicted_temperature_K": pred_temperature.tolist(),
        "reference_temperature_K": reference_temperature.tolist(),
    }
    return summary, curves


def evaluate_run(run_dir, train, test, archive_sha):
    config = read_json(run_dir / "config.json")
    if config is None:
        raise FileNotFoundError(run_dir / "config.json")
    provenance = config["dataset"]
    if config["system"] != "hydrogen-stage2" or provenance.get("state_coordinates") != "normalized":
        raise ValueError(f"{run_dir}: expected author-matched normalized Hydrogen Stage 2")
    if provenance["sha256"] != archive_sha:
        raise ValueError(f"{run_dir}: current dataset does not match the training archive")
    expected = {
        "species_dim": 9, "hidden_dim": 3, "num_basis": 5, "n_mu": 3, "use_base_act": False,
    }
    if config["architecture"] != expected:
        raise ValueError(f"{run_dir}: unsupported architecture")
    checkpoint_path = run_dir / "checkpoint_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != config:
        raise ValueError(f"{run_dir}: checkpoint and disk configurations disagree")
    records, notes = load_records(run_dir, checkpoint)
    dtype = getattr(torch, config["dtype"])
    normalizer = MinMaxNormalizer(train["u_min"], train["u_max"]).to(dtype=dtype)
    saved_normalizer = checkpoint["loss_normalizer"]
    for name, value in normalizer.state_dict().items():
        if not torch.equal(value, saved_normalizer[name].to(dtype=dtype)):
            raise ValueError(f"{run_dir}: saved {name} differs from current training statistics")
    configure_system("hydrogen")
    model = build_model().to(dtype=dtype)
    if config["method"] in {"adaptive-fgd", "fixed-fgd"}:
        blocks = adaptive_edges(model, initial_intervals=config["fgd"]["initial_intervals"])
    elif config["method"] == "adam":
        blocks = {}
    else:
        raise ValueError(f"{run_dir}: unsupported method {config['method']}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.requires_grad_(False)
    metrics = read_json(run_dir / "metrics.json")
    failure = read_json(run_dir / "failure.json")
    accepted = [r for r in records if (r.get("step_size") or 0) > 0 and not r.get("stationary", False)]
    comparable_losses = [
        r for r in accepted if finite_number(r.get("loss_after")) and finite_number(r.get("loss_before"))
    ]
    trial_failures = [failure for r in records for failure in (r.get("solver_trial_failures") or [])]
    failure_types = {}
    for trial_failure in trial_failures:
        kind = trial_failure.get("type", "unknown")
        failure_types[kind] = failure_types.get(kind, 0) + 1
    training_cap = int(config.get("max_rhs_evaluations") or 0)
    evaluation_cap = max(50000, training_cap)
    latest_recorded_time = next(
        (r["elapsed_seconds"] for r in reversed(records) if finite_number(r.get("elapsed_seconds"))), None
    )
    if any(r.get("loss_after") is None or r.get("test_loss") is None for r in records):
        notes.append("Some saved updates lack a completed post-update evaluation; missing history values are omitted from comparisons and left as gaps in plots.")
    if config.get("resume_from"):
        notes.append("Recorded timing includes the source checkpoint's elapsed offset and the resumed run, but excludes work after that source checkpoint in its original process.")
    summary = {
        "run_directory": str(run_dir.resolve()),
        "method": config["method"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "dataset_sha256": provenance["sha256"],
        "initial_model_sha256": provenance.get("initial_model_sha256"),
        "stage1_checkpoint_sha256": provenance.get("stage1_from", {}).get("sha256"),
        "t_ref_seconds": provenance["t_ref_seconds"],
        "dtype": config["dtype"],
        "training_solver": config["solver"],
        "training_max_rhs_evaluations_per_solve": training_cap,
        "evaluation_max_rhs_evaluations_per_solve": evaluation_cap,
        "resume_from": config.get("resume_from"),
        "training_timing_scope": config.get("timing_scope"),
        "fgd": config["fgd"],
        "functional_input_coordinates": config.get("functional_input_coordinates"),
        "requested_steps": config["steps"],
        "completed_steps": int(checkpoint.get("completed_steps", len(records))),
        "accepted_updates": len(accepted),
        "certified_updates": sum(bool(r.get("certified", False)) for r in accepted),
        "nonmonotone_loss_increases": sum(r["loss_after"] > r["loss_before"] for r in comparable_losses),
        "updates_with_comparable_before_after_losses": len(comparable_losses),
        "updates_with_complete_train_test_evaluation": sum(
            finite_number(r.get("loss_after")) and finite_number(r.get("test_loss")) for r in records
        ),
        "solver_trial_failure_count": len(trial_failures),
        "solver_trial_failures_by_type": failure_types,
        "solver_trial_failure_count_scope": "Failures recorded inside saved update records; a final rejected/interrupted update may have no saved trial diagnostics.",
        "final_intervals": {name: block.intervals for name, block in blocks.items()},
        "recorded_elapsed_seconds": first_available(
            (metrics or {}).get("elapsed_seconds"), checkpoint.get("elapsed_seconds"), latest_recorded_time
        ),
        "status": "failed_or_interrupted" if failure else metrics.get("status", "unknown") if metrics else "unknown",
        "stop_reason": failure or (metrics or {}).get("stop_reason"),
        "failure": failure,
        "recorded_metrics": metrics,
        "initial_train_loss": first_available(
            (metrics or {}).get("initial_train_loss"), checkpoint.get("initial_train_loss"),
            records[0].get("loss_before") if records else None,
        ),
        "initial_test_loss": first_available(
            (metrics or {}).get("initial_test_loss"), checkpoint.get("initial_test_loss")
        ),
        "notes": notes,
        "evaluations": {},
    }
    solvers = {
        "training_solver": SolverConfig(**config["solver"]),
        "strict_solver": SolverConfig("tsit5", 1e-6, 1e-8, "direct_autograd"),
    }
    curves = {}
    for solver_name, solver in solvers.items():
        summary["evaluations"][solver_name] = {
            "solver": asdict(solver), "max_rhs_evaluations_per_solve": evaluation_cap,
        }
        curves[solver_name] = {}
        for split, data in [("train", train), ("test", test)]:
            count = int(provenance[f"{split}_trajectories"])
            nt = int(provenance["time_points"])
            if count > data["full_TBm1"].shape[1] or nt > data["t"].numel():
                raise ValueError(f"{run_dir}: saved split dimensions exceed the current dataset")
            print(f"Evaluating {run_dir.name}: {solver_name}, {split}", flush=True)
            try:
                values, curve = evaluate_split(
                    model, normalizer, data, count, nt,
                    float(provenance["t_ref_seconds"]), solver, config["pinn"], dtype, evaluation_cap,
                )
                summary["evaluations"][solver_name][split] = values
                if split == "test":
                    curves[solver_name] = curve
            except (AssertionError, FloatingPointError, RuntimeError, ValueError) as exc:
                summary["evaluations"][solver_name][split] = {
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
    average_report_losses(summary, config)
    return summary, records, curves


def average_report_losses(summary, config):
    """Convert summed evaluation metrics; retain original training provenance.

    evaluate_split returns time sums regardless of the training reduction.
    Saved histories and recorded_metrics always retain their original units.
    """
    if summary.get("reported_loss_reduction") == "mean_over_time":
        raise ValueError("report losses are already averaged over time")
    training_reduction = config.get("loss_reduction", "sum_over_time")
    if training_reduction not in {"sum_over_time", "mean_over_time"}:
        raise ValueError(f"unknown training loss reduction: {training_reduction}")
    nt = int(config["dataset"]["time_points"])
    divisor = nt if training_reduction == "sum_over_time" else 1
    summary["training_loss_reduction"] = training_reduction
    summary["reported_loss_reduction"] = "mean_over_time"
    summary["recorded_loss_plot_divisor"] = divisor
    for key in ("initial_train_loss", "initial_test_loss"):
        if summary.get(key) is not None:
            summary[key] /= divisor
    keys = (
        "loss", "trajectory_mse", "species_mse_contribution",
        "temperature_mse_contribution", "raw_element_conservation_penalty",
        "weighted_element_conservation_penalty",
    )
    for evaluation in summary["evaluations"].values():
        for split in ("train", "test"):
            values = evaluation[split]
            if "error" in values:
                continue
            values["time_summed_metrics"] = {key: values[key] for key in keys}
            for key in keys:
                values[key] /= values["time_points"]
    summary["notes"].append(
        "Reported MSE and conservation penalty are averaged over time points; "
        "recorded_metrics and source histories retain the original training reduction. "
        "Rescaling a report does not retrain the saved model."
    )


def format_number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def write_markdown(report, path):
    lines = [
        "# Hydrogen Stage 2 FGD pilot comparison", "",
        "These runs start from the same author-matched Stage 1 archive and train in "
        "normalized coordinates. The complete RHS is divided by the saved t_ref exactly once.", "",
        "The pilots use initial learning rate 0.002, Tsit5 rtol=1e-4/atol=1e-6, "
        "a five-loss nonmonotone line-search window, and Armijo c1=1e-6. "
        "These are learning-rate and numerical-solver settings, not starting-loss targets.", "",
        "All reported losses and MSE contributions below are divided by the number of "
        "observation time points (100 here). The conservation penalty is divided by the "
        "same count to preserve its relative weight. Physical temperature metrics are unchanged. "
        "This report conversion does not retrain the models.", "",
        "The functional-gradient sample centers are now clipped to [-1,1], matching "
        "the additive edge functions' evaluation coordinates. Comparisons against older "
        "runs are confounded by this correction as well as the changed optimization settings.", "",
        "Fixed-grid FGD is an approximation on a fixed interpolation grid and uses the same "
        "line search. It does not enforce adaptive representation certification. "
        "Both methods still reject nonfinite trajectories and stop if no acceptable "
        "representable update is found.", "",
        f"Shared initial model hash: **{report['checks']['same_initial_model_sha256']}**. "
        f"Shared dataset hash: **{report['checks']['same_dataset_sha256']}**. "
        f"Shared Stage 1 archive hash: **{report['checks']['same_stage1_checkpoint_sha256']}**.", "",
        "| Run | Completed / requested | Certified updates | Temporary train-loss increases | Recorded solver trial failures | Stop |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        stop = run["stop_reason"]
        if isinstance(stop, dict):
            stop = f"{stop.get('type')}: {stop.get('message')}"
        lines.append(
            f"| {Path(run['run_directory']).name} | {run['completed_steps']} / {run['requested_steps']} "
            f"| {run['certified_updates']} | {run['nonmonotone_loss_increases']} "
            f"| {run['solver_trial_failure_count']} | {stop} |"
        )
    lines += ["", "Evaluation solves have a finite work cap: max(50,000, training cap) RHS "
              "evaluations per solve. Hitting that cap is recorded as an evaluation failure, "
              "rather than producing an unbounded solve or a fabricated loss. Trial-failure "
              "counts include only diagnostics attached to saved updates; an interrupted "
              "or rejected final update can have no saved trial diagnostics.", ""]
    for run in report["runs"]:
        name = Path(run["run_directory"]).name
        cap = run["evaluation_max_rhs_evaluations_per_solve"]
        train_cap = run["training_max_rhs_evaluations_per_solve"]
        lines.append(f"- {name}: original training reduction `{run['training_loss_reduction']}`; "
                     f"evaluation cap {cap:,}; training cap {train_cap:,} (0 means unbounded).")
        resume = run["resume_from"]
        if resume:
            lines.append(
                f"  Resumed from {resume.get('path')} at update {resume.get('completed_steps')}; "
                f"elapsed offset {format_number(resume.get('elapsed_offset_seconds'))} s. "
                "Recorded timing excludes work performed after the source checkpoint in its original process."
            )
    lines += ["", "Missing post-update loss/evaluation records are left as gaps in the history. "
              "The saved weights are still evaluated directly below, so incomplete logging "
              "does not cause an older checkpoint to be substituted."]
    for solver_name, title in [("training_solver", "Training solver"), ("strict_solver", "Strict solver reevaluation")]:
        lines += ["", f"## {title}", "",
                  "| Run | Split | Total loss | Species MSE contribution | Temperature MSE contribution | Temperature MAE (K) | Predicted / reference median peak rise (K) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for run in report["runs"]:
            for split in ("train", "test"):
                row = run["evaluations"][solver_name][split]
                name = Path(run["run_directory"]).name
                if "error" in row:
                    lines.append(f"| {name} | {split} | evaluation failed: {row['error']['message']} | — | — | — | — |")
                    continue
                values = [format_number(row[k]) for k in (
                    "loss", "species_mse_contribution", "temperature_mse_contribution", "temperature_mae_K",
                )]
                rise = format_number(row["median_predicted_max_temperature_rise_K"])
                reference = format_number(row["median_reference_max_temperature_rise_K"])
                lines.append(f"| {name} | {split} | {' | '.join(values)} | {rise} / {reference} |")
    lines += ["", "Species and temperature MSE contributions both retain division by all ten state "
              "outputs and averaging over observation time points; their sum is the trajectory MSE. "
              "Total loss adds the weighted time-averaged elemental conservation penalty. "
              "Peak rise is max(T(t)) − T(0), computed per condition and "
              "then summarized by its median. Test curves and losses describe only the saved "
              "final checkpoint, with no selection using the held-out condition.", "",
              "The loss plot uses a logarithmic loss axis and logged training-solver objectives "
              "converted to the time-averaged reduction after each update. "
              "Its strict-evaluation markers reevaluate only the final saved model. "
              "The temperature plot shows the median test trajectory (one held-out condition "
              "in the current dataset), with both solver tolerances.", "",
              "The pilots overlapped on the same machine, so their recorded elapsed times "
              "are not an isolated speed comparison. A short pilot cannot establish convergence or show that longer temperature "
              "training will succeed. The Stage 1 archive and the single held-out condition "
              "remain limitations of these comparisons.", "",
              "[Loss vs epochs](loss_vs_epochs.png) · [Test temperature](test_temperature.png) · "
              "[Machine-readable comparison](comparison.json)", ""]
    path.write_text("\n".join(lines))


def make_plots(report, all_records, all_curves, output_dir):
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for index, (run, records) in enumerate(zip(report["runs"], all_records)):
        color = f"C{index % 10}"
        label = run["method"]
        for axis, split, key in zip(axes, ["train", "test"], ["loss_after", "test_loss"]):
            epochs = [r.get("step", number) for number, r in enumerate(records, start=1)]
            divisor = run["recorded_loss_plot_divisor"]
            losses = [r[key] / divisor if finite_number(r.get(key)) else float("nan") for r in records]
            initial = run[f"initial_{split}_loss"]
            if finite_number(initial):
                epochs.insert(0, 0)
                losses.insert(0, initial)
            axis.plot(epochs, losses, label=label, color=color)
            strict = run["evaluations"]["strict_solver"][split]
            if "loss" in strict:
                axis.scatter([run["completed_steps"]], [strict["loss"]], color=color,
                             marker="x", s=50, zorder=4, label=f"{label}: strict final")
            axis.set(title=f"{split.capitalize()} total loss", xlabel="Completed updates", ylabel="Time-averaged loss")
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle("Hydrogen Stage 2: lr 0.002, loose ODE, nonmonotone window 5")
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"loss_vs_epochs.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    reference_drawn = False
    for index, (run, curves) in enumerate(zip(report["runs"], all_curves)):
        for solver_name, style in [("training_solver", "--"), ("strict_solver", "-")]:
            curve = curves[solver_name]
            if not curve:
                continue
            times = np.asarray(curve["time_microseconds"])
            if not reference_drawn:
                reference = np.median(np.asarray(curve["reference_temperature_K"]), axis=1)
                axis.plot(times, reference, color="black", linewidth=2, label="Reference")
                reference_drawn = True
            predicted = np.median(np.asarray(curve["predicted_temperature_K"]), axis=1)
            tolerance = "strict" if solver_name == "strict_solver" else "training tolerances"
            axis.plot(times, predicted, style, color=f"C{index % 10}",
                      label=f"{run['method']} ({run['completed_steps']} updates), {tolerance}")
    axis.set(title="Final saved model: median held-out temperature trajectory",
             xlabel="Time (µs)", ylabel="Temperature (K)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"test_temperature.{suffix}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        parser.error("--run-dirs requires at least two run directories")
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists; choose a new output directory")
    torch.set_num_threads(1)
    train, test = load_hydrogen("train"), load_hydrogen("test")
    assert_species_order(train["species"])
    archive_sha = hashlib.sha256((DATA_DIR / "hydrogen.npz").read_bytes()).hexdigest()
    runs, records, curves = [], [], []
    for run_dir in args.run_dirs:
        summary, history, curve = evaluate_run(run_dir, train, test, archive_sha)
        runs.append(summary)
        records.append(history)
        curves.append(curve)
    checks = {}
    for key in ("initial_model_sha256", "dataset_sha256", "stage1_checkpoint_sha256"):
        values = [r[key] for r in runs]
        checks[f"same_{key}"] = all(v is not None for v in values) and len(set(values)) == 1
    checks["all_datasets_match_current_archive"] = all(r["dataset_sha256"] == archive_sha for r in runs)
    checks["same_requested_steps"] = len({r["requested_steps"] for r in runs}) == 1
    checks["same_completed_steps"] = len({r["completed_steps"] for r in runs}) == 1
    report = {
        "experiment": "Hydrogen Stage 2 FGD pilots with relaxed nonmonotone line search",
        "evaluation_device": "cpu",
        "evaluation_threads": 1,
        "evaluation_has_training_updates": False,
        "checks": checks,
        "comparison_caveats": [
            "A different completed step count means the final checkpoints do not have equal update budgets.",
            "The pilot changes learning rate, line search, and clipped functional-gradient centers; comparisons to older runs cannot isolate one change.",
            "Fixed-grid FGD remains an approximate functional-gradient method with the same line-search safeguard.",
            "The single held-out condition is used for reporting, not checkpoint selection.",
            "Evaluation solves are capped at max(50000, training RHS cap); failures are reported per split/solver.",
            "Resume timing excludes original-process work after the source checkpoint, and trial-failure counts cover saved update records only.",
            "The pilots overlapped on the same machine; elapsed times are not an isolated speed comparison.",
        ],
        "runs": runs,
        "test_temperature_curves": {str(path.resolve()): curve for path, curve in zip(args.run_dirs, curves)},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(report, args.output_dir / "comparison.md")
    make_plots(report, records, curves, args.output_dir)
    print(f"Comparison saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
