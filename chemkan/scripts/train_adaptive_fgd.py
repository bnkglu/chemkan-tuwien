"""Local adaptive-FGD experiment using ChemKAN's data, dynamics and trajectory loss.

Run from the repository root with PYTHONPATH=chemkan/src. This separate entry
point leaves the reproduction trainers unchanged. See docs/adaptive_fgd.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from _chemistry import ATOMIC_WEIGHTS, ELEMENT_COUNTS, MOLAR_WEIGHTS, assert_species_order
from _data import DATA_DIR, load_biodiesel, load_hydrogen, load_hydrogen_temperature

from chemkan.adaptive_fgd import AdaptiveFGD, AdaptiveFGDConfig
from chemkan.dynamics import ChemKANDynamics, KineticDynamics
from chemkan.losses import element_conservation_loss, trajectory_mse
from chemkan.model import ChemKAN, KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate
from chemkan.temperature import ConstantTemperature, ObservedTemperature


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--system",
        choices=["biodiesel", "hydrogen-stage1", "hydrogen-full", "hydrogen-stage2"],
        default="biodiesel",
    )
    p.add_argument(
        "--method", choices=["adaptive-fgd", "fixed-fgd", "adam"], default="adaptive-fgd"
    )
    p.add_argument(
        "--run-dir", type=Path, required=True, help="a new directory; never overwritten"
    )
    p.add_argument("--data-file", type=Path, help="biodiesel archive; default biodiesel_v2.npz")
    p.add_argument(
        "--stage1-from",
        type=Path,
        help="compatible Stage-1 checkpoint/archive; required for hydrogen-stage2",
    )
    p.add_argument("--no-pinn-stage2", dest="pinn_stage2", action="store_false", default=True)
    p.add_argument("--alpha-pinn", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--resume-from", type=Path, help="resume saved weights and optimizer state into a new run directory")
    p.add_argument(
        "--max-rhs-evaluations", type=int, default=0,
        help="per-solve RHS evaluation limit; expensive FGD trials backtrack; 0 disables",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save checkpoint_progress.pt every N completed updates; 0 disables it",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        help="stop at an update boundary after this wall-time budget; includes evaluation",
    )
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=100.0)
    p.add_argument("--initial-intervals", type=int, default=4)
    p.add_argument("--max-intervals", type=int, default=4096)
    p.add_argument("--max-backtracks", type=int, default=24)
    p.add_argument("--armijo", type=float, default=1e-4)
    p.add_argument(
        "--line-search-window", type=int, default=1,
        help="1: monotone Armijo; >1: compare against the maximum of this many training losses",
    )
    p.add_argument("--hidden-dim", type=int, default=4)
    p.add_argument("--num-basis", type=int, default=3)
    p.add_argument("--n-mu", type=int, default=2)
    p.add_argument("--use-base-act", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--solver-method", default="tsit5")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--train-limit", type=int, help="first N training trajectories for a smoke run")
    p.add_argument("--test-limit", type=int, help="first N test trajectories for a smoke run")
    p.add_argument(
        "--time-points", type=int, help="first N observations for a short-horizon smoke"
    )
    return p


def prepare(args):
    """Use existing physical-state conventions and train-only normalization."""
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    biodiesel = args.system == "biodiesel"
    if not biodiesel and args.data_file is not None:
        raise ValueError("--data-file is currently supported only for biodiesel")
    path = (
        (args.data_file or DATA_DIR / "biodiesel_v2.npz")
        if biodiesel
        else DATA_DIR / "hydrogen.npz"
    )
    load = (lambda split: load_biodiesel(split, data_file=path)) if biodiesel else load_hydrogen
    train, test = load("train"), load("test")
    species_dim = train["species_TBm"].shape[-1]
    lo, hi = train["u_min"], train["u_max"]
    if biodiesel:
        lo = torch.cat([lo, train["T_const"].min().reshape(1)])
        hi = torch.cat([hi, train["T_const"].max().reshape(1)])
    normalizer = MinMaxNormalizer(lo, hi).to(device=device, dtype=dtype)
    if args.system == "hydrogen-stage2" and args.stage1_from is not None:
        checkpoint = torch.load(args.stage1_from, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") == 1 and "config" in checkpoint:
            return prepare_author_matched_stage2(
                args, path, train, test, lo, hi, device, dtype, checkpoint
            )
    full = args.system in {"hydrogen-full", "hydrogen-stage2"}
    loss_norm = normalizer if full else normalizer.subset(slice(0, species_dim))
    architecture = {
        "species_dim": species_dim,
        "hidden_dim": args.hidden_dim,
        "num_basis": args.num_basis,
        "n_mu": args.n_mu,
        "use_base_act": args.use_base_act,
    }
    model = (ChemKAN if full else KineticCore)(**architecture).to(device=device, dtype=dtype)
    stage1 = None
    if args.system == "hydrogen-stage2":
        if args.stage1_from is None:
            raise ValueError("hydrogen-stage2 requires --stage1-from")
        checkpoint = torch.load(args.stage1_from, map_location="cpu", weights_only=False)
        expected = {k: v for k, v in architecture.items() if k != "species_dim"}
        if checkpoint.get("architecture") != expected:
            raise ValueError(f"Stage-1 architecture must match {expected}")
        if list(checkpoint.get("data", {}).get("species", [])) != list(train["species"]):
            raise ValueError("Stage-1 species order does not match the hydrogen data")
        scaling = checkpoint.get("input_scaling", {})
        if scaling.get("method") != "minmax" or any(
            k not in scaling
            or torch.as_tensor(scaling[k]).shape != lo.shape
            or not bool(torch.isfinite(torch.as_tensor(scaling[k])).all())
            for k in ("u_min", "u_max")
        ):
            raise ValueError("Stage-1 input normalization is missing or invalid")
        saved_lo, saved_hi = torch.as_tensor(scaling["u_min"]), torch.as_tensor(scaling["u_max"])
        if bool((saved_hi < saved_lo).any()):
            raise ValueError("Stage-1 input normalization has inverted bounds")
        if not (torch.equal(saved_lo.cpu(), lo.cpu()) and torch.equal(saved_hi.cpu(), hi.cpu())):
            raise ValueError(
                "Stage-1 checkpoint normalization differs from the current hydrogen archive; "
                "refusing to mix checkpoints and datasets (the legacy 10k checkpoint used "
                "the old 50-point archive)"
            )
        normalizer = MinMaxNormalizer(saved_lo, saved_hi).to(device=device, dtype=dtype)
        if checkpoint.get("solver", {}).get("sensitivity") != "direct_autograd":
            raise ValueError("Stage-1 checkpoint must use direct_autograd for this comparison")
        model.kinetic.load_state_dict(checkpoint["kinetic_state"], strict=True)
        assert_species_order(train["species"])
        stage1 = {
            "path": str(args.stage1_from.resolve()),
            "sha256": hashlib.sha256(args.stage1_from.read_bytes()).hexdigest(),
            "epochs": checkpoint.get("stage1_epochs"),
            "final_loss": checkpoint.get("stage1_final_loss"),
            "temperature": checkpoint.get("stage1_temperature"),
            "protocol": "standard physical-state ChemKAN",
            "input_normalization": "restored from Stage-1 checkpoint",
            "checkpoint_stats_match_current_archive": bool(
                torch.equal(saved_lo.cpu(), lo.cpu()) and torch.equal(saved_hi.cpu(), hi.cpu())
            ),
            "thermodynamic_initialization": "fresh default initialization with this run's seed",
        }
    elif args.stage1_from is not None:
        raise ValueError("--stage1-from is only supported with --system hydrogen-stage2")
    initial_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    problems = []
    for split, data, limit in [
        ("train", train, args.train_limit),
        ("test", test, args.test_limit),
    ]:
        source = data["full_TBm1"] if full else data["species_TBm"]
        count = source.shape[1] if limit is None else min(limit, source.shape[1])
        nt = (
            source.shape[0] if args.time_points is None else min(args.time_points, source.shape[0])
        )
        target = source[:nt, :count].to(device=device, dtype=dtype)
        times = data["t"][:nt].to(device=device, dtype=dtype)
        if full:
            dynamics = ChemKANDynamics(model, input_normalizer=normalizer)
        else:
            if biodiesel:
                temperature = ConstantTemperature(data["T_const"][:count])
            else:
                dense = load_hydrogen_temperature(split)
                temperature = ObservedTemperature(
                    dense["t_dense"], dense["T_dense_TB1"][:, :count]
                )
            dynamics = KineticDynamics(model, temperature, input_normalizer=normalizer)
        dynamics.to(device=device, dtype=dtype)
        problems.append((dynamics, target[0], times, loss_norm.normalize(target)))
    provenance = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_trajectories": problems[0][1].shape[0],
        "test_trajectories": problems[1][1].shape[0],
        "time_points": problems[0][2].numel(),
        "normalization": (
            "loss: current archive train-only min/max; inputs: saved Stage-1 statistics"
            if stage1
            else "archive train-only min/max; temperature from full train split"
        ),
        "hydrogen_stage1_temperature": "dense 20000-point cache"
        if args.system == "hydrogen-stage1"
        else None,
        "stage1_from": stage1,
        "initial_model_sha256": initial_hash.hexdigest(),
    }
    return model, architecture, normalizer, loss_norm, problems, provenance


def prepare_author_matched_stage2(args, path, train, test, lo, hi, device, dtype, checkpoint):
    """Load a current-grid Stage-1 archive using its normalized-coordinate protocol."""
    from author_repo_match.train_author_matched_hydrogen import (
        Stage2Dynamics,
        build_model,
        init_thermo_,
    )

    cfg = checkpoint["config"]
    dataset = cfg.get("dataset", {})
    current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_dataset = {
        "sha256": current_sha,
        "n_points": int(train["t"].numel()),
        "n_conditions": int(train["full_TBm1"].shape[1]),
    }
    for key, value in expected_dataset.items():
        if dataset.get(key) != value:
            raise ValueError(
                f"Stage-1 archive {key}={dataset.get(key)!r} does not match "
                f"the current dataset value {value!r}"
            )
    if cfg.get("dtype") != args.dtype:
        raise ValueError(
            f"author-matched Stage-1 archive requires --dtype {cfg.get('dtype')}, "
            f"not {args.dtype}"
        )
    normalization = cfg.get("normalization", {})
    saved_lo = torch.as_tensor(normalization.get("u_min", []), dtype=lo.dtype)
    saved_hi = torch.as_tensor(normalization.get("u_max", []), dtype=hi.dtype)
    if not (torch.equal(saved_lo, lo) and torch.equal(saved_hi, hi)):
        raise ValueError("Stage-1 archive normalization differs from the current dataset")
    architecture_cfg = cfg.get("architecture", {})
    required = {"hidden_dim": 3, "num_basis": 5, "n_mu": 3, "use_base_act": False}
    if any(architecture_cfg.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported author-matched Stage-1 architecture")
    if cfg.get("state_space", {}).get("stage2") != "normalized species and temperature u_hat (10)":
        raise ValueError("Stage-1 archive does not declare the normalized Stage-2 state space")

    normalizer = MinMaxNormalizer(lo, hi).to(device=device, dtype=dtype)
    model = build_model().to(device=device, dtype=dtype)
    kinetic = {
        key.removeprefix("kinetic."): value
        for key, value in checkpoint["model_state"].items()
        if key.startswith("kinetic.")
    }
    model.kinetic.load_state_dict(kinetic, strict=True)
    init_thermo_(model, torch.Generator().manual_seed(args.seed))
    assert_species_order(train["species"])
    initial_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())

    t_ref = float(cfg.get("t_ref", {}).get("value_s"))
    problems = []
    for data, limit in [(train, args.train_limit), (test, args.test_limit)]:
        source = data["full_TBm1"]
        count = source.shape[1] if limit is None else min(limit, source.shape[1])
        nt = source.shape[0] if args.time_points is None else min(args.time_points, source.shape[0])
        physical = source[:nt, :count].to(device=device, dtype=dtype)
        target = normalizer.normalize(physical)
        times = data["t"][:nt].to(device=device, dtype=dtype)
        dynamics = Stage2Dynamics(model, t_ref).to(device=device, dtype=dtype)
        problems.append((dynamics, target[0], times, target))

    architecture = {"species_dim": 9, **required}
    stage1 = {
        "path": str(args.stage1_from.resolve()),
        "sha256": hashlib.sha256(args.stage1_from.read_bytes()).hexdigest(),
        "run_id": checkpoint.get("run_id"),
        "epochs": checkpoint.get("epochs_completed"),
        "protocol": "author-matched normalized-coordinate ChemKAN",
        "dataset_sha256_match": True,
        "time_points_match": True,
        "thermodynamic_initialization": "author-matched seed initialization",
    }
    provenance = {
        "path": str(path.resolve()),
        "sha256": current_sha,
        "train_trajectories": problems[0][1].shape[0],
        "test_trajectories": problems[1][1].shape[0],
        "time_points": problems[0][2].numel(),
        "normalization": "current archive train-only min/max; normalized-coordinate dynamics",
        "state_coordinates": "normalized",
        "t_ref_seconds": t_ref,
        "hydrogen_stage1_temperature": None,
        "stage1_from": stage1,
        "initial_model_sha256": initial_hash.hexdigest(),
    }
    return model, architecture, normalizer, normalizer, problems, provenance


def integrate_with_budget(dynamics, initial, times, solver, max_evaluations=0):
    """Bound one numerical solve without modifying successful solve calculations."""
    if not max_evaluations:
        return integrate(dynamics, initial, times, solver)
    count = 0

    def count_evaluation(module, inputs):
        nonlocal count
        count += 1
        if count > max_evaluations:
            raise FloatingPointError(f"ODE solve exceeded {max_evaluations} RHS evaluations")

    handle = dynamics.register_forward_pre_hook(count_evaluation)
    try:
        return integrate(dynamics, initial, times, solver)
    finally:
        handle.remove()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.steps < 0 or args.threads < 1 or args.checkpoint_every < 0:
        raise ValueError("steps/checkpoint-every must be nonnegative and threads positive")
    if args.max_rhs_evaluations < 0:
        raise ValueError("max-rhs-evaluations must be nonnegative")
    if not math.isfinite(args.alpha_pinn) or args.alpha_pinn < 0:
        raise ValueError("--alpha-pinn must be finite and nonnegative")
    if args.max_seconds is not None and (
        not math.isfinite(args.max_seconds) or args.max_seconds <= 0
    ):
        raise ValueError("--max-seconds must be finite and positive")
    for name in ("train_limit", "test_limit", "time_points"):
        value = getattr(args, name)
        if value is not None and value < (2 if name == "time_points" else 1):
            raise ValueError(f"invalid --{name.replace('_', '-')}")
    if args.run_dir.exists():
        raise FileExistsError(f"{args.run_dir} already exists; choose a new --run-dir")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model, architecture, normalizer, loss_norm, problems, provenance = prepare(args)
    solver = SolverConfig(args.solver_method, args.rtol, args.atol, "direct_autograd")
    cfg = AdaptiveFGDConfig(
        lr=args.lr,
        epsilon=args.epsilon,
        gamma=args.gamma,
        initial_intervals=args.initial_intervals,
        max_intervals=args.max_intervals,
        max_backtracks=args.max_backtracks,
        armijo=args.armijo,
        line_search_window=args.line_search_window,
        refine=args.method == "adaptive-fgd",
    )
    opt = (
        torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.method == "adam"
        else AdaptiveFGD(model, cfg)
    )

    use_pinn = args.system == "hydrogen-stage2" and args.pinn_stage2
    ec = ELEMENT_COUNTS.to(device=args.device, dtype=getattr(torch, args.dtype))
    aw = ATOMIC_WEIGHTS.to(device=args.device, dtype=getattr(torch, args.dtype))
    mw = MOLAR_WEIGHTS.to(device=args.device, dtype=getattr(torch, args.dtype))

    def objective(problem):
        dynamics, initial, times, target = problem
        prediction = integrate_with_budget(
            dynamics, initial, times, solver, args.max_rhs_evaluations
        )
        normalized_coordinates = provenance.get("state_coordinates") == "normalized"
        prediction_for_loss = prediction if normalized_coordinates else loss_norm.normalize(prediction)
        loss = trajectory_mse(prediction_for_loss, target)
        if use_pinn:
            physical_prediction = (
                loss_norm.denormalize(prediction) if normalized_coordinates else prediction
            )
            loss = loss + args.alpha_pinn * element_conservation_loss(
                physical_prediction[..., :-1], ec, aw, mw
            )
        # Both terms sum over observation times; average the complete objective
        # to preserve their relative weighting across different time grids.
        return loss / target.shape[0]

    def closure():
        return objective(problems[0])

    config = {
        "format": "chemkan-adaptive-fgd-v1",
        "system": args.system,
        "method": args.method,
        "architecture": architecture,
        "fgd": asdict(cfg),
        "solver": asdict(solver),
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "threads": args.threads,
        "steps": args.steps,
        "max_seconds": args.max_seconds,
        "checkpoint_every": args.checkpoint_every,
        "max_rhs_evaluations": args.max_rhs_evaluations,
        "functional_input_coordinates": "clamped to [-1,1] for both evaluation and gradient samples",
        "timing_scope": (
            "initial train/test evaluation, training updates, post-update train/test evaluation, "
            "and per-step logging; excludes setup and final artifact serialization; "
            "a running update completes before the budget is checked again"
        ),
        "dataset": provenance,
        "torch_version": str(torch.__version__),
        "loss_reduction": "mean_over_time",
        "loss": "time-averaged normalized trajectory MSE, clean targets, full rollout"
        + (" + alpha*time-averaged element-conservation PINN" if use_pinn else ""),
        "pinn": {"enabled": use_pinn, "alpha": args.alpha_pinn if use_pinn else None},
        "reference_repository": "https://github.com/dccsillag/experiments-adaptive-fgd",
        "reference_commit": "57f0bab8df7bd3fe5e4d1ff8c23953989c62befc",
    }
    records = []
    initial_train = initial_test = None
    elapsed_offset = 0.0
    if args.resume_from is not None:
        saved = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        previous = saved["config"]
        for key in (
            "system", "method", "architecture", "fgd", "solver", "dtype", "seed",
            "device", "threads", "torch_version", "pinn", "functional_input_coordinates",
            "loss_reduction",
        ):
            if previous.get(key) != config.get(key):
                raise ValueError(f"resume configuration mismatch: {key}")
        for key in ("sha256", "initial_model_sha256", "time_points", "train_trajectories", "test_trajectories", "t_ref_seconds", "state_coordinates"):
            if previous["dataset"].get(key) != provenance.get(key):
                raise ValueError(f"resume dataset mismatch: {key}")
        if saved.get("optimizer_state") is None:
            raise ValueError("checkpoint lacks optimizer state; cannot resume exactly")
        for key, current in (("input_normalizer", normalizer), ("loss_normalizer", loss_norm)):
            if any(not torch.equal(value.cpu(), saved[key][name].cpu()) for name, value in current.state_dict().items()):
                raise ValueError(f"resume normalization mismatch: {key}")
        records = list(saved["records"])
        if len(records) != saved["completed_steps"] or any(not record.get("evaluation_complete", False) for record in records):
            raise ValueError("resume requires a checkpoint with completely evaluated update records")
        if args.steps <= len(records):
            raise ValueError("--steps must exceed the checkpoint's completed updates")
        model.load_state_dict(saved["model_state"], strict=True)
        opt.load_state_dict(saved["optimizer_state"])
        initial_train = saved.get("initial_train_loss", records[0]["loss_before"] if records else None)
        initial_test = saved.get("initial_test_loss")
        elapsed_offset = saved.get("elapsed_seconds", records[-1]["elapsed_seconds"] if records else 0.0)
        config["resume_from"] = {
            "path": str(args.resume_from.resolve()),
            "sha256": hashlib.sha256(args.resume_from.read_bytes()).hexdigest(),
            "completed_steps": len(records),
            "elapsed_offset_seconds": elapsed_offset,
            "note": "timing excludes work after the source checkpoint; solve-work cap may differ",
        }
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.perf_counter()

    def elapsed():
        return elapsed_offset + time.perf_counter() - started

    def save_checkpoint(path=None):
        path = path or args.run_dir / "checkpoint_final.pt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "config": config,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "input_normalizer": normalizer.state_dict(),
                "loss_normalizer": loss_norm.state_dict(),
                "completed_steps": len(records),
                "records": records,
                "initial_train_loss": initial_train,
                "initial_test_loss": initial_test,
                "elapsed_seconds": elapsed(),
            },
            temporary,
        )
        temporary.replace(path)

    try:
        with torch.no_grad():
            current_train = float(closure())
            current_test = float(objective(problems[1]))
            if not records:
                initial_train, initial_test = current_train, current_test
        with (args.run_dir / "diagnostics.jsonl").open("w") as diagnostics:
            for record in records:
                diagnostics.write(json.dumps(record, allow_nan=False) + "\n")
            diagnostics.flush()
            for step in range(len(records) + 1, args.steps + 1):
                if (
                    args.max_seconds is not None
                    and elapsed() >= args.max_seconds
                ):
                    break
                if args.method == "adam":
                    opt.zero_grad(set_to_none=True)
                    loss = closure()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite Adam training loss")
                    loss.backward()
                    opt.step()
                    record = {
                        "loss_before": float(loss.detach()),
                        "loss_after": None,
                        "step_size": args.lr,
                        "stationary": False,
                    }
                else:
                    record = opt.step(closure)
                # Count the accepted state before evaluation: an interruption
                # during evaluation must not label new weights as the old step.
                record.update(step=step, test_loss=None, evaluation_complete=False)
                records.append(record)
                with torch.no_grad():
                    if args.method == "adam":
                        record["loss_after"] = float(closure())
                    record["test_loss"] = float(objective(problems[1]))
                record.update(
                    evaluation_complete=True, elapsed_seconds=elapsed()
                )
                diagnostics.write(json.dumps(record, allow_nan=False) + "\n")
                diagnostics.flush()
                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    save_checkpoint(args.run_dir / "checkpoint_progress.pt")
                print(
                    f"step={step} train={record['loss_after']:.6g} "
                    f"test={record['test_loss']:.6g} lr={record['step_size']:.3g} "
                    f"intervals={record.get('intervals_after', {})}",
                    flush=True,
                )
                if record["stationary"]:
                    break
        metrics = {
            "initial_train_loss": initial_train,
            "initial_test_loss": initial_test,
            "final_train_loss": records[-1]["loss_after"] if records else initial_train,
            "final_test_loss": records[-1]["test_loss"] if records else initial_test,
            "completed_steps": len(records),
            "elapsed_seconds": elapsed(),
            "status": "completed",
            "stop_reason": (
                "stationary"
                if records and records[-1]["stationary"]
                else "step_limit"
                if len(records) == args.steps
                else "time_budget"
            ),
        }
        if args.max_seconds is not None:
            within_budget = [r for r in records if r["elapsed_seconds"] <= args.max_seconds]
            metrics["wall_time_budget_seconds"] = args.max_seconds
            metrics["last_update_within_budget"] = (
                {
                    k: within_budget[-1][k]
                    for k in ("step", "elapsed_seconds", "loss_after", "test_loss")
                }
                if within_budget
                else None
            )
        with (args.run_dir / "history.csv").open("w", newline="") as stream:
            columns = [
                "step",
                "loss_before",
                "loss_after",
                "test_loss",
                "step_size",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        (args.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
        save_checkpoint()
        print(json.dumps(metrics, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        save_checkpoint()
        (args.run_dir / "failure.json").write_text(
            json.dumps(
                {"type": type(exc).__name__, "message": str(exc), "completed_steps": len(records)},
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
