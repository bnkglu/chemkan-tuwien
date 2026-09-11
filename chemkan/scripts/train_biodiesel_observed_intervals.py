r"""Biodiesel experiment: train on observed states at every adjacent interval.

This is an ADDITIVE experiment script. ``train_biodiesel.py`` is untouched and remains
the original trainer; nothing here changes shared library code, datasets, or any earlier
result.

What differs from the original trainer
--------------------------------------
The original trainer integrates ONE trajectory per condition, from the first observed
state at t=0 all the way to t_end, and compares the whole predicted trajectory with the
observations.

This script instead integrates each adjacent pair of observation times INDEPENDENTLY:
interval j starts from the observed state at t[j], is integrated over the absolute-time
grid ``t[j:j+2]``, and only its endpoint is compared with the observation at t[j+1]. The
next interval starts from its own observation again -- a prediction is never fed into the
following interval. Gradients from all N-1 intervals are accumulated, then ONE Adam step
is taken per epoch, so the update count matches the original trainer exactly.

The error formula is unchanged: ``chemkan.losses.trajectory_mse`` (Eq. 18) is called once
per interval on ``(1, B, m)`` tensors, i.e. mean over species, mean over trajectories, and
the per-interval values are summed over the N-1 endpoints. The objective differs from
full-trajectory training only because the predictions start from fresh observations, NOT
because a new loss has been introduced.

Provenance note
---------------
Restarting from observed states at each interval is an additional experiment. The
ChemKAN paper (Sec. II C 3-5, III A 1, Eq. 18) does not describe it. It is not claimed
here to be a correction of the reproduction, nor to be the authors' undocumented
procedure.

Evaluation
----------
Training-time logging always reports FULL ROLLOUTS from the original initial conditions
(no intermediate observed-state resets, no test observation ever fed to the model), so the
recorded ``full_rollout_*`` columns are directly comparable with the original trainer's
loss and with ``evaluate_biodiesel.py``. Evaluation runs under ``no_grad`` and saves and
restores the RNG state, so it cannot perturb training.

Example
-------
    python chemkan/scripts/train_biodiesel_observed_intervals.py \
        --seed 0 --epochs 10000 --snapshot-epochs 5000 \
        --run-dir results/experiments/biodiesel_observed_intervals/seed0
"""

from __future__ import annotations

import argparse
import logging
import time

import torch
from _data import input_scaling_meta, load_biodiesel, resolve_device
from _run import RunManager, check_resume_config

from chemkan.dynamics import KineticDynamics
from chemkan.losses import trajectory_mse
from chemkan.model import KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate
from chemkan.temperature import ConstantTemperature

try:                                                    # optional: interactive progress bar
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:
    _tqdm = None

# Recorded verbatim in config.json / the checkpoint so a reader of the artifact alone
# knows exactly which prediction procedure produced the training gradients.
_PROCEDURE = (
    "observed-interval training: for each adjacent pair of observation times (j, j+1) the "
    "solver is started from the OBSERVED state at t[j] and integrated over the absolute-time "
    "grid t[j:j+2]; only the endpoint is compared with the observation at t[j+1] via "
    "chemkan.losses.trajectory_mse on (1, B, m) tensors. A prediction is never fed into the "
    "next interval. Gradients from all N-1 intervals are accumulated before ONE Adam step "
    "per epoch. Not a procedure specified by the paper."
)
_TRAIN_OBJECTIVE = ("sum over the N-1 interval endpoints of the Eq. 18 normalized MSE "
                    "(mean over species, mean over trajectories)")
_EVAL_CONVENTION = ("full rollout from the ORIGINAL initial conditions over the whole saved "
                    "time grid, scored with chemkan.losses.trajectory_mse under train-only "
                    "min-max normalization -- the same quantity the original trainer reports")


def accumulate_interval_gradients(dynamics, obs, obs_norm, t, loss_norm, solver, *,
                                  backward: bool = True):
    r"""The observed-interval objective:  sum_j Eq.18-MSE(endpoint_j, observation_{j+1}).

        obs      : (N, B, m) PHYSICAL observations -- obs[j] is interval j's initial state
        obs_norm : (N, B, m) the same observations, normalized -- the endpoint targets
        t        : (N,)      ABSOLUTE observation times (spacing may be irregular)

    Interval ``j`` is integrated from ``obs[j]`` over ``t[j:j+2]`` and only its endpoint
    ``pred[-1:]`` (shape ``(1, B, m)``) is scored, so ``trajectory_mse``'s time sum runs
    over a single endpoint and its state/trajectory reductions are unchanged. A prediction
    is never carried into the next interval.

    With ``backward=True`` each interval is backpropagated immediately, so only one
    interval graph is alive at a time and the parameter ``.grad`` buffers accumulate the
    sum's gradient exactly; a DETACHED scalar is returned. With ``backward=False`` the
    graph-connected sum is returned instead (used for the final no-grad recomputation and
    by the focused checks).
    """
    n_intervals = int(obs.shape[0]) - 1
    if n_intervals < 1:
        raise ValueError("need at least two observation times to form an interval")
    total = None
    for j in range(n_intervals):
        pred = integrate(dynamics, obs[j], t[j:j + 2], solver)       # (2, B, m) physical
        loss_j = trajectory_mse(loss_norm.normalize(pred[-1:]),      # (1, B, m)
                                obs_norm[j + 1:j + 2])               # (1, B, m)
        if backward:
            loss_j.backward()                       # frees THIS interval's graph
            loss_j = loss_j.detach()
        total = loss_j if total is None else total + loss_j
        del pred                                    # drop the interval's graph reference
    return total


def full_rollout_mse(dynamics, y0, t, reference_norm, loss_norm, solver):
    """Eq. 18 loss of ONE rollout from ``y0`` over the whole grid ``t``.

    ``y0`` is the trajectory's ORIGINAL initial condition; no intermediate observation is
    supplied to the model anywhere in this function. Called under ``no_grad`` by the
    training loop.
    """
    return trajectory_mse(loss_norm.normalize(integrate(dynamics, y0, t, solver)),
                          reference_norm)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # architecture -- same paper count-matching defaults as train_biodiesel.py
    p.add_argument("--hidden-dim", type=int, default=4)
    p.add_argument("--num-basis", type=int, default=3)
    p.add_argument("--n-mu", type=int, default=2)
    p.add_argument("--use-base-act", action="store_true",
                   help="literal Eq. 11 base path (default OFF = paper count-matching)")
    # training
    p.add_argument("--epochs", type=int, default=10000,
                   help="optimizer updates; ONE per epoch, as in the original trainer")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for model init / training reproducibility. The initial "
                        "parameter tensors match train_biodiesel.py at the same seed.")
    p.add_argument("--input-scaling", default="minmax", choices=["minmax", "none"])
    p.add_argument("--eval-every", type=int, default=100,
                   help="full-rollout evaluation every N updates (0 = only at the "
                        "initial and final states). An experiment choice, not a paper "
                        "requirement. Evaluation only: no step, no parameter change.")
    p.add_argument("--snapshot-epochs", default="",
                   help="comma-separated update counts to save as checkpoint_epoch_<N>.pt "
                        "(e.g. 5000 for the Figure-4 budget). Never overwritten.")
    # solver -- explicit PyTorch implementation settings, NOT paper values
    p.add_argument("--solver-method", default="tsit5")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    # run directory -- required: this experiment writes only into its own new area
    p.add_argument("--run-dir", required=True,
                   help="one directory per run, e.g. "
                        "results/experiments/biodiesel_observed_intervals/seed0")
    p.add_argument("--experiment-name", default="observed_intervals")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="overwrite checkpoint_resume.pt every N updates")
    p.add_argument("--resume", action="store_true",
                   help="resume from RUN_DIR/checkpoint_resume.pt if present")
    # NOTE: there is deliberately no --overwrite. A completed run is never replaced.
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- data: CLEAN biodiesel only (this experiment adds no noise dimension) ---------
    data = load_biodiesel(split="train", noise_percent=None)
    test = load_biodiesel(split="test", noise_percent=None)
    species_dim = data["species_TBm"].shape[-1]         # data-derived, never hard-coded
    t = data["t"].to(device)                            # (N,) absolute observation times
    n_times = int(t.numel())
    n_intervals = n_times - 1                           # derived from the archive, not assumed
    if n_intervals < 1:
        raise SystemExit("need at least two observation times to form an interval")

    solver = SolverConfig(method=args.solver_method, rtol=args.rtol, atol=args.atol,
                          sensitivity="direct_autograd")

    core = KineticCore(species_dim=species_dim, hidden_dim=args.hidden_dim,
                       num_basis=args.num_basis, n_mu=args.n_mu,
                       use_base_act=args.use_base_act).to(device)

    # Full-state [Y1..Ym, T] train-only input normalizer, built exactly as in
    # train_biodiesel.py (species stats from the archive + T's TRAIN min/max).
    T_const = data["T_const"]
    u_min_full = torch.cat([data["u_min"], T_const.min().reshape(1)]).to(device)
    u_max_full = torch.cat([data["u_max"], T_const.max().reshape(1)]).to(device)
    full_norm = MinMaxNormalizer(u_min_full, u_max_full).to(device)      # (m+1,)
    loss_norm = full_norm.subset(slice(0, species_dim))                  # species-only (m,)
    input_normalizer = full_norm if args.input_scaling == "minmax" else None

    dynamics = KineticDynamics(core, ConstantTemperature(T_const),
                               input_normalizer=input_normalizer).to(device)

    # Training observations. Clean run -> targets_TBm is the clean trajectory; it is both
    # the per-interval initial state and the per-interval endpoint target.
    obs = data["targets_TBm"].to(device)                        # (N, B, m) physical
    obs_norm = loss_norm.normalize(obs)                         # (N, B, m) normalized target

    # Full-rollout evaluation inputs: ORIGINAL initial conditions only.
    Y0 = data["Y0"].to(device)                                  # (B, m)
    train_ref = obs_norm                                        # (N, B, m)
    test_dyn = KineticDynamics(core, ConstantTemperature(test["T_const"].to(device)),
                               input_normalizer=input_normalizer).to(device)
    test_Y0 = test["Y0"].to(device)
    test_t = test["t"].to(device)
    test_ref = loss_norm.normalize(test["species_TBm"].to(device))   # clean (Eq. 22)

    def interval_objective() -> torch.Tensor:
        """Accumulate every interval's gradient into ``.grad``; return the summed value."""
        return accumulate_interval_gradients(dynamics, obs, obs_norm, t, loss_norm, solver)

    def full_rollout_metrics() -> dict:
        """Full-rollout train + clean-test losses at the CURRENT parameter state.

        Original initial conditions only; no intermediate observation is ever supplied.
        The model has no train/eval-dependent layers, so its mode is left untouched. The
        RNG state is saved and restored so evaluation cannot shift the training run.
        """
        rng_state = torch.get_rng_state()
        started = time.perf_counter()
        try:
            with torch.no_grad():
                tr = full_rollout_mse(dynamics, Y0, t, train_ref, loss_norm, solver)
                te = full_rollout_mse(test_dyn, test_Y0, test_t, test_ref, loss_norm, solver)
        finally:
            torch.set_rng_state(rng_state)
        return {"full_rollout_train_mse": float(tr), "full_rollout_test_mse_clean": float(te),
                "eval_seconds": round(time.perf_counter() - started, 4)}

    # --- run-directory plumbing (organization only; math unchanged) -------------------
    run = RunManager(args.run_dir, "biodiesel_observed_intervals", resume=args.resume)
    run.start()

    start_epoch, optimizer_state = 0, None
    resume_state = run.load_resume() if args.resume else None
    if resume_state is not None:
        core.load_state_dict(resume_state["model_state"])
        start_epoch = int(resume_state["epoch"])
        optimizer_state = resume_state.get("optimizer_state")
        if resume_state.get("rng_state") is not None:
            torch.set_rng_state(resume_state["rng_state"])

    n_params = sum(p.numel() for p in core.parameters())
    config = {
        "model": "ChemKAN-KineticCore", "chemical_system": "biodiesel",
        "experiment": "observed_interval_training",
        "experiment_name": args.experiment_name, "seed": args.seed,
        "sensitivity_backend": solver.sensitivity, "device": str(device),
        "architecture": {"hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
                         "n_mu": args.n_mu, "use_base_act": args.use_base_act},
        "parameter_count": n_params, "optimizer": "Adam", "learning_rate": args.lr,
        "epochs": args.epochs,
        "solver": {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol},
        "loss": "normalized trajectory MSE (Eq. 18), unchanged formula",
        "training_procedure": _PROCEDURE,
        "training_objective": _TRAIN_OBJECTIVE,
        "updates_per_epoch": 1,
        "time_grid": {"n_saved_times": n_times, "n_intervals": n_intervals,
                      "t_start_s": float(t[0]), "t_end_s": float(t[-1]),
                      "dt_min_s": float((t[1:] - t[:-1]).min()),
                      "dt_max_s": float((t[1:] - t[:-1]).max()),
                      "note": "read from biodiesel.npz; the paper states 1 s sampling, the "
                              "archive stores 30 points over 30 s (dt ~ 1.03448 s). The "
                              "archive is used as-is; no data is regenerated."},
        "pinn": {"enabled": False},
        "normalization": {"input_scaling": args.input_scaling, "stats": "train-only min-max"},
        "dataset": "biodiesel.npz (train split, clean)",
        "noise": None,
        "evaluation": {"eval_every": args.eval_every,
                       "columns": ["full_rollout_train_mse", "full_rollout_test_mse_clean"],
                       "convention": _EVAL_CONVENTION,
                       "parameter_state": "before this epoch's optimizer update -- the same "
                                          "state that produced the epoch's training loss; "
                                          "the row at epoch == epochs is the final weights",
                       "normalizer": "train-only min-max (never refit on test data)",
                       "test_reference": "clean test trajectories (Eq. 22)"},
    }
    if resume_state is not None:
        check_resume_config(resume_state.get("config", {}), config)
        if args.epochs < start_epoch:
            raise SystemExit(f"--resume: requested epochs {args.epochs} < already-completed "
                             f"{start_epoch}; the epoch total may grow, never shrink.")
        config = resume_state.get("config", config)
        logging.info("resuming from update %d (original config preserved)", start_epoch)
    else:
        run.write_config(config)

    history = run.history(
        "history.csv",
        ["epoch", "total_loss", "full_rollout_train_mse", "full_rollout_test_mse_clean",
         "eval_seconds", "elapsed_seconds"],
        resume_from=start_epoch)

    provenance = {
        "architecture": {"hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
                         "n_mu": args.n_mu, "use_base_act": args.use_base_act},
        "data": {"species": data["species"], "species_dim": species_dim},
        "training": {"learning_rate": args.lr, "epochs": args.epochs,
                     "seed": args.seed, "alpha_pinn": None, "use_pinn": False,
                     "noise_percent": None},
        "solver": {"method": solver.method, "rtol": solver.rtol,
                   "atol": solver.atol, "sensitivity": solver.sensitivity},
        "state_representation": "physical",
        "input_scaling": input_scaling_meta(args.input_scaling, full_norm),
        # Extra keys; ignored by the existing loaders, which read only the six above.
        "experiment": "observed_interval_training",
        "training_procedure": _PROCEDURE,
        "n_intervals": n_intervals,
    }

    snap = None
    if args.snapshot_epochs.strip() and run.enabled:
        from diagnostics._instrumentation import Stage2Snapshot
        snap = Stage2Snapshot(core, run.run_dir,
                              [int(e) for e in args.snapshot_epochs.split(",") if e.strip()],
                              extra={**provenance, "run_id": run.run_id},
                              name_fmt="checkpoint_epoch_{epoch}.pt")
        logging.info("snapshots at updates %s", sorted(snap.epochs))

    opt = torch.optim.Adam(core.parameters(), lr=args.lr)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)

    def should_eval(epoch: int) -> bool:
        return epoch == 0 or not args.eval_every or epoch % args.eval_every == 0

    rng = range(start_epoch, args.epochs)
    use_bar = _tqdm is not None
    epoch_iter = (_tqdm(rng, desc="observed-interval", unit="ep",
                        initial=start_epoch, total=args.epochs) if use_bar else rng)
    started = time.perf_counter()
    eval_total = 0.0
    objective = float("nan")
    try:
        for epoch in epoch_iter:
            opt.zero_grad()                                   # once per epoch, before all intervals
            objective = float(interval_objective())           # accumulates all interval gradients
            components = full_rollout_metrics() if should_eval(epoch) else {}
            eval_total += components.get("eval_seconds", 0.0)
            opt.step()                                        # exactly one update per epoch
            history.on_epoch(epoch, objective, components, time.perf_counter() - started)
            if snap is not None:
                snap.on_epoch(epoch, objective, components, 0.0)
            if use_bar:
                epoch_iter.set_postfix(obj=f"{objective:.3e}")
            elif epoch % 100 == 0:
                logging.info("update %5d  interval objective %.6e", epoch, objective)
            if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                run.save_resume({"stage": "observed_intervals", "epoch": epoch + 1,
                                 "model_state": core.state_dict(),
                                 "optimizer_state": opt.state_dict(),
                                 "config": config, "rng_state": torch.get_rng_state()})
    except KeyboardInterrupt:
        if use_bar:
            epoch_iter.close()
        history.close()
        logging.warning("interrupted; resume checkpoint preserved for --resume")
        run.finish(ok=False)
        raise SystemExit(130)
    if use_bar:
        epoch_iter.close()

    # Final row at the FINAL weights (after the last update), so the saved checkpoint's
    # metrics are on record. The objective here is recomputed without gradients.
    train_seconds = time.perf_counter() - started
    final_components = full_rollout_metrics()
    with torch.no_grad():
        final_objective = float(accumulate_interval_gradients(
            dynamics, obs, obs_norm, t, loss_norm, solver, backward=False))
    history.on_epoch(args.epochs, final_objective, final_components, train_seconds)
    history.close()

    logging.info("final interval objective: %.6e", final_objective)
    logging.info("final full-rollout train MSE: %.6e | clean test MSE: %.6e",
                 final_components["full_rollout_train_mse"],
                 final_components["full_rollout_test_mse_clean"])

    checkpoint = {"model_state": core.state_dict(), **provenance,
                  "final_metrics": {
                      "interval_objective": final_objective,
                      **{k: v for k, v in final_components.items() if k != "eval_seconds"},
                      "updates": args.epochs,
                      "evaluation_convention": _EVAL_CONVENTION},
                  "runtime": {"train_seconds_this_segment": round(train_seconds, 3),
                              "evaluation_seconds_this_segment": round(eval_total, 3),
                              "updates_this_segment": args.epochs - start_epoch,
                              "note": "segment wall time; resumed runs measure only the "
                                      "current segment. Evaluation time is included in "
                                      "train_seconds and reported separately here."}}
    if snap is not None:
        for path in snap.written:
            logging.info("wrote %s", path)
    checkpoint["run_id"] = run.run_id
    run.save_final(checkpoint)
    run.finish(ok=True)


if __name__ == "__main__":
    main()
