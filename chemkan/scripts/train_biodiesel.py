r"""Train a biodiesel kinetic core (isothermal, MSE only) -- ChemKAN Eq. 13, 18.

This is a benchmark configuration of the GENERIC ChemKAN library: ``species_dim`` is
inferred from the data, architecture is chosen here (paper count-matching defaults,
overridable via CLI), and the MSE-only loss is constructed in this script.

Flow:  Y0 --odeint--> KineticDynamics --> [Y,T] physical --> train-minmax --> KineticCore
       --> physical dY/dt.  Only the kinetic-core parameters are optimized; no thermo,
no PINN.

The paper reproduction trains for 1e4 epochs (the default here). For a quick smoke
run use e.g. ``python scripts/train_biodiesel.py --epochs 100``. Compare input scaling
with ``--input-scaling none`` (raw-input ablation) vs. the default ``minmax``.
"""

from __future__ import annotations

import argparse
import logging

import torch
from _data import available_noise_percents, input_scaling_meta, load_biodiesel, resolve_device
from _run import RunManager, check_resume_config

from chemkan.dynamics import KineticDynamics
from chemkan.losses import trajectory_mse
from chemkan.model import KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate
from chemkan.temperature import ConstantTemperature
from chemkan.training import train_kinetic_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # architecture (paper count-matching defaults; species_dim comes from data)
    p.add_argument("--hidden-dim", type=int, default=4)
    p.add_argument("--num-basis", type=int, default=3)
    p.add_argument("--n-mu", type=int, default=2)
    p.add_argument("--use-base-act", action="store_true",
                   help="literal Eq. 11 base path (default OFF = paper count-matching)")
    # training
    p.add_argument("--epochs", type=int, default=10000,
                   help="paper reproduction uses 1e4; use e.g. 100 for a smoke run")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for model init / training reproducibility")
    p.add_argument("--input-scaling", default="minmax", choices=["minmax", "none"],
                   help="pre-KAN scaling of the physical state ('none' = raw ablation)")
    # noise + in-training evaluation (Fig. 5). Both default OFF, so an unflagged run
    # reproduces the existing clean runs exactly, including the history columns.
    p.add_argument("--noise-percent", type=int, default=None,
                   help="train against the stored deterministic noisy observations at this "
                        "whole-percent level (clean when omitted). The level must already "
                        "exist in biodiesel.npz.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="log test MSE every N epochs (0 = off, 1 = every epoch). "
                        "Evaluation only: no optimizer step, no parameter change.")
    p.add_argument("--snapshot-epochs", default="",
                   help="comma-separated epoch counts to save as checkpoint_epoch_<N>.pt. "
                        "Snapshot N is the model after exactly N optimizer steps, so a "
                        "5000-epoch snapshot of a 10,000-epoch run IS a 5,000-epoch result "
                        "for the same architecture/settings. Analysis only: changes no "
                        "training state and is never overwritten within a run.")
    # solver (explicit PyTorch implementation defaults, NOT paper values)
    p.add_argument("--solver-method", default="tsit5")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--out", default="biodiesel_kinetic.pt",
                   help="legacy flat checkpoint path (used only when --run-dir is not given)")
    # run-directory layout (organized reproduction runs)
    p.add_argument("--run-dir", default=None,
                   help="one directory per run; writes checkpoint_final.pt, config.json, "
                        "run.log, history.csv, checkpoint_resume.pt. Overrides --out.")
    p.add_argument("--experiment-name", default="main",
                   help="recorded in config.json (e.g. main, noise_15, scaling_...).")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="overwrite checkpoint_resume.pt every N epochs (run-dir mode).")
    p.add_argument("--resume", action="store_true",
                   help="resume from RUN_DIR/checkpoint_resume.pt if present.")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing an existing completed run (checkpoint_final.pt).")
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.noise_percent is not None and args.noise_percent not in available_noise_percents():
        raise SystemExit(f"noise level {args.noise_percent}% is not stored in biodiesel.npz "
                         f"(have {available_noise_percents()}). Add it with "
                         f"scripts/data_gen/add_biodiesel_noise_level.py first.")
    data = load_biodiesel(split="train", noise_percent=args.noise_percent)
    species_dim = data["species_TBm"].shape[-1]              # data-derived, not hard-coded
    test = load_biodiesel(split="test", noise_percent=args.noise_percent) if args.eval_every else None
    solver = SolverConfig(method=args.solver_method, rtol=args.rtol, atol=args.atol,
                          sensitivity="direct_autograd")     # every field explicit

    core = KineticCore(species_dim=species_dim, hidden_dim=args.hidden_dim,
                       num_basis=args.num_basis, n_mu=args.n_mu,
                       use_base_act=args.use_base_act).to(device)

    # Full-state [Y1..Ym, T] train-only input normalizer. The biodiesel archive stores
    # species-only stats + per-trajectory constant T, so we append T's TRAIN min/max.
    T_const = data["T_const"]
    u_min_full = torch.cat([data["u_min"], T_const.min().reshape(1)]).to(device)
    u_max_full = torch.cat([data["u_max"], T_const.max().reshape(1)]).to(device)
    full_norm = MinMaxNormalizer(u_min_full, u_max_full).to(device)     # (m+1,)
    loss_norm = full_norm.subset(slice(0, species_dim))                 # species-only (m,)
    input_normalizer = full_norm if args.input_scaling == "minmax" else None

    dynamics = KineticDynamics(core, ConstantTemperature(T_const),
                               input_normalizer=input_normalizer).to(device)

    Y0 = data["Y0"].to(device)
    t = data["t"].to(device)
    # Training targets are the OBSERVATIONS (noisy when --noise-percent is given);
    # data["species_TBm"] stays the clean trajectory and is never a training target.
    target = loss_norm.normalize(data["targets_TBm"].to(device))    # (T, B, m) normalized

    if test is not None:
        test_dyn = KineticDynamics(core, ConstantTemperature(test["T_const"].to(device)),
                                   input_normalizer=input_normalizer).to(device)
        test_Y0, test_t = test["Y0"].to(device), test["t"].to(device)
        test_obs = loss_norm.normalize(test["targets_TBm"].to(device))    # noisy observations
        test_clean = loss_norm.normalize(test["species_TBm"].to(device))  # Eq. 22 reference

    # ``_optimize`` calls loss_fn exactly once per epoch, in order, from ``start_epoch``.
    # The counter is seeded from start_epoch below, after the resume state is known.
    epoch_state = {"epoch": 0}

    def evaluate_test(epoch: int) -> dict:
        """Evaluation-only test metrics at the CURRENT parameter state.

        Called from ``loss_fn``, i.e. before this epoch's optimizer step, so the logged
        test values belong to exactly the parameter state that produced the epoch's
        training loss. Runs under ``no_grad``; takes no optimizer step and writes no
        parameter. The model has no train/eval-dependent layers (no dropout or batch
        norm), so its mode is deliberately left untouched. The RNG state is saved and
        restored so the evaluation cannot shift the training run.
        """
        if test is None or not args.eval_every or epoch % args.eval_every:
            return {}
        rng_state = torch.get_rng_state()
        try:
            with torch.no_grad():
                pred = integrate(test_dyn, test_Y0, test_t, solver)   # (T, B, m) physical
                pred_norm = loss_norm.normalize(pred)
                # One prediction, two targets -- only the reference changes (Eq. 18 vs Eq. 22).
                return {"test_mse_noisy": float(trajectory_mse(pred_norm, test_obs)),
                        "test_mse_clean": float(trajectory_mse(pred_norm, test_clean))}
        finally:
            torch.set_rng_state(rng_state)

    def loss_fn(pred):                                       # pred: (T, B, m) physical
        mse = trajectory_mse(loss_norm.normalize(pred), target)
        components = {"mse_loss": mse.detach()}             # total + component (biodiesel = MSE only)
        components.update(evaluate_test(epoch_state["epoch"]))
        epoch_state["epoch"] += 1
        return mse, components

    # --- run-directory plumbing (organization only; math unchanged) ----------------
    run = RunManager(args.run_dir, "biodiesel", resume=args.resume, overwrite=args.overwrite)
    run.start()

    start_epoch, optimizer_state = 0, None
    resume_state = run.load_resume() if args.resume else None
    if resume_state is not None:
        core.load_state_dict(resume_state["model_state"])
        start_epoch = int(resume_state["epoch"])
        optimizer_state = resume_state.get("optimizer_state")
        if resume_state.get("rng_state") is not None:
            torch.set_rng_state(resume_state["rng_state"])

    epoch_state["epoch"] = start_epoch          # in-loop evaluation follows the real epoch

    n_params = sum(p.numel() for p in core.parameters())
    config = {
        "model": "ChemKAN-KineticCore", "chemical_system": "biodiesel",
        "experiment_name": args.experiment_name, "seed": args.seed,
        "sensitivity_backend": solver.sensitivity, "device": str(device),
        "architecture": {"hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
                         "n_mu": args.n_mu, "use_base_act": args.use_base_act},
        "parameter_count": n_params, "optimizer": "Adam", "learning_rate": args.lr,
        "epochs": args.epochs,
        "solver": {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol},
        "loss": "normalized trajectory MSE (Eq. 18)", "pinn": {"enabled": False},
        "normalization": {"input_scaling": args.input_scaling, "stats": "train-only min-max"},
        "dataset": "biodiesel.npz (train split)",
        "noise": None if args.noise_percent is None else {
            "percent": args.noise_percent,
            "source": f"biodiesel.npz train_states_noise{args.noise_percent:02d}",
        },
        "in_training_evaluation": None if not args.eval_every else {
            "eval_every": args.eval_every,
            "columns": ["test_mse_noisy", "test_mse_clean"],
            "split": "test", "conditions": int(test["species_TBm"].shape[1]),
            "parameter_state": "before this epoch's optimizer update -- the same state "
                               "that produced the epoch's training loss",
            "normalizer": "train-only min-max (never refit on test data)",
        },
    }
    if resume_state is not None:
        # A resumed run may not silently change its science. Epoch total may GROW but
        # never fall below the already-completed epoch; original provenance is preserved.
        check_resume_config(resume_state.get("config", {}), config)
        if args.epochs < start_epoch:
            raise SystemExit(f"--resume: requested epochs {args.epochs} < already-completed "
                             f"{start_epoch}; the epoch total may grow on resume, never shrink.")
        config = resume_state.get("config", config)     # keep the original run's config.json
        logging.info("resuming biodiesel from epoch %d (original config preserved)", start_epoch)
    else:
        run.write_config(config)

    # Extra columns only when evaluation is on, so an unflagged run keeps the exact
    # four-column history of the existing clean runs.
    columns = ["epoch", "total_loss", "mse_loss"]
    if args.eval_every:
        columns += ["test_mse_noisy", "test_mse_clean"]
    history = run.history("history.csv", columns + ["elapsed_seconds"],
                          resume_from=start_epoch)

    def save_resume(next_epoch, opt_state):
        run.save_resume({"stage": "main", "epoch": next_epoch,
                         "model_state": core.state_dict(), "optimizer_state": opt_state,
                         "config": config, "rng_state": torch.get_rng_state()})

    # Shared checkpoint body: the final checkpoint and every snapshot describe the same
    # run, so evaluation loads a snapshot exactly as it loads checkpoint_final.pt.
    provenance = {
        "architecture": {"hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
                         "n_mu": args.n_mu, "use_base_act": args.use_base_act},
        "data": {"species": data["species"], "species_dim": species_dim},
        "training": {"learning_rate": args.lr, "epochs": args.epochs,
                     "seed": args.seed, "alpha_pinn": None, "use_pinn": False,
                     "noise_percent": args.noise_percent},
        "solver": {"method": solver.method, "rtol": solver.rtol,
                   "atol": solver.atol, "sensitivity": solver.sensitivity},
        "state_representation": "physical",
        "input_scaling": input_scaling_meta(args.input_scaling, full_norm),
    }

    epoch_cb, snap = history.on_epoch, None
    if args.snapshot_epochs.strip() and run.enabled:
        from diagnostics._instrumentation import Stage2Snapshot
        from diagnostics._stage2_probe import chain
        snap = Stage2Snapshot(
            core, run.run_dir,
            [int(e) for e in args.snapshot_epochs.split(",") if e.strip()],
            extra={**provenance, "run_id": run.run_id},
            name_fmt="checkpoint_epoch_{epoch}.pt")
        epoch_cb = chain(history.on_epoch, snap.on_epoch)
        logging.info("snapshots at epochs %s", sorted(snap.epochs))

    try:
        final = train_kinetic_stage(dynamics, Y0, t, loss_fn, epochs=args.epochs,
                                    lr=args.lr, solver=solver, start_epoch=start_epoch,
                                    optimizer_state=optimizer_state, on_epoch=epoch_cb,
                                    checkpoint_every=args.checkpoint_every, save_resume=save_resume)
    except KeyboardInterrupt:
        history.close()
        logging.warning("interrupted; resume checkpoint preserved for --resume")
        run.finish(ok=False)                            # closes log handler; keeps checkpoint_resume.pt
        raise SystemExit(130)
    history.close()
    if snap is not None:
        for path in snap.written:
            logging.info("wrote %s", path)
    logging.info("final training loss: %.6e", final)

    checkpoint = {"model_state": core.state_dict(), **provenance}
    if run.enabled:
        checkpoint["run_id"] = run.run_id
        run.save_final(checkpoint)                          # writes final, deletes resume
        run.finish(ok=True)
    else:
        torch.save(checkpoint, args.out)                    # legacy flat-file behavior
        logging.info("saved kinetic core (+ metadata) -> %s", args.out)


if __name__ == "__main__":
    main()
