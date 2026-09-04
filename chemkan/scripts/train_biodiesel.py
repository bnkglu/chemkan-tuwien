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
from _data import input_scaling_meta, load_biodiesel, resolve_device
from _run import RunManager, check_resume_config

from chemkan.dynamics import KineticDynamics
from chemkan.losses import trajectory_mse
from chemkan.model import KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig
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

    data = load_biodiesel(split="train")
    species_dim = data["species_TBm"].shape[-1]              # data-derived, not hard-coded
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
    target = loss_norm.normalize(data["species_TBm"].to(device))    # (T, B, m) normalized

    def loss_fn(pred):                                       # pred: (T, B, m) physical
        mse = trajectory_mse(loss_norm.normalize(pred), target)
        return mse, {"mse_loss": mse.detach()}              # total + component (biodiesel = MSE only)

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
        "dataset": "biodiesel.npz (train split)", "noise": None,
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

    history = run.history("history.csv",
                          ["epoch", "total_loss", "mse_loss", "elapsed_seconds"],
                          resume_from=start_epoch)

    def save_resume(next_epoch, opt_state):
        run.save_resume({"stage": "main", "epoch": next_epoch,
                         "model_state": core.state_dict(), "optimizer_state": opt_state,
                         "config": config, "rng_state": torch.get_rng_state()})

    try:
        final = train_kinetic_stage(dynamics, Y0, t, loss_fn, epochs=args.epochs,
                                    lr=args.lr, solver=solver, start_epoch=start_epoch,
                                    optimizer_state=optimizer_state, on_epoch=history.on_epoch,
                                    checkpoint_every=args.checkpoint_every, save_resume=save_resume)
    except KeyboardInterrupt:
        history.close()
        logging.warning("interrupted; resume checkpoint preserved for --resume")
        run.finish(ok=False)                            # closes log handler; keeps checkpoint_resume.pt
        raise SystemExit(130)
    history.close()
    logging.info("final training loss: %.6e", final)

    checkpoint = {
        "model_state": core.state_dict(),
        "architecture": {
            "hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
            "n_mu": args.n_mu, "use_base_act": args.use_base_act,
        },
        "data": {"species": data["species"], "species_dim": species_dim},
        "training": {"learning_rate": args.lr, "epochs": args.epochs,
                     "seed": args.seed, "alpha_pinn": None, "use_pinn": False},
        "solver": {"method": solver.method, "rtol": solver.rtol,
                   "atol": solver.atol, "sensitivity": solver.sensitivity},
        "state_representation": "physical",
        "input_scaling": input_scaling_meta(args.input_scaling, full_norm),
    }
    if run.enabled:
        checkpoint["run_id"] = run.run_id
        run.save_final(checkpoint)                          # writes final, deletes resume
        run.finish(ok=True)
    else:
        torch.save(checkpoint, args.out)                    # legacy flat-file behavior
        logging.info("saved kinetic core (+ metadata) -> %s", args.out)


if __name__ == "__main__":
    main()
