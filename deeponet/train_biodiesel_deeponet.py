r"""Train the biodiesel DeepONet baseline (ChemKAN Figs. 4, 5, 6).

Same dataset, same train-only min-max normalizer and the same Eq. 18 loss reduction as
the ChemKAN runs, so the two models are scored identically. The architecture and its
provenance labels live in ``biodiesel_deeponet.py``; the optimizer settings are the
reference example's (Adam, lr = 1e-3) and are marked as such in ``config.json``.

Artifacts follow the ChemKAN run-directory layout (``chemkan/scripts/_run.py``):
``checkpoint_final.pt``, ``config.json``, ``run.log``, ``history.csv``, and a transient
``checkpoint_resume.pt``.

    # Fig. 5 noise sweep (10,000 epochs, per-epoch clean-test history)
    python train_biodiesel_deeponet.py --noise-percent 15 --epochs 10000 --eval-every 1 \
        --run-dir ../results/reproduction/baselines/deeponet/biodiesel/reference_final_trunk_relu/noise/noise15_seed0

    # Fig. 4 width sweep (50,000 epochs, no per-epoch test history needed)
    python train_biodiesel_deeponet.py --width 6 --epochs 50000 \
        --run-dir ../results/reproduction/baselines/deeponet/biodiesel/reference_final_trunk_relu/scaling/w6_seed0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "chemkan" / "scripts"))

from _data import available_noise_percents, load_biodiesel, resolve_device   # noqa: E402
from _run import RunManager, check_resume_config                             # noqa: E402

from biodiesel_deeponet import (COMPARISON_WIDTH, PAPER_PARAMS, architecture,  # noqa: E402
                                LEGACY_ARCHITECTURE_VERSION, REFERENCE_ARCHITECTURE_VERSION,
                                build, prepare_inputs)
from chemkan.losses import trajectory_mse                                    # noqa: E402
from chemkan.normalization import MinMaxNormalizer                           # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=COMPARISON_WIDTH,
                   help="hidden width w; w=8 reproduces the paper-described architecture")
    p.add_argument("--epochs", type=int, default=10000,
                   help="10000 for the Fig. 5 noise sweep; 50000 for the Fig. 4 width sweep")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="reference-example value, NOT a ChemKAN-paper value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-percent", type=int, default=None,
                   help="train against the stored deterministic noisy observations")
    p.add_argument("--eval-every", type=int, default=0,
                   help="log test MSE every N epochs (0 = off, 1 = every epoch)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--experiment-name", default="main")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def require_reference_run_directory(run_dir):
    """Protect legacy artifacts even when --resume or --overwrite is requested."""
    root = Path(run_dir)
    paths = ([root / "config.json"] if (root / "config.json").exists() else [])
    paths += sorted(root.glob("checkpoint*.pt"))
    for path in paths:
        record = (json.loads(path.read_text()) if path.suffix == ".json" else
                  torch.load(path, map_location="cpu", weights_only=False))
        arch = record.get("architecture", record.get("config", {}).get("architecture", {}))
        version = arch.get("architecture_version", LEGACY_ARCHITECTURE_VERSION)
        if version != REFERENCE_ARCHITECTURE_VERSION:
            raise SystemExit(
                f"{path} has architecture_version={version!r}. "
                "Refusing to resume or overwrite a legacy or unknown architecture. "
                "Use a new run directory for reference_final_trunk_relu.")


def main():
    args = build_parser().parse_args()
    require_reference_run_directory(args.run_dir)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    if args.noise_percent is not None and args.noise_percent not in available_noise_percents():
        raise SystemExit(f"noise level {args.noise_percent}% is not stored in biodiesel.npz "
                         f"(have {available_noise_percents()}). Add it with "
                         f"chemkan/scripts/data_gen/add_biodiesel_noise_level.py first.")

    train = load_biodiesel(split="train", noise_percent=args.noise_percent)
    test = load_biodiesel(split="test", noise_percent=args.noise_percent)

    # Train-only statistics, exactly as the ChemKAN runs use them (Eq. 18): the (m+1,)
    # full-state normalizer scales the branch input, its species subset the loss space.
    T_train = train["T_const"]
    full_norm = MinMaxNormalizer(
        torch.cat([train["u_min"], T_train.min().reshape(1)]),
        torch.cat([train["u_max"], T_train.max().reshape(1)])).to(device)
    norm = full_norm.subset(slice(0, len(train["species"]))).to(device)
    t_end = float(train["t"][-1])

    u0, tau = prepare_inputs(train, full_norm, t_end)
    target = norm.normalize(train["targets_TBm"].to(device))     # observations
    test_u0, test_tau = prepare_inputs(test, full_norm, t_end)
    test_obs = norm.normalize(test["targets_TBm"].to(device))
    test_clean = norm.normalize(test["species_TBm"].to(device))  # Eq. 22 reference

    model = build(args.width, seed=args.seed).to(device)

    run = RunManager(args.run_dir, "biodiesel-deeponet",
                     resume=args.resume, overwrite=args.overwrite)
    run.start()

    start_epoch, optimizer_state = 0, None
    resume_state = run.load_resume() if args.resume else None
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        start_epoch = int(resume_state["epoch"])
        optimizer_state = resume_state.get("optimizer_state")
        if resume_state.get("rng_state") is not None:
            torch.set_rng_state(resume_state["rng_state"])

    arch = architecture(model)
    config = {
        "model": "DeepONet-biodiesel", "chemical_system": "biodiesel",
        "experiment_name": args.experiment_name, "seed": args.seed,
        "device": str(device), "architecture": arch,
        "parameter_count": arch["parameter_count"],
        "paper_parameter_count": PAPER_PARAMS,
        "optimizer": "Adam", "learning_rate": args.lr, "epochs": args.epochs,
        "loss": "normalized trajectory MSE (Eq. 18)",
        "normalization": {
            "input_scaling": "min-max on the branch input [Y0, T]; trunk input tau = t/t_end",
            "output": "normalized species u_hat (compared directly with normalized targets)",
            "stats": "train-only min-max", "t_end_s": t_end,
        },
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
        "provenance": {
            "architecture": "literal reconstruction of the paper's prose description; "
                            "340 parameters at w=8 vs the 308 reported (unexplained)",
            "optimizer_and_init": "REFERENCE-DERIVED from deeponet/src/deeponet_dataset.py "
                                  "(relu, Glorot normal, biased Linear, Adam lr=1e-3); "
                                  "not stated by the ChemKAN paper",
        },
    }
    if resume_state is not None:
        check_resume_config(resume_state.get("config", {}), config)
        if args.epochs < start_epoch:
            raise SystemExit(f"--resume: requested epochs {args.epochs} < already-completed "
                             f"{start_epoch}")
        config = resume_state.get("config", config)
        logging.info("resuming DeepONet from epoch %d (original config preserved)", start_epoch)
    else:
        run.write_config(config)

    logging.info("DeepONet w=%d: %d trainable parameters; branch inputs [Y0, T] use "
                 "train-only min-max scaling; trunk time input tau=t/%.1fs; "
                 "outputs are normalized species concentrations",
                 args.width, arch["parameter_count"], t_end)
    if args.width == COMPARISON_WIDTH:
        logging.info("Paper main-model comparison: %d reported parameters; "
                     "our reconstructed w=%d model has %d",
                     PAPER_PARAMS, COMPARISON_WIDTH, arch["parameter_count"])

    columns = ["epoch", "total_loss", "mse_loss"]
    if args.eval_every:
        columns += ["test_mse_noisy", "test_mse_clean"]
    history = run.history("history.csv", columns + ["elapsed_seconds"], resume_from=start_epoch)

    def evaluate_test(epoch: int) -> dict:
        """Evaluation-only test metrics at the pre-update parameter state (see trainer)."""
        if not args.eval_every or epoch % args.eval_every:
            return {}
        rng_state = torch.get_rng_state()
        try:
            with torch.no_grad():
                pred = model(test_u0, test_tau)          # already in normalized space
                return {"test_mse_noisy": float(trajectory_mse(pred, test_obs)),
                        "test_mse_clean": float(trajectory_mse(pred, test_clean))}
        finally:
            torch.set_rng_state(rng_state)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)

    started = time.perf_counter()
    loss = torch.tensor(float("nan"))
    try:
        for epoch in range(start_epoch, args.epochs):
            opt.zero_grad()
            loss = trajectory_mse(model(u0, tau), target)
            components = {"mse_loss": float(loss.detach())}
            components.update(evaluate_test(epoch))     # BEFORE the update
            loss.backward()
            opt.step()
            history.on_epoch(epoch, float(loss.detach()), components,
                             time.perf_counter() - started)
            if epoch % 1000 == 0:
                logging.info("epoch %6d  loss %.6e", epoch, float(loss.detach()))
            if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                run.save_resume({"stage": "main", "epoch": epoch + 1,
                                 "architecture": arch,
                                 "model_state": model.state_dict(),
                                 "optimizer_state": opt.state_dict(),
                                 "config": config, "rng_state": torch.get_rng_state()})
    except KeyboardInterrupt:
        history.close()
        logging.warning("interrupted; resume checkpoint preserved for --resume")
        run.finish(ok=False)
        raise SystemExit(130)
    history.close()
    logging.info("final training loss: %.6e", float(loss.detach()))

    run.save_final({
        "model_state": model.state_dict(),
        "architecture": arch,
        "data": {"species": train["species"], "species_dim": len(train["species"])},
        "training": {"learning_rate": args.lr, "epochs": args.epochs, "seed": args.seed,
                     "noise_percent": args.noise_percent},
        "normalization": {"u_min": full_norm.u_min.detach().cpu(),      # (m+1,) branch stats
                          "u_max": full_norm.u_max.detach().cpu(),
                          "t_end_s": t_end,
                          "output_space": "normalized species"},
    })
    run.finish(ok=True)


if __name__ == "__main__":
    main()
