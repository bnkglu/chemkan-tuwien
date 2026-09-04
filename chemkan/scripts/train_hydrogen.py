r"""Train hydrogen ChemKAN in two explicit stages -- ChemKAN Eq. 13-18, Sec. II C 5.

Benchmark configuration of the GENERIC ChemKAN library. ``species_dim`` is inferred
from the data; architecture is chosen here (paper count-matching defaults). PINN
usage per stage is an explicit CLI choice, and both stage loss functions are built
in THIS script -- the training library knows nothing about hydrogen or PINN.

Stage 1: integrate species only; temperature observed/interpolated; kinetic core.
Stage 2: integrate full [Y, T] with the COMPLETE model; update ALL parameters.

Stage-1 external temperature source is selectable (``--stage1-temperature-source``).
The supervisor-approved default ``dense-cantera`` reads a precomputed dense Cantera
trajectory (default 20000 points over 0.6 ms, generated once by
``generate_hydrogen.py --temperature-only``) through the existing linear
``ObservedTemperature``; ``training-data`` reads the original sparse 50-point
observed trajectory instead (ablation). Either way the Stage-1 output grid and
species targets stay on the canonical 50-point dataset -- only ``T(t)`` becomes
dense. 20000 is a reproduction choice, not a paper-specified value.

Architecture defaults are ``num_basis = 4`` with the Eq. 11 base path ON, which
reproduces the reported 344 parameters. ``num_basis`` is INFERRED either way -- the
paper states neither the grid size nor whether the base path is counted. The earlier
``--num-basis 5 --no-use-base-act`` reading also gives 344 and stays reachable from
the CLI. See ASSUMPTIONS.md §2-3.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # diagnostics/ helpers
from _chemistry import (
    ATOMIC_WEIGHTS,
    ELEMENT_COUNTS,
    MOLAR_WEIGHTS,
    assert_species_order,
)
from _data import (
    DATA_DIR,
    input_scaling_meta,
    load_hydrogen,
    load_hydrogen_temperature,
    resolve_device,
)
from _predictions import checkpoint_sha256
from _run import RunManager, check_resume_config

from chemkan.dynamics import ChemKANDynamics, KineticDynamics
from chemkan.losses import chemkan_loss, element_conservation_loss, trajectory_mse
from chemkan.model import ChemKAN
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig
from chemkan.temperature import ObservedTemperature
from chemkan.training import train_full_chemkan, train_kinetic_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden-dim", type=int, default=3)
    p.add_argument("--num-basis", type=int, default=4)      # inferred (see docstring)
    p.add_argument("--n-mu", type=int, default=3)
    # Eq. 11 base path ON by default; applies to the kinetic KAN and KAN_cor alike
    # (never to the thermo Linear layer, which has no edge functions).
    p.add_argument("--use-base-act", dest="use_base_act", action="store_true", default=True)
    p.add_argument("--no-use-base-act", dest="use_base_act", action="store_false",
                   help="historical N=5/base-OFF reading (use with --num-basis 5)")
    # stage epoch counts are implementation choices (paper does not specify them here)
    p.add_argument("--stage1-epochs", type=int, default=10000)
    p.add_argument("--stage2-epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for model init / training reproducibility")
    # Paper-aligned H2 interpretation: PINN ON in both Stage 1 and Stage 2.
    p.add_argument("--no-pinn-stage1", dest="pinn_stage1", action="store_false", default=True)
    p.add_argument("--no-pinn-stage2", dest="pinn_stage2", action="store_false", default=True)
    p.add_argument("--alpha-pinn", type=float, default=1e-4)
    p.add_argument("--input-scaling", default="minmax", choices=["minmax", "none"],
                   help="pre-KAN scaling of the physical state ('none' = raw ablation)")
    p.add_argument("--solver-method", default="tsit5")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    # Stage-1 external temperature source. The supervisor-approved default is a
    # dense precomputed Cantera trajectory read through ObservedTemperature; the
    # original sparse 50-point training-data trajectory remains as an ablation.
    p.add_argument("--stage1-temperature-source",
                   choices=["dense-cantera", "training-data"], default="dense-cantera",
                   help="Stage-1 temperature: 'dense-cantera' (default) reads a precomputed "
                        "dense Cantera trajectory; 'training-data' reads the original 50-point "
                        "observed trajectory from the canonical hydrogen dataset.")
    p.add_argument("--stage1-temperature-points", type=int, default=20000,
                   help="dense Stage-1 temperature resolution (dense-cantera source); loads "
                        "hydrogen_temperature_<N>.npz. Not a paper-specified value.")
    # DIAGNOSTIC: initialization of the Eq. 14 thermodynamic coefficients.
    # 'random' (default) = current behavior exactly; 'cantera' = physics-seeded (hypothesis test).
    p.add_argument("--thermo-init", choices=["random", "cantera", "scaled-random"],
                   default="random",
                   help="thermo.linear init: 'random' (default, unchanged), 'cantera' "
                        "(-h_k/cp at a reference state), or 'scaled-random' (default random "
                        "vector rescaled to the Cantera L2 norm -- magnitude-only control). "
                        "Diagnostic, not paper reproduction.")
    p.add_argument("--thermo-init-temperature", type=float, default=1050.0,
                   help="reference T [K] USED to evaluate the Cantera enthalpies and to select "
                        "the training initial condition (cantera / scaled-random).")
    p.add_argument("--thermo-init-phi", type=float, default=0.9,
                   help="reference equivalence ratio USED to select the training initial "
                        "condition whose composition sets cp (cantera / scaled-random). Must be "
                        "an exact training IC or the run fails loudly.")
    # scaled-random scaling mode (mutually exclusive)
    p.add_argument("--thermo-init-scale-factor", type=float, default=None,
                   help="scaled-random: multiply the random vector by this explicit factor.")
    p.add_argument("--thermo-match-cantera-norm", action="store_true",
                   help="scaled-random: rescale the random vector to ||w_cantera|| (norm only).")
    # DIAGNOSTIC: isolated RNG for the thermo.linear DIRECTION only
    p.add_argument("--thermo-linear-init-seed", type=int, default=0,
                   help="isolated seed for the thermo.linear random direction only. 0 keeps the "
                        "as-constructed default-random vector; other values redraw from the same "
                        "U(-1/sqrt(fan_in), 1/sqrt(fan_in)) using a private generator, leaving "
                        "kinetic.*, thermo.correction.* and the global RNG untouched.")
    # Stage-1 reuse: a completed run saves checkpoint_stage1.pt; --stage1-from loads that
    # kinetic core and SKIPS Stage 1, so several Stage-2 experiments share one kinetic model.
    p.add_argument("--stage1-from", default=None,
                   help="path to a checkpoint_stage1.pt; loads the kinetic core and skips "
                        "Stage 1 (thermo init policy is still applied fresh).")
    # DIAGNOSTIC Stage-2 probe (no effect on training; runs under no_grad)
    p.add_argument("--stage2-probe", action="store_true",
                   help="write stage2_probe.csv with loss / T-MSE / peak T / ignition / "
                        "thermo.linear at selected epochs.")
    p.add_argument("--stage2-snapshot-epochs", default="",
                   help="comma-separated Stage-2 epochs at which to save an immutable "
                        "checkpoint_stage2_epoch_<N>.pt snapshot (analysis only; does not "
                        "change training). Empty = none.")
    p.add_argument("--count-nfe", action="store_true",
                   help="record per-epoch ODE RHS evaluations (nfe) and epoch_wall_time_s "
                        "in the history CSVs. Behavior-neutral (forward hook).")
    p.add_argument("--stage2-probe-epochs", default="0,10,50,100,250,500",
                   help="comma-separated probe epochs (0 = before any update).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--out", default="hydrogen_chemkan.pt",
                   help="legacy flat checkpoint path (used only when --run-dir is not given)")
    # run-directory layout (organized reproduction runs)
    p.add_argument("--run-dir", default=None,
                   help="one directory per run; writes checkpoint_final.pt, config.json, "
                        "run.log, history_stage1.csv, history_stage2.csv, checkpoint_resume.pt. "
                        "Overrides --out.")
    p.add_argument("--experiment-name", default="main",
                   help="recorded in config.json (e.g. main, generalization).")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="overwrite checkpoint_resume.pt every N epochs within a stage.")
    p.add_argument("--resume", action="store_true",
                   help="resume from RUN_DIR/checkpoint_resume.pt (correct stage + epoch).")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing an existing completed run (checkpoint_final.pt).")
    return p


def stage1_temperature_metadata(source: str, n_points: int,
                                cache_file: str | None = None) -> dict:
    """Checkpoint record of the Stage-1 temperature-provider configuration.

    ``source`` is the resolved metadata tag ('dense_cantera' or 'training_data').
    The dense variant additionally records the cache file name (no absolute path).
    """
    meta = {
        "source": source,
        "n_points": int(n_points),
        "provider": "ObservedTemperature",
        "interpolation": "linear",
    }
    if source == "dense_cantera":
        meta["cache_file"] = cache_file
    return meta


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data = load_hydrogen(split="train")
    assert_species_order(data["species"])                  # PINN matrices are positional
    m = data["species_TBm"].shape[-1]                       # data-derived species dim
    solver = SolverConfig(method=args.solver_method, rtol=args.rtol, atol=args.atol,
                          sensitivity="direct_autograd")     # every field explicit

    model = ChemKAN(species_dim=m, hidden_dim=args.hidden_dim, num_basis=args.num_basis,
                    n_mu=args.n_mu, use_base_act=args.use_base_act).to(device)

    # Optional DIAGNOSTIC initialization of the Eq. 14 coefficients. Default 'random'
    # leaves PyTorch's nn.Linear init untouched (exact current behavior); 'cantera' seeds
    # them with the physical -h_k/cp at a documented reference state. Hypothesis test only.
    thermo_init_meta = {"thermo_init": args.thermo_init}
    if args.thermo_init == "cantera":
        from diagnostics._thermo_coeffs import MECH as _MECH
        from diagnostics._thermo_coeffs import cantera_coefficients
        from diagnostics._thermo_coeffs import resolve_training_ic
        _arch = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
        ref_Y = resolve_training_ic(_arch["train_ics"], _arch["train_states"],
                                    args.thermo_init_temperature, args.thermo_init_phi,
                                    species_dim=m)
        coeffs, cp_mass = cantera_coefficients(
            args.thermo_init_temperature, ref_Y, species=list(data["species"]), mech=_MECH)
        with torch.no_grad():
            w = model.thermo.linear.weight
            w.copy_(torch.as_tensor(coeffs.reshape(w.shape), dtype=w.dtype, device=w.device))
        thermo_init_meta.update({
            "mechanism": _MECH,
            "reference_T_K": float(args.thermo_init_temperature),
            "reference_phi": float(args.thermo_init_phi),
            "reference_state": f"initial composition of training IC (T0={args.thermo_init_temperature:g} K, "
                               f"phi={args.thermo_init_phi:g}); enthalpies evaluated at that T",
            "cp_mass_J_per_kg_K": cp_mass,
            "coefficients": [float(c) for c in coeffs],
            "species_order": [str(s) for s in data["species"]],
            "formula": "-partial_molar_enthalpies[k]/molecular_weights[k]/cp_mass  [K]",
        })
        logging.info("thermo.linear initialized from Cantera at T=%.0f K (|w| max %.3g)",
                     args.thermo_init_temperature, float(np.abs(coeffs).max()))
    elif args.thermo_init == "scaled-random":
        # DIAGNOSTIC CONTROL for "scale vs structure". The model is already fully built with
        # the ordinary random init; here we (optionally) redraw ONLY the thermo.linear
        # direction from an isolated generator, then multiply by ONE scalar. Direction, signs
        # and species ratios are whatever the random draw gave -- no Cantera values are copied.
        if (args.thermo_init_scale_factor is None) == (not args.thermo_match_cantera_norm):
            raise SystemExit("--thermo-init scaled-random needs exactly one of "
                             "--thermo-init-scale-factor or --thermo-match-cantera-norm")
        w = model.thermo.linear.weight
        if args.thermo_linear_init_seed != 0:
            # Isolated RNG: a private Generator never advances the global RNG stream, so every
            # other parameter stays bit-identical to the seed-0 arm. Same formula/distribution
            # as nn.Linear's default init: U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
            g = torch.Generator(device="cpu").manual_seed(int(args.thermo_linear_init_seed))
            bound = 1.0 / np.sqrt(w.shape[-1])
            with torch.no_grad():
                w.copy_(torch.empty(w.shape, dtype=torch.float32).uniform_(-bound, bound,
                                                                           generator=g).to(w.device))
        w_random = w.detach().cpu().numpy().ravel().copy()
        original_norm = float(np.linalg.norm(w_random))

        from diagnostics._thermo_coeffs import MECH as _MECH
        from diagnostics._thermo_coeffs import cantera_coefficients
        from diagnostics._thermo_coeffs import resolve_training_ic
        _arch = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
        _ref_Y = resolve_training_ic(_arch["train_ics"], _arch["train_states"],
                                     args.thermo_init_temperature, args.thermo_init_phi,
                                     species_dim=m)
        coeffs, cp_mass = cantera_coefficients(
            args.thermo_init_temperature, _ref_Y, species=list(data["species"]), mech=_MECH)
        cantera_norm = float(np.linalg.norm(coeffs))

        if args.thermo_match_cantera_norm:
            scale_factor = cantera_norm / original_norm
            mode = "match-cantera-norm"
        else:
            scale_factor = float(args.thermo_init_scale_factor)
            mode = "explicit-scale-factor"
        with torch.no_grad():
            w.mul_(scale_factor)                                  # magnitude only
        w_scaled = w.detach().cpu().numpy().ravel().copy()
        sign_match = int((np.sign(w_scaled) == np.sign(coeffs)).sum())

        thermo_init_meta.update({
            "mechanism": _MECH, "scaling_mode": mode,
            "thermo_linear_init_seed": int(args.thermo_linear_init_seed),
            "original_random_vector": [float(x) for x in w_random],
            "original_random_norm": original_norm,
            "requested_scale_factor": (None if args.thermo_init_scale_factor is None
                                       else float(args.thermo_init_scale_factor)),
            "effective_scale_factor": scale_factor,
            "resulting_vector": [float(x) for x in w_scaled],
            "resulting_norm": float(np.linalg.norm(w_scaled)),
            "cantera_reference_norm": cantera_norm,
            "sign_match_vs_cantera": f"{sign_match}/{len(coeffs)}",
            "reference_temperature": float(args.thermo_init_temperature),
            "reference_phi": float(args.thermo_init_phi),
            "cp_mass_J_per_kg_K": cp_mass,
            "species_order": [str(x) for x in data["species"]],
            "note": "Cantera vector used for its L2 NORM ONLY; direction/signs/ratios are random. "
                    "sign_match is descriptive only and never used to accept/reject a draw.",
        })
        logging.info("thermo.linear scaled-random [%s, dir seed %d]: norm %.6g -> %.6g "
                     "(x%.6g) | cantera norm %.6g | sign match %s",
                     mode, args.thermo_linear_init_seed, original_norm,
                     float(np.linalg.norm(w_scaled)), scale_factor, cantera_norm,
                     f"{sign_match}/{len(coeffs)}")

    stage1_from_meta = None
    if args.stage1_from:
        s1ck = torch.load(args.stage1_from, map_location=device, weights_only=False)
        if s1ck.get("architecture") != {"hidden_dim": args.hidden_dim,
                                        "num_basis": args.num_basis, "n_mu": args.n_mu,
                                        "use_base_act": args.use_base_act}:
            raise SystemExit(f"--stage1-from architecture {s1ck.get('architecture')} does not "
                             f"match the requested architecture")
        if list(s1ck.get("data", {}).get("species", [])) != [str(x) for x in data["species"]]:
            raise SystemExit("--stage1-from species order does not match the dataset")
        # Load ONLY the kinetic core: the thermo modules keep this run's own init policy,
        # which is what makes random-vs-cantera a controlled comparison.
        model.kinetic.load_state_dict(s1ck["kinetic_state"])
        stage1_from_meta = {"source": str(Path(args.stage1_from).name),
                            "path": str(args.stage1_from),
                            # provable fairness: every arm records the hash of the exact
                            # Stage-1 file it started from, so sharing can be verified later
                            "stage1_checkpoint_sha256": checkpoint_sha256(args.stage1_from),
                            "run_id": s1ck.get("run_id"),
                            "architecture": s1ck.get("architecture"),
                            "stage1_final_loss": s1ck.get("stage1_final_loss"),
                            "stage1_epochs": s1ck.get("stage1_epochs")}
        logging.info("loaded Stage-1 kinetic core from %s (stage1 loss %.6e); Stage 1 will be skipped",
                     args.stage1_from, float(s1ck.get("stage1_final_loss", float("nan"))))

    # Full-state [Y1..Ym, T] train-only normalizer (archive already stores m+1 stats).
    full_norm = MinMaxNormalizer(data["u_min"], data["u_max"]).to(device)
    species_norm = full_norm.subset(slice(0, m))           # species-only for Stage-1 loss
    input_normalizer = full_norm if args.input_scaling == "minmax" else None

    t = data["t"].to(device)
    ec = ELEMENT_COUNTS.to(device)
    aw = ATOMIC_WEIGHTS.to(device)
    mw = MOLAR_WEIGHTS.to(device)

    # --- Stage 1: species only, observed temperature ------------------------------
    # Only the EXTERNAL temperature provider changes here. The Stage-1 output grid
    # (t = data["t"], 50 points) and species targets stay on the canonical dataset.
    if args.stage1_temperature_source == "dense-cantera":
        dense = load_hydrogen_temperature(split="train",
                                          n_points=args.stage1_temperature_points)
        temp = ObservedTemperature(dense["t_dense"], dense["T_dense_TB1"]).to(device)
        stage1_temp_meta = stage1_temperature_metadata(
            "dense_cantera", args.stage1_temperature_points,
            cache_file=f"hydrogen_temperature_{args.stage1_temperature_points}.npz")
    else:  # "training-data": original sparse 50-point observed trajectory (ablation)
        temp = ObservedTemperature(data["t"], data["T_obs_TB1"]).to(device)
        stage1_temp_meta = stage1_temperature_metadata(
            "training_data", int(data["t"].shape[0]))
    logging.info("Stage-1 temperature provider: %s", stage1_temp_meta)

    kin_dyn = KineticDynamics(model.kinetic, temp,
                              input_normalizer=input_normalizer).to(device)
    Y0 = data["species_TBm"][0].to(device)                  # (B, m)
    tgt_species = species_norm.normalize(data["species_TBm"].to(device))   # (T, B, m)

    def stage1_loss(pred):                                  # pred: (T, B, m)
        species_mse = trajectory_mse(species_norm.normalize(pred), tgt_species)
        if args.pinn_stage1:                                # physical species for PINN
            pinn = args.alpha_pinn * element_conservation_loss(pred, ec, aw, mw)
            total = species_mse + pinn
        else:
            pinn = torch.zeros((), device=species_mse.device, dtype=species_mse.dtype)
            total = species_mse
        return total, {"species_mse": species_mse.detach(), "pinn_loss": pinn.detach()}

    # --- Stage 2 pieces (built now; used after Stage 1) ---------------------------
    chem_dyn = ChemKANDynamics(model, input_normalizer=input_normalizer).to(device)
    u0 = data["full_TBm1"][0].to(device)                    # (B, m+1)
    tgt_full = full_norm.normalize(data["full_TBm1"].to(device))           # (T, B, m+1)

    def stage2_loss(pred):                                  # pred: (T, B, m+1)
        pred_norm = full_norm.normalize(pred)
        total = chemkan_loss(
            pred_norm, tgt_full,
            use_pinn=args.pinn_stage2, alpha_pinn=args.alpha_pinn,
            Y_phys=pred[..., :m], element_counts=ec,
            atomic_weights=aw, molar_weights=mw)
        state_mse = trajectory_mse(pred_norm, tgt_full)     # component for history only
        return total, {"state_mse": state_mse.detach(),
                       "pinn_loss": (total - state_mse).detach()}   # exact weighted PINN part

    # --- run-directory plumbing (organization only; math unchanged) ---------------
    run = RunManager(args.run_dir, "hydrogen", resume=args.resume, overwrite=args.overwrite)
    run.start()

    resume_state = run.load_resume() if args.resume else None
    resume_stage = resume_state.get("stage") if resume_state is not None else None
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        if resume_state.get("rng_state") is not None:
            torch.set_rng_state(resume_state["rng_state"])

    config = {
        "model": "ChemKAN", "chemical_system": "hydrogen",
        "experiment_name": args.experiment_name, "seed": args.seed,
        "sensitivity_backend": solver.sensitivity, "device": str(device),
        "architecture": {"hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
                         "n_mu": args.n_mu, "use_base_act": args.use_base_act},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam", "learning_rate": args.lr,
        "epochs": {"stage1": args.stage1_epochs, "stage2": args.stage2_epochs},
        "solver": {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol},
        "loss": "normalized trajectory MSE (Eq. 18) + alpha*PINN",
        "pinn": {"stage1": args.pinn_stage1, "stage2": args.pinn_stage2,
                 "alpha_pinn": args.alpha_pinn},
        "normalization": {"input_scaling": args.input_scaling, "stats": "train-only min-max"},
        "dataset": "hydrogen.npz (train split, 50 points, 35 conditions)", "noise": None,
        "stage1_temperature": stage1_temp_meta,
        "thermo_init": thermo_init_meta,
        "stage1_from": stage1_from_meta,
    }
    if resume_state is not None:
        # A resumed run may not silently change its science (incl. the Stage-1 temperature
        # provider). Epoch totals may GROW but never fall below the completed epoch; the
        # original run's config.json is preserved rather than rewritten.
        check_resume_config(resume_state.get("config", {}), config)
        completed = int(resume_state.get("epoch", 0))
        if resume_stage == "stage1" and args.stage1_epochs < completed:
            raise SystemExit(f"--resume: requested stage1-epochs {args.stage1_epochs} < "
                             f"already-completed {completed}; epoch total may grow, never shrink.")
        if resume_stage == "stage2" and args.stage2_epochs < completed:
            raise SystemExit(f"--resume: requested stage2-epochs {args.stage2_epochs} < "
                             f"already-completed {completed}; epoch total may grow, never shrink.")
        config = resume_state.get("config", config)     # preserve original provenance
        logging.info("resuming hydrogen at stage=%s epoch=%s (original config preserved)",
                     resume_stage, completed)
    else:
        run.write_config(config)

    def resume_payload(stage, epoch, opt_state, **extra):
        return {"stage": stage, "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": opt_state, "config": config,
                "rng_state": torch.get_rng_state(), **extra}

    hist1 = hist2 = None
    try:
        # --- Stage 1: species only, observed temperature --------------------------
        if stage1_from_meta is not None:
            l1 = float(stage1_from_meta.get("stage1_final_loss") or float("nan"))
            logging.info("Stage 1 skipped (loaded kinetic core); recorded loss %.6e", l1)
        elif resume_stage == "stage2":
            l1 = float(resume_state.get("stage1_final_loss", float("nan")))
            logging.info("resume: Stage 1 already complete (final loss %.6e); skipping", l1)
        else:
            s1_start = int(resume_state["epoch"]) if resume_stage == "stage1" else 0
            s1_opt = resume_state.get("optimizer_state") if resume_stage == "stage1" else None
            hist1 = run.history(
                "history_stage1.csv",
                ["epoch", "total_loss", "species_mse", "pinn_loss", "elapsed_seconds"]
            + (["epoch_wall_time_s", "nfe"] if args.count_nfe else []),
                resume_from=s1_start)
            s1_cb, _s1_nfe = hist1.on_epoch, None
            if args.count_nfe:
                from diagnostics._instrumentation import EpochInstrumentation, NFECounter
                from diagnostics._stage2_probe import chain as _chain
                _s1_nfe = NFECounter(kin_dyn)
                s1_cb = _chain(EpochInstrumentation(_s1_nfe).on_epoch, hist1.on_epoch)
            l1 = train_kinetic_stage(
                kin_dyn, Y0, t, stage1_loss, epochs=args.stage1_epochs, lr=args.lr, solver=solver,
                start_epoch=s1_start, optimizer_state=s1_opt, on_epoch=s1_cb,
                checkpoint_every=args.checkpoint_every,
                save_resume=lambda e, o: run.save_resume(resume_payload("stage1", e, o)))
            hist1.close()
            if _s1_nfe is not None:
                _s1_nfe.detach()
            logging.info("stage 1 final loss: %.6e", l1)
            # Permanent Stage-1 artifact: lets later Stage-2 experiments start from the
            # EXACT same kinetic model (see --stage1-from). Never overwrites checkpoint_final.
            if run.enabled:
                torch.save({"kinetic_state": model.kinetic.state_dict(),
                            "architecture": {"hidden_dim": args.hidden_dim,
                                             "num_basis": args.num_basis, "n_mu": args.n_mu,
                                             "use_base_act": args.use_base_act},
                            "data": {"species": data["species"], "species_dim": m},
                            "stage1_final_loss": l1, "stage1_epochs": args.stage1_epochs,
                            "seed": args.seed, "run_id": run.run_id,
                            "stage1_temperature": stage1_temp_meta,
                            "solver": {"method": solver.method, "rtol": solver.rtol,
                                       "atol": solver.atol, "sensitivity": solver.sensitivity},
                            "input_scaling": input_scaling_meta(args.input_scaling, full_norm)},
                           run.run_dir / "checkpoint_stage1.pt")
                logging.info("wrote %s", run.run_dir / "checkpoint_stage1.pt")
            # Mark Stage 1 done so an interrupted Stage 2 resumes into Stage 2, not Stage 1.
            run.save_resume(resume_payload("stage2", 0, None, stage1_final_loss=l1))

        # --- Stage 2: full [Y, T], all params, MSE (+ alpha * PINN) ----------------
        s2_start = int(resume_state["epoch"]) if resume_stage == "stage2" else 0
        s2_opt = resume_state.get("optimizer_state") if resume_stage == "stage2" else None
        hist2 = run.history(
            "history_stage2.csv",
            ["epoch", "total_loss", "state_mse", "pinn_loss", "elapsed_seconds"]
            + (["epoch_wall_time_s", "nfe"] if args.count_nfe else []),
            resume_from=s2_start)
        probe = None
        s2_cb = hist2.on_epoch
        if args.stage2_probe and run.enabled:
            from diagnostics._stage2_probe import Stage2Probe, chain
            from evaluate_hydrogen import integrate_hydrogen as _integrate_h2
            _npz = np.load(DATA_DIR / "hydrogen.npz", allow_pickle=True)
            _ics, _states = _npz["ics"], _npz["states"]
            _i = int(np.argmin(np.abs(_ics[:, 0] - 1050.0) + np.abs(_ics[:, 1] - 0.9)))
            probe = Stage2Probe(model, integrate_fn=_integrate_h2, input_norm=input_normalizer,
                                solver=solver, full_norm=full_norm, t=_npz["t"],
                                ref=_states[_i], species=[str(x) for x in data["species"]],
                                path=run.run_dir / "stage2_probe.csv",
                                epochs=[int(e) for e in args.stage2_probe_epochs.split(",")])
            # Epoch-0 stability check: a large arbitrary thermo.linear can make the ODE
            # blow up. That is a VALID diagnostic outcome -- record it and exit cleanly.
            # Nothing is clipped, reduced or otherwise stabilized.
            try:
                _row0 = probe.probe(0, None)
                _bad = not (np.isfinite(float(_row0["peak_T_K"]))
                            and np.isfinite(float(_row0["temperature_MSE"])))
                _why = "non-finite peak T / temperature MSE at epoch 0" if _bad else ""
            except Exception as exc:                      # solver failure counts as unstable
                _bad, _why = True, f"{type(exc).__name__}: {exc}"
            if _bad:
                probe.close()
                hist2.close()
                (run.run_dir / "epoch0_status.json").write_text(json.dumps(
                    {"status": "unstable_at_initialization", "reason": _why,
                     "thermo_init": thermo_init_meta, "stage2_epochs_run": 0}, indent=2, default=str))
                logging.error("EPOCH-0 UNSTABLE (%s); initialization provenance saved to %s. "
                              "Exiting before optimization -- no clipping applied.",
                              _why, run.run_dir / "epoch0_status.json")
                run.finish(ok=False)
                raise SystemExit(3)
            s2_cb = chain(hist2.on_epoch, probe.on_epoch)
            logging.info("stage-2 probe -> %s", run.run_dir / "stage2_probe.csv")

        # Behaviour-neutral instrumentation. EpochInstrumentation must run BEFORE the
        # history writer because it injects nfe / epoch_wall_time_s into `components`.
        _s2_nfe, _snap = None, None
        _extra_cbs = []
        if args.count_nfe:
            from diagnostics._instrumentation import EpochInstrumentation, NFECounter
            _s2_nfe = NFECounter(chem_dyn)
            _extra_cbs.append(EpochInstrumentation(_s2_nfe).on_epoch)
        if args.stage2_snapshot_epochs.strip() and run.enabled:
            from diagnostics._instrumentation import Stage2Snapshot
            _snap = Stage2Snapshot(
                model, run.run_dir,
                [int(e) for e in args.stage2_snapshot_epochs.split(",") if e.strip()],
                extra={"architecture": {"hidden_dim": args.hidden_dim,
                                        "num_basis": args.num_basis, "n_mu": args.n_mu,
                                        "use_base_act": args.use_base_act},
                       "data": {"species": data["species"], "species_dim": m},
                       "seed": args.seed, "run_id": run.run_id,
                       "solver": {"method": solver.method, "rtol": solver.rtol,
                                  "atol": solver.atol, "sensitivity": solver.sensitivity},
                       "input_scaling": input_scaling_meta(args.input_scaling, full_norm),
                       "state_representation": "physical",   # required by load_input_scaling
                       "thermo_init": thermo_init_meta})
            logging.info("stage-2 snapshots at epochs %s", sorted(_snap.epochs))
        if _extra_cbs or _snap is not None:
            from diagnostics._stage2_probe import chain as _chain2
            _tail = [_snap.on_epoch] if _snap is not None else []
            s2_cb = _chain2(*_extra_cbs, s2_cb, *_tail)

        l2 = train_full_chemkan(
            chem_dyn, u0, t, stage2_loss, epochs=args.stage2_epochs, lr=args.lr, solver=solver,
            start_epoch=s2_start, optimizer_state=s2_opt, on_epoch=s2_cb,
            checkpoint_every=args.checkpoint_every,
            save_resume=lambda e, o: run.save_resume(
                resume_payload("stage2", e, o, stage1_final_loss=l1)))
        hist2.close()
        if probe is not None:
            probe.close()
        if _s2_nfe is not None:
            _s2_nfe.detach()
        if _snap is not None and _snap.written:
            for _p in _snap.written:
                logging.info("wrote %s", _p)
        logging.info("stage 2 final loss: %.6e", l2)
    except KeyboardInterrupt:
        if hist1 is not None:
            hist1.close()
        if hist2 is not None:
            hist2.close()
        logging.warning("interrupted; resume checkpoint preserved for --resume")
        run.finish(ok=False)                            # closes log handler; keeps checkpoint_resume.pt
        raise SystemExit(130)

    checkpoint = {
        "model_state": model.state_dict(),
        "architecture": {
            "hidden_dim": args.hidden_dim, "num_basis": args.num_basis,
            "n_mu": args.n_mu, "use_base_act": args.use_base_act,
        },
        "data": {"species": data["species"], "species_dim": m},
        "training": {"learning_rate": args.lr,
                     "stage1_epochs": args.stage1_epochs,
                     "stage2_epochs": args.stage2_epochs,
                     "seed": args.seed,
                     "alpha_pinn": args.alpha_pinn,
                     "use_pinn_stage1": args.pinn_stage1,
                     "use_pinn_stage2": args.pinn_stage2},
        "solver": {"method": solver.method, "rtol": solver.rtol,
                   "atol": solver.atol, "sensitivity": solver.sensitivity},
        "stage1_temperature": stage1_temp_meta,
        "thermo_init": thermo_init_meta,
        "state_representation": "physical",
        "input_scaling": input_scaling_meta(args.input_scaling, full_norm),
    }
    if run.enabled:
        checkpoint["run_id"] = run.run_id
        run.save_final(checkpoint)                          # writes final, deletes resume
        run.finish(ok=True)
    else:
        torch.save(checkpoint, args.out)                    # legacy flat-file behavior
        logging.info("saved full ChemKAN (+ metadata) -> %s", args.out)


if __name__ == "__main__":
    main()
