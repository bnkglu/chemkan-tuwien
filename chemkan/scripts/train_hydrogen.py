r"""Train hydrogen ChemKAN in two explicit stages -- ChemKAN Eq. 13-18, Sec. II C 5.

Benchmark configuration of the GENERIC ChemKAN library. ``species_dim`` is inferred
from the data; architecture is chosen here (paper count-matching defaults). PINN
usage per stage is an explicit CLI choice, and both stage loss functions are built
in THIS script -- the training library knows nothing about hydrogen or PINN.

Stage 1: integrate species only; temperature observed/interpolated; kinetic core.
Stage 2: integrate full [Y, T] with the COMPLETE model; update ALL parameters.

``num_basis = 5`` (default) is INFERRED for the count-matching reproduction from the
reported architecture + 344-parameter count, not an explicitly stated grid size.
"""

from __future__ import annotations

import argparse
import logging

import torch
from _chemistry import (
    ATOMIC_WEIGHTS,
    ELEMENT_COUNTS,
    MOLAR_WEIGHTS,
    assert_species_order,
)
from _data import input_scaling_meta, load_hydrogen, resolve_device

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
    p.add_argument("--num-basis", type=int, default=5)      # inferred (see docstring)
    p.add_argument("--n-mu", type=int, default=3)
    p.add_argument("--use-base-act", action="store_true")
    # stage epoch counts are implementation choices (paper does not specify them here)
    p.add_argument("--stage1-epochs", type=int, default=10000)
    p.add_argument("--stage2-epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for model init / training reproducibility")
    # explicit PINN choices: main interpretation is Stage 1 OFF, Stage 2 ON.
    p.add_argument("--pinn-stage1", action="store_true", default=False)
    p.add_argument("--no-pinn-stage2", dest="pinn_stage2", action="store_false",
                   default=True)
    p.add_argument("--alpha-pinn", type=float, default=1e-4)
    p.add_argument("--input-scaling", default="minmax", choices=["minmax", "none"],
                   help="pre-KAN scaling of the physical state ('none' = raw ablation)")
    p.add_argument("--solver-method", default="tsit5")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--out", default="hydrogen_chemkan.pt")
    return p


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
    # Full-state [Y1..Ym, T] train-only normalizer (archive already stores m+1 stats).
    full_norm = MinMaxNormalizer(data["u_min"], data["u_max"]).to(device)
    species_norm = full_norm.subset(slice(0, m))           # species-only for Stage-1 loss
    input_normalizer = full_norm if args.input_scaling == "minmax" else None

    t = data["t"].to(device)
    ec = ELEMENT_COUNTS.to(device)
    aw = ATOMIC_WEIGHTS.to(device)
    mw = MOLAR_WEIGHTS.to(device)

    # --- Stage 1: species only, observed temperature ------------------------------
    temp = ObservedTemperature(data["t"], data["T_obs_TB1"]).to(device)
    kin_dyn = KineticDynamics(model.kinetic, temp,
                              input_normalizer=input_normalizer).to(device)
    Y0 = data["species_TBm"][0].to(device)                  # (B, m)
    tgt_species = species_norm.normalize(data["species_TBm"].to(device))   # (T, B, m)

    def stage1_loss(pred):                                  # pred: (T, B, m)
        mse = trajectory_mse(species_norm.normalize(pred), tgt_species)
        if args.pinn_stage1:                                # physical species for PINN
            mse = mse + args.alpha_pinn * element_conservation_loss(pred, ec, aw, mw)
        return mse

    l1 = train_kinetic_stage(kin_dyn, Y0, t, stage1_loss,
                             epochs=args.stage1_epochs, lr=args.lr, solver=solver)
    logging.info("stage 1 final loss: %.6e", l1)

    # --- Stage 2: full [Y, T], all params, MSE (+ alpha * PINN) -------------------
    chem_dyn = ChemKANDynamics(model, input_normalizer=input_normalizer).to(device)
    u0 = data["full_TBm1"][0].to(device)                    # (B, m+1)
    tgt_full = full_norm.normalize(data["full_TBm1"].to(device))           # (T, B, m+1)

    def stage2_loss(pred):                                  # pred: (T, B, m+1)
        return chemkan_loss(
            full_norm.normalize(pred), tgt_full,
            use_pinn=args.pinn_stage2, alpha_pinn=args.alpha_pinn,
            Y_phys=pred[..., :m], element_counts=ec,
            atomic_weights=aw, molar_weights=mw)

    l2 = train_full_chemkan(chem_dyn, u0, t, stage2_loss,
                            epochs=args.stage2_epochs, lr=args.lr, solver=solver)
    logging.info("stage 2 final loss: %.6e", l2)

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
        "state_representation": "physical",
        "input_scaling": input_scaling_meta(args.input_scaling, full_norm),
    }
    torch.save(checkpoint, args.out)
    logging.info("saved full ChemKAN (+ metadata) -> %s", args.out)


if __name__ == "__main__":
    main()
