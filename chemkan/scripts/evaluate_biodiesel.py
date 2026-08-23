r"""Evaluate a trained biodiesel kinetic core: reconstruct from checkpoint, report MSE.

The model is rebuilt from the dataset's species dimension + the checkpoint's stored
architecture -- there is no dependency on any global config module.

Besides the CLI (``main``), this module exposes REUSABLE functions so notebooks and other
callers can obtain predictions without re-implementing the evaluation mathematics:

* ``build_kinetic_core(ckpt, species_dim, device)`` -- rebuild the trained core.
* ``solver_from_ckpt(ckpt)`` -- reconstruct the training ``SolverConfig``.
* ``integrate_biodiesel(core, input_normalizer, solver, Y0, T_const, t, device)`` --
  integrate an ARBITRARY initial condition (e.g. the paper's Fig. 3 case) with the same
  physical/scaling semantics the training used.
* ``evaluate_biodiesel(ckpt_path, split, device)`` -- full split evaluation returning
  ``{t, truth, pred, mse, species, T_const, n_params, ckpt, ...}``.
"""

from __future__ import annotations

import argparse
import logging

import torch
from _data import load_biodiesel, load_input_scaling, resolve_device

from chemkan.dynamics import KineticDynamics
from chemkan.losses import trajectory_mse
from chemkan.model import KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate
from chemkan.temperature import ConstantTemperature


def _validate_species(ckpt, species_dim, species):
    cdata = ckpt.get("data", {})
    if cdata.get("species_dim") not in (None, species_dim):
        raise ValueError(
            f"checkpoint species_dim {cdata['species_dim']} != dataset {species_dim}")
    if cdata.get("species") is not None and list(cdata["species"]) != list(species):
        raise ValueError("checkpoint species names/order differ from the dataset")


def solver_from_ckpt(ckpt) -> SolverConfig:
    if "solver" not in ckpt:
        raise ValueError("checkpoint has no 'solver' metadata; cannot reconstruct the "
                         "solver settings the model was trained with")
    s = ckpt["solver"]
    return SolverConfig(method=s["method"], rtol=s["rtol"], atol=s["atol"],
                        sensitivity=s["sensitivity"])


_solver_from_ckpt = solver_from_ckpt        # backwards-compatible alias


def build_kinetic_core(ckpt, species_dim, device) -> KineticCore:
    """Rebuild the trained kinetic core from ``data.species_dim`` + checkpoint architecture."""
    arch = ckpt["architecture"]
    core = KineticCore(species_dim=species_dim, hidden_dim=arch["hidden_dim"],
                       num_basis=arch["num_basis"], n_mu=arch["n_mu"],
                       use_base_act=arch["use_base_act"]).to(device)
    core.load_state_dict(ckpt["model_state"])
    core.eval()
    return core


def integrate_biodiesel(core, input_normalizer, solver, Y0, T_const, t, device="cpu"):
    """Integrate species from ANY initial condition with the trained core.

    ``Y0``:(B,m)  ``T_const``:(B,) or (B,1)  ``t``:(T,)  ->  (T,B,m) physical.
    Uses the same ``KineticDynamics`` (physical state, pre-KAN input scaling) as training.
    """
    Y0 = torch.as_tensor(Y0, dtype=torch.get_default_dtype())
    T_const = torch.as_tensor(T_const, dtype=torch.get_default_dtype())
    t = torch.as_tensor(t, dtype=torch.get_default_dtype())
    dyn = KineticDynamics(core, ConstantTemperature(T_const),
                          input_normalizer=input_normalizer).to(device)
    with torch.no_grad():
        return integrate(dyn, Y0.to(device), t.to(device), solver)


def evaluate_biodiesel(ckpt_path="biodiesel_kinetic.pt", split="test", device="cpu") -> dict:
    """Full-split evaluation. Returns predictions + metric + reusable model pieces."""
    dev = resolve_device(device)
    data = load_biodiesel(split=split)                  # normalization stats stay train-only
    species_dim = data["species_TBm"].shape[-1]
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    _validate_species(ckpt, species_dim, data["species"])

    core = build_kinetic_core(ckpt, species_dim, dev)
    input_normalizer = load_input_scaling(ckpt, dev)    # reconstruct pre-KAN scaling (never refit)
    solver = solver_from_ckpt(ckpt)
    loss_norm = MinMaxNormalizer(data["u_min"], data["u_max"]).to(dev)  # species-only, train stats

    pred = integrate_biodiesel(core, input_normalizer, solver,
                               data["Y0"], data["T_const"], data["t"], dev)   # (T,B,m)
    mse = trajectory_mse(loss_norm.normalize(pred),
                         loss_norm.normalize(data["species_TBm"].to(dev))).item()
    return {
        "t": data["t"], "truth": data["species_TBm"], "pred": pred.cpu(),
        "species": data["species"], "T_const": data["T_const"], "split": split,
        "mse": mse, "n_params": sum(p.numel() for p in core.parameters()),
        "ckpt": ckpt, "core": core, "input_normalizer": input_normalizer,
        "solver": solver, "loss_normalizer": loss_norm,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="biodiesel_kinetic.pt")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = evaluate_biodiesel(args.ckpt, args.split, args.device)
    logging.info("biodiesel [%s] normalized trajectory MSE: %.6e", args.split, res["mse"])


if __name__ == "__main__":
    main()
