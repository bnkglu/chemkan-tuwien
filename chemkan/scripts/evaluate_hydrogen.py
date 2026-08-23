r"""Evaluate a trained hydrogen ChemKAN: reconstruct from checkpoint, report full MSE.

The model is rebuilt from the dataset's species dimension + the checkpoint's stored
architecture -- no dependency on any global config module.

Besides the CLI (``main``), this module exposes REUSABLE functions so notebooks and other
callers can obtain predictions without re-implementing the evaluation mathematics:

* ``build_chemkan(ckpt, species_dim, device)`` -- rebuild the trained full model.
* ``solver_from_ckpt(ckpt)`` -- reconstruct the training ``SolverConfig``.
* ``integrate_hydrogen(model, input_normalizer, solver, u0, t, device)`` -- integrate an
  arbitrary full initial state ``[Y0, T0]`` with the trained ChemKAN.
* ``evaluate_hydrogen(ckpt_path, split, device)`` -- full split evaluation returning
  ``{t, truth, pred, mse, species, n_params, ckpt, ...}``.
"""

from __future__ import annotations

import argparse
import logging

import torch
from _data import load_hydrogen, load_input_scaling, resolve_device

from chemkan.dynamics import ChemKANDynamics
from chemkan.losses import trajectory_mse
from chemkan.model import ChemKAN
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate


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


def build_chemkan(ckpt, species_dim, device) -> ChemKAN:
    """Rebuild the trained full ChemKAN from ``data.species_dim`` + checkpoint architecture."""
    arch = ckpt["architecture"]
    model = ChemKAN(species_dim=species_dim, hidden_dim=arch["hidden_dim"],
                    num_basis=arch["num_basis"], n_mu=arch["n_mu"],
                    use_base_act=arch["use_base_act"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def integrate_hydrogen(model, input_normalizer, solver, u0, t, device="cpu"):
    """Integrate the full ``[Y,T]`` state from ANY initial state with the trained model.

    ``u0``:(B,m+1)  ``t``:(T,)  ->  (T,B,m+1) physical.
    """
    u0 = torch.as_tensor(u0, dtype=torch.get_default_dtype())
    t = torch.as_tensor(t, dtype=torch.get_default_dtype())
    dyn = ChemKANDynamics(model, input_normalizer=input_normalizer).to(device)
    with torch.no_grad():
        return integrate(dyn, u0.to(device), t.to(device), solver)


def evaluate_hydrogen(ckpt_path="hydrogen_chemkan.pt", split="test", device="cpu") -> dict:
    """Full-split evaluation. Returns predictions + metric + reusable model pieces."""
    dev = resolve_device(device)
    data = load_hydrogen(split=split)                   # normalization stats stay train-only
    m = data["species_TBm"].shape[-1]
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    _validate_species(ckpt, m, data["species"])

    model = build_chemkan(ckpt, m, dev)
    input_normalizer = load_input_scaling(ckpt, dev)
    solver = solver_from_ckpt(ckpt)
    norm = MinMaxNormalizer(data["u_min"], data["u_max"]).to(dev)     # full-state, train stats

    pred = integrate_hydrogen(model, input_normalizer, solver,
                              data["full_TBm1"][0], data["t"], dev)          # (T,B,m+1)
    mse = trajectory_mse(norm.normalize(pred),
                         norm.normalize(data["full_TBm1"].to(dev))).item()
    return {
        "t": data["t"], "truth": data["full_TBm1"], "pred": pred.cpu(),
        "species": data["species"], "m": m, "split": split,
        "mse": mse, "n_params": sum(p.numel() for p in model.parameters()),
        "ckpt": ckpt, "model": model, "input_normalizer": input_normalizer,
        "solver": solver, "full_normalizer": norm,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hydrogen_chemkan.pt")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = evaluate_hydrogen(args.ckpt, args.split, args.device)
    logging.info("hydrogen [%s] normalized trajectory MSE: %.6e", args.split, res["mse"])


if __name__ == "__main__":
    main()
