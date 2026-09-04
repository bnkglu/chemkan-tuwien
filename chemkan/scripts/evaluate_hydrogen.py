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
import json
import logging
import time
from pathlib import Path

import torch
from _data import load_hydrogen, load_input_scaling, resolve_device
from _predictions import checkpoint_sha256, save_predictions
from _run import METRICS_JSON, PREDICTIONS_DIR, utc_now

_METRIC_CONVENTION = ("normalized trajectory MSE (Eq. 18): mean over modeled state variables "
                      "(full [Y, T] state), summed over observation times, then mean over "
                      "trajectories, using train-only min-max normalization "
                      "(chemkan.losses.trajectory_mse).")

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


def write_metrics(run_dir: Path, res: dict, split: str, ckpt_path: Path,
                  evaluation_wall_time_s: float) -> Path:
    """Merge measured metrics into RUN_DIR/metrics.json (no huge arrays).

    ``evaluation_wall_time_s`` is the whole-command wall time (data + checkpoint loading +
    model reconstruction + integration + metric), NOT pure model inference. The paper
    speed-up benchmark (Table I) will use a dedicated warm-up/repeated-integration timer.
    """
    path = Path(run_dir) / METRICS_JSON
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics.setdefault("run_id", res["ckpt"].get("run_id"))
    metrics[f"{split}_mse"] = res["mse"]
    metrics["n_params"] = res["n_params"]
    metrics["evaluation_wall_time_s"] = round(evaluation_wall_time_s, 4)
    metrics["evaluated_conditions"] = int(res["truth"].shape[1])
    metrics["metric_convention"] = _METRIC_CONVENTION
    metrics["solver"] = {"method": res["solver"].method, "rtol": res["solver"].rtol,
                         "atol": res["solver"].atol, "sensitivity": res["solver"].sensitivity}
    if "stage1_temperature" in res["ckpt"]:
        metrics["stage1_temperature"] = res["ckpt"]["stage1_temperature"]
    metrics[f"evaluated_at_{split}"] = utc_now()
    path.write_text(json.dumps(metrics, indent=2, default=str))
    logging.info("wrote %s", path)
    return path


def save_prediction_artifact(run_dir: Path, res: dict, split: str, ckpt_path: Path,
                             force: bool = False) -> Path:
    """Persist model predictions for this split with full provenance (see _predictions)."""
    fn = res["full_normalizer"]
    out = Path(run_dir) / PREDICTIONS_DIR / f"{split}_predictions.npz"
    return save_predictions(
        out, force=force,
        run_id=res["ckpt"].get("run_id"),
        checkpoint_sha256=checkpoint_sha256(ckpt_path),
        architecture=res["ckpt"]["architecture"],
        predictions=res["pred"].numpy(),
        reference=res["truth"].numpy(),               # canonical ground truth (not recomputed)
        t=res["t"].numpy(),
        initial_conditions=res["truth"][0].numpy(),   # u0 = [Y0, T0] (B, m+1)
        species=res["species"],
        u_min=fn.u_min.detach().cpu().numpy(),
        u_max=fn.u_max.detach().cpu().numpy(),
        metric_convention=_METRIC_CONVENTION,
        eval_config={"split": split, "solver": {"method": res["solver"].method,
                     "rtol": res["solver"].rtol, "atol": res["solver"].atol}},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hydrogen_chemkan.pt",
                   help="checkpoint path (ignored if --run-dir is given).")
    p.add_argument("--run-dir", default=None,
                   help="evaluate RUN_DIR/checkpoint_final.pt and write metrics/predictions there.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--metrics", action="store_true", help="write RUN_DIR/metrics.json")
    p.add_argument("--save-predictions", action="store_true",
                   help="write RUN_DIR/predictions/<split>_predictions.npz")
    p.add_argument("--force", action="store_true", help="overwrite an existing prediction artifact")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ckpt_path = Path(args.run_dir) / "checkpoint_final.pt" if args.run_dir else Path(args.ckpt)
    started = time.perf_counter()
    res = evaluate_hydrogen(str(ckpt_path), args.split, args.device)
    elapsed = time.perf_counter() - started
    logging.info("hydrogen [%s] normalized trajectory MSE: %.6e", args.split, res["mse"])

    if args.metrics:
        if not args.run_dir:
            raise SystemExit("--metrics requires --run-dir")
        write_metrics(Path(args.run_dir), res, args.split, ckpt_path, elapsed)
    if args.save_predictions:
        if not args.run_dir:
            raise SystemExit("--save-predictions requires --run-dir")
        try:
            path = save_prediction_artifact(Path(args.run_dir), res, args.split, ckpt_path, args.force)
        except FileExistsError as e:
            raise SystemExit(str(e))
        logging.info("wrote %s", path)


if __name__ == "__main__":
    main()
