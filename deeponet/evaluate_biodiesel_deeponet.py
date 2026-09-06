r"""Evaluate a trained biodiesel DeepONet -- the Fig. 5A metric triple, from the checkpoint.

Mirrors ``chemkan/scripts/evaluate_biodiesel.py`` so ChemKAN and DeepONet numbers are
produced by identical conventions: the same train-only normalizer, the same Eq. 18
reduction, one set of predicted trajectories scored against two targets (the noisy
observations and the clean underlying trajectories, paper Eq. 22).

The scaling is reconstructed from the checkpoint, never refitted.

    python evaluate_biodiesel_deeponet.py --run-dir <run> --split test --noise-percent 15 \
        --metrics --save-predictions
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

from _data import load_biodiesel, resolve_device                    # noqa: E402
from _predictions import checkpoint_sha256, save_predictions        # noqa: E402
from _run import METRICS_JSON, PREDICTIONS_DIR, utc_now             # noqa: E402

from biodiesel_deeponet import (PAPER_PARAMS, BiodieselDeepONet,   # noqa: E402
                                prepare_inputs)
from chemkan.losses import trajectory_mse                           # noqa: E402
from chemkan.normalization import MinMaxNormalizer                  # noqa: E402

_METRIC_CONVENTION = ("normalized trajectory MSE (Eq. 18): mean over modeled state variables "
                      "(species-only), summed over observation times, then mean over "
                      "trajectories, using train-only min-max normalization "
                      "(chemkan.losses.trajectory_mse).")


def build_model(ckpt, device) -> BiodieselDeepONet:
    model = BiodieselDeepONet(width=ckpt["architecture"]["width"],
                              out_dim=ckpt["architecture"]["out_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate(ckpt_path, split="test", device="cpu", noise_percent=None) -> dict:
    dev = resolve_device(device)
    data = load_biodiesel(split=split, noise_percent=noise_percent)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if list(ckpt["data"]["species"]) != list(data["species"]):
        raise ValueError("checkpoint species names/order differ from the dataset")

    model = build_model(ckpt, dev)
    n = ckpt["normalization"]
    full_norm = MinMaxNormalizer(n["u_min"], n["u_max"]).to(dev)     # never refit
    loss_norm = full_norm.subset(slice(0, len(data["species"]))).to(dev)
    t_end = float(n["t_end_s"])

    u0, tau = prepare_inputs(data, full_norm, t_end)
    with torch.no_grad():
        pred_norm = model(u0, tau)                                   # normalized species
    pred = loss_norm.denormalize(pred_norm)                          # physical, for plots

    mse = trajectory_mse(pred_norm, loss_norm.normalize(data["targets_TBm"].to(dev))).item()
    mse_clean = trajectory_mse(pred_norm,
                               loss_norm.normalize(data["species_TBm"].to(dev))).item()
    return {
        "t": data["t"], "truth": data["species_TBm"], "observations": data["targets_TBm"],
        "pred": pred.cpu(), "species": data["species"], "split": split,
        "noise_percent": data["noise_percent"], "mse": mse, "mse_clean": mse_clean,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ckpt": ckpt, "model": model, "loss_normalizer": loss_norm, "t_end": t_end,
    }


def write_metrics(run_dir: Path, res: dict, split: str, wall_time_s: float) -> Path:
    path = Path(run_dir) / METRICS_JSON
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics.setdefault("run_id", res["ckpt"].get("run_id"))
    if res["noise_percent"] is None:
        metrics[f"{split}_mse"] = res["mse"]
    else:
        metrics["noise_percent"] = res["noise_percent"]
        metrics[f"{split}_mse_noisy"] = res["mse"]
        metrics[f"{split}_mse_clean"] = res["mse_clean"]
    metrics["n_params"] = res["n_params"]
    metrics["paper_parameter_count"] = PAPER_PARAMS      # the count the paper reports
    metrics["evaluation_wall_time_s"] = round(wall_time_s, 4)
    # Split-scoped: train and test are evaluated by separate invocations that merge
    # into one file, so a shared key would report whichever ran last.
    metrics[f"{split}_conditions"] = int(res["truth"].shape[1])
    metrics["metric_convention"] = _METRIC_CONVENTION
    metrics[f"evaluated_at_{split}"] = utc_now()
    path.write_text(json.dumps(metrics, indent=2, default=str))
    logging.info("wrote %s", path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--noise-percent", type=int, default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ckpt_path = Path(args.run_dir) / "checkpoint_final.pt"
    started = time.perf_counter()
    res = evaluate(str(ckpt_path), args.split, args.device, args.noise_percent)
    elapsed = time.perf_counter() - started
    if args.noise_percent is None:
        logging.info("deeponet [%s] normalized trajectory MSE: %.6e", args.split, res["mse"])
    else:
        logging.info("deeponet [%s @ %d%% noise] observations MSE: %.6e | "
                     "noise-free MSE (Eq. 22): %.6e",
                     args.split, args.noise_percent, res["mse"], res["mse_clean"])

    if args.metrics:
        write_metrics(Path(args.run_dir), res, args.split, elapsed)
    if args.save_predictions:
        ln = res["loss_normalizer"]
        out = Path(args.run_dir) / PREDICTIONS_DIR / f"{args.split}_predictions.npz"
        try:
            path = save_predictions(
                out, force=args.force,
                run_id=res["ckpt"].get("run_id"),
                checkpoint_sha256=checkpoint_sha256(ckpt_path),
                architecture=res["ckpt"]["architecture"],
                predictions=res["pred"].numpy(),
                reference=res["truth"].numpy(),          # clean canonical ground truth
                t=res["t"].numpy(),
                initial_conditions=res["truth"][0].numpy(),
                species=res["species"],
                u_min=ln.u_min.detach().cpu().numpy(),
                u_max=ln.u_max.detach().cpu().numpy(),
                metric_convention=_METRIC_CONVENTION,
                eval_config={"split": args.split, "noise_percent": res["noise_percent"],
                             "reference": "clean underlying trajectories (Eq. 22 target)",
                             "model": "DeepONet", "t_end_s": res["t_end"]},
            )
        except FileExistsError as e:
            raise SystemExit(str(e))
        logging.info("wrote %s", path)


if __name__ == "__main__":
    main()
