r"""Smoke check for the author-matched hydrogen Stage 1: batched vs per-trajectory solves and
float32 vs float64, from ONE fixed set of initial parameters (Glorot seed 0, drawn in float64
exactly as ``run_stage1`` draws them, then cast), t_ref 6e-4 s, direct autograd.

1a  at the initial parameters: predictions, loss and gradient, batched vs per-trajectory
    (each dtype) and float32 vs float64 (batched): relative L2 and cosine.
1b  50 Adam steps each for batched float32, per_trajectory float32 and batched float64:
    loss per step and seconds per epoch.

    python smoke_check_solve_mode_dtype.py --out <dir>/solve_mode_dtype.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_author_matched_hydrogen as amh  # noqa: E402

from chemkan.solver import integrate  # noqa: E402

T_REF = 6e-4


def initial_state() -> dict:
    torch.set_default_dtype(torch.float64)
    model = amh.build_model()
    gen = torch.Generator().manual_seed(0)
    amh.init_kinetic_(model, gen)
    amh.init_thermo_(model, gen)
    return model.state_dict()


def setup(dtype, state):
    torch.set_default_dtype(dtype)
    problem = amh.load_problem(dtype)
    model = amh.build_model()
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()})
    return problem, model


def predictions(model, problem, mode) -> torch.Tensor:
    units = amh.stage1_units(model, problem, T_REF, mode)
    with torch.no_grad():
        return torch.cat([integrate(f, y0, problem["t"], amh._solver("direct_autograd"))
                          for f, y0, _ in units], dim=1).double()


def loss_and_grad(model, problem, mode):
    params = list(model.kinetic.parameters())
    for p in params:
        p.grad = None
    vals = amh.epoch_gradients(amh.stage1_units(model, problem, T_REF, mode), problem["t"],
                               params, amh._solver("direct_autograd"))
    return vals, torch.cat([p.grad.reshape(-1) for p in params]).double()


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.reshape(-1), b.reshape(-1)
    return {"rel_l2": float((a - b).norm() / b.norm()),
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "max_abs": float((a - b).abs().max())}


def adam_steps(model, problem, mode, steps) -> dict:
    params = list(model.kinetic.parameters())
    opt = amh._adam(params)
    losses, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        vals = amh.epoch_gradients(amh.stage1_units(model, problem, T_REF, mode), problem["t"],
                                   params, amh._solver("direct_autograd"))
        opt.step()
        times.append(time.perf_counter() - t0)
        losses.append(vals["loss"])
    return {"loss_per_step": losses, "seconds_per_epoch_mean": sum(times) / steps,
            "seconds_per_epoch_last10": sum(times[-10:]) / 10}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    state = initial_state()
    result = {"settings": {"stage": 1, "t_ref_s": T_REF, "glorot_seed": 0,
                           "sensitivity": "direct_autograd", "rtol": amh.RTOL, "atol": amh.ATOL,
                           "initial_parameters": "drawn once in float64, cast to float32"},
              "at_initial_parameters": {}, "adam_steps": {}}
    pred, grad, loss = {}, {}, {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        problem, model = setup(dtype, state)
        for mode in ("batched", "per_trajectory"):
            pred[name, mode] = predictions(model, problem, mode)
            loss[name, mode], grad[name, mode] = loss_and_grad(model, problem, mode)
    for name in ("float32", "float64"):
        result["at_initial_parameters"][f"{name}: per_trajectory vs batched"] = {
            "prediction": compare(pred[name, "per_trajectory"], pred[name, "batched"]),
            "gradient": compare(grad[name, "per_trajectory"], grad[name, "batched"]),
            "loss": [loss[name, "batched"]["loss"], loss[name, "per_trajectory"]["loss"]]}
    result["at_initial_parameters"]["batched: float32 vs float64"] = {
        "prediction": compare(pred["float32", "batched"], pred["float64", "batched"]),
        "gradient": compare(grad["float32", "batched"], grad["float64", "batched"]),
        "loss": [loss["float64", "batched"]["loss"], loss["float32", "batched"]["loss"]]}
    for name, dtype, mode in (("batched_float32", torch.float32, "batched"),
                              ("per_trajectory_float32", torch.float32, "per_trajectory"),
                              ("batched_float64", torch.float64, "batched")):
        problem, model = setup(dtype, state)
        result["adam_steps"][name] = adam_steps(model, problem, mode, args.steps)
        print(name, result["adam_steps"][name]["seconds_per_epoch_mean"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["at_initial_parameters"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
