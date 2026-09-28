r"""Authors-repo-matched PyTorch biodiesel run (controlled experiment, NOT the canonical
reproduction; no repository default changes).

Matches the authors' RELEASED biodiesel example (DENG-MIT/ChemKAN, d7aa5ab):
  data      experiment-local ``data/author_repo_match_biodiesel.npz`` (the example's arrays);
            ``--data canonical`` instead trains on OUR ``biodiesel.npz`` with legacy B0's
            train-only normalization, to test the settings on our problem
  model     ``_author_match.build_core()``: 7 -> 4 -> 6, G = 3, 156 params, base off,
            layer-1 normalizer off, layer-2 tanh, n_mu = 2, Julia RBF exp(-(x - c)^2)
  dynamics  normalized coordinates, ``dz/dt = [KAN(z), 0] / 50``
  init      Lux glorot_uniform on each C (out x G*in)
  optimizer Adam lr 1e-2, betas (0.9, 0.999), eps 1e-8; full batch; one step per epoch
  solver    torchdiffeq tsit5, rtol 1e-2, atol 1e-6, one trajectory per solve (as Julia);
            gradients by direct autograd through the solver; ``--sensitivity fsa`` uses
            ``chemkan.fsa`` forward sensitivity analysis instead (one augmented solve per
            trajectory, same loss, same single Adam step); ``--solve-mode batched`` solves
            all trajectories in one call instead (the final rtol 1e-10 check stays per
            trajectory)
  loss      released-example Flux ``mse`` (mean over species and times, mean over
            trajectories); Eq. 18 convention = 30 x that value, logged beside it

Per epoch, like the Julia loop: train and validation losses at the epoch's STARTING
parameters, then one Adam step. Residual differences that cannot be removed: the
PyTorch RNG (so a different initial draw), torchdiffeq's step-size controller and dense
output vs OrdinaryDiffEq's ``AutoTsit5(Rosenbrock23)``, float64 throughout here.

    python train_author_repo_match.py --run-dir <dir> [--epochs 10000] [--seed 0] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # chemkan/scripts
from _author_match import (EXP, NormalizedDynamics, build_core, glorot_uniform_,  # noqa: E402
                           julia_mse, load_julia_params)
from _data import load_biodiesel  # noqa: E402
from chemkan.fsa import (NonFiniteFSAError, ParameterPacking,  # noqa: E402
                         fsa_loss_and_gradients, fsa_provenance)
from chemkan.solver import SolverConfig, integrate  # noqa: E402

DATA = EXP / "data/author_repo_match_biodiesel.npz"
CANONICAL_DATA = "chemkan/data/generated/biodiesel.npz"
N_T = 30                                   # Eq. 18 sums over these 30 times; Flux mse averages
JULIA = {"train_mse_final_logged": 4.9196e-5, "val_mse_final_logged": 2.12588e-4,
         "train_mse_min": 4.18160e-5, "train_mse_min_epoch": 9957,
         "val_mse_min": 1.92183e-4, "val_mse_min_epoch": 9454,
         "wall_clock_s": 36 * 60 + 47.66}
# Legacy B0 (results/reproduction/legacy/biodiesel/chemkan/main/direct_autograd_seed0) on
# the SAME canonical data and normalization; Eq. 18 values from its history/metrics, mse = /30.
LEGACY_B0 = {"run": "results/reproduction/legacy/biodiesel/chemkan/main/direct_autograd_seed0",
             "train_eq18_last20pct_median": 3.578290529549122e-2,
             "train_mse_last20pct_median": 3.578290529549122e-2 / N_T,
             "train_eq18_min": 3.1129175797104836e-2, "train_mse_min": 3.1129175797104836e-2 / N_T,
             "test_eq18_final_checkpoint": 8.140052855014801e-2,
             "test_mse_final_checkpoint": 8.140052855014801e-2 / N_T}
HISTORY_COLUMNS = ["epoch", "train_mse", "val_mse", "train_eq18", "val_eq18", "elapsed_seconds"]


def load_data(device, path: Path = DATA) -> dict:
    """The released example's own arrays (experiment-local dataset), or another file in the
    same layout (``--data-file``, e.g. the reaction-order datasets)."""
    d = np.load(path)
    n_train = int(d["n_train"])
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    source = {"data_source": "author_repo",
              "path": str(path.resolve().relative_to(EXP.parents[2])),
              "normalization": "released example: species min/max over train+test, "
                               "T min/max over all ICs"}
    if path != DATA:
        source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"t": as_t(d["t"]), "u0": as_t(d["u0_norm"]), "target": as_t(d["normdata"]),
            "n_train": n_train, "n_test": int(d["n_test"]),
            "train": slice(0, n_train), "test": slice(n_train, n_train + int(d["n_test"])),
            "source": source}


def load_canonical_data(device) -> dict:
    """Our canonical ``biodiesel.npz``, normalized with legacy B0's exact statistics.

    Species: the archive's train-only ``u_min``/``u_max``; T: min/max of the TRAIN
    temperatures -- the full-state normalizer ``train_biodiesel.py`` builds. Test states use
    the same (train) statistics.
    """
    train, test = load_biodiesel(split="train"), load_biodiesel(split="test")
    f64 = lambda x: x.to(device=device, dtype=torch.float64)
    y_min, y_max = f64(train["u_min"]), f64(train["u_max"])
    T_min, T_max = f64(train["T_const"].min()), f64(train["T_const"].max())

    def norm(d):
        Y0 = (f64(d["Y0"]) - y_min) / (y_max - y_min)
        T0 = ((f64(d["T_const"]) - T_min) / (T_max - T_min)).reshape(-1, 1)
        target = ((f64(d["species_TBm"]) - y_min) / (y_max - y_min)).permute(1, 2, 0)
        return torch.cat([Y0, T0], dim=-1), target                    # (B, 7), (B, 6, T)

    (u0_tr, y_tr), (u0_te, y_te) = norm(train), norm(test)
    n_train, n_test = u0_tr.shape[0], u0_te.shape[0]
    return {"t": f64(train["t"]), "u0": torch.cat([u0_tr, u0_te]), "target": torch.cat([y_tr, y_te]),
            "n_train": n_train, "n_test": n_test,
            "train": slice(0, n_train), "test": slice(n_train, n_train + n_test),
            "source": {"data_source": "canonical", "path": CANONICAL_DATA,
                       "normalization": "legacy B0: archive train-only species u_min/u_max, "
                                        "T min/max over train temperatures",
                       "species_min": y_min.tolist(), "species_max": y_max.tolist(),
                       "T_min": float(T_min), "T_max": float(T_max)}}


def predict(dyn, u0, t, solver, batched: bool = False) -> torch.Tensor:
    """(B, 7) initial states -> (B, 6, T). Default: each trajectory in its own adaptive
    solve (as Julia). ``batched``: ONE solve of all B trajectories (shared steps, RMS error
    norm over the batch), as the canonical trainer does."""
    if batched:
        return integrate(dyn, u0, t, solver)[:, :, :6].permute(1, 2, 0)
    return torch.stack([integrate(dyn, u0[i:i + 1], t, solver)[:, 0, :6].T
                        for i in range(u0.shape[0])])


def split_mse(dyn, data, split, solver, batched: bool = False) -> torch.Tensor:
    return julia_mse(predict(dyn, data["u0"][split], data["t"], solver, batched),
                     data["target"][split])


def fsa_train_mse(dyn, data, solver, batched: bool = False) -> torch.Tensor:
    """Train loss with FSA gradients accumulated into ``.grad``: the same solves and the
    same ``julia_mse`` as ``split_mse`` (per trajectory: each term / n_train)."""
    if batched:
        target = data["target"][data["train"]]
        loss = fsa_loss_and_gradients(
            dyn, dyn.core.parameters(), data["u0"][data["train"]], data["t"], solver,
            lambda pred: julia_mse(pred[:, :, :6].permute(1, 2, 0), target))
        return loss.detach()
    n, total = data["n_train"], 0.0
    for i in range(n):
        target = data["target"][data["train"]][i]
        term = fsa_loss_and_gradients(
            dyn, dyn.core.parameters(), data["u0"][data["train"]][i:i + 1], data["t"], solver,
            lambda pred, y=target: ((pred[:, 0, :6].T - y) ** 2).mean() / n)
        total += float(term.detach())
    return torch.tensor(total)


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent).stdout.strip()


def summarize(history: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in history])
    late = v[int(0.8 * len(v)):]
    i = int(v.argmin())
    return {"first": float(v[0]), "final_logged": float(v[-1]),
            "min": float(v[i]), "min_epoch": int(history[i]["epoch"]),
            "last20pct_median": float(np.median(late)), "last20pct_min": float(late.min()),
            "last20pct_max": float(late.max()), "all_finite": bool(np.isfinite(v).all())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data", choices=["author_repo", "canonical"], default="author_repo",
                    help="author_repo: the released example's own arrays (default); "
                         "canonical: our biodiesel.npz with legacy B0's train-only statistics")
    ap.add_argument("--sensitivity", choices=["direct_autograd", "fsa"], default="direct_autograd",
                    help="training gradients: backprop through the solver (default) or "
                         "chemkan.fsa forward sensitivity analysis (the paper's method)")
    ap.add_argument("--solve-mode", choices=["per_trajectory", "batched"],
                    default="per_trajectory",
                    help="per_trajectory: one solve per trajectory, as Julia (default); "
                         "batched: one solve of all trajectories, as the canonical trainer")
    ap.add_argument("--data-file", type=Path, default=None,
                    help="with --data author_repo: train on this .npz (same layout as the "
                         "default author_repo_match_biodiesel.npz) instead")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from these 156 flat Julia parameters (p.txt layout, e.g. "
                         "julia_reference/p_init.txt) instead of the seeded glorot draw")
    args = ap.parse_args()
    batched = args.solve_mode == "batched"
    if args.init_from is not None and not args.init_from.is_file():
        raise SystemExit(f"--init-from {args.init_from} not found")

    run = args.run_dir
    final_path, resume_path = run / "checkpoint_final.pt", run / "checkpoint_resume.pt"
    if final_path.exists():
        raise SystemExit(f"{final_path} exists; choose another --run-dir")
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise SystemExit(f"{run} is not empty; pass --resume or choose another --run-dir")
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run / "run.log"), logging.StreamHandler()])

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    if args.data_file is not None and args.data != "author_repo":
        raise SystemExit("--data-file requires --data author_repo")
    data = (load_data(device, args.data_file or DATA) if args.data == "author_repo"
            else load_canonical_data(device))
    solver = SolverConfig(method="tsit5", rtol=1e-2, atol=1e-6, sensitivity=args.sensitivity)

    core = build_core().to(device)
    init_gen = torch.Generator().manual_seed(args.seed)
    glorot_uniform_(core, init_gen)
    if args.init_from is not None:       # Julia values are Float32: parse as such, widen
        load_julia_params(core, np.loadtxt(args.init_from, dtype=np.float32).astype(np.float64))
    dyn = NormalizedDynamics(core)
    opt = torch.optim.Adam(core.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    config = {
        "experiment": "biodiesel/author_repo_match",
        "purpose": "controlled match to the authors' RELEASED GitHub biodiesel example; "
                   "not the canonical reproduction, not the paper protocol, not the "
                   "2026-09-22 email clarifications",
        "reference": {"repository": "DENG-MIT/ChemKAN",
                      "commit": "d7aa5abecbb595d504c86d031cd478a549c614cb",
                      "file": "ChemKAN_biodiesel_example.jl (run via the approved "
                              "3-line API compatibility copy, Julia 1.11.1)"},
        "seed": args.seed, "dtype": "float64", "torch_threads": args.threads,
        "data": data["source"]["path"],
        "data_source": data["source"],
        "architecture": {"layers": "7 -> 4 -> 6", "num_basis": 3, "parameter_count": 156,
                         "use_base_act": False, "layer1_input_tanh": False,
                         "layer2_input_tanh": True, "n_mu": 2, "centers": [-1.0, 0.0, 1.0],
                         "rbf": "exp(-((x - c)/1)^2) (library gaussian with h = 1/sqrt(2))"},
        "dynamics": "normalized coordinates z = [Y_hat, T_hat]; dz/dt = [KAN(z), 0] / 50",
        "initialization": (
            "glorot_uniform on C (out x G*in): U(-a, a), a = sqrt(6/(G*in + out))"
            if args.init_from is None else
            {"source": "Julia initial parameters (Random.seed!(1234), Lux.setup), exported "
                       "by export_julia_init.jl", "path": str(args.init_from),
             "sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest()}),
        "optimizer": {"name": "Adam", "lr": 1e-2, "betas": [0.9, 0.999], "eps": 1e-8},
        "epochs": args.epochs, "batching": "full batch, one Adam step per epoch",
        "solver": {"method": "tsit5", "rtol": 1e-2, "atol": 1e-6, "library": "torchdiffeq",
                   "per_trajectory_solves": not batched, "sensitivity": args.sensitivity},
        "loss": "Flux-style mse: mean over 6 species and 30 times per trajectory, then mean "
                "over trajectories; *_eq18 columns = 30 x that value",
        "logged_state": "train/val losses at each epoch's STARTING parameters (before the "
                        "Adam step), as in the Julia loop",
        "noise": 0.0, "git_commit": git_commit(),
        "python": platform.python_version(), "torch": torch.__version__,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.sensitivity == "fsa":        # direct-autograd configs keep their exact keys
        config["fsa"] = fsa_provenance(
            {"kinetic": ParameterPacking.from_module(dyn, core.parameters())})
        config["fsa"]["per_trajectory"] = ("one augmented [state, sensitivity] solve per "
                                           "trajectory; gradients summed, each / n_train")

    start_epoch, prior_elapsed = 0, 0.0
    history: list[dict] = []
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, weights_only=False)
        core.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["rng_state"])
        start_epoch, prior_elapsed = state["epoch"], state["elapsed_seconds"]
        with open(run / "history.csv") as f:
            history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)
                       if int(r["epoch"]) <= start_epoch]
        logging.info("resumed at epoch %d", start_epoch)
    else:
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        logging.info("run start | %s | seed %d | params %d", run, args.seed,
                     sum(p.numel() for p in core.parameters()))

    with open(run / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        w.writeheader()
        for r in history:
            w.writerow({**r, "epoch": int(r["epoch"])})
    hist_file = open(run / "history.csv", "a", newline="")
    writer = csv.DictWriter(hist_file, fieldnames=HISTORY_COLUMNS)

    started = time.perf_counter()
    status = "completed"
    for epoch in range(start_epoch + 1, args.epochs + 1):       # 1-based, as Julia's iter
        opt.zero_grad()
        if args.sensitivity == "fsa":                            # gradients land in .grad
            try:
                train = fsa_train_mse(dyn, data, solver, batched)
            except NonFiniteFSAError as err:
                status = f"stopped at epoch {epoch}: {err}"
                logging.info(status)
                break
        else:
            train = split_mse(dyn, data, data["train"], solver, batched)
        with torch.no_grad():
            val = split_mse(dyn, data, data["test"], solver, batched)
        tr, va = float(train.detach()), float(val)
        elapsed = prior_elapsed + time.perf_counter() - started
        row = {"epoch": epoch, "train_mse": tr, "val_mse": va, "train_eq18": N_T * tr,
               "val_eq18": N_T * va, "elapsed_seconds": round(elapsed, 4)}
        writer.writerow(row)
        hist_file.flush()
        history.append(row)
        if not (math.isfinite(tr) and math.isfinite(va)):
            status = f"stopped: non-finite loss at epoch {epoch}"
            logging.info(status)
            break
        if args.sensitivity == "direct_autograd":
            train.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == 1:
            logging.info("epoch %5d  train %.4e  val %.4e  (%.0f s)", epoch, tr, va, elapsed)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0 and epoch < args.epochs:
            torch.save({"epoch": epoch, "model_state": core.state_dict(),
                        "optimizer_state": opt.state_dict(), "rng_state": torch.get_rng_state(),
                        "elapsed_seconds": elapsed, "config": config}, resume_path)
    hist_file.close()
    wall = prior_elapsed + time.perf_counter() - started

    # Final parameters (after the last Adam step): training solver and a near-exact solve.
    tight = SolverConfig(method="tsit5", rtol=1e-10, atol=1e-12, sensitivity="direct_autograd")
    with torch.no_grad():
        post = {name: {"train_mse": float(split_mse(dyn, data, data["train"], s)),
                       "val_mse": float(split_mse(dyn, data, data["test"], s))}
                for name, s in (("training_solver_rtol1e-2", solver),
                                ("tight_solver_rtol1e-10", tight))}
    for v in post.values():
        v.update(train_eq18=N_T * v["train_mse"], val_eq18=N_T * v["val_mse"])

    train_s, val_s = summarize(history, "train_mse"), summarize(history, "val_mse")
    metrics = {
        "status": status, "epochs_completed": int(history[-1]["epoch"]),
        "wall_clock_s": round(wall, 2),
        "loss_convention": "primary: released-example Flux mse; eq18 = 30 x mse",
        "train_mse": train_s, "val_mse": val_s,
        "train_eq18_final_logged": N_T * train_s["final_logged"],
        "val_eq18_final_logged": N_T * val_s["final_logged"],
        "final_parameters": post,
    }
    if args.data == "author_repo":                   # same data as the Julia run
        metrics["julia_reference"] = JULIA
        metrics["ratio_to_julia"] = {
            "train_final_logged": train_s["final_logged"] / JULIA["train_mse_final_logged"],
            "val_final_logged": val_s["final_logged"] / JULIA["val_mse_final_logged"]}
    else:                                            # same data as legacy B0
        metrics["legacy_b0_reference"] = LEGACY_B0
        metrics["ratio_to_legacy_b0"] = {
            "train_last20pct_median": train_s["last20pct_median"] / LEGACY_B0["train_mse_last20pct_median"],
            "train_min": train_s["min"] / LEGACY_B0["train_mse_min"],
            "val_last20pct_median_vs_b0_final_test": val_s["last20pct_median"]
                                                     / LEGACY_B0["test_mse_final_checkpoint"]}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if status == "completed":
        torch.save({"model_state": core.state_dict(), "config": config, "metrics": metrics,
                    "optimizer_state": opt.state_dict()}, final_path)
        resume_path.unlink(missing_ok=True)
    logging.info("%s | %d epochs | %.1f s | train %.4e val %.4e", status,
                 metrics["epochs_completed"], wall, train_s["final_logged"], val_s["final_logged"])
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
