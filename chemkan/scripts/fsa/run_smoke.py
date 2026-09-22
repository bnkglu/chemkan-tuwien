r"""Short FSA smoke tests through the real training CLIs (DIAGNOSTIC; not scientific runs).

Runs, in order, each into the gitignored ``results/experiments/validation/fsa/smoke/runs``:

  bd_full      biodiesel, B0 settings, full batch, 20 updates + state-only evaluation
  bd_batch5    biodiesel trajectory mini-batches of 5 (support check), 1 epoch = 4 updates
  bd_interval  biodiesel observed-interval accumulation (support check), 2 updates
  h_stage1     hydrogen Stage 1, H_STAGE1 flags, 20 updates
  h_H0         hydrogen Stage 2 from h_stage1's FSA checkpoint, H0 init, 5 updates
  h_Hnorm1     hydrogen Stage 2 from h_stage1's FSA checkpoint, Hnorm1 init, 5 updates
  h_2stage     one process: Stage 1 (3 updates) -> Stage 2 (2 updates) hand-over
  guard        a direct-autograd Stage 2 must refuse the FSA Stage-1 checkpoint

Explicit checks per training run: exit status; finite logged losses; finite physical
states, sensitivities, loss and parameter gradients in an in-process FSA evaluation of the
final smoke model on the full training batch; successful Adam updates (optimizer step
count == updates, every optimized tensor changed from its recorded initial hash); runtime
per update vs the archived direct-autograd baseline at the same epochs ("reasonable" =
at most RUNTIME_RATIO_MAX times slower); NFE per update where the trainer records it.

Smoke models and optimizer states are never used to start a scientific run.
Writes ``results/experiments/validation/fsa/smoke_summary.json``.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
import torch                                                        # noqa: E402
from _run import git_commit, tensor_sha256, utc_now                 # noqa: E402

from chemkan.fsa import FunctionalDynamics, integrate_with_sensitivities  # noqa: E402
from chemkan.solver import SolverConfig                             # noqa: E402
from chemkan.training import loss_and_gradients                     # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1]
OUT = P.RESULTS / "experiments/validation/fsa"
WORK = P.RESULTS / "experiments/validation/fsa/smoke/runs"
PROBE_EPOCHS = "0,1,2,5,10,20,50,100,200,500,1000,2000,3000,5000,7500,10000"
H_COMMON = ["--seed", "0", "--experiment-name", "base_on_n4", "--sensitivity", "fsa",
            "--count-nfe", "--stage1-temperature-source", "dense-cantera",
            "--stage1-temperature-points", "20000"]
RUNTIME_RATIO_MAX = 30.0      # FSA seconds/update vs archived direct-autograd seconds/epoch
PRODUCTION = SolverConfig(method="tsit5", rtol=1e-6, atol=1e-8, sensitivity="fsa")
HNORM1 = ["--thermo-init", "scaled-random", "--thermo-match-cantera-norm",
          "--thermo-linear-init-seed", "1", "--thermo-init-temperature", "1050",
          "--thermo-init-phi", "0.5"]


def cli(*args, expect=0):
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, *args], cwd=SCRIPTS, capture_output=True, text=True)
    return r, time.perf_counter() - t0


def train(*args):
    """Training CLI through the resume driver, which keeps every resume state (Adam)."""
    return cli("fsa/_resume_driver.py", *args)


def fsa_probe(kind: str, ckpt_path: Path) -> dict:
    """One explicit FSA gradient evaluation of the final smoke model (float32, full batch)."""
    ck = P.load(ckpt_path)
    prob = {"biodiesel": P.biodiesel, "stage1": P.stage1, "stage2": P.stage2}[kind](
        "init" if kind != "stage2" else "H0-init", torch.float32)
    prob.net.load_state_dict(ck["model_state"])
    fd = FunctionalDynamics(prob.dynamics, prob.params)
    x, S = integrate_with_sensitivities(fd, prob.x0, prob.t, PRODUCTION)
    for q in prob.params:
        q.grad = None
    loss = loss_and_gradients(prob.dynamics, prob.x0, prob.t, prob.params, prob.loss_fn, PRODUCTION)
    g = torch.cat([q.grad.reshape(-1) for q in fd.params])
    return {"check_finite_states": bool(torch.isfinite(x).all()),
            "check_finite_sensitivities": bool(torch.isfinite(S).all()),
            "check_finite_loss": bool(torch.isfinite(loss.detach())),
            "check_finite_gradients": bool(torch.isfinite(g).all()),
            "probe_max_abs_sensitivity": float(S.abs().max()),
            "probe_gradient_norm": float(g.norm()), "probe_loss": float(loss.detach())}


def adam_updates(run_dir: Path, stage: str, n: int, packing_key: str) -> dict:
    """Optimizer steps after the last update, and every optimized tensor moved.

    A tensor that did not change passes only if its Adam state received a non-zero finite
    gradient AND replaying the last Adam step from the saved state leaves the float32
    tensor bit-identical (the update is below float32 resolution -- e.g. the ~1e5
    norm-matched thermo coefficients with lr 2e-3, as in the archived Hnorm1 baseline).
    """
    st = P.load(run_dir / f"resume_history/{stage}_{n}.pt")
    opt = st["optimizer_state"]
    steps = sorted({int(v["step"]) for v in opt["state"].values()})
    group = opt["param_groups"][0]
    cfg = json.loads((run_dir / "config.json").read_text())
    init = cfg["fsa"]["initial_model"]["tensors"]
    final = P.load(run_dir / "checkpoint_final.pt")["model_state"]
    names = [o["name"] for o in cfg["fsa"]["parameter_packing"][packing_key]["order"]]
    moved, below = {}, {}
    for i, name in enumerate(names):
        key = name[len("model."):] if name.startswith("model.") else name
        if packing_key == "kinetic":          # biodiesel: dynamics prefix 'kinetic.' -> core
            key = name[len("kinetic."):]
        w = final[key]
        moved[name] = init[f"param:{key}"]["sha256"] != tensor_sha256(w)
        if not moved[name]:
            s_i = opt["state"][i]
            t = float(s_i["step"])
            b1, b2 = group["betas"]
            m_hat = s_i["exp_avg"] / (1 - b1 ** t)
            v_hat = s_i["exp_avg_sq"] / (1 - b2 ** t)
            delta = group["lr"] * m_hat / (v_hat.sqrt() + group["eps"])
            got_grad = bool(torch.isfinite(s_i["exp_avg"]).all() and s_i["exp_avg"].abs().max() > 0)
            below[name] = {"gradient_received": got_grad,
                           "max_abs_last_update": float(delta.abs().max()),
                           "update_below_float32_resolution": bool(torch.equal((w - delta).to(w.dtype), w))}
    ok = all(moved[k] or (below[k]["gradient_received"] and below[k]["update_below_float32_resolution"])
             for k in names)
    return {"check_optimizer_steps": steps == [n], "optimizer_steps": steps,
            "check_all_parameters_updated": ok,
            "parameters_unchanged": below}


def runtime(run_dir: Path, hist: str, baseline: Path, epochs: int, first: int = 1) -> dict:
    """Median seconds/update (skipping warm-up) vs the archived direct-autograd baseline."""
    def per_epoch(path):
        h = rows(path)[:epochs]
        if "epoch_wall_time_s" in h[0]:
            v = [float(r["epoch_wall_time_s"]) for r in h]
        else:
            e = [float(r["elapsed_seconds"]) for r in h]
            v = [b - a for a, b in zip([0.0] + e[:-1], e)]
        v = v[first:] or v
        return sorted(v)[len(v) // 2]
    fsa, da = per_epoch(run_dir / hist), per_epoch(baseline / hist)
    out = {"median_seconds_per_update_fsa": fsa, "median_seconds_per_epoch_direct_autograd_baseline": da,
           "runtime_ratio": fsa / da, "check_reasonable_runtime": fsa / da <= RUNTIME_RATIO_MAX}
    hf, hb = rows(run_dir / hist)[:epochs], rows(baseline / hist)[:epochs]
    if "nfe" in hf[0] and "nfe" in hb[0]:
        out["nfe_fsa"] = [int(r["nfe"]) for r in hf]
        out["nfe_direct_autograd_baseline"] = [int(r["nfe"]) for r in hb]
    return out


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(name, run_dir, hist, r, seconds, extra=None):
    ok = r.returncode == 0
    out = {"name": name, "run_dir": str(run_dir.relative_to(P.ROOT)), "returncode": r.returncode,
           "wall_seconds": round(seconds, 1)}
    if ok and hist:
        h = rows(run_dir / hist)
        losses = [float(x["total_loss"]) for x in h]
        out.update({"updates_logged": len(h), "first_loss": losses[0], "last_loss": losses[-1],
                    "all_losses_finite": all(math.isfinite(v) for v in losses)})
        if "nfe" in h[0]:
            out["nfe_per_update"] = [int(x["nfe"]) for x in h]
        if "epoch_wall_time_s" in h[0]:
            out["seconds_per_update"] = [float(x["epoch_wall_time_s"]) for x in h]
        cfg = json.loads((run_dir / "config.json").read_text())
        out["sensitivity_backend"] = cfg["sensitivity_backend"]
        ok &= out["all_losses_finite"] and cfg["sensitivity_backend"] == "fsa"
    else:
        out["stderr_tail"] = r.stderr[-1500:]
    out.update(extra or {})
    out["pass"] = bool(ok and all(v for k, v in (extra or {}).items() if k.startswith("check_")))
    print(f"{name:12s} {'PASS' if out['pass'] else 'FAIL'}  {seconds:6.1f}s "
          f"{out.get('first_loss', '')} -> {out.get('last_loss', '')}", flush=True)
    return out


def evaluate(script, run_dir):
    r, _ = cli(script, "--run-dir", str(run_dir), "--split", "test", "--metrics")
    m = json.loads((run_dir / "metrics.json").read_text()) if r.returncode == 0 else {}
    return {"check_state_only_inference": r.returncode == 0 and math.isfinite(m.get("test_mse", math.nan)),
            "inference_test_mse": m.get("test_mse"),
            "inference_solver": m.get("solver")}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    res = []

    d = WORK / "bd_full"
    r, s = train("train_biodiesel.py", "--seed", "0", "--epochs", "20", "--sensitivity", "fsa",
                 "--checkpoint-every", "20", "--run-dir", str(d))
    extra = {}
    if r.returncode == 0:
        extra.update(evaluate("evaluate_biodiesel.py", d))
        extra.update(fsa_probe("biodiesel", d / "checkpoint_final.pt"))
        extra.update(adam_updates(d, "main", 20, "kinetic"))
        extra.update(runtime(d, "history.csv", P.B0_DIR, 20))
    res.append(summarize("bd_full", d, "history.csv", r, s, extra))

    d = WORK / "bd_batch5"
    r, s = train("train_biodiesel.py", "--seed", "0", "--epochs", "1", "--batch-size", "5",
                 "--sensitivity", "fsa", "--checkpoint-every", "1", "--run-dir", str(d))
    extra = {}
    if r.returncode == 0:
        extra["check_four_updates"] = rows(d / "history.csv")[-1]["optimizer_step"] == "4"
        extra.update(adam_updates(d, "main", 1, "kinetic"))
        extra["check_optimizer_steps"] = extra.pop("optimizer_steps") == [4]
        extra.update(fsa_probe("biodiesel", d / "checkpoint_final.pt"))
    res.append(summarize("bd_batch5", d, "history.csv", r, s, extra))

    d = WORK / "bd_interval"
    r, s = train("train_biodiesel_observed_intervals.py", "--seed", "0", "--epochs", "2",
                 "--eval-every", "1", "--sensitivity", "fsa", "--checkpoint-every", "2",
                 "--run-dir", str(d))
    extra = {}
    if r.returncode == 0:
        extra.update(adam_updates(d, "observed_intervals", 2, "kinetic"))
        extra.update(fsa_probe("biodiesel", d / "checkpoint_final.pt"))
    res.append(summarize("bd_interval", d, "history.csv", r, s, extra))

    s1 = WORK / "h_stage1"
    r, s = train("train_hydrogen.py", *H_COMMON, "--stage1-epochs", "20", "--stage2-epochs", "0",
                 "--stage2-probe", "--stage2-probe-epochs", PROBE_EPOCHS,
                 "--stage2-snapshot-epochs", "500", "--checkpoint-every", "20", "--run-dir", str(s1))
    extra = {}
    if r.returncode == 0:
        extra.update(fsa_probe("stage1", s1 / "checkpoint_final.pt"))
        extra.update(adam_updates(s1, "stage1", 20, "stage1"))
        extra.update(runtime(s1, "history_stage1.csv", P.H_STAGE1_DIR, 20))
    res.append(summarize("h_stage1", s1, "history_stage1.csv", r, s, extra))
    s1_ckpt = s1 / "checkpoint_stage1.pt"

    for name, init, base in (("h_H0", [], P.H0_DIR), ("h_Hnorm1", HNORM1, P.HNORM1_DIR)):
        d = WORK / name
        r, s = train("train_hydrogen.py", *H_COMMON, "--stage1-from", str(s1_ckpt),
                     "--stage2-epochs", "5", "--stage2-probe", "--stage2-probe-epochs", PROBE_EPOCHS,
                     "--stage2-snapshot-epochs", "500", *init, "--checkpoint-every", "5",
                     "--run-dir", str(d))
        extra = {}
        if r.returncode == 0:
            cfg = json.loads((d / "config.json").read_text())
            extra["check_consumes_fsa_stage1"] = cfg["fsa"]["stage1_from_sensitivity"] == "fsa"
            if init:
                want = json.loads((P.HNORM1_DIR / "config.json").read_text())["thermo_init"]["resulting_vector"]
                extra["check_hnorm1_vector_matches_archive"] = cfg["thermo_init"]["resulting_vector"] == want
            extra.update(evaluate("evaluate_hydrogen.py", d))
            extra.update(fsa_probe("stage2", d / "checkpoint_final.pt"))
            extra.update(adam_updates(d, "stage2", 5, "stage2"))
            extra.update(runtime(d, "history_stage2.csv", base, 5, first=0))
        res.append(summarize(name, d, "history_stage2.csv", r, s, extra))

    d = WORK / "h_2stage"
    r, s = train("train_hydrogen.py", *H_COMMON, "--stage1-epochs", "3", "--stage2-epochs", "2",
                 "--checkpoint-every", "1", "--run-dir", str(d))
    extra = {}
    if r.returncode == 0:
        extra["check_stage1_rows"] = len(rows(d / "history_stage1.csv")) == 3
        extra["check_stage2_rows"] = len(rows(d / "history_stage2.csv")) == 2
        s2_start = P.load(d / "resume_history/stage2_0.pt")
        extra["check_stage2_starts_fresh_optimizer"] = s2_start["optimizer_state"] is None
        extra.update(adam_updates(d, "stage2", 2, "stage2"))
    res.append(summarize("h_2stage", d, "history_stage2.csv", r, s, extra))

    d = WORK / "guard"
    r, s = cli("train_hydrogen.py", "--stage1-from", str(s1_ckpt), "--stage2-epochs", "1",
               "--run-dir", str(d))
    guard_ok = r.returncode != 0 and "stage1-backend-mismatch" in (r.stderr + r.stdout)
    res.append({"name": "guard", "returncode": r.returncode, "pass": guard_ok,
                "what": "direct-autograd Stage 2 refuses an FSA Stage-1 checkpoint"})
    print(f"{'guard':12s} {'PASS' if guard_ok else 'FAIL'}")

    ok = all(x["pass"] for x in res)
    payload = {"created": utc_now(), "git_commit": git_commit(), "code_state": P.code_state(),
               "status": "PASS" if ok else "FAIL", "runtime_ratio_max": RUNTIME_RATIO_MAX,
               "runs": res,
               "note": "diagnostic smoke runs; never used to initialize scientific runs. "
                       "bd_batch5 and bd_interval are support checks of the mini-batch and "
                       "observed-interval paths, not scientific runs."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"-> {(OUT / 'smoke_summary.json').relative_to(P.ROOT)} [{payload['status']}]")


if __name__ == "__main__":
    main()
