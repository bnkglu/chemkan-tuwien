r"""FSA resume validation: 10 uninterrupted updates vs 5 + checkpoint/resume + 5 (DIAGNOSTIC).

For each case the real training CLI runs through ``_resume_driver.py``:

* U: 10 updates without interruption;
* I: the same command interrupted (simulated Ctrl+C) right after the epoch-5 resume
  checkpoint, then restarted with ``--resume`` for the remaining 5 updates.

Compared (timing fields excluded): final model tensors, the Adam state and the torch RNG
state saved after update 10, and the history rows. Cases: biodiesel full batch, hydrogen
Stage 1, and hydrogen Stage 2 started from the FSA Stage-1 checkpoint of case 2 (which
also exercises the Stage-1 -> Stage-2 hand-over of an FSA-trained kinetic core).

Writes ``results/experiments/validation/fsa/resume_validation.json``; the runs themselves
go to the gitignored ``results/experiments/validation/fsa/smoke/resume``.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
from _run import git_commit, utc_now                                # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1]
OUT = P.RESULTS / "experiments/validation/fsa"
WORK = P.RESULTS / "experiments/validation/fsa/smoke/resume"
TIMING = {"elapsed_seconds", "epoch_wall_time_s"}


def run(args, run_dir, interrupt=None, resume=False):
    cmd = [sys.executable, "fsa/_resume_driver.py"]
    if interrupt:
        cmd += ["--interrupt-stage", interrupt[0], "--interrupt-epoch", str(interrupt[1])]
    cmd += [*args, "--run-dir", str(run_dir)] + (["--resume"] if resume else [])
    r = subprocess.run(cmd, cwd=SCRIPTS, capture_output=True, text=True)
    expected = 130 if interrupt else 0
    if r.returncode != expected:
        raise SystemExit(f"{' '.join(cmd)} exited {r.returncode} (expected {expected}):\n"
                         f"{r.stderr[-3000:]}")
    return r


def tensors_equal(a: dict, b: dict) -> tuple[bool, list]:
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    return (not bad and a.keys() == b.keys()), bad


def optimizer_equal(a: dict, b: dict) -> tuple[bool, list]:
    bad = []
    for pid, st in a["state"].items():
        for k, v in st.items():
            w = b["state"][pid][k]
            if not (torch.equal(v, w) if torch.is_tensor(v) else v == w):
                bad.append(f"{pid}.{k}")
    return (not bad and a["param_groups"] == b["param_groups"]), bad


def history(path):
    with open(path) as f:
        return [{k: v for k, v in r.items() if k not in TIMING} for r in csv.DictReader(f)]


def compare(case, u_dir, i_dir, stage, hist_name, extra_ckpt=None) -> dict:
    load = P.load
    u_final, i_final = load(u_dir / "checkpoint_final.pt"), load(i_dir / "checkpoint_final.pt")
    m_ok, m_bad = tensors_equal(u_final["model_state"], i_final["model_state"])
    u10, i10 = load(u_dir / f"resume_history/{stage}_10.pt"), load(i_dir / f"resume_history/{stage}_10.pt")
    o_ok, o_bad = optimizer_equal(u10["optimizer_state"], i10["optimizer_state"])
    r_ok = torch.equal(u10["rng_state"], i10["rng_state"])
    hu, hi = history(u_dir / hist_name), history(i_dir / hist_name)
    h_ok = hu == hi and len(hu) == 10
    res = {"case": case, "model_tensors_equal": m_ok, "differing_tensors": m_bad,
           "optimizer_state_after_10_equal": o_ok, "differing_optimizer_entries": o_bad,
           "optimizer_steps_after_10": sorted({int(v["step"]) for v in u10["optimizer_state"]["state"].values()}),
           "rng_state_equal": r_ok, "history_rows_equal": h_ok, "history_rows": len(hu),
           "interrupted_segment_rows": None,
           "sensitivity_backend": u_final["solver"]["sensitivity"]}
    if extra_ckpt:
        a, b = load(u_dir / extra_ckpt), load(i_dir / extra_ckpt)
        res[f"{extra_ckpt}_equal"] = tensors_equal(a["kinetic_state"], b["kinetic_state"])[0]
    res["pass"] = bool(m_ok and o_ok and r_ok and h_ok and res["sensitivity_backend"] == "fsa"
                       and res.get(f"{extra_ckpt}_equal", True)
                       and res["optimizer_steps_after_10"] == [10])
    return res


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    cases = []

    bd = ["train_biodiesel.py", "--sensitivity", "fsa", "--epochs", "10", "--checkpoint-every", "5"]
    run(bd, WORK / "biodiesel_U")
    run(bd, WORK / "biodiesel_I", interrupt=("main", 5))
    run(bd, WORK / "biodiesel_I", resume=True)
    cases.append(compare("biodiesel full batch", WORK / "biodiesel_U", WORK / "biodiesel_I",
                         "main", "history.csv"))

    h1 = ["train_hydrogen.py", "--sensitivity", "fsa", "--stage1-epochs", "10",
          "--stage2-epochs", "0", "--checkpoint-every", "5", "--count-nfe"]
    run(h1, WORK / "stage1_U")
    run(h1, WORK / "stage1_I", interrupt=("stage1", 5))
    run(h1, WORK / "stage1_I", resume=True)
    cases.append(compare("hydrogen Stage 1", WORK / "stage1_U", WORK / "stage1_I", "stage1",
                         "history_stage1.csv", extra_ckpt="checkpoint_stage1.pt"))

    s1 = WORK / "stage1_U" / "checkpoint_stage1.pt"
    h2 = ["train_hydrogen.py", "--sensitivity", "fsa", "--stage1-from", str(s1),
          "--stage2-epochs", "10", "--checkpoint-every", "5", "--count-nfe"]
    run(h2, WORK / "stage2_U")
    run(h2, WORK / "stage2_I", interrupt=("stage2", 5))
    run(h2, WORK / "stage2_I", resume=True)
    c = compare("hydrogen Stage 2 from FSA Stage 1", WORK / "stage2_U", WORK / "stage2_I",
                "stage2", "history_stage2.csv")
    cfg = json.loads((WORK / "stage2_U" / "config.json").read_text())
    c["stage1_from_sensitivity"] = cfg["fsa"]["stage1_from_sensitivity"]
    kin_loaded = P.load(s1)["kinetic_state"]
    init = cfg["fsa"]["initial_model"]["tensors"]
    from _run import tensor_sha256
    c["stage2_initial_kinetic_equals_fsa_stage1"] = all(
        init[f"param:kinetic.{k}"]["sha256"] == tensor_sha256(v) for k, v in kin_loaded.items()
        if f"param:kinetic.{k}" in init)
    c["pass"] = c["pass"] and c["stage1_from_sensitivity"] == "fsa" and c["stage2_initial_kinetic_equals_fsa_stage1"]
    cases.append(c)

    ok = all(c["pass"] for c in cases)
    payload = {"created": utc_now(), "git_commit": git_commit(), "code_state": P.code_state(),
               "status": "PASS" if ok else "FAIL",
               "protocol": "10 uninterrupted updates vs 5 updates + simulated interrupt + "
                           "--resume + 5 updates; timing fields excluded", "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resume_validation.json").write_text(json.dumps(payload, indent=2))
    for c in cases:
        print(f"{c['case']:36s} {'PASS' if c['pass'] else 'FAIL'}")
    print(f"-> {(OUT / 'resume_validation.json').relative_to(P.ROOT)} [{payload['status']}]")


if __name__ == "__main__":
    main()
