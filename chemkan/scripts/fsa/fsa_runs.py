r"""The three FSA comparisons (four training jobs): resolve, launch, verify.

    python fsa/fsa_runs.py resolve            # record resolved configurations (no training)
    python fsa/fsa_runs.py launch JOB         # run one job exactly as recorded
    python fsa/fsa_runs.py verify             # configs + initial tensors vs the baselines

Jobs (the ONLY scientific FSA training in this task):

    B0-FSA          biodiesel, B0 settings                     -> fsa/biodiesel/fsa_seed0
    H_STAGE1-FSA    hydrogen Stage 1, H_STAGE1 settings        -> fsa/hydrogen/stage1_fsa_seed0
    H0-FSA          Stage 2 from H_STAGE1-FSA, H0 init         -> fsa/hydrogen/random_stage2_10000_fsa_seed0
    Hnorm1-FSA      Stage 2 from H_STAGE1-FSA, Hnorm1 init     -> fsa/hydrogen/normmatched_dir1_stage2_10000_fsa

Every CLI value is taken from the baseline's saved config.json (never from a trainer
default) and cross-checked by parsing the command with the trainer's own parser. The only
intended change is ``--sensitivity fsa`` (and, for Stage 2, the FSA Stage-1 checkpoint).
Diagnostic flags the baselines used (NFE counting, Stage-2 probe, epoch-500 snapshot) are
reproduced because they are behavior-neutral and keep the artifacts comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
from _run import git_commit, model_tensor_record, tensor_sha256, utc_now  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1]
# Legacy (pre-author-clarification) FSA runs: per-domain run dirs, shared records in fsa_runs/.
LEGACY = P.RESULTS / "experiments/legacy"
FSA = LEGACY / "fsa_runs"
RESOLVED = FSA / "resolved_configurations.json"
OUT = {"B0-FSA": LEGACY / "biodiesel/fsa/fsa_seed0",
       "H_STAGE1-FSA": LEGACY / "hydrogen/fsa/stage1_fsa_seed0",
       "H0-FSA": LEGACY / "hydrogen/fsa/random_stage2_10000_fsa_seed0",
       "Hnorm1-FSA": LEGACY / "hydrogen/fsa/normmatched_dir1_stage2_10000_fsa"}
BASELINE = {"B0-FSA": P.B0_DIR, "H_STAGE1-FSA": P.H_STAGE1_DIR, "H0-FSA": P.H0_DIR,
            "Hnorm1-FSA": P.HNORM1_DIR}
PROBE_EPOCHS = "0,1,2,5,10,20,50,100,200,500,1000,2000,3000,5000,7500,10000"
THREADS = "2"          # torch/OMP threads per job: two concurrent jobs on 4 performance
                       # cores; 2 threads measured ~12% faster than 1 for a Stage-2 FSA step


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(P.ROOT))


def cfg_of(job):
    return json.loads((BASELINE[job] / "config.json").read_text())


def biodiesel_argv(c: dict) -> list[str]:
    a = c["architecture"]
    argv = ["chemkan/scripts/train_biodiesel.py", "--seed", str(c["seed"]), "--epochs", str(c["epochs"]),
            "--hidden-dim", str(a["hidden_dim"]), "--num-basis", str(a["num_basis"]),
            "--n-mu", str(a["n_mu"]), "--lr", repr(c["learning_rate"]),
            "--input-scaling", c["normalization"]["input_scaling"],
            "--solver-method", c["solver"]["method"], "--rtol", repr(c["solver"]["rtol"]),
            "--atol", repr(c["solver"]["atol"]), "--device", c["device"],
            "--experiment-name", c["experiment_name"], "--init", "default"]
    if a["use_base_act"]:
        argv.append("--use-base-act")
    assert c["noise"] is None and not c["pinn"]["enabled"]
    return argv


def hydrogen_argv(c: dict, *, stage1_from: Path | None) -> list[str]:
    a = c["architecture"]
    t = c["stage1_temperature"]
    argv = ["chemkan/scripts/train_hydrogen.py", "--seed", str(c["seed"]),
            "--hidden-dim", str(a["hidden_dim"]), "--num-basis", str(a["num_basis"]),
            "--n-mu", str(a["n_mu"]),
            "--use-base-act" if a["use_base_act"] else "--no-use-base-act",
            "--stage1-epochs", str(c["epochs"]["stage1"]),
            "--stage2-epochs", str(c["epochs"]["stage2"]), "--lr", repr(c["learning_rate"]),
            "--alpha-pinn", repr(c["pinn"]["alpha_pinn"]),
            "--input-scaling", c["normalization"]["input_scaling"],
            "--solver-method", c["solver"]["method"], "--rtol", repr(c["solver"]["rtol"]),
            "--atol", repr(c["solver"]["atol"]), "--device", c["device"],
            "--stage1-temperature-source", t["source"].replace("_", "-"),
            "--stage1-temperature-points", str(t["n_points"]),
            "--experiment-name", c["experiment_name"],
            "--count-nfe", "--stage2-probe", "--stage2-probe-epochs", PROBE_EPOCHS,
            "--stage2-snapshot-epochs", "500"]
    if not c["pinn"]["stage1"]:
        argv.append("--no-pinn-stage1")
    if not c["pinn"]["stage2"]:
        argv.append("--no-pinn-stage2")
    ti = c["thermo_init"]
    if ti["thermo_init"] == "scaled-random":
        assert ti["scaling_mode"] == "match-cantera-norm"
        argv += ["--thermo-init", "scaled-random", "--thermo-match-cantera-norm",
                 "--thermo-linear-init-seed", str(ti["thermo_linear_init_seed"]),
                 "--thermo-init-temperature", repr(ti["reference_temperature"]),
                 "--thermo-init-phi", repr(ti["reference_phi"])]
    else:
        assert ti["thermo_init"] == "random"
        argv += ["--thermo-init", "random"]
    if stage1_from is not None:
        argv += ["--stage1-from", rel(stage1_from)]
    return argv


def check_parsed(job: str, argv: list[str], c: dict) -> dict:
    """Parse with the trainer's own parser; every scientific value must equal the baseline."""
    if job == "B0-FSA":
        import train_biodiesel as tr
        ns = tr.build_parser().parse_args(argv[1:])
        want = {"seed": c["seed"], "epochs": c["epochs"], "lr": c["learning_rate"],
                "hidden_dim": c["architecture"]["hidden_dim"], "num_basis": c["architecture"]["num_basis"],
                "n_mu": c["architecture"]["n_mu"], "use_base_act": c["architecture"]["use_base_act"],
                "input_scaling": c["normalization"]["input_scaling"], "noise_percent": None,
                "batch_size": None, "eval_every": 0, "init": "default",
                "solver_method": c["solver"]["method"], "rtol": c["solver"]["rtol"],
                "atol": c["solver"]["atol"], "sensitivity": "fsa"}
    else:
        import train_hydrogen as tr
        ns = tr.build_parser().parse_args(argv[1:])
        ti = c["thermo_init"]
        want = {"seed": c["seed"], "lr": c["learning_rate"], "hidden_dim": c["architecture"]["hidden_dim"],
                "num_basis": c["architecture"]["num_basis"], "n_mu": c["architecture"]["n_mu"],
                "use_base_act": c["architecture"]["use_base_act"],
                "stage1_epochs": c["epochs"]["stage1"], "stage2_epochs": c["epochs"]["stage2"],
                "pinn_stage1": c["pinn"]["stage1"], "pinn_stage2": c["pinn"]["stage2"],
                "alpha_pinn": c["pinn"]["alpha_pinn"], "input_scaling": c["normalization"]["input_scaling"],
                "solver_method": c["solver"]["method"], "rtol": c["solver"]["rtol"], "atol": c["solver"]["atol"],
                "stage1_temperature_points": c["stage1_temperature"]["n_points"],
                "thermo_init": ti["thermo_init"], "sensitivity": "fsa"}
        if ti["thermo_init"] == "scaled-random":
            want.update(thermo_match_cantera_norm=True, thermo_init_scale_factor=None,
                        thermo_linear_init_seed=ti["thermo_linear_init_seed"],
                        thermo_init_temperature=ti["reference_temperature"],
                        thermo_init_phi=ti["reference_phi"])
    got = {k: getattr(ns, k) for k in want}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise SystemExit(f"{job}: resolved command disagrees with the baseline: {bad}")
    return got


def jobs() -> dict:
    s1 = OUT["H_STAGE1-FSA"] / "checkpoint_stage1.pt"
    spec = {"B0-FSA": biodiesel_argv(cfg_of("B0-FSA")),
            "H_STAGE1-FSA": hydrogen_argv(cfg_of("H_STAGE1-FSA"), stage1_from=None),
            "H0-FSA": hydrogen_argv(cfg_of("H0-FSA"), stage1_from=s1),
            "Hnorm1-FSA": hydrogen_argv(cfg_of("Hnorm1-FSA"), stage1_from=s1)}
    for job, argv in spec.items():
        spec[job] = argv + ["--sensitivity", "fsa", "--checkpoint-every", "500",
                            "--run-dir", rel(OUT[job])]
    return spec


def resolve():
    if RESOLVED.exists():
        raise SystemExit(f"{RESOLVED} exists; resolved configurations are recorded once")
    spec, out = jobs(), {}
    for job, argv in spec.items():
        c = cfg_of(job)
        out[job] = {"baseline": rel(BASELINE[job]), "baseline_config": c,
                    "output_dir": rel(OUT[job]), "argv": argv,
                    "cwd": "repository root (as the baselines were launched)",
                    "env": {"OMP_NUM_THREADS": THREADS},
                    "parsed_and_checked": check_parsed(job, argv, c),
                    "intended_changes": ["sensitivity: direct_autograd -> fsa"] + (
                        ["stage1_from: archived direct-autograd Stage 1 -> H_STAGE1-FSA's "
                         "checkpoint_stage1.pt (the same file for H0-FSA and Hnorm1-FSA)"]
                        if job in ("H0-FSA", "Hnorm1-FSA") else [])}
    payload = {"created": utc_now(), "git_commit": git_commit(), "code_state": P.code_state(FSA / "code_patches"),
               "validation_summary": json.loads((FSA / "validation/summary.json").read_text()),
               "order": ["B0-FSA and H_STAGE1-FSA (parallel)", "H0-FSA and Hnorm1-FSA (parallel, "
                         "after H_STAGE1-FSA wrote checkpoint_stage1.pt)"],
               "jobs": out}
    RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED.write_text(json.dumps(payload, indent=2))
    print(f"-> {rel(RESOLVED)}")
    for job, v in out.items():
        print(f"{job}: OMP_NUM_THREADS={THREADS} python {' '.join(v['argv'])}")


MANIFEST = FSA / "training_code_manifest.json"
# Files training never imports; a change confined to them does not change the training code.
NON_TRAINING_PREFIXES = ("chemkan/scripts/figures/",)


def figure_only_change() -> str:
    """Accept a training-code hash change only if every changed file is figure code.

    Compares per-file hashes against ``training_code_manifest.json``, which was written
    while the code still matched the resolved hash. Anything else is refused.
    """
    if not MANIFEST.exists():
        raise SystemExit("training code changed since the configurations were resolved")
    old = json.loads(MANIFEST.read_text())["files"]
    new = P.training_code_manifest()
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    bad = [k for k in changed if not k.startswith(NON_TRAINING_PREFIXES)]
    if bad:
        raise SystemExit(f"training code changed since the configurations were resolved: {bad}")
    return ("training-code hash differs from the resolved one only through non-training "
            f"figure files: {changed}")


def launch(job: str, resume: bool = False):
    """Run the recorded command; ``resume`` appends only ``--resume`` (the trainer then
    refuses any scientific-configuration difference from the interrupted run)."""
    rec = json.loads(RESOLVED.read_text())
    note = None
    if rec["code_state"]["training_code_sha256"] != P.code_state()["training_code_sha256"]:
        note = figure_only_change()
    
    argv = rec["jobs"][job]["argv"] + (["--resume"] if resume else [])
    env = {**os.environ, **rec["jobs"][job]["env"]}
    OUT[job].mkdir(parents=True, exist_ok=True)
    with open(OUT[job] / "stdout.log", "a") as log:
        log.write(f"# {utc_now()} {'resume' if resume else 'launch'} {job}: {' '.join(argv)}\n")
        if note:
            log.write(f"# {note}\n")
        log.flush()
        r = subprocess.run([sys.executable, *argv], cwd=P.ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
    print(f"{job} exited {r.returncode}")
    sys.exit(r.returncode)


# ----------------------------------------------------------------------- verification
ALLOWED_ALWAYS = {"run_id", "git_commit", "created", "sensitivity_backend", "fsa", "dtype",
                  "torch_num_threads"}
# keys the biodiesel trainer added after B0 was run, with their legacy-equivalent values
B0_NEWER_KEYS = {"initialization": "default", "training_formulation": "full_rollout"}


def config_diff(job: str) -> dict:
    new = json.loads((OUT[job] / "config.json").read_text())
    old = cfg_of(job)
    diffs, notes = [], {}
    for k in sorted(set(new) | set(old)):
        if k in ALLOWED_ALWAYS or new.get(k) == old.get(k):
            continue
        if job == "B0-FSA" and k in B0_NEWER_KEYS and k not in old:
            ok = new[k] == B0_NEWER_KEYS[k]
        elif job == "B0-FSA" and k == "batching" and k not in old:
            b = new[k]
            ok = b["batch_size"] == 20 and b["legacy_full_batch_path"] and b["total_optimizer_steps"] == 10000
        elif k == "stage1_from" and old.get(k):
            a, b = new[k], old[k]
            ok = a["architecture"] == b["architecture"] and a["stage1_epochs"] == b["stage1_epochs"]
            notes["stage1_from"] = {"new_sha256": a["stage1_checkpoint_sha256"],
                                    "baseline_sha256": b["stage1_checkpoint_sha256"]}
        else:
            ok = False
        (notes.setdefault("explained", []) if ok else diffs).append(k)
    return {"sensitivity_backend": new["sensitivity_backend"], "unexplained_differences": diffs,
            **notes}


def _hashes(state: dict, prefix="") -> dict:
    return {f"{prefix}{k}": tensor_sha256(v) for k, v in state.items()}


def verify():
    out, ok = {}, True
    init = lambda job: json.loads((OUT[job] / "config.json").read_text())["fsa"]["initial_model"]
    archived_stage2_init = P.load(P.H_STAGE1_DIR / "checkpoint_final.pt")["model_state"]  # 0 Stage-2 epochs

    # B0: initial kinetic core == seed-0 construction (the B0 start; confirmed by the DA regression)
    torch.manual_seed(0)
    from chemkan.model import ChemKAN, KineticCore
    core = KineticCore(6, 4, 3, 2, use_base_act=False)
    rec = init("B0-FSA")
    out["B0-FSA"] = {"config": config_diff("B0-FSA"),
                     "initial_tensors_equal_seed0_construction":
                         rec["tensors"] == model_tensor_record(core)["tensors"],
                     "rbf_grids": rec["rbf_grids"]}

    # Stage 1: full initial model == seed-0 ChemKAN construction; its thermo tensors equal
    # the archived untrained Stage-2 tensors of the baseline pipeline
    torch.manual_seed(0)
    h = ChemKAN(9, 3, 4, 3, use_base_act=True)
    rec1 = init("H_STAGE1-FSA")
    thermo_hash = {k: tensor_sha256(v) for k, v in archived_stage2_init.items() if k.startswith("thermo.")}
    out["H_STAGE1-FSA"] = {"config": config_diff("H_STAGE1-FSA"),
                           "initial_tensors_equal_seed0_construction":
                               rec1["tensors"] == model_tensor_record(h)["tensors"],
                           "rbf_grids_equal_baseline_construction": rec1["rbf_grids"] == model_tensor_record(h)["rbf_grids"]}
    s1 = OUT["H_STAGE1-FSA"] / "checkpoint_stage1.pt"
    s1_ok = s1.exists()
    if s1_ok:
        s1_kin = P.load(s1)["kinetic_state"]
        s1_sha = P.checkpoint_sha256(s1) if hasattr(P, "checkpoint_sha256") else None
    for job in ("H0-FSA", "Hnorm1-FSA"):
        cfgp = OUT[job] / "config.json"
        if not cfgp.exists():
            out[job] = {"status": "not started"}
            continue
        rec2 = init(job)["tensors"]
        new_cfg = json.loads(cfgp.read_text())
        kin_ok = s1_ok and all(rec2[f"param:kinetic.{k}"]["sha256"] == tensor_sha256(v)
                               for k, v in s1_kin.items() if f"param:kinetic.{k}" in rec2)
        entry = lambda k: rec2.get(f"param:{k}") or rec2[f"buffer:{k}"]   # params and buffers
        cor_ok = all(entry(k)["sha256"] == v for k, v in thermo_hash.items()
                     if k.startswith("thermo.correction"))
        if job == "H0-FSA":
            lin_ok = rec2["param:thermo.linear.weight"]["sha256"] == thermo_hash["thermo.linear.weight"]
        else:
            vec = cfg_of(job)["thermo_init"]["resulting_vector"]
            lin_ok = (rec2["param:thermo.linear.weight"]["sha256"]
                      == tensor_sha256(torch.tensor(vec, dtype=torch.float32).reshape(1, 9)))
            lin_ok &= new_cfg["thermo_init"] == cfg_of(job)["thermo_init"]
        out[job] = {"config": config_diff(job),
                    "initial_kinetic_equals_fsa_stage1_checkpoint": kin_ok,
                    "initial_thermo_linear_equals_archived_baseline_init": lin_ok,
                    "initial_kan_cor_equals_archived_baseline_init": cor_ok,
                    "stage1_checkpoint_sha256": new_cfg["stage1_from"]["stage1_checkpoint_sha256"],
                    "stage1_from_sensitivity": new_cfg["fsa"]["stage1_from_sensitivity"]}
    if all(j in out and "config" in out[j] for j in ("H0-FSA", "Hnorm1-FSA")):
        out["shared_stage1_checkpoint"] = (out["H0-FSA"]["stage1_checkpoint_sha256"]
                                           == out["Hnorm1-FSA"]["stage1_checkpoint_sha256"])
    for job, v in out.items():
        if isinstance(v, dict) and "config" in v:
            good = (not v["config"]["unexplained_differences"]
                    and v["config"]["sensitivity_backend"] == "fsa"
                    and all(x for k, x in v.items() if k.startswith("initial_")))
            v["pass"] = good
            ok &= good
    payload = {"created": utc_now(), "status": "PASS" if ok else "FAIL", "jobs": out}
    (FSA / "validation/run_configuration_verification.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["resolve", "launch", "verify", "show"])
    ap.add_argument("job", nargs="?", choices=list(OUT))
    ap.add_argument("--resume", action="store_true",
                    help="launch: continue an interrupted run from its checkpoint_resume.pt")
    a = ap.parse_args()
    if a.action == "resolve":
        resolve()
    elif a.action == "launch":
        launch(a.job, resume=a.resume)
    elif a.action == "verify":
        verify()
    else:
        for job, argv in jobs().items():
            check_parsed(job, argv, cfg_of(job))
            print(f"{job}: {' '.join(argv)}")


if __name__ == "__main__":
    main()
