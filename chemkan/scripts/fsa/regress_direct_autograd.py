r"""Direct-autograd regression through the CURRENT training interface (DIAGNOSTIC).

After FSA was added, re-run the first epochs of each baseline -- B0, H_STAGE1, H0 and
Hnorm1 -- with direct autograd and the baseline's saved settings, then compare the new
history rows with the archived ones. Any drift in model dynamics, loss, normalization,
initialization, data or optimizer behavior shows up in these rows. Historical artifacts
are only read; the short runs go to gitignored ``*smoke*`` directories.

Writes ``results/experiments/fsa/validation/da_regression.json``.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _problems as P                                               # noqa: E402
from _run import git_commit, utc_now                                # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1]
OUT = P.RESULTS / "experiments/fsa/validation"
WORK = P.RESULTS / "experiments/fsa/smoke/da_regression"
PROBE_EPOCHS = "0,1,2,5,10,20,50,100,200,500,1000,2000,3000,5000,7500,10000"
REL_TOL = 1e-5            # allowed relative drift per logged value (bit-identity reported)
TIMING = {"epoch", "elapsed_seconds", "epoch_wall_time_s"}


def thermo_args(cfg: dict) -> list[str]:
    ti = cfg["thermo_init"]
    if ti["thermo_init"] == "random":
        return []
    assert ti["thermo_init"] == "scaled-random" and ti["scaling_mode"] == "match-cantera-norm"
    return ["--thermo-init", "scaled-random", "--thermo-match-cantera-norm",
            "--thermo-linear-init-seed", str(ti["thermo_linear_init_seed"]),
            "--thermo-init-temperature", str(ti["reference_temperature"]),
            "--thermo-init-phi", str(ti["reference_phi"])]


def jobs() -> list[dict]:
    b0 = json.loads((P.B0_DIR / "config.json").read_text())
    h1 = json.loads((P.H_STAGE1_DIR / "config.json").read_text())
    h0 = json.loads((P.H0_DIR / "config.json").read_text())
    hn = json.loads((P.HNORM1_DIR / "config.json").read_text())
    common_h = ["--seed", "0", "--experiment-name", "base_on_n4", "--count-nfe",
                "--stage1-temperature-source", "dense-cantera",
                "--stage1-temperature-points", "20000"]
    return [
        {"name": "B0", "baseline": P.B0_DIR, "history": "history.csv", "rows": 20,
         "argv": ["train_biodiesel.py", "--seed", str(b0["seed"]), "--epochs", "20",
                  "--experiment-name", b0["experiment_name"]]},
        {"name": "H_STAGE1", "baseline": P.H_STAGE1_DIR, "history": "history_stage1.csv",
         "rows": 20,
         "argv": ["train_hydrogen.py", *common_h, "--stage1-epochs", "20",
                  "--stage2-epochs", "0", *thermo_args(h1)]},
        {"name": "H0", "baseline": P.H0_DIR, "history": "history_stage2.csv", "rows": 5,
         "argv": ["train_hydrogen.py", *common_h, "--stage1-from",
                  str(P.ROOT / h0["stage1_from"]["path"]), "--stage2-epochs", "5",
                  "--stage2-probe", "--stage2-probe-epochs", PROBE_EPOCHS, *thermo_args(h0)]},
        {"name": "Hnorm1", "baseline": P.HNORM1_DIR, "history": "history_stage2.csv", "rows": 5,
         "argv": ["train_hydrogen.py", *common_h, "--stage1-from",
                  str(P.ROOT / hn["stage1_from"]["path"]), "--stage2-epochs", "5",
                  "--stage2-probe", "--stage2-probe-epochs", PROBE_EPOCHS, *thermo_args(hn)]},
    ]


def read_rows(path: Path, n: int) -> list[dict]:
    with path.open() as f:
        return [r for _, r in zip(range(n), csv.DictReader(f))]


def compare_rows(new: list[dict], old: list[dict]) -> dict:
    worst, bitwise, cols = 0.0, True, set()
    for a, b in zip(new, old):
        for k in a:
            if k in TIMING or k not in b:
                continue
            cols.add(k)
            x, y = float(a[k]), float(b[k])
            if a[k] != b[k]:
                bitwise = False
            if k == "nfe":
                worst = max(worst, abs(x - y) / max(abs(y), 1.0))
            elif not (x == y or (math.isnan(x) and math.isnan(y))):
                worst = max(worst, abs(x - y) / max(abs(y), 1e-30))
    return {"rows_compared": len(new), "columns": sorted(cols), "max_relative_diff": worst,
            "bit_identical": bitwise}


# Keys the biodiesel trainer started recording after B0 was run; explained only when they
# hold the values equivalent to B0's legacy full-batch, default-init, full-trajectory setup.
NEWER_SCHEMA = {"initialization": lambda v: v == "default",
                "training_formulation": lambda v: v == "full_rollout",
                "batching": lambda v: v["batch_size"] == 20 and v["legacy_full_batch_path"]}


def compare_config(new: dict, old: dict) -> dict:
    allowed = {"run_id", "git_commit", "created", "epochs", "stage1_from"}
    diffs, explained = [], []
    for k in sorted(set(new) | set(old)):
        if k in allowed or new.get(k) == old.get(k):
            continue
        if k not in old and k in NEWER_SCHEMA and NEWER_SCHEMA[k](new[k]):
            explained.append(k)
        else:
            diffs.append(k)
    same_stage1 = None
    if old.get("stage1_from"):
        same_stage1 = (new["stage1_from"]["stage1_checkpoint_sha256"]
                       == old["stage1_from"]["stage1_checkpoint_sha256"])
    return {"differing_keys": diffs, "explained_newer_schema_keys": explained,
            "stage1_checkpoint_sha256_equal": same_stage1}


def main():
    results, ok = [], True
    for job in jobs():
        run_dir = WORK / job["name"]
        argv = [sys.executable, *job["argv"], "--run-dir", str(run_dir), "--overwrite"]
        print(f"== {job['name']}: {' '.join(job['argv'])}", flush=True)
        r = subprocess.run(argv, cwd=SCRIPTS, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-3000:])
            results.append({"name": job["name"], "status": "FAIL", "error": r.stderr[-2000:]})
            ok = False
            continue
        new = read_rows(run_dir / job["history"], job["rows"])
        old = read_rows(job["baseline"] / job["history"], job["rows"])
        rows = compare_rows(new, old)
        cfg = compare_config(json.loads((run_dir / "config.json").read_text()),
                             json.loads((job["baseline"] / "config.json").read_text()))
        good = (rows["rows_compared"] == job["rows"] and rows["max_relative_diff"] <= REL_TOL
                and not cfg["differing_keys"] and cfg["stage1_checkpoint_sha256_equal"] in (None, True))
        ok &= good
        results.append({"name": job["name"], "baseline": str(job["baseline"].relative_to(P.ROOT)),
                        "argv": job["argv"], "history": job["history"], **rows,
                        "config_comparison": cfg, "status": "PASS" if good else "FAIL"})
        print(f"   rows {rows['rows_compared']} max rel {rows['max_relative_diff']:.2e} "
              f"bitwise {rows['bit_identical']} config diffs {cfg['differing_keys']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"created": utc_now(), "git_commit": git_commit(), "code_state": P.code_state(),
               "status": "PASS" if ok else "FAIL", "relative_tolerance": REL_TOL,
               "note": "timing columns are excluded; nfe compared exactly via relative diff",
               "results": results}
    (OUT / "da_regression.json").write_text(json.dumps(payload, indent=2))
    print(f"-> {(OUT / 'da_regression.json').relative_to(P.ROOT)} [{payload['status']}]")


if __name__ == "__main__":
    main()
