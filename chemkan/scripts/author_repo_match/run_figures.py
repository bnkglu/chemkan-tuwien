r"""Figure runs on ``biodiesel_v2.npz`` (Figs. 4 and 5; Figs. 3 and 6 are evaluated from the
Fig. 5 runs). Results: ``results/experiments/biodiesel/author_repo_match/figures/``.

    fig05/  noise 0, 1, 2, 3, 5, 7, 10, 15 %, 10,000 epochs each:
            ChemKAN  train_author_repo_match.py --data canonical --hidden 4
                     --solve-mode batched --sensitivity fsa
            DeepONet train_biodiesel_deeponet.py --width 8 --eval-every 1   (308 parameters)
    fig04/  noise-free:
            ChemKAN  --hidden 2, 3, 4, 9, 11 (same settings as fig05), 5,000 epochs
            DeepONet FIG4_ARCHITECTURES 78 ... 456 (--width w --trunk-hidden q), 50,000 epochs

Every run uses ``--data-file chemkan/data/generated/biodiesel_v2.npz`` and seed 0. A run
whose ``checkpoint_final.pt`` exists is skipped; one with ``checkpoint_resume.pt`` is
resumed. A directory holding neither (a run stopped before its first checkpoint) is moved
aside to ``<dir>.incomplete-<time>`` and started again -- nothing is deleted.

    python run_figures.py --list                     # runs, status, estimated time
    python run_figures.py --dry-run                  # print commands only
    python run_figures.py --only fig05_chemkan_noise05 fig04_deeponet_p456
    python run_figures.py --shard 1/3                # first of 3 time-balanced groups
    python run_figures.py --jobs 2                   # 2 parallel runs, 1 torch thread each

Start and finish of every run are appended to ``figures/run_log_<hostname>.txt``.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "deeponet"))
from biodiesel_deeponet import FIG4_ARCHITECTURES  # noqa: E402

OUT = ROOT / "results/experiments/biodiesel/author_repo_match/figures"
DATA = "chemkan/data/generated/biodiesel_v2.npz"
PY = sys.executable
CHEMKAN = "chemkan/scripts/author_repo_match/train_author_repo_match.py"
DEEPONET = "deeponet/train_biodiesel_deeponet.py"
NOISE = (0, 1, 2, 3, 5, 7, 10, 15)
FIG04_HIDDEN = (2, 3, 4, 9, 11)

# Seconds per epoch, measured on this machine (torch default threads):
#   ChemKAN batched FSA, H = 4: run_batched_fsa_juliainit, 1,588 s / 10,000 epochs; other
#   widths scaled by H/4 (the FSA sensitivities grow with the 39*H parameters).
#   DeepONet: slowest legacy reference_final_trunk_relu run (noise10: 42 s / 10,000 epochs).
CHEMKAN_S_PER_EPOCH_H4 = 1588.36 / 10000
DEEPONET_S_PER_EPOCH = 42.0 / 10000
ESTIMATE_BASIS = ("ChemKAN: run_batched_fsa_juliainit (0.159 s/epoch at H = 4, noise-free "
                  "author data) scaled by H/4; DeepONet: slowest legacy reference run "
                  "(0.0042 s/epoch). Measured with torch default threads; noisy data and "
                  "--jobs > 1 (one thread per run) can be slower.")


def plan() -> list[dict]:
    runs = []
    for p in NOISE:
        runs.append({"name": f"fig05_chemkan_noise{p:02d}", "dir": OUT / f"fig05/chemkan_noise{p:02d}_seed0",
                     "model": "chemkan", "epochs": 10000,
                     "args": ["--data", "canonical", "--noise-percent", str(p), "--hidden", "4"],
                     "estimate_s": 10000 * CHEMKAN_S_PER_EPOCH_H4})
        runs.append({"name": f"fig05_deeponet_noise{p:02d}", "dir": OUT / f"fig05/deeponet_noise{p:02d}_seed0",
                     "model": "deeponet", "epochs": 10000,
                     "args": ["--noise-percent", str(p), "--width", "8", "--eval-every", "1",
                              "--experiment-name", "fig05"],
                     "estimate_s": 10000 * DEEPONET_S_PER_EPOCH})
    for h in FIG04_HIDDEN:
        runs.append({"name": f"fig04_chemkan_h{h:02d}", "dir": OUT / f"fig04/chemkan_h{h:02d}_seed0",
                     "model": "chemkan", "epochs": 5000,
                     "args": ["--data", "canonical", "--hidden", str(h)],
                     "estimate_s": 5000 * CHEMKAN_S_PER_EPOCH_H4 * h / 4})
    for n, (w, q, _p) in sorted(FIG4_ARCHITECTURES.items()):
        runs.append({"name": f"fig04_deeponet_p{n:03d}", "dir": OUT / f"fig04/deeponet_p{n:03d}_seed0",
                     "model": "deeponet", "epochs": 50000,
                     "args": ["--width", str(w), "--trunk-hidden", str(q),
                              "--experiment-name", "fig04"],
                     "estimate_s": 50000 * DEEPONET_S_PER_EPOCH})
    return runs


def status(run: dict) -> str:
    d = run["dir"]
    if (d / "checkpoint_final.pt").exists():
        return "done"
    if (d / "checkpoint_resume.pt").exists():
        return "resume"
    if d.exists() and any(d.iterdir()):
        return "incomplete"
    return "new"


def _rel(path: Path) -> str:
    """Repository-relative when inside the repository (commands run with cwd = ROOT)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def command(run: dict, threads: int | None) -> list[str]:
    common = ["--data-file", DATA, "--seed", "0", "--epochs", str(run["epochs"]),
              "--run-dir", _rel(run["dir"])]
    if run["model"] == "chemkan":
        cmd = [PY, CHEMKAN, *run["args"], "--solve-mode", "batched", "--sensitivity", "fsa",
               *common]
        if threads is not None:
            cmd += ["--threads", str(threads)]
    else:
        cmd = [PY, DEEPONET, *run["args"], *common]
    if status(run) == "resume":
        cmd.append("--resume")
    return cmd


def shards(runs: list[dict], n: int) -> list[list[dict]]:
    """Greedy longest-first balance by estimated time, over ALL planned runs, so every
    machine computes the same partition whatever has finished already."""
    groups = [[] for _ in range(n)]
    loads = [0.0] * n
    for run in sorted(runs, key=lambda r: (-r["estimate_s"], r["name"])):
        k = loads.index(min(loads))
        groups[k].append(run)
        loads[k] += run["estimate_s"]
    return groups


def fmt(seconds: float) -> str:
    return f"{seconds / 60:6.1f} min" if seconds < 3600 else f"{seconds / 3600:6.2f} h"


def log_line(text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(OUT / f"run_log_{socket.gethostname()}.txt", "a") as f:
        f.write(f"{stamp} {text}\n")


def execute(run: dict, threads: int | None, dry_run: bool) -> str:
    st = status(run)
    if st == "done":
        return f"{run['name']}: skipped (checkpoint_final.pt exists)"
    if st == "incomplete" and not dry_run:
        aside = run["dir"].with_name(run["dir"].name + ".incomplete-"
                                     + datetime.now().strftime("%Y%m%dT%H%M%S"))
        run["dir"].rename(aside)
        log_line(f"MOVED {run['name']} partial directory (no checkpoint) -> {aside.name}")
    cmd = command(run, threads)
    if dry_run:
        return " ".join(cmd)
    run["dir"].parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if threads == 1:
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_line(f"START {run['name']} {' '.join(cmd[1:])}")
    t0 = time.perf_counter()
    with open(run["dir"].parent / f"{run['dir'].name}.console.log", "a") as console:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT).returncode
    result = "ok" if rc == 0 and (run["dir"] / "checkpoint_final.pt").exists() else f"FAILED rc={rc}"
    log_line(f"FINISH {run['name']} {result} {time.perf_counter() - t0:.0f}s")
    return f"{run['name']}: {result}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print runs, status and estimates")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--shard", metavar="K/N", help="run the K-th of N time-balanced groups")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel runs in this session; with > 1 each gets one torch thread")
    args = ap.parse_args(argv)

    runs = plan()
    if args.only:
        known = {r["name"] for r in runs}
        unknown = sorted(set(args.only) - known)
        if unknown:
            raise SystemExit(f"unknown run names: {unknown}; see --list")
        runs = [r for r in runs if r["name"] in args.only]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= k <= n:
            raise SystemExit("--shard K/N needs 1 <= K <= N")
        runs = shards(runs, n)[k - 1]
        print(f"shard {k}/{n}: {len(runs)} runs, estimated {fmt(sum(r['estimate_s'] for r in runs)).strip()}")
        for r in runs:
            print(f"  {r['name']:28s} {fmt(r['estimate_s'])}  [{status(r)}]")

    if args.list:
        print(f"{'run':28s} {'epochs':>7s} {'estimate':>11s}  status")
        for r in runs:
            print(f"{r['name']:28s} {r['epochs']:7d} {fmt(r['estimate_s'])}  {status(r)}")
        todo = [r for r in runs if status(r) != "done"]
        print(f"\n{len(runs)} runs, {len(todo)} not done; estimated serial time of those: "
              f"{fmt(sum(r['estimate_s'] for r in todo)).strip()}\nbasis: {ESTIMATE_BASIS}")
        return 0

    threads = 1 if args.jobs > 1 else None
    if args.jobs > 1 and not args.dry_run:
        print(f"note: {args.jobs} runs in parallel, one torch thread each")
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for line in pool.map(lambda r: execute(r, threads, args.dry_run), runs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
