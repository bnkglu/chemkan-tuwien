"""Train the two pending biodiesel experiments; rendering follows in Notebook 07.

The existing run configs define the comparison settings. New runs retain those settings
except for the explicit experiment changes (final trunk ReLU, or fixed n_mu=2) and CLI
overrides. Completed results are reused only after config and checkpoint validation.
No legacy DeepONet weights enter training. No hydrogen commands are generated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
for directory in ("chemkan/src", "chemkan/scripts", "deeponet"):
    sys.path.insert(0, str(ROOT / directory))
try:
    import torch
except ImportError as exc:
    raise SystemExit("Set CHEMKAN_PYTHON to the project Python environment with torch installed.") from exc

from biodiesel_deeponet import REFERENCE_ARCHITECTURE_VERSION, architecture, build
from evaluate_biodiesel_deeponet import build_model
from evaluate_biodiesel import build_kinetic_core
from _data import load_biodiesel, available_noise_percents
import train_biodiesel_deeponet as don_trainer
import train_biodiesel as ck_trainer

RESULTS = ROOT / "results/reproduction"
CK = RESULTS / "chemkan/biodiesel"
DON = RESULTS / "baselines/deeponet/biodiesel"
STAMP_KEYS = {"run_id", "created", "git_commit", "device"}


def path_label(path):
    return os.path.relpath(path, ROOT)


def read_config(path):
    return json.loads(path.read_text())


def compare_configs(saved, expected, *, reused=False, snapshot=False):
    ignored = set(STAMP_KEYS)
    if reused:
        # Logging does not change optimizer steps; the clean replay has test history.
        ignored.update(("experiment_name", "in_training_evaluation"))
    if snapshot:
        ignored.add("epochs")  # checked against the checkpoint's actual step below
    for key in sorted((saved.keys() | expected.keys()) - ignored):
        if saved.get(key) != expected.get(key):
            raise ValueError(f"config mismatch on {key}: {saved.get(key)!r} != {expected.get(key)!r}")


def validate_checkpoint(path, job, *, snapshot=False):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = job["config"]
    if saved["architecture"] != expected["architecture"]:
        raise ValueError(f"{path}: checkpoint architecture differs from requested config")
    training = saved["training"]
    steps = saved.get("stage2_epoch") if snapshot else training["epochs"]
    if steps != expected["epochs"]:
        raise ValueError(f"{path}: expected {expected['epochs']} optimizer steps, found {steps}")
    for key, value in (("seed", expected["seed"]), ("learning_rate", expected["learning_rate"]),
                       ("noise_percent", (expected["noise"] or {}).get("percent"))):
        if training.get(key) != value:
            raise ValueError(f"{path}: checkpoint training.{key} differs")
    data = load_biodiesel(split="train")
    if saved["data"]["species"] != data["species"]:
        raise ValueError(f"{path}: species order differs from the canonical dataset")
    full_min = torch.cat([data["u_min"], data["T_const"].min().reshape(1)])
    full_max = torch.cat([data["u_max"], data["T_const"].max().reshape(1)])
    norm = saved["normalization"] if job["model"] == "DeepONet" else saved["input_scaling"]
    if not torch.equal(norm["u_min"], full_min) or not torch.equal(norm["u_max"], full_max):
        raise ValueError(f"{path}: checkpoint scaling statistics differ from the dataset")
    if job["model"] == "DeepONet":
        if norm["t_end_s"] != float(data["t"][-1]) or norm["output_space"] != "normalized species":
            raise ValueError(f"{path}: time/output scaling differs")
        model = build_model(saved, "cpu")
    else:
        if norm["method"] != "minmax" or saved["state_representation"] != "physical":
            raise ValueError(f"{path}: input/output scaling differs")
        if saved["solver"] != {**expected["solver"], "sensitivity": "direct_autograd"}:
            raise ValueError(f"{path}: solver differs")
        model = build_kinetic_core(saved, len(data["species"]), "cpu")
    if sum(p.numel() for p in model.parameters()) != expected["parameter_count"]:
        raise ValueError(f"{path}: parameter count differs")


def plan_action(job):
    """Read-only preflight. Never fall back to overwriting an incompatible result."""
    directory = job["directory"]
    final, resume = directory / "checkpoint_final.pt", directory / "checkpoint_resume.pt"
    if final.exists() or resume.exists():
        compare_configs(read_config(directory / "config.json"), job["config"])
        if final.exists():
            validate_checkpoint(final, job)
            return "skip", final
        state = torch.load(resume, map_location="cpu", weights_only=False)
        compare_configs(state["config"], job["config"])
        arch = state.get("architecture", state["config"]["architecture"])
        if arch != job["config"]["architecture"]:
            raise ValueError(f"{resume}: checkpoint architecture differs from its config")
        if not {"model_state", "optimizer_state", "rng_state"} <= state.keys():
            raise ValueError(f"{resume}: missing resumable training state")
        if job["model"] == "DeepONet":
            build_model({"architecture": arch, "model_state": state["model_state"]}, "cpu")
        else:
            build_kinetic_core({"architecture": arch, "model_state": state["model_state"]}, 6, "cpu")
        if not 0 <= state["epoch"] <= job["config"]["epochs"]:
            raise ValueError(f"{resume}: invalid resume epoch")
        return "resume", final
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory}: contains an incomplete run without a resumable checkpoint; "
                         "preserving it. Select another --output-root.")
    candidate = job.get("reuse")
    if candidate and candidate.exists():
        snapshot = candidate.name != "checkpoint_final.pt"
        compare_configs(read_config(candidate.parent / "config.json"), job["config"],
                        reused=True, snapshot=snapshot)
        validate_checkpoint(candidate, job, snapshot=snapshot)
        return "reuse", candidate
    return "train", final


def command(job, args):
    config = job["config"]
    if job["model"] == "DeepONet":
        cli = [str(ROOT / "deeponet/train_biodiesel_deeponet.py")]
        if job["group"] == "noise":
            cli += ["--noise-percent", str(job["noise"]), "--eval-every", "1"]
        else:
            cli += ["--width", str(job["width"]), "--experiment-name", "scaling"]
        parser = don_trainer.build_parser()
    else:
        cli = [str(ROOT / "chemkan/scripts/train_biodiesel.py"), "--hidden-dim", str(job["width"]),
               "--n-mu", "2", "--experiment-name", "scaling_nmu2"]
        parser = ck_trainer.build_parser()
    cli += ["--epochs", str(config["epochs"]), "--run-dir", str(job["directory"])]
    if args.seed != 0:
        cli += ["--seed", str(args.seed)]
    if args.device != "cpu":
        cli += ["--device", args.device]
    defaults = parser.parse_args(cli[1:])
    # A changed trainer default must not silently change this reproduction experiment.
    for key, value in (("lr", config["learning_rate"]), ("seed", config["seed"]),
                       ("epochs", config["epochs"]), ("noise_percent", job.get("noise"))):
        if getattr(defaults, key) != value:
            raise ValueError(f"trainer default {key} changed; review the reproduction config")
    if job["model"] == "ChemKAN":
        for key, value in (("num_basis", 3), ("use_base_act", False), ("input_scaling", "minmax"),
                           ("solver_method", "tsit5"), ("rtol", 1e-6), ("atol", 1e-8)):
            if getattr(defaults, key) != value:
                raise ValueError(f"trainer default {key} changed; review the reproduction config")
    elif defaults.width != job["width"]:
        raise ValueError("DeepONet default width changed; review the reproduction config")
    return [sys.executable, *cli]


def make_jobs(args):
    jobs = []
    if args.experiment == "deeponet":
        root = args.output_root or DON / REFERENCE_ARCHITECTURE_VERSION
        specs = []
        if args.only in ("all", "noise"):
            specs += [("noise", 8, n, args.noise_epochs) for n in (0, 1, 2, 3, 5, 7, 10, 15)]
        if args.only in ("all", "scaling"):
            specs += [("scaling", w, None, args.scaling_epochs) for w in (3, 5, 6, 8, 10, 13)]
        for group, width, noise, epochs in specs:
            name = f"noise{noise:02d}" if group == "noise" else f"w{width:02d}"
            config = read_config(DON / group / f"{name}_seed0/config.json")
            config["architecture"] = architecture(build(width, seed=args.seed))
            config.update(seed=args.seed, device=args.device, epochs=epochs)
            jobs.append(dict(model="DeepONet", group=group, width=width, noise=noise, config=config,
                             directory=root / group / f"{name}_seed{args.seed}"))
    else:
        root = args.output_root or CK / "scaling_nmu2"
        for width in (2, 3, 4, 10, 17):
            config = read_config(CK / "scaling/h03_seed0/config.json")
            config["architecture"]["hidden_dim"] = width
            config.update(parameter_count=39 * width, epochs=args.epochs, seed=args.seed,
                          device=args.device, experiment_name="scaling_nmu2")
            reuse = None
            if args.epochs == 5000 and args.seed == 0 and args.device == "cpu":
                if width == 3:
                    reuse = CK / "scaling/h03_seed0/checkpoint_final.pt"
                elif width == 4:
                    reuse = CK / "noise/clean_replay_seed0/checkpoint_epoch_5000.pt"
            jobs.append(dict(model="ChemKAN", group="scaling", width=width, config=config,
                             directory=root / f"h{width:02d}_seed{args.seed}", reuse=reuse))
    for job in jobs:
        for key in STAMP_KEYS - {"device"}:
            job["config"].pop(key, None)
        job["command"] = command(job, args)
    return root, jobs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="experiment", required=True)
    for mode in ("deeponet", "nmu2"):
        p = sub.add_parser(mode)
        p.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN") == "1")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        p.add_argument("--output-root", type=Path, help="use a separate root for altered budgets/settings")
        if mode == "deeponet":
            p.add_argument("--only", choices=("all", "noise", "scaling"), default="all")
            p.add_argument("--noise-epochs", type=int, default=10000)
            p.add_argument("--scaling-epochs", type=int, default=50000)
        else:
            p.add_argument("--epochs", type=int, default=5000,
                           help="Figure 4 uses 5000; 10000 is a separate experiment")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output_root:
        args.output_root = args.output_root.resolve()
    if any(value <= 0 for key, value in vars(args).items() if key.endswith("epochs")):
        raise SystemExit("Epoch budgets must be positive.")
    root, jobs = make_jobs(args)
    dataset_hash = hashlib.sha256((ROOT / "chemkan/data/generated/biodiesel.npz").read_bytes()).hexdigest()
    for existing_manifest in root.glob("manifest_*.json"):
        if read_config(existing_manifest)["dataset_sha256"] != dataset_hash:
            raise SystemExit(f"{existing_manifest}: dataset has changed; use a separate --output-root.")
    if args.experiment == "deeponet":
        missing = {j["noise"] for j in jobs if j["noise"] is not None} - set(available_noise_percents())
        if missing:
            raise SystemExit(f"Missing noise levels {sorted(missing)} in biodiesel.npz; data was not changed.")
    # Validate the entire plan before starting any job.
    plan = [(job, *plan_action(job)) for job in jobs]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in ("chemkan/src", "chemkan/scripts", "deeponet"))
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]
    manifest_jobs = []
    for job, action, checkpoint in plan:
        cmd = job["command"] + (["--resume"] if action == "resume" else [])
        print(f"{action:6s} {job['model']} width={job['width']} epochs={job['config']['epochs']} "
              f"{path_label(checkpoint)}", flush=True)
        if action in ("train", "resume"):
            print("       " + shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                compare_configs(read_config(job["directory"] / "config.json"), job["config"])
                validate_checkpoint(checkpoint, job)
        manifest_jobs.append(dict(model=job["model"], group=job["group"], width=job["width"],
                                  noise_percent=job.get("noise"), epochs=job["config"]["epochs"],
                                  parameters=job["config"]["parameter_count"], action=action,
                                  checkpoint=path_label(checkpoint), config=deepcopy(job["config"])))
    print(f"{len(jobs)} points: " + ", ".join(f"{sum(a == kind for _, a, _ in plan)} {kind}"
                                              for kind in ("train", "resume", "skip", "reuse")))
    if args.dry_run:
        print("Dry run only. No files written and no training started.")
        return
    for item in manifest_jobs:
        item["checkpoint_sha256"] = hashlib.sha256((ROOT / item["checkpoint"]).read_bytes()).hexdigest()
    manifest = dict(experiment=args.experiment, seed=args.seed,
                    dataset="chemkan/data/generated/biodiesel.npz",
                    dataset_sha256=dataset_hash,
                    jobs=manifest_jobs)
    root.mkdir(parents=True, exist_ok=True)
    suffix = args.only if args.experiment == "deeponet" else "scaling"
    path = root / f"manifest_{suffix}_seed{args.seed}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {path_label(path)}. Ready for Notebook 07 evaluation; existing figures remain unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"Preflight/validation failed: {exc}. No existing runs were overwritten.") from exc
