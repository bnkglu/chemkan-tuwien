"""Post-hoc helpers for the ReLU reachability batch (values at epoch e = model after e updates:
history row e for e < run length, the final checkpoint at the run length)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "deeponet"))
import biodiesel_deeponet_repro as rp    # noqa: E402
from biodiesel_deeponet import build    # noqa: E402

DIAG = Path(__file__).resolve().parent.parent
DATA = rp.load_author_data(noise_percent=0)
STATS = rp.author_global_stats(DATA)


def branch_tensor(mode="normalized"):
    return torch.as_tensor(rp.branch_input_mode(DATA["train_branch_raw"], mode,
                                                "author_global_normalized_states", STATS),
                           dtype=torch.float32)


def load_model(path, seed):
    m = build(8, seed=seed)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"])
    return m


def history(run):
    rows = list(csv.DictReader((Path(run) / "history.csv").open()))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("train_mean_mse", "test_mean_mse",
                                                                 "train_eq18_obs", "test_eq18_obs")}


def at(run, epoch, key="train_mean_mse"):
    """Metric of the model after ``epoch`` updates."""
    h = history(run)
    n = len(h[key])
    if epoch < n:
        return float(h[key][epoch])
    if epoch == n:
        m = json.loads((Path(run) / "metrics.json").read_text())["final"]
        split, metric = key.split("_", 1)
        return float(m[split][metric])
    raise ValueError(f"{run} has only {n} epochs")


def trunk_diag(model, mode="normalized"):
    td = torch.linspace(0, 30, 301)
    dead = rp.dead_dimensions(rp.trunk_activations(model, td))
    ub = rp.useful_breakpoints(model, branch_tensor(mode), 30.0)
    return {"dead1": dead["relu0"], "dead2": dead["relu1"], "n_useful": ub["n_useful"],
            "useful_s": ub["useful_positions_s"], "n_interior": ub["n_interior"],
            "layer1": ub["layer1"], "layer2": ub["layer2"]}
