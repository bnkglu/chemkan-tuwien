"""chemkan/scripts/train.py: the one-file example computes what the biodiesel trainer does
(with the trainer's default RHS divisor, 50), and trains on other .npz datasets."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "chemkan/scripts"


def test_train_example_equals_the_biodiesel_trainer(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO / "chemkan/src")}
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=REPO, env=env,  # noqa: E731
                                    capture_output=True, text=True, check=True)
    run(SCRIPTS / "train.py", "--epochs", "3", "--out-dir", tmp_path / "example")
    run(SCRIPTS / "train_biodiesel.py", "--data", "canonical", "--data-file",
        REPO / "chemkan/data/generated/biodiesel_v2.npz", "--hidden", "4", "--solve-mode", "batched",
        "--sensitivity", "direct_autograd", "--seed", "0", "--epochs", "3",
        "--run-dir", tmp_path / "trainer")
    a = torch.load(tmp_path / "example/chemkan_kinetic.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "trainer/checkpoint_final.pt", weights_only=False)["model_state"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    rhs = json.loads((tmp_path / "trainer/config.json").read_text())["rhs_divisor"]
    assert rhs["value"] == 50.0 and rhs["source"].startswith("default")   # the trainer's default


# --------------------------------------------------------------------------- other datasets

def _train_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("train")


def _abc_dataset(path, n_train=6, n_test=2, n_t=20):
    """A -> B -> C, first order, Arrhenius rates, one constant temperature per trajectory."""
    t = np.linspace(0.0, 10.0, n_t)
    temps = np.linspace(300.0, 340.0, n_train + n_test)
    states = []
    for T in temps:
        k1, k2 = 0.6 * np.exp(-1500 * (1 / T - 1 / 320)), 0.25 * np.exp(-2500 * (1 / T - 1 / 320))
        a = np.exp(-k1 * t)
        b = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
        states.append(np.stack([a, b, 1.0 - a - b], axis=-1))
    states = np.array(states)
    np.savez(path, t=t, train_states=states[:n_train], test_states=states[n_train:],
             train_T=temps[:n_train], test_T=temps[n_train:])
    return path


def test_train_example_on_a_three_species_system(tmp_path):
    train = _train_module()
    out = train.main({**train.SETTINGS, "data_file": _abc_dataset(tmp_path / "abc.npz"),
                      "epochs": 25, "hidden": 2, "rhs_divisor": 10.0, "out_dir": tmp_path / "out"})
    ckpt = torch.load(out["checkpoint"], weights_only=False)
    assert ckpt["species_dim"] == 3
    core = train.build_core(2, species_dim=3)
    core.load_state_dict(ckpt["model_state"])
    assert sum(p.numel() for p in core.parameters()) == 3 * 2 * (2 * 3 + 1)   # 3 H (2m + 1)
    losses = out["train_loss"]
    assert len(losses) == 25 and all(np.isfinite(losses)) and losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("change, message", [
    (lambda d: d.pop("test_T"), "missing arrays"),
    (lambda d: d.update(test_states=d["test_states"][:, :-1]), "must share"),
    (lambda d: d.update(train_T=d["train_T"][:-1]), "one temperature per trajectory"),
    (lambda d: d.update(t=d["t"][::-1].copy()), "strictly increasing"),
    (lambda d: d["train_states"].__setitem__((slice(None), slice(None), 2), 0.5), "constant"),
])
def test_load_npz_rejects_malformed_files(tmp_path, change, message):
    train = _train_module()
    d = dict(np.load(_abc_dataset(tmp_path / "abc.npz")))
    change(d)
    np.savez(tmp_path / "bad.npz", **d)
    with pytest.raises(ValueError, match=message):
        train.load_npz(tmp_path / "bad.npz")
