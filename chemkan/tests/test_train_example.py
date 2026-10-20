"""chemkan/scripts/train.py: the one-file example computes what the biodiesel trainer does
(with the trainer's default RHS divisor, 50)."""

import json
import os
import subprocess
import sys
from pathlib import Path

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
    a = torch.load(tmp_path / "example/chemkan_biodiesel.pt", weights_only=False)["model_state"]
    b = torch.load(tmp_path / "trainer/checkpoint_final.pt", weights_only=False)["model_state"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    rhs = json.loads((tmp_path / "trainer/config.json").read_text())["rhs_divisor"]
    assert rhs["value"] == 50.0 and rhs["source"].startswith("default")   # the trainer's default
