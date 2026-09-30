"""run_figures.py (figure-run plan, shards, skip/resume) and the DeepONet trainer's
--data-file option. No real training: the trainer check runs 2 epochs in pytest's temp dir."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "chemkan/scripts/author_repo_match"))
sys.path.insert(0, str(REPO / "deeponet"))

import run_figures as rf  # noqa: E402
import train_biodiesel_deeponet as trainer  # noqa: E402

V2 = REPO / "chemkan/data/generated/biodiesel_v2.npz"


# --------------------------------------------------------------------------- DeepONet --data-file

def _deeponet_run(tmp_path, monkeypatch, *extra):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_biodiesel_deeponet.py", "--epochs", "2",
                                      "--run-dir", str(run_dir), *extra])
    trainer.main()
    return json.loads((run_dir / "config.json").read_text())


def test_deeponet_reads_v2_with_data_file(tmp_path, monkeypatch):
    cfg = _deeponet_run(tmp_path, monkeypatch, "--data-file", str(V2), "--noise-percent", "5")
    assert cfg["dataset"] == "biodiesel_v2.npz (train split)"
    assert cfg["dataset_file"]["sha256"] == hashlib.sha256(V2.read_bytes()).hexdigest()
    assert cfg["dataset_file"]["normalization_keys"] == ["u_min_noise05", "u_max_noise05"]
    assert cfg["noise"]["source"] == "biodiesel_v2.npz train_states_noise05"


def test_deeponet_legacy_default_unchanged(tmp_path, monkeypatch):
    cfg = _deeponet_run(tmp_path, monkeypatch)
    assert cfg["dataset"] == "biodiesel.npz (train split)"
    assert "dataset_file" not in cfg


# --------------------------------------------------------------------------- run_figures

def test_plan_contents():
    runs = rf.plan()
    names = [r["name"] for r in runs]
    assert len(runs) == 27 and len(set(names)) == 27
    fig05 = [r for r in runs if r["name"].startswith("fig05")]
    assert len(fig05) == 16 and all(r["epochs"] == 10000 for r in fig05)
    ck4 = [r for r in runs if r["name"].startswith("fig04_chemkan")]
    assert [r["args"][-1] for r in ck4] == ["2", "3", "4", "9", "11"]
    assert all(r["epochs"] == 5000 for r in ck4)
    dn4 = [r for r in runs if r["name"].startswith("fig04_deeponet")]
    assert all(r["epochs"] == 50000 for r in dn4) and len(dn4) == 6
    for r in runs:
        cmd = rf.command(r, None)
        assert cmd[cmd.index("--data-file") + 1] == rf.DATA
        assert cmd[cmd.index("--seed") + 1] == "0"
        assert "--noise-percent" not in cmd or r["name"].startswith("fig05")
    ck = rf.command(runs[0], None)
    assert ck[ck.index("--solve-mode") + 1] == "batched" and ck[ck.index("--sensitivity") + 1] == "fsa"


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_shards_partition_and_balance(n):
    runs = rf.plan()
    groups = rf.shards(runs, n)
    flat = [r["name"] for g in groups for r in g]
    assert sorted(flat) == sorted(r["name"] for r in runs)             # each run exactly once
    loads = [sum(r["estimate_s"] for r in g) for g in groups]
    assert max(loads) - min(loads) <= max(r["estimate_s"] for r in runs)


def test_skip_resume_and_incomplete(tmp_path, monkeypatch):
    run = dict(rf.plan()[0], dir=tmp_path / "chemkan_noise00_seed0")
    assert rf.status(run) == "new"
    run["dir"].mkdir()
    (run["dir"] / "history.csv").write_text("x")
    assert rf.status(run) == "incomplete"
    (run["dir"] / "checkpoint_resume.pt").write_text("x")
    assert rf.status(run) == "resume" and rf.command(run, None)[-1] == "--resume"
    (run["dir"] / "checkpoint_final.pt").write_text("x")
    assert rf.status(run) == "done"
    monkeypatch.setattr(rf.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert "skipped" in rf.execute(run, None, dry_run=False)


def test_dry_run_starts_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rf, "OUT", tmp_path)                     # no finished runs here
    monkeypatch.setattr(rf.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(rf, "log_line", lambda *a: pytest.fail("must not log"))
    assert rf.main(["--dry-run", "--only", "fig04_deeponet_p308"]) == 0
    out = capsys.readouterr().out
    assert "--width 8 --trunk-hidden 7" in out and "biodiesel_v2.npz" in out
