"""Preflight and reuse checks for the pending biodiesel runs; never train."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "biodiesel_reproduction_training", ROOT / "chemkan/scripts/reproduction/biodiesel/_training.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def jobs_for(mode, output, *args):
    parsed = runner.build_parser().parse_args([mode, "--output-root", str(output), *args])
    return runner.make_jobs(parsed)[1]


def test_reference_plan_contains_only_the_14_affected_runs(tmp_path):
    jobs = jobs_for("deeponet", tmp_path / "reference")
    assert len(jobs) == 14
    assert [j["noise"] for j in jobs[:8]] == [0, 1, 2, 3, 5, 7, 10, 15]
    assert [j["width"] for j in jobs[8:]] == [3, 5, 6, 8, 10, 13]
    for job in jobs:
        config = job["config"]
        assert config["epochs"] == (10000 if job["group"] == "noise" else 50000)
        assert config["architecture"]["architecture_version"] == "reference_final_trunk_relu"
        assert config["optimizer"] == "Adam" and config["learning_rate"] == 1e-3
        assert config["seed"] == 0
        assert job["command"][1].endswith("train_biodiesel_deeponet.py")
        assert "--overwrite" not in job["command"] and "--resume" not in job["command"]
        assert runner.plan_action(job)[0] == "train"


def test_nmu2_plan_reuses_verified_5000_step_checkpoints(tmp_path):
    jobs = jobs_for("nmu2", tmp_path / "nmu2")
    assert [j["width"] for j in jobs] == [2, 3, 4, 10, 17]
    assert [j["config"]["parameter_count"] for j in jobs] == [78, 117, 156, 390, 663]
    assert all(j["config"]["architecture"]["n_mu"] == 2 for j in jobs)
    actions = [runner.plan_action(j) for j in jobs]
    assert [a for a, _ in actions] == ["train", "reuse", "reuse", "train", "train"]
    assert actions[2][1].name == "checkpoint_epoch_5000.pt"
    for override in (["--epochs", "10000"], ["--seed", "1"]):
        changed = jobs_for("nmu2", tmp_path / "changed", *override)
        assert all(j["reuse"] is None for j in changed)


def test_dry_run_writes_nothing_and_never_launches_training(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run launched a subprocess")
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    output = tmp_path / "must_not_exist"
    runner.main(["deeponet", "--dry-run", "--output-root", str(output)])
    assert not output.exists()
    assert "14 train, 0 resume, 0 skip, 0 reuse" in capsys.readouterr().out


def test_completed_reference_run_is_skipped_only_when_full_config_matches(tmp_path):
    job = jobs_for("deeponet", tmp_path / "reference")[0]
    directory = job["directory"]
    directory.mkdir(parents=True)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(job["config"]))
    # Synthetic completed-checkpoint fixture: all writes stay in pytest's temp dir.
    source = runner.DON / "noise/noise00_seed0/checkpoint_final.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["architecture"] = deepcopy(job["config"]["architecture"])
    checkpoint["model_state"] = runner.build(8, seed=0).state_dict()
    path = directory / "checkpoint_final.pt"
    torch.save(checkpoint, path)
    assert runner.plan_action(job) == ("skip", path)
    config = deepcopy(job["config"])
    config["learning_rate"] = 0.1
    config_path.write_text(json.dumps(config))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="learning_rate"):
        runner.plan_action(job)
    assert path.read_bytes() == before


def test_corrected_resume_and_legacy_resume_are_distinguished(tmp_path):
    job = jobs_for("deeponet", tmp_path / "reference")[0]
    directory = job["directory"]
    directory.mkdir(parents=True)
    config = deepcopy(job["config"])
    (directory / "config.json").write_text(json.dumps(config))
    path = directory / "checkpoint_resume.pt"
    model = runner.build(8, seed=0)
    state = {"config": config, "epoch": 1000, "model_state": model.state_dict(),
             "optimizer_state": torch.optim.Adam(model.parameters(), lr=1e-3).state_dict(),
             "rng_state": torch.get_rng_state()}
    torch.save(state, path)
    assert runner.plan_action(job)[0] == "resume"
    del config["architecture"]["architecture_version"]
    torch.save(state, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="architecture"):
        runner.plan_action(job)
    assert path.read_bytes() == before


def test_all_jobs_are_validated_before_first_training_command(tmp_path, monkeypatch):
    root = tmp_path / "reference"
    jobs = jobs_for("deeponet", root)
    corrupt = jobs[1]["directory"]
    corrupt.mkdir(parents=True)
    (corrupt / "run.log").write_text("An interrupted run without a checkpoint.\n")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("training started"))
    with pytest.raises(ValueError, match="without a resumable checkpoint"):
        runner.main(["deeponet", "--output-root", str(root)])
    assert not jobs[0]["directory"].exists()
    assert (corrupt / "run.log").read_text() == "An interrupted run without a checkpoint.\n"


def test_changed_dataset_manifest_blocks_reuse(tmp_path):
    (tmp_path / "manifest_all_seed0.json").write_text(json.dumps({"dataset_sha256": "different"}))
    with pytest.raises(SystemExit, match="dataset has changed"):
        runner.main(["deeponet", "--dry-run", "--output-root", str(tmp_path)])
