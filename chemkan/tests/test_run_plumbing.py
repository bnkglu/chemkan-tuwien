"""Run-directory + resume + history plumbing (organization only, no model math).

Unit tests exercise RunManager/HistoryWriter directly. Two guarded end-to-end tests
run the real CLIs for a couple of epochs (skipped if the generated data is absent).
"""

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_DATA = _ROOT / "data" / "generated"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _run import HistoryWriter, RunManager, check_resume_config  # noqa: E402


# --------------------------------------------------------------------------
# RunManager / HistoryWriter units
# --------------------------------------------------------------------------

def test_run_dir_created_and_paths(tmp_path):
    rm = RunManager(tmp_path / "run", "biodiesel")
    rm.start()
    assert (tmp_path / "run").is_dir()                          # 1
    assert rm.final_path.name == "checkpoint_final.pt"          # 2
    rm.finish()


def test_config_written_with_run_id(tmp_path):
    rm = RunManager(tmp_path / "run", "hydrogen")
    rm.start()
    rm.write_config({"seed": 0, "sensitivity_backend": "direct_autograd"})
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())   # 3
    assert cfg["run_id"] == rm.run_id and cfg["seed"] == 0
    assert cfg["sensitivity_backend"] == "direct_autograd"
    rm.finish()


def test_history_schema_and_flush(tmp_path):
    h = HistoryWriter(tmp_path / "history.csv",
                      ["epoch", "total_loss", "species_mse", "pinn_loss", "elapsed_seconds"])
    h.on_epoch(0, 1.0, {"species_mse": torch.tensor(1.0), "pinn_loss": 0.0}, 0.1)
    with (tmp_path / "history.csv").open() as f:                # flushed immediately
        rows = list(csv.DictReader(f))
    assert rows[0]["epoch"] == "0" and "species_mse" in rows[0]  # 4
    h.close()


def test_history_resume_no_duplicate_epochs(tmp_path):
    cols = ["epoch", "total_loss", "elapsed_seconds"]
    h = HistoryWriter(tmp_path / "h.csv", cols)
    for e in range(4):
        h.on_epoch(e, float(e), {}, 0.0)
    h.close()
    # resume from epoch 2: rows for epochs 2,3 are dropped, then continued
    h2 = HistoryWriter(tmp_path / "h.csv", cols, resume_from=2)
    h2.on_epoch(2, 2.0, {}, 0.0)
    h2.close()
    with (tmp_path / "h.csv").open() as f:
        epochs = [r["epoch"] for r in csv.DictReader(f)]
    assert epochs == ["0", "1", "2"]                            # no dup, no lost history


def test_resume_write_and_load(tmp_path):
    rm = RunManager(tmp_path / "run", "hydrogen")
    rm.start()
    rm.save_resume({"stage": "stage1", "epoch": 3, "model_state": {}})  # 6
    loaded = rm.load_resume()                                            # 7
    assert loaded["stage"] == "stage1" and loaded["epoch"] == 3
    rm.finish()


def test_success_removes_resume(tmp_path):
    rm = RunManager(tmp_path / "run", "biodiesel")
    rm.start()
    rm.save_resume({"stage": "main", "epoch": 1})
    assert rm.resume_path.exists()
    rm.save_final({"model_state": {}})                          # 8
    assert rm.final_path.exists() and not rm.resume_path.exists()
    rm.finish()


def test_interrupted_preserves_resume(tmp_path):
    rm = RunManager(tmp_path / "run", "biodiesel")
    rm.start()
    rm.save_resume({"stage": "main", "epoch": 2})
    # simulate crash: no save_final called
    assert rm.resume_path.exists()                              # 9
    assert not rm.final_path.exists()


def test_overwrite_protection(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "checkpoint_final.pt").write_bytes(b"x")
    with pytest.raises(SystemExit):                             # 10
        RunManager(tmp_path / "run", "biodiesel").start()
    # --overwrite bypasses
    RunManager(tmp_path / "run", "biodiesel", overwrite=True).start()


def test_resume_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        RunManager(tmp_path / "run", "biodiesel", resume=True).start()


# --------------------------------------------------------------------------
# Fix 1: --overwrite starts a genuinely fresh run
# --------------------------------------------------------------------------

def test_overwrite_clears_all_prior_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # a completed previous run with the full artifact set
    (run_dir / "checkpoint_final.pt").write_bytes(b"old")
    (run_dir / "checkpoint_resume.pt").write_bytes(b"stale-resume")
    (run_dir / "metrics.json").write_text('{"test_mse": 1.0}')
    (run_dir / "config.json").write_text("{}")
    (run_dir / "run.log").write_text("old log\n")
    (run_dir / "predictions").mkdir()
    (run_dir / "predictions" / "test_predictions.npz").write_bytes(b"x")
    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "total_loss", "elapsed_seconds"])
        w.writeheader()
        for e in range(3):
            w.writerow({"epoch": e, "total_loss": e, "elapsed_seconds": 0})

    RunManager(run_dir, "biodiesel", overwrite=True).start()

    # old resume/metrics/predictions/history do NOT survive as if they were the new run's
    assert not (run_dir / "checkpoint_resume.pt").exists()
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "predictions").exists()
    assert not (run_dir / "history.csv").exists()

    # a fresh history is NOT appended to the old one
    rm = RunManager(run_dir, "biodiesel", overwrite=True)
    rm.start()
    h = rm.history("history.csv", ["epoch", "total_loss", "elapsed_seconds"], resume_from=0)
    h.on_epoch(0, 0.0, {}, 0.0)
    h.close()
    with (run_dir / "history.csv").open() as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["0"]   # fresh, not 0,1,2,0
    rm.finish()


# --------------------------------------------------------------------------
# Fix 2: --resume rejects scientific-configuration changes
# --------------------------------------------------------------------------

_BASE_CFG = {
    "chemical_system": "hydrogen", "seed": 0, "sensitivity_backend": "direct_autograd",
    "architecture": {"hidden_dim": 3, "num_basis": 5, "n_mu": 3, "use_base_act": False},
    "learning_rate": 2e-3,
    "solver": {"method": "tsit5", "rtol": 1e-6, "atol": 1e-8},
    "pinn": {"stage1": False, "stage2": True, "alpha_pinn": 1e-4},
    "normalization": {"input_scaling": "minmax", "stats": "train-only min-max"},
    "dataset": "hydrogen.npz (train split, 50 points, 35 conditions)", "noise": None,
    "stage1_temperature": {"source": "dense_cantera", "n_points": 20000,
                           "provider": "ObservedTemperature", "interpolation": "linear",
                           "cache_file": "hydrogen_temperature_20000.npz"},
    "device": "cpu", "epochs": {"stage1": 10000, "stage2": 10000}, "checkpoint_every": 500,
}


def _mut(**over):
    c = copy.deepcopy(_BASE_CFG)
    c.update(over)
    return c


def test_resume_same_config_ok():
    check_resume_config(_BASE_CFG, _mut())                     # no raise


def test_resume_rejects_stage1_temperature_change():
    new = _mut(stage1_temperature={**_BASE_CFG["stage1_temperature"],
                                   "source": "training_data", "n_points": 50})
    with pytest.raises(SystemExit):
        check_resume_config(_BASE_CFG, new)


def test_resume_rejects_solver_tolerance_change():
    with pytest.raises(SystemExit):
        check_resume_config(_BASE_CFG, _mut(solver={"method": "tsit5", "rtol": 1e-5, "atol": 1e-8}))


def test_resume_rejects_pinn_change():
    with pytest.raises(SystemExit):
        check_resume_config(_BASE_CFG, _mut(pinn={"stage1": True, "stage2": True, "alpha_pinn": 1e-4}))


def test_resume_rejects_architecture_change():
    with pytest.raises(SystemExit):
        check_resume_config(_BASE_CFG, _mut(architecture={**_BASE_CFG["architecture"], "hidden_dim": 4}))


def test_resume_rejects_seed_change():
    with pytest.raises(SystemExit):
        check_resume_config(_BASE_CFG, _mut(seed=1))


def test_resume_allows_runtime_only_changes():
    # device, checkpoint cadence, and epoch totals may differ without rejection
    new = _mut(device="cuda", checkpoint_every=1000, epochs={"stage1": 20000, "stage2": 10000})
    check_resume_config(_BASE_CFG, new)                        # no raise


def test_legacy_mode_is_inert():
    rm = RunManager(None, "biodiesel")
    assert rm.enabled is False and rm.run_id is None            # 11
    rm.start(); rm.write_config({}); rm.save_final({}); rm.finish()  # all no-ops


# --------------------------------------------------------------------------
# End-to-end CLI runs (guarded by generated data)
# --------------------------------------------------------------------------

def _run(cmd, **kw):
    return subprocess.run([sys.executable, *cmd], cwd=_SCRIPTS,
                          capture_output=True, text=True, **kw)


@pytest.mark.skipif(not (_DATA / "biodiesel.npz").exists(), reason="biodiesel.npz absent")
def test_biodiesel_train_eval_end_to_end(tmp_path):
    run_dir = tmp_path / "bd"
    r = _run(["train_biodiesel.py", "--epochs", "2", "--run-dir", str(run_dir)])
    assert r.returncode == 0, r.stderr
    assert (run_dir / "checkpoint_final.pt").exists()
    assert not (run_dir / "checkpoint_resume.pt").exists()      # removed on success (28)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["sensitivity_backend"] == "direct_autograd"     # 12
    with (run_dir / "history.csv").open() as f:
        assert csv.DictReader(f).fieldnames == ["epoch", "total_loss", "mse_loss", "elapsed_seconds"]
    ck = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert ck["run_id"] == "biodiesel/bd"
    assert ck["solver"]["sensitivity"] == "direct_autograd"    # 12

    e = _run(["evaluate_biodiesel.py", "--run-dir", str(run_dir), "--split", "test",
              "--metrics", "--save-predictions"])
    assert e.returncode == 0, e.stderr
    metrics = json.loads((run_dir / "metrics.json").read_text())  # 14
    assert "test_mse" in metrics and metrics["run_id"] == "biodiesel/bd"
    assert "evaluation_wall_time_s" in metrics and "inference_time_s" not in metrics  # Fix 3
    assert (run_dir / "predictions" / "test_predictions.npz").exists()  # 15


@pytest.mark.skipif(
    not ((_DATA / "hydrogen.npz").exists() and (_DATA / "hydrogen_temperature_20000.npz").exists()),
    reason="hydrogen data / 20k temperature cache absent")
def test_hydrogen_two_stage_histories_and_metadata(tmp_path):
    run_dir = tmp_path / "h2"
    r = _run(["train_hydrogen.py", "--stage1-temperature-source", "dense-cantera",
              "--stage1-temperature-points", "20000", "--stage1-epochs", "2",
              "--stage2-epochs", "2", "--run-dir", str(run_dir)])
    assert r.returncode == 0, r.stderr
    with (run_dir / "history_stage1.csv").open() as f:         # 5
        assert csv.DictReader(f).fieldnames == \
            ["epoch", "total_loss", "species_mse", "pinn_loss", "elapsed_seconds"]
    with (run_dir / "history_stage2.csv").open() as f:         # 5
        assert csv.DictReader(f).fieldnames == \
            ["epoch", "total_loss", "state_mse", "pinn_loss", "elapsed_seconds"]
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["stage1_temperature"]["source"] == "dense_cantera"   # 13
    ck = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert ck["stage1_temperature"]["source"] == "dense_cantera"    # 13
    assert ck["solver"]["sensitivity"] == "direct_autograd"
