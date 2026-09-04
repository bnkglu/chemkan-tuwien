"""Prediction-artifact provenance + compatibility (organization plumbing)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _predictions import (  # noqa: E402
    IncompatiblePredictionError,
    canonical_arch,
    checkpoint_sha256,
    load_compatible_predictions,
    load_predictions,
    save_predictions,
    validate,
)

ARCH = {"hidden_dim": 3, "num_basis": 5, "n_mu": 3, "use_base_act": False}


def _fake_checkpoint(tmp_path: Path) -> Path:
    p = tmp_path / "checkpoint_final.pt"
    p.write_bytes(b"not-a-real-checkpoint-but-hashable")
    return p


def _save(tmp_path, *, run_id="chemkan/hydrogen/main/direct_autograd_seed0", sha="abc123",
          force=False, name="test_predictions.npz"):
    return save_predictions(
        tmp_path / "predictions" / name, force=force,
        run_id=run_id, checkpoint_sha256=sha, architecture=ARCH,
        predictions=np.zeros((5, 3, 9)), reference=np.ones((5, 3, 9)),
        t=np.linspace(0, 6e-4, 5), initial_conditions=np.ones((3, 10)),
        species=["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2"],
        u_min=np.zeros(10), u_max=np.ones(10),
        metric_convention="Eq. 18 species MSE: mean over normalized species, over times.",
        eval_config={"split": "test", "solver": {"method": "tsit5"}})


def test_save_and_load_records_provenance(tmp_path):
    _save(tmp_path)
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    assert art["run_id"] == "chemkan/hydrogen/main/direct_autograd_seed0"   # 16
    assert art["checkpoint_sha256"] == "abc123"                            # 17
    assert art["architecture"] == ARCH                                     # 18
    assert art["predictions"].shape == (5, 3, 9)
    assert art["reference"].shape == (5, 3, 9)


def test_actual_normalization_arrays_preserved(tmp_path):
    _save(tmp_path)
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    assert np.allclose(art["u_min"], np.zeros(10))     # 25: actual arrays, not a string label
    assert np.allclose(art["u_max"], np.ones(10))


def test_reference_is_stored_ground_truth(tmp_path):
    _save(tmp_path)
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    assert np.allclose(art["reference"], np.ones((5, 3, 9)))   # 26: copied truth


def test_metadata_needs_no_pickle(tmp_path):
    # load_predictions uses allow_pickle=False; a direct np.load must also succeed.
    _save(tmp_path)
    with np.load(tmp_path / "predictions" / "test_predictions.npz", allow_pickle=False) as d:
        assert "provenance" in d.files                 # 27: JSON scalar, not a pickled object


def test_validate_accepts_matching_checkpoint(tmp_path):
    ckpt = _fake_checkpoint(tmp_path)
    sha = checkpoint_sha256(ckpt)
    _save(tmp_path, sha=sha)
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    ok, _ = validate(art, run_id="chemkan/hydrogen/main/direct_autograd_seed0",
                     architecture=ARCH, checkpoint_sha256=sha)
    assert ok                                          # 19


def test_validate_rejects_sha_mismatch(tmp_path):
    _save(tmp_path, sha="one")
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    ok, reason = validate(art, run_id="chemkan/hydrogen/main/direct_autograd_seed0",
                          architecture=ARCH, checkpoint_sha256="two")
    assert not ok and "sha256" in reason               # 20 (different checkpoint)


def test_validate_rejects_arch_mismatch(tmp_path):
    _save(tmp_path)
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    ok, reason = validate(art, run_id="chemkan/hydrogen/main/direct_autograd_seed0",
                          architecture={**ARCH, "hidden_dim": 99}, checkpoint_sha256="abc123")
    assert not ok and "architecture" in reason          # 21


def test_validate_rejects_run_id_mismatch(tmp_path):
    _save(tmp_path)
    art = load_predictions(tmp_path / "predictions" / "test_predictions.npz")
    ok, reason = validate(art, run_id="chemkan/other/run",
                          architecture=ARCH, checkpoint_sha256="abc123")
    assert not ok and "run_id" in reason                # 22


def test_overwrite_protection_and_force(tmp_path):
    _save(tmp_path)
    with pytest.raises(FileExistsError):                # 23
        _save(tmp_path)
    _save(tmp_path, force=True)                         # 24: --force allows replacement


def test_canonical_arch_is_order_independent():
    assert canonical_arch({"a": 1, "b": 2}) == canonical_arch({"b": 2, "a": 1})


# --------------------------------------------------------------------------
# Fix 6: reusable compatible-loader that cannot silently return stale predictions
# --------------------------------------------------------------------------

def test_load_compatible_accepts_matching_checkpoint(tmp_path):
    ckpt = _fake_checkpoint(tmp_path)
    _save(tmp_path, sha=checkpoint_sha256(ckpt))
    checkpoint = {"run_id": "chemkan/hydrogen/main/direct_autograd_seed0", "architecture": ARCH}
    art = load_compatible_predictions(tmp_path / "predictions" / "test_predictions.npz",
                                      checkpoint=checkpoint, checkpoint_path=ckpt)
    assert art["predictions"].shape == (5, 3, 9)


def test_load_compatible_rejects_wrong_checkpoint(tmp_path):
    ckpt = _fake_checkpoint(tmp_path)
    _save(tmp_path, sha="made-from-a-different-file")   # artifact sha != real file's sha
    checkpoint = {"run_id": "chemkan/hydrogen/main/direct_autograd_seed0", "architecture": ARCH}
    with pytest.raises(IncompatiblePredictionError):
        load_compatible_predictions(tmp_path / "predictions" / "test_predictions.npz",
                                    checkpoint=checkpoint, checkpoint_path=ckpt)
