"""Guardrails for the data/model/solver boundary refactor."""

import importlib
from pathlib import Path

import pytest

import chemkan
from chemkan import training
from chemkan.solver import SolverConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_config_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chemkan.config")


def test_training_api_is_generic():
    assert hasattr(training, "train_kinetic_stage")
    assert hasattr(training, "train_full_chemkan")
    assert not hasattr(training, "train_biodiesel")        # dataset-named funcs gone
    assert not hasattr(training, "train_hydrogen_stage1")


def test_single_solverconfig():
    assert chemkan.SolverConfig is SolverConfig             # one owner: solver.py


def test_scripts_infer_species_dim_from_data():
    for name in ["train_biodiesel.py", "train_hydrogen.py"]:
        src = (SCRIPTS / name).read_text()
        assert "shape[-1]" in src                           # dimension is data-derived
        assert "species_dim = 6" not in src and "species_dim=6" not in src
        assert "species_dim = 9" not in src and "species_dim=9" not in src


def test_train_scripts_seed_and_record_it():
    for name in ["train_biodiesel.py", "train_hydrogen.py"]:
        src = (SCRIPTS / name).read_text()
        assert "--seed" in src
        assert "manual_seed" in src
        assert '"seed"' in src                              # persisted in checkpoint


def test_biodiesel_epoch_default_is_paper_1e4():
    src = (SCRIPTS / "train_biodiesel.py").read_text()
    assert "default=10000" in src
