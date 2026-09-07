"""Regression coverage for the final-trunk-ReLU fix, without optimization steps."""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "deeponet"))

from biodiesel_deeponet import (  # noqa: E402
    ARCHITECTURE_VERSIONS, LEGACY_ARCHITECTURE_VERSION as LEGACY,
    REFERENCE_ARCHITECTURE_VERSION as REFERENCE,
    BiodieselDeepONet, architecture, build, n_params,
)
from evaluate_biodiesel_deeponet import build_model  # noqa: E402
import train_biodiesel_deeponet as trainer  # noqa: E402


def _forward_from_weights(state, branch_input, tau, *, final_trunk_relu):
    """Independent forward oracle using the pre-fix state-dict layout."""
    def linear(x, name):
        return F.linear(x, state[name + ".weight"], state[name + ".bias"])

    branch = F.relu(linear(branch_input, "branch.0"))
    branch = F.relu(linear(branch, "branch.2"))
    branch = linear(branch, "branch.4")
    trunk = F.relu(linear(tau.reshape(-1, 1), "trunk.0"))
    trunk = linear(trunk, "trunk.2")
    if final_trunk_relu:
        trunk = F.relu(trunk)
    return linear(branch.unsqueeze(0) * trunk.unsqueeze(1), "head")


@pytest.mark.parametrize("version", ARCHITECTURE_VERSIONS)
def test_final_activation_placement_and_linear_head(version):
    model = build(seed=0, architecture_version=version)
    assert [type(layer) for layer in model.branch] == [
        nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    expected_trunk = [nn.Linear, nn.ReLU, nn.Linear]
    if version == REFERENCE:
        expected_trunk.append(nn.ReLU)
    assert [type(layer) for layer in model.trunk] == expected_trunk
    assert type(model.head) is nn.Linear
    assert (model.head.in_features, model.head.out_features) == (8, 6)

    # Deliberately negative branch/trunk/head values expose unintended clipping.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        model.branch[-1].bias.fill_(-2)
        model.trunk[2].bias.fill_(-3)
        model.head.weight.fill_(1)
        model.head.bias.fill_(-0.5)
        u0, tau = torch.zeros(2, 7), torch.tensor([0.0, 0.5, 1.0])
        assert torch.equal(model.branch(u0), torch.full((2, 8), -2.0))
        expected = -0.5 if version == REFERENCE else 47.5
        assert torch.equal(model(u0, tau), torch.full((3, 2, 6), expected))


@pytest.mark.parametrize("stored_version", [None, LEGACY, REFERENCE])
def test_checkpoint_round_trip_reconstructs_saved_forward(tmp_path, stored_version):
    version = stored_version or LEGACY
    original = build(seed=0, architecture_version=version)
    arch = architecture(original)
    if stored_version is None:
        del arch["architecture_version"]
        assert arch["activation"] == (
            "relu between layers only (none on branch/trunk output or head)")
    path = tmp_path / "checkpoint.pt"
    torch.save({"architecture": arch, "model_state": original.state_dict()}, path)
    before = path.read_bytes()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded = build_model(saved, "cpu")
    assert loaded.architecture_version == version
    assert not loaded.training
    generator = torch.Generator().manual_seed(19)
    u0 = torch.rand(4, 7, generator=generator)
    tau = torch.tensor([-0.25, 0.0, 0.1, 0.4, 1.0, 1.5])
    with torch.no_grad():
        expected = _forward_from_weights(saved["model_state"], u0, tau,
                                         final_trunk_relu=(version == REFERENCE))
        other_variant = _forward_from_weights(saved["model_state"], u0, tau,
                                              final_trunk_relu=(version != REFERENCE))
        assert torch.equal(loaded(u0, tau), expected)
        assert not torch.equal(expected, other_variant)
    assert path.read_bytes() == before


def test_unknown_versions_are_rejected():
    assert ARCHITECTURE_VERSIONS == (LEGACY, REFERENCE)
    ckpt = {"architecture": {"width": 8, "out_dim": 6, "architecture_version": "typo"}}
    with pytest.raises(ValueError, match="unknown DeepONet architecture_version"):
        build_model(ckpt, "cpu")


@pytest.mark.parametrize("width", [3, 5, 6, 8, 10, 13])
def test_parameters_and_initialization_are_identical_between_versions(width):
    legacy = build(width, seed=0, architecture_version=LEGACY)
    reference = build(width, seed=0, architecture_version=REFERENCE)
    assert legacy.branch_dims == reference.branch_dims == (7, width, width, width)
    assert legacy.trunk_dims == reference.trunk_dims == (1, width - 1, width)
    assert n_params(legacy) == n_params(reference) == 3 * width**2 + 18 * width + 4
    if width == 8:
        assert n_params(reference) == 340
    assert legacy.state_dict().keys() == reference.state_dict().keys()
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key]), key

    # Reproduce the existing initializer independently; ReLU must consume no RNG.
    torch.manual_seed(0)
    expected = BiodieselDeepONet(width, architecture_version=LEGACY)
    for layer in expected.modules():
        if isinstance(layer, nn.Linear):
            assert layer.bias is not None
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    for key, value in expected.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key]), key
    for layer in reference.modules():
        if isinstance(layer, nn.Linear):
            assert layer.bias is not None and torch.count_nonzero(layer.bias) == 0


def test_new_config_and_checkpoint_record_version_without_training(tmp_path, monkeypatch):
    """Exercise actual serialization at zero epochs; fail if any optimizer step occurs."""
    run_dir = tmp_path / "reference"
    optimizers = []
    adam = torch.optim.Adam

    def capture_adam(*args, **kwargs):
        optimizer = adam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def forbid_step(*args, **kwargs):
        pytest.fail("This serialization test must not train the model")

    monkeypatch.setattr(adam, "step", forbid_step)
    monkeypatch.setattr(trainer.torch.optim, "Adam", capture_adam)
    monkeypatch.setattr(sys, "argv", ["train_biodiesel_deeponet.py", "--epochs", "0",
                                     "--run-dir", str(run_dir)])
    trainer.main()

    config = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert config["architecture"] == ckpt["architecture"]
    assert config["architecture"]["architecture_version"] == REFERENCE
    assert "after every trunk layer" in config["architecture"]["activation"]
    assert config["parameter_count"] == ckpt["architecture"]["parameter_count"] == 340
    assert config["architecture"]["weight_init"] == "glorot_normal (xavier_normal_), zero bias"
    assert config["optimizer"] == "Adam"
    assert config["learning_rate"] == ckpt["training"]["learning_rate"] == 1e-3
    assert len(optimizers) == 1 and type(optimizers[0]) is adam
    assert optimizers[0].defaults["lr"] == 1e-3
    assert optimizers[0].defaults["betas"] == (0.9, 0.999)
    assert optimizers[0].defaults["eps"] == 1e-8
    assert config["loss"] == "normalized trajectory MSE (Eq. 18)"
    assert config["normalization"]["stats"] == "train-only min-max"
    assert ckpt["normalization"]["output_space"] == "normalized species"
    assert ckpt["training"]["epochs"] == 0
    assert len((run_dir / "history.csv").read_text().splitlines()) == 1
    loaded = build_model(ckpt, "cpu")
    expected = build(seed=0)
    for key, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    trainer.require_reference_run_directory(run_dir)


@pytest.mark.parametrize("artifact", ["config.json", "checkpoint_final.pt", "checkpoint_resume.pt"])
@pytest.mark.parametrize("flag", [None, "--resume", "--overwrite"])
def test_legacy_run_is_rejected_before_any_mutation(tmp_path, monkeypatch, artifact, flag):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    model = build(seed=0, architecture_version=LEGACY)
    arch = architecture(model)
    del arch["architecture_version"]
    config = {"architecture": arch}
    if artifact == "config.json":
        (run_dir / artifact).write_text(json.dumps(config))
    else:
        record = {"config": config} if artifact == "checkpoint_resume.pt" else config
        torch.save({**record, "model_state": model.state_dict()}, run_dir / artifact)
    (run_dir / "run.log").write_text("legacy log\n")
    (run_dir / "history.csv").write_text("epoch,total_loss\n0,0.5\n")
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    argv = ["train_biodiesel_deeponet.py", "--run-dir", str(run_dir), "--epochs", "0"]
    if flag:
        argv.append(flag)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="Use a new run directory"):
        trainer.main()
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before
