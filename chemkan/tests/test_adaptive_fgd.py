"""Mathematical and ODE integration checks for the experimental FGD path."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from chemkan.adaptive_fgd import (
    AdaptiveEdgeFunctions,
    AdaptiveFGD,
    AdaptiveFGDConfig,
    LineSearchError,
    RefinementLimitError,
    adaptive_edges,
    interpolate_edges,
    interpolation_error_bound,
    kernel_gradient,
)
from chemkan.dynamics import KineticDynamics
from chemkan.model import ChemKAN, KineticCore
from chemkan.solver import SolverConfig, integrate
from chemkan.temperature import ConstantTemperature


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def core():
    torch.manual_seed(12)
    return KineticCore(2, 3, 3, 1, use_base_act=True).double()


def regression(model):
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)[:, None].expand(-1, 3)
    target = torch.cat([torch.sin(3 * x[:, :1]), x[:, :1].square()], dim=1)
    return lambda: (model(x) - target).square().mean()


def ode_problem(model):
    dynamics = KineticDynamics(
        model, ConstantTemperature(torch.tensor([0.3, 0.7])), input_normalizer=None
    ).double()
    y0 = torch.tensor([[0.2, 0.4], [0.8, 0.6]], dtype=torch.float64)
    times = torch.linspace(0, 0.2, 4, dtype=torch.float64)
    target = y0[None] * torch.exp(-times[:, None, None])
    solver = SolverConfig("rk4", 1e-8, 1e-10, "direct_autograd")
    return lambda: (integrate(dynamics, y0, times, solver) - target).square().sum()


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conversion_is_exact_and_preserves_full_chemkan(full, dtype):
    model = ChemKAN(2, 3, 3, 1, use_base_act=True) if full else core()
    model = model.to(dtype=dtype)
    x = torch.randn(11, 3, dtype=dtype)
    expected = model(x).detach()
    blocks = adaptive_edges(model)
    assert torch.equal(model(x), expected)
    assert len(blocks) == (3 if full else 2)
    assert all(not p.requires_grad for b in blocks.values() for p in b.base.parameters())
    adaptive_edges(model)  # idempotent, does not wrap the frozen base again
    assert torch.equal(model(x), expected)


def test_adaptive_wrapper_uses_no_tanh_edge_coordinates():
    from author_repo_match._author_match import NoInputTanhRBFEdges

    base = NoInputTanhRBFEdges(2, 1, 3, use_base_act=False).double()
    block = AdaptiveEdgeFunctions(base, intervals=2)
    block.capture = True
    x = torch.tensor([[0.25, 0.75], [0.5, 1.0]], dtype=torch.float64, requires_grad=True)
    block(x).sum().backward()
    points, _ = block.samples[0]
    assert torch.equal(points, x.detach())


def test_nested_linear_refinement_preserves_values_and_state_derivatives():
    torch.manual_seed(2)
    values = torch.randn(2, 3, 5, dtype=torch.float64)
    fine = F.interpolate(values, size=17, mode="linear", align_corners=True)
    z = (torch.rand(19, 3, dtype=torch.float64) * 2 - 1).requires_grad_()
    coarse_out, fine_out = interpolate_edges(values, z), interpolate_edges(fine, z)
    torch.testing.assert_close(coarse_out, fine_out, atol=1e-14, rtol=1e-14)
    a = torch.autograd.grad(coarse_out.sum(), z)[0]
    b = torch.autograd.grad(fine_out.sum(), z)[0]
    torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)
    endpoints = torch.tensor([[-1.0] * 3, [1.0] * 3], dtype=torch.float64)
    torch.testing.assert_close(interpolate_edges(values, endpoints)[0], values[..., 0])
    torch.testing.assert_close(interpolate_edges(values, endpoints)[1], values[..., -1])


def test_kernel_gradient_and_analytic_error_bound():
    torch.manual_seed(3)
    x = torch.rand(17, 3, dtype=torch.float64) * 2 - 1
    a = torch.randn(17, 2, 3, dtype=torch.float64)
    cfg = AdaptiveFGDConfig(gamma=7, sample_chunk=5, node_chunk=7)
    nodes = torch.linspace(-1, 1, 17, dtype=torch.float64)
    g = kernel_gradient(x, a, nodes, cfg)
    direct = torch.einsum("qni,noi->oiq", torch.exp(-7 * (nodes[:, None, None] - x).square()), a)
    torch.testing.assert_close(g, direct)
    dense = torch.linspace(-1, 1, 2001, dtype=torch.float64)
    true = kernel_gradient(x, a, dense, cfg)
    approximate = interpolate_edges(g, dense[:, None].expand(-1, 3)).permute(1, 2, 0)
    bound = interpolation_error_bound(a, 16, 7)
    assert float((true - approximate).abs().max()) <= bound
    assert interpolation_error_bound(a, 32, 7) < bound


def test_functional_gradient_through_ode_matches_finite_difference():
    model = core()
    blocks = adaptive_edges(model)
    closure = ode_problem(model)
    block = blocks["add.edges"]
    block.capture = True
    closure().backward()
    block.capture = False
    points, adjoints = (torch.cat(parts) for parts in zip(*block.samples))
    gamma, center = 9.0, 0.15
    derivative = float(
        (adjoints[:, 0, 0] * torch.exp(-gamma * (points[:, 0] - center).square())).sum()
    )

    def perturbation(sign):
        def hook(module, args, output):
            z = torch.tanh(args[0][:, 0])
            delta = torch.zeros_like(output)
            delta[:, 0, 0] = sign * 1e-5 * torch.exp(-gamma * (z - center).square())
            return output + delta

        return hook

    values = []
    for sign in [1, -1]:
        handle = block.register_forward_hook(perturbation(sign))
        try:
            with torch.no_grad():
                values.append(float(closure()))
        finally:
            handle.remove()
    assert derivative == pytest.approx((values[0] - values[1]) / 2e-5, rel=1e-6, abs=1e-10)


def test_refined_ode_step_is_certified_and_descends():
    model = core()
    original = copy.deepcopy(model.state_dict())
    opt = AdaptiveFGD(model, AdaptiveFGDConfig(lr=100.0, initial_intervals=1))
    closure = ode_problem(model)
    result = opt.step(closure)
    assert result["certified"]
    assert 1.5 * result["error_bound"] < 0.5 * result["gradient_norm"]
    assert result["loss_after"] < result["loss_before"]
    assert result["loss_after"] == pytest.approx(float(closure().detach()))
    assert max(result["intervals_after"].values()) > 1
    assert result["backtracks"] > 0
    for name, block in opt.blocks.items():
        for key, value in block.base.state_dict().items():
            assert torch.equal(value, original[f"{name}.{key}"])
        assert not block.samples and not block.capture


def test_cap_rejects_instead_of_using_uncertified_update():
    model = core()
    opt = AdaptiveFGD(model, AdaptiveFGDConfig(initial_intervals=1, max_intervals=1))
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(RefinementLimitError, match="no update applied"):
        opt.step(regression(model))
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    assert all(not b.samples and not b.capture for b in opt.blocks.values())


def test_failed_line_search_restores_function_and_grid():
    model = core()
    opt = AdaptiveFGD(model, AdaptiveFGDConfig(lr=1e10, initial_intervals=1, max_backtracks=0))
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(LineSearchError):
        opt.step(regression(model))
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_exception_during_trial_restores_state():
    model = core()
    opt = AdaptiveFGD(model, AdaptiveFGDConfig())
    original = copy.deepcopy(model.state_dict())
    objective = regression(model)
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("trial failed")
        return objective()

    with pytest.raises(RuntimeError, match="trial failed"):
        opt.step(closure)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_zero_gradient_terminates_without_refining():
    model = core()
    opt = AdaptiveFGD(model, AdaptiveFGDConfig(initial_intervals=1, max_intervals=1))
    result = opt.step(lambda: regression(model)() * 0)
    assert result["stationary"] and result["step_size"] == 0
    assert result["intervals_before"] == result["intervals_after"]


def test_fixed_baseline_reports_missing_certificate():
    model = core()
    opt = AdaptiveFGD(model, AdaptiveFGDConfig(initial_intervals=1, refine=False))
    result = opt.step(regression(model))
    assert not result["certified"]
    assert result["intervals_after"] == result["intervals_before"]


def test_refined_checkpoint_roundtrip_and_next_step(tmp_path):
    model = core()
    cfg = AdaptiveFGDConfig(initial_intervals=1)
    opt = AdaptiveFGD(model, cfg)
    opt.step(regression(model))
    path = tmp_path / "adaptive.pt"
    torch.save(model.state_dict(), path)
    restored = core()
    adaptive_edges(restored, initial_intervals=1)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert float(regression(model)().detach()) == float(regression(restored)().detach())
    other = AdaptiveFGD(restored, cfg)
    a, b = opt.step(regression(model)), other.step(regression(restored))
    assert a == b
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_full_model_updates_thermo_linear_and_functional_edges():
    torch.manual_seed(8)
    model = ChemKAN(2, 3, 3, 1, use_base_act=True).double()
    opt = AdaptiveFGD(model, AdaptiveFGDConfig())
    x = torch.randn(10, 3, dtype=torch.float64)
    original = model.thermo.linear.weight.detach().clone()
    result = opt.step(lambda: (model(x) - torch.ones_like(x)).square().mean())
    assert result["loss_after"] < result["loss_before"]
    assert not torch.equal(original, model.thermo.linear.weight)
    assert len(result["intervals_after"]) == 3


@pytest.mark.parametrize(
    "setting",
    [
        {"epsilon": 1},
        {"gamma": 0},
        {"lr": float("nan")},
        {"initial_intervals": 0},
        {"sample_chunk": 0},
        {"max_backtracks": -1},
    ],
)
def test_invalid_settings_fail_early(setting):
    with pytest.raises(ValueError):
        replace(AdaptiveFGDConfig(), **setting)


@pytest.mark.parametrize(
    "system,method",
    [
        ("biodiesel", "adaptive-fgd"),
        ("hydrogen-stage1", "adaptive-fgd"),
        ("hydrogen-full", "adaptive-fgd"),
        ("biodiesel", "fixed-fgd"),
        ("biodiesel", "adam"),
    ],
)
def test_experiment_entry_point(tmp_path, system, method):
    from train_adaptive_fgd import main

    output = tmp_path / "run"
    main(
        [
            "--run-dir",
            str(output),
            "--system",
            system,
            "--method",
            method,
            "--steps",
            "1",
            "--train-limit",
            "2",
            "--test-limit",
            "2",
            "--time-points",
            "4",
        ]
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "completed" and metrics["completed_steps"] == 1
    if method == "adaptive-fgd":
        assert metrics["final_train_loss"] < metrics["initial_train_loss"]
    checkpoint = torch.load(output / "checkpoint_final.pt", weights_only=True)
    assert checkpoint["config"]["system"] == system
    assert (output / "history.csv").is_file()
    with pytest.raises(FileExistsError):
        main(["--run-dir", str(output)])


@pytest.mark.parametrize("update_elapsed", [0.75, 1.25])
def test_wall_time_budget_stops_between_updates(tmp_path, monkeypatch, update_elapsed):
    from types import SimpleNamespace

    import train_adaptive_fgd as experiment

    # Setup starts at 0; one whole update fits or crosses the one-second budget.
    # The next update must not start. This clock avoids a timing-dependent test.
    ticks = iter([0.0, 0.0, update_elapsed, 1.25, 1.25])
    monkeypatch.setattr(experiment, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    output = tmp_path / "budget"
    experiment.main(
        [
            "--run-dir",
            str(output),
            "--method",
            "adam",
            "--steps",
            "5",
            "--max-seconds",
            "1",
            "--train-limit",
            "2",
            "--test-limit",
            "2",
            "--time-points",
            "4",
        ]
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["completed_steps"] == 1
    assert metrics["stop_reason"] == "time_budget"
    assert metrics["elapsed_seconds"] == 1.25
    if update_elapsed <= 1:
        assert metrics["last_update_within_budget"]["step"] == 1
        assert metrics["last_update_within_budget"]["loss_after"] == metrics["final_train_loss"]
    else:
        assert metrics["last_update_within_budget"] is None


def test_hydrogen_stage2_starts_from_supplied_stage1_checkpoint(tmp_path):
    from _data import load_hydrogen
    from train_adaptive_fgd import main

    data = load_hydrogen("train")
    torch.manual_seed(0)
    model = ChemKAN(9, 2, 3, 2, use_base_act=False)
    stage1 = tmp_path / "stage1.pt"
    torch.save(
        {
            "kinetic_state": model.kinetic.state_dict(),
            "architecture": {
                "hidden_dim": 2,
                "num_basis": 3,
                "n_mu": 2,
                "use_base_act": False,
            },
            "data": {"species": data["species"], "species_dim": 9},
            "stage1_final_loss": 0.25,
            "stage1_epochs": 10,
            "solver": {"sensitivity": "direct_autograd"},
            "input_scaling": {
                "method": "minmax",
                "u_min": data["u_min"],
                "u_max": data["u_max"],
            },
        },
        stage1,
    )
    output = tmp_path / "stage2"
    main(
        [
            "--run-dir",
            str(output),
            "--system",
            "hydrogen-stage2",
            "--stage1-from",
            str(stage1),
            "--method",
            "adam",
            "--steps",
            "1",
            "--hidden-dim",
            "2",
            "--num-basis",
            "3",
            "--n-mu",
            "2",
            "--train-limit",
            "2",
            "--test-limit",
            "1",
            "--time-points",
            "4",
        ]
    )
    config = json.loads((output / "config.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert config["dataset"]["stage1_from"]["epochs"] == 10
    assert config["dataset"]["stage1_from"]["final_loss"] == 0.25
    assert config["pinn"] == {"enabled": True, "alpha": 1e-4}
    assert metrics["completed_steps"] == 1


def test_hydrogen_stage2_requires_stage1_checkpoint(tmp_path):
    from train_adaptive_fgd import main

    with pytest.raises(ValueError, match="requires --stage1-from"):
        main(["--run-dir", str(tmp_path / "missing"), "--system", "hydrogen-stage2"])


def test_hydrogen_stage2_rejects_mismatched_dataset_stats(tmp_path):
    from _data import load_hydrogen
    from train_adaptive_fgd import main

    data = load_hydrogen("train")
    model = ChemKAN(9, 2, 3, 2, use_base_act=False)
    wrong_min = data["u_min"].clone()
    wrong_min[0] += 1e-6
    stage1 = tmp_path / "old-stage1.pt"
    torch.save(
        {
            "kinetic_state": model.kinetic.state_dict(),
            "architecture": {
                "hidden_dim": 2,
                "num_basis": 3,
                "n_mu": 2,
                "use_base_act": False,
            },
            "data": {"species": data["species"]},
            "solver": {"sensitivity": "direct_autograd"},
            "input_scaling": {
                "method": "minmax",
                "u_min": wrong_min,
                "u_max": data["u_max"],
            },
        },
        stage1,
    )
    with pytest.raises(ValueError, match="refusing to mix checkpoints and datasets"):
        main(
            [
                "--run-dir",
                str(tmp_path / "rejected"),
                "--system",
                "hydrogen-stage2",
                "--stage1-from",
                str(stage1),
                "--hidden-dim",
                "2",
                "--num-basis",
                "3",
                "--n-mu",
                "2",
            ]
        )


def test_hydrogen_stage2_accepts_current_author_matched_archive(tmp_path):
    from _data import DATA_DIR, load_hydrogen
    from _predictions import checkpoint_sha256
    from author_repo_match.train_author_matched_hydrogen import build_model
    from train_adaptive_fgd import main

    data = load_hydrogen("train")
    model = build_model()
    stage1 = tmp_path / "author-stage1.pt"
    torch.save(
        {
            "stage": 1,
            "run_id": "test-current-grid",
            "epochs_completed": 7,
            "config": {
                "dtype": "float32",
                "dataset": {
                    "sha256": checkpoint_sha256(DATA_DIR / "hydrogen.npz"),
                    "n_points": 100,
                    "n_conditions": 35,
                },
                "normalization": {
                    "u_min": [float(value) for value in data["u_min"]],
                    "u_max": [float(value) for value in data["u_max"]],
                },
                "architecture": {
                    "hidden_dim": 3,
                    "num_basis": 5,
                    "n_mu": 3,
                    "use_base_act": False,
                },
                "state_space": {
                    "stage2": "normalized species and temperature u_hat (10)"
                },
                "t_ref": {"value_s": 6e-4},
            },
            "model_state": model.state_dict(),
        },
        stage1,
    )
    output = tmp_path / "accepted"
    main(
        [
            "--run-dir",
            str(output),
            "--system",
            "hydrogen-stage2",
            "--stage1-from",
            str(stage1),
            "--dtype",
            "float32",
            "--method",
            "adam",
            "--steps",
            "1",
            "--train-limit",
            "2",
            "--test-limit",
            "1",
            "--time-points",
            "4",
        ]
    )
    config = json.loads((output / "config.json").read_text())
    assert config["dataset"]["state_coordinates"] == "normalized"
    assert config["dataset"]["stage1_from"]["time_points_match"] is True
