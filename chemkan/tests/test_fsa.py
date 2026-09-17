"""Forward sensitivity analysis (chemkan.fsa): unit tests.

Fast versions of the validation in ``scripts/fsa/validate_fsa.py``: the analytic scalar
problem, parameter packing, functional equivalence with the existing dynamics wrappers,
trajectory independence, FSA-vs-direct-autograd gradients on small ChemKAN problems,
gradient accumulation / explicit zeros, and the non-finite abort.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.func import jacrev

from chemkan.dynamics import ChemKANDynamics, KineticDynamics
from chemkan.fsa import (
    GROUP_COR,
    GROUP_KIN,
    GROUP_THERMO,
    FunctionalDynamics,
    NonFiniteFSAError,
    ParameterPacking,
    fsa_loss_and_gradients,
    integrate_with_sensitivities,
)
from chemkan.losses import element_conservation_loss, trajectory_mse
from chemkan.model import ChemKAN, KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate
from chemkan.temperature import ConstantTemperature, ObservedTemperature
from chemkan.training import loss_and_gradients


@pytest.fixture
def f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def tight(backend):
    return SolverConfig(method="tsit5", rtol=1e-10, atol=1e-12, sensitivity=backend)


class Exponential(nn.Module):
    """dy/dt = theta * y  (plus an unused parameter to test explicit zero gradients)."""

    def __init__(self, theta: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta))
        self.unused = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, t, y):
        return self.theta * y


# ------------------------------------------------------------------ analytic problem
def test_analytic_state_sensitivity_and_gradient(f64):
    theta, y0 = -0.7, 1.3
    dyn = Exponential(theta)
    t = torch.linspace(0.0, 2.0, 9)
    fd = FunctionalDynamics(dyn, [dyn.theta])
    x, S = integrate_with_sensitivities(fd, torch.tensor([[y0]]), t, tight("fsa"))
    y_exact = y0 * torch.exp(theta * t)
    assert torch.allclose(x[:, 0, 0], y_exact, rtol=1e-8, atol=1e-12)
    assert torch.allclose(S[:, 0, 0, 0], t * y_exact, rtol=1e-7, atol=1e-12)   # dy/dtheta = t y
    assert S[0].abs().max() == 0                                   # S(t0) = 0
    assert x.grad_fn is None and S.grad_fn is None                 # no graph through solver

    # L = 0.5 sum_j (y_j - d_j)^2  ->  dL/dtheta = sum_j (y_j - d_j) t_j y_j
    data = torch.linspace(1.0, 0.2, 9)
    exact = float(((y_exact - data) * t * y_exact).sum())
    loss = lambda pred: 0.5 * ((pred[:, 0, 0] - data) ** 2).sum()
    fsa_loss_and_gradients(dyn, [dyn.theta], torch.tensor([[y0]]), t, tight("fsa"), loss)
    assert math.isclose(float(dyn.theta.grad), exact, rel_tol=1e-7)

    dyn.theta.grad = None                                          # direct autograd reference
    loss(integrate(dyn, torch.tensor([[y0]]), t, tight("direct_autograd"))).backward()
    assert math.isclose(float(dyn.theta.grad), exact, rel_tol=1e-7)


def test_gradients_accumulate_and_unused_parameters_get_explicit_zeros(f64):
    dyn = Exponential(-0.3)
    t = torch.linspace(0.0, 1.0, 4)
    loss = lambda pred: pred.pow(2).sum()
    params = [dyn.theta, dyn.unused]
    fsa_loss_and_gradients(dyn, params, torch.tensor([[1.0]]), t, tight("fsa"), loss)
    once = dyn.theta.grad.clone()
    assert dyn.unused.grad is not None and torch.equal(dyn.unused.grad, torch.zeros(2))
    fsa_loss_and_gradients(dyn, params, torch.tensor([[1.0]]), t, tight("fsa"), loss)
    assert torch.allclose(dyn.theta.grad, 2 * once)                # added, not overwritten


def test_non_finite_state_aborts(f64):
    class Blowup(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(1.0))

        def forward(self, t, y):
            return self.a * y / (1.0 - t)                           # singular at t = 1

    dyn = Blowup()
    cfg = SolverConfig(method="tsit5", rtol=1e-3, atol=1e-6, sensitivity="fsa")
    t = torch.tensor([0.0, 0.5, 0.9])
    with pytest.raises(NonFiniteFSAError):
        fsa_loss_and_gradients(dyn, [dyn.a], torch.tensor([[1.0]]), t, cfg,
                               lambda p: p.sum() * float("inf"))


# ------------------------------------------------------------------ packing
def test_packing_is_deterministic_complete_and_grouped():
    torch.manual_seed(0)
    model = ChemKAN(9, 3, 4, 3, use_base_act=True)
    norm = MinMaxNormalizer(torch.zeros(10), torch.ones(10))
    chem = ChemKANDynamics(model, input_normalizer=norm)
    pk = ParameterPacking.from_module(chem, model.parameters())
    assert pk.total == 344 == sum(p.numel() for p in model.parameters())
    assert pk.describe()["group_counts"] == {GROUP_KIN: 285, GROUP_THERMO: 9, GROUP_COR: 50}
    assert pk == ParameterPacking.from_module(chem, reversed(list(model.parameters())))
    flat = pk.pack(dict(chem.named_parameters()))
    for name, v in pk.unpack(flat).items():
        assert torch.equal(v, dict(chem.named_parameters())[name])

    kin = KineticDynamics(model.kinetic, ConstantTemperature(torch.ones(2)),
                          input_normalizer=norm)
    pk1 = ParameterPacking.from_module(kin, model.kinetic.parameters())
    assert pk1.total == 285 and set(pk1.groups) == {GROUP_KIN}
    with pytest.raises(ValueError):                        # thermo is not part of Stage 1
        ParameterPacking.from_module(kin, model.parameters())
    with pytest.raises(ValueError):                        # a tensor twice
        p0 = next(model.kinetic.parameters())
        ParameterPacking.from_module(kin, [p0, p0])


def test_biodiesel_packing_counts():
    core = KineticCore(6, 4, 3, 2, use_base_act=False)
    norm = MinMaxNormalizer(torch.zeros(7), torch.ones(7))
    kin = KineticDynamics(core, ConstantTemperature(torch.ones(3)), input_normalizer=norm)
    assert ParameterPacking.from_module(kin, core.parameters()).total == 156


# ------------------------------------------------------------------ small ChemKAN problems
def _stage2(B=3):
    torch.manual_seed(1)
    model = ChemKAN(4, 3, 4, 2, use_base_act=True)
    u_min = torch.cat([torch.zeros(4), torch.tensor([900.0])])
    u_max = torch.cat([torch.full((4,), 0.5), torch.tensor([2500.0])])
    norm = MinMaxNormalizer(u_min, u_max)
    with torch.no_grad():
        model.thermo.linear.weight.mul_(50.0)              # a visible temperature response
    u0 = torch.cat([torch.rand(B, 4) * 0.3, 1000.0 + 100.0 * torch.rand(B, 1)], dim=-1)
    return model, ChemKANDynamics(model, input_normalizer=norm), u0, norm


def _stage1(B=3):
    model, _, u0, norm = _stage2(B)
    ts = torch.linspace(0.0, 1.0, 7)
    T_obs = 1000.0 + 300.0 * ts[:, None, None] ** 2 * torch.arange(1, B + 1)[None, :, None]
    temp = ObservedTemperature(ts, T_obs)
    return model, KineticDynamics(model.kinetic, temp, input_normalizer=norm), u0[:, :4], norm


def _biodiesel(B=3):
    torch.manual_seed(2)
    core = KineticCore(6, 4, 3, 2, use_base_act=False)
    norm = MinMaxNormalizer(torch.zeros(7), torch.cat([torch.ones(6), torch.tensor([350.0])]))
    temp = ConstantTemperature(torch.linspace(300.0, 340.0, B))
    return core, KineticDynamics(core, temp, input_normalizer=norm), torch.rand(B, 6), norm


CASES = {"biodiesel": _biodiesel, "stage1": _stage1, "stage2": _stage2}


def _params(case, net):
    return list(net.kinetic.parameters()) if case == "stage1" else list(net.parameters())


@pytest.mark.parametrize("case", sorted(CASES))
def test_functional_rhs_equals_existing_wrapper(case, f64):
    net, dyn, x0, _ = CASES[case]()
    fd = FunctionalDynamics(dyn, _params(case, net))
    for t in (torch.tensor(0.0), torch.tensor(0.37), torch.tensor(1.0)):
        with torch.no_grad():
            assert torch.allclose(fd.rhs(t, x0), dyn(t, x0), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("case", sorted(CASES))
def test_trajectories_are_independent(case, f64):
    net, dyn, x0, _ = CASES[case]()
    fd = FunctionalDynamics(dyn, _params(case, net))
    t = torch.tensor(0.4)
    J = jacrev(lambda x: fd.rhs(t, x))(x0)                 # (B, n, B, n)
    B = x0.shape[0]
    for b in range(B):
        for c in range(B):
            if b != c:
                assert J[b, :, c, :].abs().max() == 0      # structurally zero cross-coupling
    _, J_x, _ = fd.rhs_and_jacobians(t, x0, fd.theta())
    for b in range(B):
        assert torch.allclose(J_x[b], J[b, :, b, :], rtol=1e-12, atol=1e-12)


def test_stage1_uses_each_trajectorys_own_temperature(f64):
    net, dyn, x0, _ = _stage1()
    fd = FunctionalDynamics(dyn, net.kinetic.parameters())
    t = torch.tensor(0.6)
    base = fd.rhs(t, x0)
    temps = dyn.temperature.temperatures
    with torch.no_grad():
        temps[:, 1] += 250.0                               # change ONLY trajectory 1's history
    changed = fd.rhs(t, x0)
    assert torch.equal(changed[0], base[0]) and torch.equal(changed[2], base[2])
    assert not torch.equal(changed[1], base[1])
    # T_obs(t) is time dependent: the same state gives a different rate at another time.
    assert not torch.allclose(fd.rhs(torch.tensor(0.1), x0), changed)


@pytest.mark.parametrize("case", sorted(CASES))
def test_fsa_gradient_matches_direct_autograd(case, f64):
    net, dyn, x0, norm = CASES[case]()
    params = _params(case, net)
    t = torch.linspace(0.0, 1.0, 6)
    n = x0.shape[1]
    sub = norm.subset(slice(0, n))
    with torch.no_grad():
        target = sub.normalize(integrate(dyn, x0, t, tight("direct_autograd"))) + 0.05

    def loss(pred):
        mse = trajectory_mse(sub.normalize(pred), target)
        if case == "biodiesel":
            return mse
        counts = torch.tensor([[2.0, 1.0, 0.0, 2.0], [0.0, 1.0, 2.0, 1.0]])
        return mse + 1e-2 * element_conservation_loss(
            pred[..., :4], counts, torch.tensor([1.0, 16.0]), torch.tensor([2.0, 17.0, 32.0, 18.0]))

    for p in params:
        p.grad = None
    loss_and_gradients(dyn, x0, t, params, loss, tight("direct_autograd"))
    g_da = torch.cat([p.grad.reshape(-1) for p in params])
    for p in params:
        p.grad = None
    loss_and_gradients(dyn, x0, t, params, loss, tight("fsa"))
    g_fsa = torch.cat([p.grad.reshape(-1) for p in params])
    rel = float((g_fsa - g_da).norm() / g_da.norm())
    assert rel < 1e-6, rel


@pytest.mark.parametrize("case", sorted(CASES))
def test_batched_solve_equals_individual_solves(case, f64):
    net, dyn, x0, _ = CASES[case]()
    fd = FunctionalDynamics(dyn, _params(case, net))
    t = torch.linspace(0.0, 1.0, 4)
    x, S = integrate_with_sensitivities(fd, x0, t, tight("fsa"))
    b = 1
    if case == "biodiesel":
        one = KineticDynamics(net, ConstantTemperature(dyn.temperature.temperature[b:b + 1]),
                              input_normalizer=dyn.input_normalizer)
    elif case == "stage1":
        one = KineticDynamics(net.kinetic,
                              ObservedTemperature(dyn.temperature.saved_times,
                                                  dyn.temperature.temperatures[:, b:b + 1]),
                              input_normalizer=dyn.input_normalizer)
    else:
        one = dyn
    fd1 = FunctionalDynamics(one, _params(case, net))
    x1, S1 = integrate_with_sensitivities(fd1, x0[b:b + 1], t, tight("fsa"))
    assert torch.allclose(x1[:, 0], x[:, b], rtol=1e-8, atol=1e-10)
    assert torch.allclose(S1[:, 0], S[:, b], rtol=1e-6, atol=1e-8 * S.abs().max())


def test_augmented_state_matches_state_only_solve(f64):
    net, dyn, u0, _ = _stage2()
    fd = FunctionalDynamics(dyn, net.parameters())
    t = torch.linspace(0.0, 1.0, 5)
    x, _ = integrate_with_sensitivities(fd, u0, t, tight("fsa"))
    with torch.no_grad():
        ref = integrate(dyn, u0, t, tight("fsa"))
    assert torch.allclose(x, ref, rtol=1e-8, atol=1e-8)


# ------------------------------------------------------------------ CLI plumbing
import json                                                        # noqa: E402
import subprocess                                                  # noqa: E402
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_DATA = _ROOT / "data" / "generated"
_HAVE_BD = (_DATA / "biodiesel.npz").exists()
_HAVE_H2 = (_DATA / "hydrogen.npz").exists() and (_DATA / "hydrogen_temperature_20000.npz").exists()


def _cli(*args):
    return subprocess.run([sys.executable, *args], cwd=_SCRIPTS, capture_output=True, text=True)


def _ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.skipif(not _HAVE_BD, reason="biodiesel.npz absent")
@pytest.mark.parametrize("extra", [[], ["--batch-size", "10"]])
def test_biodiesel_fsa_cli_records_backend(tmp_path, extra):
    run = tmp_path / "bd"
    r = _cli("train_biodiesel.py", "--epochs", "2", "--sensitivity", "fsa",
             "--run-dir", str(run), *extra)
    assert r.returncode == 0, r.stderr
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["sensitivity_backend"] == "fsa"
    assert cfg["fsa"]["parameter_packing"]["kinetic"]["total"] == 156
    ck = _ck(run / "checkpoint_final.pt")
    assert ck["solver"]["sensitivity"] == "fsa" and "fsa" in ck
    # inference of an FSA-trained checkpoint is the ordinary state-only evaluation
    e = _cli("evaluate_biodiesel.py", "--run-dir", str(run), "--split", "test", "--metrics")
    assert e.returncode == 0, e.stderr
    assert json.loads((run / "metrics.json").read_text())["solver"]["sensitivity"] == "fsa"


@pytest.mark.skipif(not _HAVE_BD, reason="biodiesel.npz absent")
def test_observed_interval_fsa_cli(tmp_path):
    run = tmp_path / "oi"
    r = _cli("train_biodiesel_observed_intervals.py", "--epochs", "1", "--eval-every", "0",
             "--sensitivity", "fsa", "--run-dir", str(run))
    assert r.returncode == 0, r.stderr
    assert json.loads((run / "config.json").read_text())["sensitivity_backend"] == "fsa"


@pytest.mark.skipif(not _HAVE_BD, reason="biodiesel.npz absent")
def test_resume_refuses_a_backend_change(tmp_path):
    run = tmp_path / "bd"
    r = _cli("train_biodiesel.py", "--epochs", "2", "--checkpoint-every", "1",
             "--run-dir", str(run))
    assert r.returncode == 0, r.stderr
    # fabricate an interrupted run: restore a resume file from the completed run's state
    ck = _ck(run / "checkpoint_final.pt")
    cfg = json.loads((run / "config.json").read_text())
    torch.save({"stage": "main", "epoch": 1, "model_state": ck["model_state"],
                "optimizer_state": None, "config": cfg}, run / "checkpoint_resume.pt")
    r = _cli("train_biodiesel.py", "--epochs", "3", "--resume", "--sensitivity", "fsa",
             "--run-dir", str(run))
    assert r.returncode != 0 and "--resume configuration mismatch" in (r.stderr + r.stdout)
    from _run import check_resume_config                          # the key itself is guarded
    with pytest.raises(SystemExit, match="sensitivity_backend"):
        check_resume_config({"sensitivity_backend": "direct_autograd"},
                            {"sensitivity_backend": "fsa"})


@pytest.mark.skipif(not _HAVE_H2, reason="hydrogen data absent")
def test_hydrogen_fsa_two_stages_and_backend_guard(tmp_path):
    run = tmp_path / "h2"
    r = _cli("train_hydrogen.py", "--stage1-epochs", "2", "--stage2-epochs", "1",
             "--sensitivity", "fsa", "--count-nfe", "--run-dir", str(run))
    assert r.returncode == 0, r.stderr
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["sensitivity_backend"] == "fsa"
    packing = cfg["fsa"]["parameter_packing"]
    assert packing["stage1"]["total"] == 285 and packing["stage2"]["total"] == 344
    s1 = _ck(run / "checkpoint_stage1.pt")
    assert s1["solver"]["sensitivity"] == "fsa"
    assert _ck(run / "checkpoint_final.pt")["solver"]["sensitivity"] == "fsa"

    # a direct-autograd Stage 2 may not silently consume the FSA Stage 1 ...
    r = _cli("train_hydrogen.py", "--stage1-from", str(run / "checkpoint_stage1.pt"),
             "--stage2-epochs", "1", "--run-dir", str(tmp_path / "mixed"))
    assert r.returncode != 0 and "stage1-backend-mismatch" in (r.stderr + r.stdout)
    # ... while an FSA Stage 2 consumes it and records the shared checkpoint hash
    r = _cli("train_hydrogen.py", "--stage1-from", str(run / "checkpoint_stage1.pt"),
             "--stage2-epochs", "1", "--sensitivity", "fsa", "--run-dir", str(tmp_path / "s2"))
    assert r.returncode == 0, r.stderr
    cfg2 = json.loads((tmp_path / "s2" / "config.json").read_text())
    assert cfg2["fsa"]["stage1_from_sensitivity"] == "fsa"
    assert cfg2["fsa"]["stage2_only_ablation"] is False
    assert len(cfg2["stage1_from"]["stage1_checkpoint_sha256"]) == 64
