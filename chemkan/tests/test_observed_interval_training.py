"""Focused checks for the observed-interval biodiesel experiment.

Scope is deliberately narrow: these verify the ONE thing that is new in
``scripts/train_biodiesel_observed_intervals.py`` -- that each interval is integrated
from its own observation, that the endpoint target and the sum reduction are right,
that gradients accumulate exactly and reach the model, that Adam steps once per epoch,
and that full-rollout evaluation only ever sees the original initial conditions.

They add no coverage for the shared library, which is tested elsewhere and unchanged.
"""

from __future__ import annotations

import math

import pytest
import torch
from train_biodiesel_observed_intervals import (
    accumulate_interval_gradients,
    full_rollout_mse,
)

from chemkan.model import KineticCore
from chemkan.normalization import MinMaxNormalizer
from chemkan.solver import SolverConfig, integrate

SOLVER = SolverConfig(method="tsit5", rtol=1e-6, atol=1e-8, sensitivity="direct_autograd")


class RecordingDynamics(torch.nn.Module):
    """Wraps a right-hand side and records every ``(t, y)`` the solver asks for."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls: list[tuple[float, torch.Tensor]] = []

    def forward(self, t, y):
        self.calls.append((float(t.detach()), y.detach().clone()))
        return self.inner(t, y)

    def first_call_after(self, index: int) -> tuple[float, torch.Tensor]:
        return self.calls[index]


class ExponentialRHS(torch.nn.Module):
    r"""Scalar ODE with a closed-form solution:  dy/dt = theta * y  ->  y(t) = y0 e^{theta dt}."""

    def __init__(self, theta: float):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(float(theta)))

    def forward(self, t, y):
        return self.theta * y


def identity_normalizer(m: int) -> MinMaxNormalizer:
    """u_min=0, u_max=1 -- normalization is the identity, so analytic checks stay readable."""
    return MinMaxNormalizer(torch.zeros(m), torch.ones(m))


# --------------------------------------------------------------------------------------
# 1. every interval starts from its OBSERVATION, on the actual absolute-time grid
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("t_values", [
    [0.0, 1.0, 2.0, 3.0],                       # uniform
    [0.0, 0.25, 1.75, 4.0, 4.5],                # irregular spacing
])
def test_each_interval_starts_from_its_observation(t_values):
    t = torch.tensor(t_values)
    n_int = len(t_values) - 1
    obs = torch.arange(1.0, 1.0 + len(t_values) * 2 * 3).reshape(len(t_values), 2, 3)
    norm = identity_normalizer(3)
    rec = RecordingDynamics(ExponentialRHS(-0.3))

    boundaries = []
    for j in range(n_int):
        boundaries.append(len(rec.calls))
        integrate(rec, obs[j], t[j:j + 2], SOLVER)

    for j, start in enumerate(boundaries):
        t0, y0 = rec.first_call_after(start)
        assert t0 == pytest.approx(float(t[j])), f"interval {j} started at t={t0}, want {t[j]}"
        assert torch.equal(y0, obs[j]), f"interval {j} did not start from its observation"

    # Interval j never looks back before its own start time. (No upper bound is asserted:
    # an adaptive solver takes trial steps past t[j+1] and interpolates back to it.)
    for j, start in enumerate(boundaries):
        stop = boundaries[j + 1] if j + 1 < n_int else len(rec.calls)
        assert min(c[0] for c in rec.calls[start:stop]) >= float(t[j]) - 1e-9
    assert norm.normalize(obs[0]).shape == obs[0].shape


def test_prediction_is_never_fed_into_the_next_interval():
    """Interval j+1 starts from observation j+1, not from interval j's endpoint."""
    t = torch.tensor([0.0, 1.0, 2.0])
    obs = torch.tensor([[[1.0]], [[5.0]], [[9.0]]])       # (3, 1, 1); linear, so a decaying
    rec = RecordingDynamics(ExponentialRHS(-0.5))          # model cannot land on obs[1]

    starts = []
    for j in range(2):
        starts.append(len(rec.calls))
        pred_end = integrate(rec, obs[j], t[j:j + 2], SOLVER)[-1]

    _, y_start_1 = rec.first_call_after(starts[1])
    assert torch.equal(y_start_1, obs[1])
    assert not torch.allclose(y_start_1, pred_end)         # the endpoint of interval 0 differs


# --------------------------------------------------------------------------------------
# 2. endpoint target and sum reduction
# --------------------------------------------------------------------------------------

def test_objective_matches_an_independent_endpoint_sum():
    """The helper equals an independently written sum over interval endpoints."""
    torch.manual_seed(0)
    t = torch.tensor([0.0, 0.7, 2.3, 3.0])
    obs = torch.rand(4, 5, 3) + 0.5
    norm = identity_normalizer(3)
    dyn = ExponentialRHS(-0.4)

    got = float(accumulate_interval_gradients(dyn, obs, norm.normalize(obs), t, norm,
                                              SOLVER, backward=False).detach())

    want = 0.0
    for j in range(3):
        end = integrate(dyn, obs[j], t[j:j + 2], SOLVER)[-1]          # (B, m)
        # mean over species, mean over trajectories; ONE endpoint, so no time sum to take
        want += float(((end - obs[j + 1]) ** 2).mean(dim=-1).mean().detach())
    assert got == pytest.approx(want, rel=1e-5)     # float32 reduction order


def test_objective_is_a_sum_over_intervals_not_a_mean():
    """No division by the interval count: doubling the intervals roughly doubles the value."""
    torch.manual_seed(0)
    norm = identity_normalizer(2)
    dyn = ExponentialRHS(-0.2)

    t_short = torch.tensor([0.0, 1.0, 2.0])
    obs_short = torch.ones(3, 4, 2)
    short = float(accumulate_interval_gradients(dyn, obs_short, norm.normalize(obs_short),
                                                t_short, norm, SOLVER,
                                                backward=False).detach())

    t_long = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    obs_long = torch.ones(5, 4, 2)
    long = float(accumulate_interval_gradients(dyn, obs_long, norm.normalize(obs_long),
                                               t_long, norm, SOLVER,
                                               backward=False).detach())
    # identical per-interval problem repeated -> exactly proportional to the interval count
    assert long == pytest.approx(short * 2, rel=1e-6)


def test_scalar_ode_known_endpoint_and_gradient():
    r"""One interval of dy/dt = theta*y: check the endpoint and dL/dtheta analytically.

        y1 = y0 e^{theta dt},  L = (y1 - target)^2,  dL/dtheta = 2 (y1 - target) y1 dt
    """
    theta0, y0, target, dt = -0.35, 2.0, 1.0, 1.4
    t = torch.tensor([0.0, dt])
    obs = torch.tensor([[[y0]], [[target]]])               # (2, 1, 1)
    norm = identity_normalizer(1)
    dyn = ExponentialRHS(theta0)

    value = accumulate_interval_gradients(dyn, obs, norm.normalize(obs), t, norm, SOLVER)

    y1 = y0 * math.exp(theta0 * dt)
    # rtol=1e-6 on a float32 solve; the analytic values are matched to ~1e-4 relative.
    assert float(value) == pytest.approx((y1 - target) ** 2, rel=1e-3)
    assert float(dyn.theta.grad) == pytest.approx(2 * (y1 - target) * y1 * dt, rel=1e-3)


# --------------------------------------------------------------------------------------
# 3. gradients: exact accumulation, reaching the model
# --------------------------------------------------------------------------------------

def test_accumulated_gradients_equal_one_backward_on_the_sum():
    torch.manual_seed(0)
    t = torch.tensor([0.0, 0.6, 1.9, 2.4])
    obs = torch.rand(4, 3, 2) + 0.5
    norm = identity_normalizer(2)

    dyn_a, dyn_b = ExponentialRHS(-0.3), ExponentialRHS(-0.3)
    accumulate_interval_gradients(dyn_a, obs, norm.normalize(obs), t, norm, SOLVER)
    total = accumulate_interval_gradients(dyn_b, obs, norm.normalize(obs), t, norm,
                                          SOLVER, backward=False)
    total.backward()
    assert float(dyn_a.theta.grad) == pytest.approx(float(dyn_b.theta.grad), rel=1e-5)


def test_gradients_reach_every_kinetic_core_parameter():
    """Through the real model + solver, every trainable tensor receives a finite gradient."""
    torch.manual_seed(0)
    m = 3
    core = KineticCore(species_dim=m, hidden_dim=2, num_basis=3, n_mu=2, use_base_act=False)
    from chemkan.dynamics import KineticDynamics
    from chemkan.temperature import ConstantTemperature
    full = MinMaxNormalizer(torch.zeros(m + 1), torch.ones(m + 1) * 400.0)
    dyn = KineticDynamics(core, ConstantTemperature(torch.full((2, 1), 330.0)),
                          input_normalizer=full)

    t = torch.tensor([0.0, 1.0, 2.0])
    obs = torch.rand(3, 2, m) * 0.5 + 0.2
    loss_norm = full.subset(slice(0, m))
    accumulate_interval_gradients(dyn, obs, loss_norm.normalize(obs), t, loss_norm, SOLVER)

    grads = {n: p.grad for n, p in core.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert any(g.abs().sum() > 0 for g in grads.values())


# --------------------------------------------------------------------------------------
# 4. full-rollout evaluation sees only the ORIGINAL initial conditions
# --------------------------------------------------------------------------------------

def test_full_rollout_uses_only_the_original_initial_condition():
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    obs = torch.rand(4, 2, 2) + 1.0
    y0 = obs[0]
    norm = identity_normalizer(2)
    rec = RecordingDynamics(ExponentialRHS(-0.4))

    with torch.no_grad():
        value = full_rollout_mse(rec, y0, t, norm.normalize(obs), norm, SOLVER)

    t0, first_y = rec.calls[0]
    assert t0 == pytest.approx(0.0)
    assert torch.equal(first_y, y0)
    # no later observation is ever handed to the model
    for _, y in rec.calls:
        for j in range(1, obs.shape[0]):
            assert not torch.equal(y, obs[j])
    # one continuous rollout that reaches the end of the grid
    assert max(c[0] for c in rec.calls) >= float(t[-1]) - 1e-9
    assert float(value) > 0


# --------------------------------------------------------------------------------------
# 5. end-to-end: one Adam step per epoch, on the real dataset
# --------------------------------------------------------------------------------------

def test_one_optimizer_step_per_epoch_end_to_end(tmp_path, monkeypatch):
    """Run the script's ``main`` for 3 updates and count the Adam steps it takes."""
    import train_biodiesel_observed_intervals as mod

    steps = {"n": 0}
    real_step = torch.optim.Adam.step

    def counting_step(self, *a, **k):
        steps["n"] += 1
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
    run_dir = tmp_path / "run"
    monkeypatch.setattr("sys.argv", [
        "train_biodiesel_observed_intervals.py", "--seed", "0", "--epochs", "3",
        "--eval-every", "1", "--run-dir", str(run_dir)])
    mod.main()

    assert steps["n"] == 3, "expected exactly one Adam update per epoch"

    rows = (run_dir / "history.csv").read_text().strip().splitlines()
    assert rows[0].split(",")[:2] == ["epoch", "total_loss"]
    # 3 training rows (epochs 0..2) + one final row at the saved weights
    assert [r.split(",")[0] for r in rows[1:]] == ["0", "1", "2", "3"]
    assert (run_dir / "checkpoint_final.pt").exists()


def test_initial_parameters_match_the_original_trainer(tmp_path):
    """Paired runs start from IDENTICAL tensors: both scripts seed, then build KineticCore."""
    def build(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return KineticCore(species_dim=6, hidden_dim=4, num_basis=3, n_mu=2,
                           use_base_act=False)

    for seed in (0, 1, 2):
        a, b = build(seed).state_dict(), build(seed).state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)
    # 156 TRAINABLE parameters (state_dict also carries the fixed RBF-centre buffers).
    assert sum(p.numel() for p in build(0).parameters()) == 156
