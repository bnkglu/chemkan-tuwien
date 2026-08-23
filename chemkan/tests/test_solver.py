import pytest
import torch
import torch.nn as nn

from chemkan.solver import SolverConfig, integrate

# The actual benchmark solver: Tsit5 (the paper's integrator, from the pinned GitHub
# torchdiffeq) with direct-autograd sensitivity. FSA is intentionally NOT implemented.
TSIT5 = SolverConfig(method="tsit5", rtol=1e-6, atol=1e-8, sensitivity="direct_autograd")


class Linear1D(nn.Module):
    """dy/dt = a * y  (analytic solution y(t) = y0 * exp(a t))."""

    def __init__(self, a: float):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))

    def forward(self, t, y):
        return self.a * y


def test_solverconfig_requires_all_fields():
    with pytest.raises(TypeError):
        SolverConfig()                                    # no defaults anymore


def test_tsit5_integrates_exponential_correctly():
    func = Linear1D(-0.5)
    y0 = torch.tensor([[1.0]])                            # (B=1, dim=1)
    t = torch.linspace(0.0, 2.0, 5)
    traj = integrate(func, y0, t, TSIT5)                  # (T, B, dim), Tsit5
    expected = torch.exp(-0.5 * t).reshape(-1, 1, 1)
    assert torch.allclose(traj, expected, atol=1e-4)


def test_tsit5_gradients_propagate_through_solver():
    func = Linear1D(-0.5)
    y0 = torch.tensor([[1.0]])
    t = torch.linspace(0.0, 1.0, 4)
    integrate(func, y0, t, TSIT5).sum().backward()        # direct autograd through odeint
    assert func.a.grad is not None and torch.isfinite(func.a.grad).all()


def test_solver_is_method_agnostic():
    # SolverConfig does not hard-code a method; a second explicit-RK method integrates too.
    cfg = SolverConfig(method="dopri5", rtol=1e-6, atol=1e-8, sensitivity="direct_autograd")
    func = Linear1D(-0.5)
    y0 = torch.tensor([[1.0]])
    t = torch.linspace(0.0, 1.0, 3)
    assert integrate(func, y0, t, cfg).shape == (3, 1, 1)


def test_rejects_non_autograd_sensitivity():
    # Forward Sensitivity Analysis / adjoint are not supported; only direct_autograd is.
    with pytest.raises(ValueError):
        SolverConfig(method="tsit5", rtol=1e-6, atol=1e-8,
                     sensitivity="forward_sensitivity")


def test_solver_roundtrips_through_checkpoint_metadata():
    # Save solver metadata as a training script would, then reconstruct it.
    meta = {"method": TSIT5.method, "rtol": TSIT5.rtol,
            "atol": TSIT5.atol, "sensitivity": TSIT5.sensitivity}
    rebuilt = SolverConfig(method=meta["method"], rtol=meta["rtol"],
                           atol=meta["atol"], sensitivity=meta["sensitivity"])
    assert (rebuilt.method, rebuilt.rtol, rebuilt.atol, rebuilt.sensitivity) == \
           (TSIT5.method, TSIT5.rtol, TSIT5.atol, TSIT5.sensitivity)
