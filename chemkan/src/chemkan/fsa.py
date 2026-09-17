r"""Continuous forward sensitivity analysis (FSA) for the ChemKAN dynamics.

The ChemKAN paper trains with forward sensitivity analysis and solves "ChemKANs and their
corresponding forward sensitivity equations" with Tsit5 (p. 8), but does not print the
sensitivity equations. They are derived here by differentiating the paper's own
equations with respect to the learnable parameter groups.

State equations (paper notation, unchanged)
-------------------------------------------
    du/dt = f(u, t)                                                    (Eq. 3)
    u = [u_tilde, T]
    du_tilde/dt = KAN_kin(u, theta_kin)                                (Eq. 13)
    dT/dt = Linear(KAN_kin(u, theta_kin), theta_thermo) + KAN_cor(u, theta_cor)   (Eq. 15)

Forward sensitivity equations
-----------------------------
For the integrated state ``x`` of a stage and a parameter group ``theta_g`` define
``S_g = partial x / partial theta_g``. Differentiating the stage's state equation
``dx/dt = F(x, t)`` with respect to ``theta_g`` gives

    dS_g/dt = (partial F / partial x) @ S_g + partial F / partial theta_g,   S_g(t0) = 0

because the initial condition is observed data and does not depend on the parameters.

* Biodiesel and hydrogen Stage 1: ``x = u_tilde`` and ``F = KAN_kin([u_tilde, T], theta_kin)``
  (Eq. 13) with ``T`` supplied externally -- the per-trajectory isothermal constant for
  biodiesel, ``T_obs(t)`` read from data for hydrogen Stage 1. ``T`` is not an integrated
  state and ``partial T / partial theta_kin = 0``, so only ``S_kin`` (species rows) is
  propagated; the time dependence of ``T_obs(t)`` enters through ``F(x, t)`` evaluated
  at the current solver time.
* Hydrogen Stage 2: ``x = u = [u_tilde, T]`` and ``F`` is the stacked Eqs. 13 and 15.
  ``S_kin``, ``S_thermo`` and ``S_cor`` each have species and temperature rows, and the
  couplings (theta_kin entering dT/dt through Linear(KAN_kin); T feeding back into
  KAN_kin through u) are contained in ``partial F / partial u`` and
  ``partial F / partial theta_g``.

Implementation
--------------
``partial F / partial x`` and ``partial F / partial theta`` are obtained by automatic
differentiation of the EXISTING dynamics wrapper (``KineticDynamics`` /
``ChemKANDynamics``) through ``torch.func.functional_call``: the same temperature provider,
the same train-only min-max input normalizer and the same model code evaluate the RHS,
so no second ChemKAN implementation exists. ``torch.func.jacrev`` gives the per-trajectory
Jacobians and ``torch.func.vmap`` maps them over the independent trajectories of a batch.

The augmented state ``z = [x, S]`` of shape ``(B, n, 1 + P)`` is integrated as ONE ODE by
the same ``torchdiffeq`` Tsit5 solver with the same ``rtol``/``atol`` and torchdiffeq's
default error norm (RMS over the whole augmented tensor). The solve runs under
``torch.no_grad``: the Jacobians are local derivatives of detached values, and no autograd
graph through the solver is ever built. The loss gradient is then

    dL/dtheta = sum_j S(t_j)^T @ (partial L / partial x(t_j))

where ``partial L / partial x`` comes from differentiating the unchanged loss with respect
to a detached leaf copy of the predicted states. The time/batch reductions are therefore
exactly those of the loss.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap
from torchdiffeq import odeint

from .solver import SolverConfig

GROUP_KIN = "theta_kin"
GROUP_THERMO = "theta_thermo"
GROUP_COR = "theta_cor"
GROUP_OTHER = "theta"            # any parameter outside the ChemKAN modules (unit tests)

# Recorded in run configs / checkpoints of FSA runs.
FSA_FORMULATION = (
    "continuous forward sensitivity analysis: dS/dt = (dF/dx) S + dF/dtheta, S(t0) = 0, "
    "integrated together with the state as one augmented ODE z = [x, S] by the same Tsit5 "
    "solver; dF/dx and dF/dtheta by torch.func.jacrev of the existing dynamics wrapper "
    "(functional_call), vmapped over independent trajectories; dL/dtheta = "
    "sum_j S(t_j)^T dL/dx(t_j) with dL/dx from the unchanged loss on a detached leaf.")
FSA_ERROR_CONTROL = (
    "torchdiffeq default: RMS norm over the whole augmented [state, sensitivity] tensor, "
    "with the run's rtol/atol applied to every component (no sensitivity-specific scaling, "
    "tolerances or norm).")


class NonFiniteFSAError(FloatingPointError):
    """Raised when an FSA solve produces a non-finite state, sensitivity, loss or gradient."""


def chemkan_parameter_group(name: str) -> str:
    """Paper parameter group of a dynamics-relative parameter name.

    ``kinetic.*`` -> theta_kin, ``thermo.linear.*`` -> theta_thermo,
    ``thermo.correction.*`` -> theta_cor (any leading prefix such as ``model.``).
    """
    parts = name.split(".")
    if "kinetic" in parts:
        return GROUP_KIN
    if "thermo" in parts:
        sub = parts[parts.index("thermo") + 1]
        if sub == "linear":
            return GROUP_THERMO
        if sub == "correction":
            return GROUP_COR
        raise ValueError(f"unrecognized thermodynamic parameter {name!r}")
    return GROUP_OTHER


@dataclass(frozen=True)
class ParameterPacking:
    """Deterministic flat layout of the OPTIMIZED parameters of a dynamics module.

    Order is the dynamics module's ``named_parameters()`` order restricted to the
    optimized parameters. Every optimized parameter appears exactly once; parameters
    that are not optimized never appear.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]
    offsets: tuple[int, ...]
    groups: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(self.numels)

    @classmethod
    def from_module(cls, module: nn.Module, optimized: Iterable[torch.Tensor]):
        optimized = list(optimized)
        ids = [id(p) for p in optimized]
        if len(set(ids)) != len(ids):
            raise ValueError("the optimized parameter list contains a tensor twice")
        by_id = {id(p): p for p in optimized}
        names, shapes, numels, groups = [], [], [], []
        seen = set()
        for name, p in module.named_parameters():
            if id(p) in by_id:
                names.append(name)
                shapes.append(tuple(p.shape))
                numels.append(p.numel())
                groups.append(chemkan_parameter_group(name))
                seen.add(id(p))
        missing = [i for i in ids if i not in seen]
        if missing:
            raise ValueError(f"{len(missing)} optimized parameter(s) are not parameters of "
                             f"the dynamics module; their sensitivities cannot be formed")
        offsets, acc = [], 0
        for k in numels:
            offsets.append(acc)
            acc += k
        packing = cls(tuple(names), tuple(shapes), tuple(numels), tuple(offsets), tuple(groups))
        expected = sum(p.numel() for p in optimized)
        if packing.total != expected:
            raise AssertionError(f"packed {packing.total} parameters, optimizer holds {expected}")
        return packing

    def pack(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """name -> tensor  ->  flat (P,) vector in packing order."""
        if set(tensors) != set(self.names):
            raise KeyError("pack() needs exactly the packed parameter names")
        flat = torch.cat([tensors[n].reshape(-1) for n in self.names])
        assert flat.numel() == self.total
        return flat

    def unpack(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """flat (..., P)  ->  name -> tensor view of shape (..., *param.shape)."""
        if flat.shape[-1] != self.total:
            raise ValueError(f"expected trailing dimension {self.total}, got {flat.shape[-1]}")
        lead = flat.shape[:-1]
        chunks = torch.split(flat, list(self.numels), dim=-1)
        return {n: c.reshape(tuple(lead) + tuple(s))
                for n, c, s in zip(self.names, chunks, self.shapes)}

    def group_indices(self) -> dict[str, torch.Tensor]:
        """Group name -> LongTensor of flat indices belonging to that group."""
        out: dict[str, list[int]] = {}
        for g, off, k in zip(self.groups, self.offsets, self.numels):
            out.setdefault(g, []).extend(range(off, off + k))
        return {g: torch.tensor(v, dtype=torch.long) for g, v in out.items()}

    def describe(self) -> dict:
        """JSON-able record for configs and checkpoints."""
        counts: dict[str, int] = {}
        for g, k in zip(self.groups, self.numels):
            counts[g] = counts.get(g, 0) + k
        return {"total": self.total, "group_counts": counts,
                "order": [{"name": n, "shape": list(s), "offset": o, "numel": k, "group": g}
                          for n, s, o, k, g in zip(self.names, self.shapes, self.offsets,
                                                   self.numels, self.groups)]}


def trajectory_buffer_axes(module: nn.Module) -> dict[str, int]:
    """Buffers that carry one entry per trajectory, with their trajectory axis.

    Modules declare these through a ``trajectory_buffer_axes`` class attribute (the
    temperature providers do). Returned names are relative to ``module``.
    """
    axes = {}
    for prefix, sub in module.named_modules():
        for buf, ax in getattr(sub, "trajectory_buffer_axes", {}).items():
            axes[f"{prefix}.{buf}" if prefix else buf] = int(ax)
    return axes


class FunctionalDynamics:
    r"""torch.func view of an EXISTING dynamics wrapper, one trajectory at a time.

    ``F(t, x_b; theta)`` for trajectory ``b`` is evaluated by ``functional_call`` on the
    unchanged wrapper with a batch of one: the optimized parameters come from the flat
    vector ``theta``, every other parameter and every buffer is the module's own detached
    tensor, and per-trajectory buffers (declared via ``trajectory_buffer_axes``) are
    sliced to trajectory ``b``. ``vmap`` over ``b`` therefore cannot mix trajectories.
    """

    def __init__(self, dynamics: nn.Module, optimized: Iterable[torch.Tensor]):
        optimized = list(optimized)
        self.module = dynamics
        self.packing = ParameterPacking.from_module(dynamics, optimized)
        named = dict(dynamics.named_parameters())
        self.params = [named[n] for n in self.packing.names]
        self.constant_params = {k: v.detach() for k, v in named.items()
                                if k not in self.packing.names}
        buffers = dict(dynamics.named_buffers())
        self.trajectory_axes = trajectory_buffer_axes(dynamics)
        unknown = set(self.trajectory_axes) - set(buffers)
        if unknown:
            raise KeyError(f"declared trajectory buffers not found: {sorted(unknown)}")
        self.trajectory_buffers = {k: buffers[k] for k in self.trajectory_axes}
        self.shared_buffers = {k: v for k, v in buffers.items()
                               if k not in self.trajectory_axes}

    def theta(self) -> torch.Tensor:
        """Current optimized parameters, detached, as the flat (P,) vector."""
        return self.packing.pack({n: p.detach() for n, p in zip(self.packing.names,
                                                                 self.params)})

    def _check_batch(self, batch: int) -> None:
        for name, ax in self.trajectory_axes.items():
            size = self.trajectory_buffers[name].shape[ax]
            if size != batch:
                raise ValueError(f"trajectory buffer {name!r} has {size} trajectories "
                                 f"along axis {ax}, state batch is {batch}")

    def _single(self, x_b, theta, traj_b, t):
        """One trajectory: x_b (n,) -> F (n,), through the existing wrapper's forward."""
        tensors = dict(self.constant_params)
        tensors.update(self.shared_buffers)
        tensors.update(self.packing.unpack(theta))
        for name, ax in self.trajectory_axes.items():
            tensors[name] = traj_b[name].unsqueeze(ax)       # restore a batch of one
        out = functional_call(self.module, tensors, (t, x_b.unsqueeze(0)), strict=True)
        return out.squeeze(0)

    def _in_dims(self):
        return (0, None, dict(self.trajectory_axes), None)

    def rhs(self, t, x, theta=None) -> torch.Tensor:
        """Batched functional RHS  x (B, n) -> F (B, n)  (the equivalence test target)."""
        self._check_batch(x.shape[0])
        theta = self.theta() if theta is None else theta
        return vmap(self._single, in_dims=self._in_dims())(x, theta,
                                                           self.trajectory_buffers, t)

    def rhs_and_jacobians(self, t, x, theta):
        """F (B, n), dF/dx (B, n, n), dF/dtheta (B, n, P) at the current solver time."""
        self._check_batch(x.shape[0])

        def f(x_b, th, traj_b, t_):
            out = self._single(x_b, th, traj_b, t_)
            return out, out

        jac = jacrev(f, argnums=(0, 1), has_aux=True)
        (J_x, J_theta), F = vmap(jac, in_dims=self._in_dims())(x, theta,
                                                                self.trajectory_buffers, t)
        return F, J_x, J_theta

    def accumulate_grad(self, grad_flat: torch.Tensor) -> None:
        """Add a flat gradient to ``.grad`` of every optimized parameter (explicit zeros too).

        Accumulates like ``loss.backward()`` does, so observed-interval training can sum
        interval contributions before one optimizer step.
        """
        for p, g in zip(self.params, self.packing.unpack(grad_flat).values()):
            g = g.to(dtype=p.dtype, device=p.device)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.add_(g)


class AugmentedSensitivityRHS(nn.Module):
    r"""RHS of the augmented system  z = [x, S]  (B, n, 1 + P):

        dx/dt = F(x, t)
        dS/dt = (dF/dx) S + dF/dtheta

    ``theta`` is fixed for the whole solve.
    """

    def __init__(self, fdyn: FunctionalDynamics, theta: torch.Tensor):
        super().__init__()
        self.fdyn = fdyn
        self.theta = theta

    def forward(self, t, z):
        x, S = z[..., 0], z[..., 1:]
        F, J_x, J_theta = self.fdyn.rhs_and_jacobians(t, x, self.theta)
        dS = torch.baddbmm(J_theta, J_x, S)                   # (dF/dx) S + dF/dtheta
        return torch.cat([F.unsqueeze(-1), dS], dim=-1)


def _require_finite(what: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise NonFiniteFSAError(f"FSA produced a non-finite {what}; aborting")


def integrate_with_sensitivities(fdyn: FunctionalDynamics, x0: torch.Tensor,
                                 t: torch.Tensor, config: SolverConfig):
    r"""Solve state and forward sensitivities together.

        x0 : (B, n)   t : (T,)   ->   x (T, B, n),  S (T, B, n, P)

    ``S(t0) = 0`` is created fresh for this solve; nothing is carried between solves.
    Returned tensors carry no autograd graph.
    """
    theta = fdyn.theta()
    B, n = x0.shape
    with torch.no_grad():
        z0 = torch.cat([x0.detach().unsqueeze(-1),
                        x0.new_zeros(B, n, fdyn.packing.total)], dim=-1)
        z = odeint(AugmentedSensitivityRHS(fdyn, theta), z0, t,
                   method=config.method, rtol=config.rtol, atol=config.atol)
    return z[..., 0].contiguous(), z[..., 1:].contiguous()


def contract_sensitivities(S: torch.Tensor, dL_dx: torch.Tensor) -> torch.Tensor:
    """dL/dtheta = sum over observation times, trajectories and states of S^T dL/dx."""
    return torch.einsum("tbnp,tbn->p", S, dL_dx)


def fsa_loss_and_gradients(dynamics: nn.Module, optimized: Iterable[torch.Tensor],
                           x0: torch.Tensor, t: torch.Tensor, config: SolverConfig,
                           loss_fn: Callable):
    r"""One FSA gradient evaluation. Accumulates into ``.grad``; returns ``loss_fn``'s output.

    1. integrate the state and its sensitivities (no graph through the solver);
    2. ``dL/dx = autograd.grad(loss_fn(pred_leaf), pred_leaf)`` -- only the loss is
       differentiated, on a detached leaf copy of the prediction;
    3. ``dL/dtheta = sum_j S(t_j)^T dL/dx(t_j)``, added to the parameters' ``.grad``.
    """
    fdyn = FunctionalDynamics(dynamics, optimized)
    pred, S = integrate_with_sensitivities(fdyn, x0, t, config)
    _require_finite("state", pred)
    _require_finite("sensitivity", S)
    pred_leaf = pred.detach().requires_grad_(True)
    out = loss_fn(pred_leaf)
    loss = out[0] if isinstance(out, tuple) else out
    _require_finite("loss", loss.detach())
    (dL_dx,) = torch.autograd.grad(loss, pred_leaf)
    grad = contract_sensitivities(S, dL_dx)
    _require_finite("parameter gradient", grad)
    fdyn.accumulate_grad(grad)
    return out


def fsa_provenance(packings: Mapping[str, ParameterPacking]) -> dict:
    """Config/checkpoint record of the FSA formulation and each stage's parameter layout."""
    return {"formulation": FSA_FORMULATION, "error_control": FSA_ERROR_CONTROL,
            "sensitivity_initial_condition": "S(t0) = 0, created fresh for every solve",
            "parameter_packing": {k: v.describe() for k, v in packings.items()}}
