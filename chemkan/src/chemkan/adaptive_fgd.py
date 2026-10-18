"""Experimental adaptive functional gradient descent on ChemKAN edge functions.

Algorithm 1 of Csillag et al., arXiv:2606.16926, applied to a product of
univariate Gaussian RKHSs. Autograd supplies the coefficients of each functional
gradient through the *discretized* ODE solve; it does not optimize RBF weights.
See docs/adaptive_fgd.md for the derivation and limits of this adaptation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .kan.rbf import RBFEdgeFunctions


def interpolate_edges(values: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Uniform piecewise-linear functions on [-1, 1]: (O,I,Q), (B,I) -> (B,O,I).

    Linear interpolation retains derivatives with respect to the ODE state. A
    piecewise-constant tree (used by the reference regression experiment) would
    discard those derivatives almost everywhere.
    """
    intervals = values.shape[-1] - 1
    position = (z.clamp(-1, 1) + 1) * (intervals / 2)
    left = position.floor().long().clamp(0, intervals - 1)
    weight = position - left
    inputs = torch.arange(values.shape[1], device=values.device)[None, :]
    lo = values[:, inputs, left].permute(1, 0, 2)
    hi = values[:, inputs, left + 1].permute(1, 0, 2)
    return lo + (hi - lo) * weight[:, None, :]


class AdaptiveEdgeFunctions(nn.Module):
    """Frozen original RBF edge plus a refinable, initially zero function.

    Refinement inserts midpoint values exactly, preserving the current function.
    The values are Parameters to let autograd traverse the solver, but updates
    come from RKHS gradients, never their ordinary parameter gradients.
    """

    def __init__(self, base: RBFEdgeFunctions, intervals: int):
        super().__init__()
        if intervals < 1:
            raise ValueError("intervals must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.values = nn.Parameter(base.w_rbf.new_zeros(*base.w_rbf.shape[:2], intervals + 1))
        self.capture = False
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def intervals(self) -> int:
        return self.values.shape[-1] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match the wrapped edge's own basis coordinates. Author-matched models
        # deliberately omit tanh on their first and correction KAN layers.
        # The additive function is constant outside its represented domain.
        # Its RKHS samples must use the same coordinates as its evaluation.
        z = self.base.representation_input(x).clamp(-1, 1)
        out = self.base(x) + interpolate_edges(self.values, z)
        if self.capture and out.requires_grad:
            # Capture the adjoint of the EDGE OUTPUT, before sum/product reduction.
            # No additional mean: the loss already contains its own normalization.
            points = z.detach()

            def record(grad):
                self.samples.append((points, grad.detach()))

            out.register_hook(record)
        return out

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        value = state_dict.get(prefix + "values")
        if (
            value is not None
            and value.ndim == 3
            and value.shape[-1] >= 2
            and value.shape[:2] == self.values.shape[:2]
        ):
            # Refined checkpoints can be loaded into a fresh coarse model.
            self.values = nn.Parameter(self.values.new_empty(value.shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def adaptive_edges(model: nn.Module, *, initial_intervals: int = 4) -> dict:
    """Replace RBF edge blocks in place; predictions are unchanged initially.

    Works on KineticCore and full ChemKAN, including thermodynamic correction
    edges. Other trainable parameters (e.g. thermo.linear) stay trainable.
    Calling again preserves already-adaptive blocks.
    """
    if initial_intervals < 1:
        raise ValueError("initial_intervals must be positive")

    def convert(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, AdaptiveEdgeFunctions):
                continue
            if isinstance(child, RBFEdgeFunctions):
                setattr(parent, name, AdaptiveEdgeFunctions(child, initial_intervals))
            else:
                convert(child)

    if isinstance(model, RBFEdgeFunctions):
        raise TypeError("pass the containing KAN layer or model, not a bare RBF block")
    convert(model)
    blocks = {n: m for n, m in model.named_modules() if isinstance(m, AdaptiveEdgeFunctions)}
    if not blocks:
        raise ValueError("model contains no ChemKAN RBF edge functions")
    return blocks


@dataclass(frozen=True)
class AdaptiveFGDConfig:
    lr: float = 0.01
    epsilon: float = 0.5
    gamma: float = 100.0
    initial_intervals: int = 4
    max_intervals: int = 4096
    gradient_atol: float = 1e-12
    max_backtracks: int = 24
    armijo: float = 1e-4
    line_search_window: int = 1
    sample_chunk: int = 2048
    node_chunk: int = 128
    refine: bool = True

    def __post_init__(self):
        for name in ("lr", "gamma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("epsilon", "armijo"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
        for name in (
            "initial_intervals", "max_intervals", "sample_chunk", "node_chunk",
            "line_search_window",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_intervals < self.initial_intervals:
            raise ValueError("max_intervals must be >= initial_intervals")
        if (
            not math.isfinite(self.gradient_atol)
            or self.gradient_atol < 0
            or not isinstance(self.max_backtracks, int)
            or self.max_backtracks < 0
        ):
            raise ValueError("invalid gradient_atol or max_backtracks")


class RefinementLimitError(RuntimeError):
    """The requested approximation certificate cannot be met within the budget."""


class LineSearchError(RuntimeError):
    """No finite sufficient-descent step was found; the model was restored."""


@torch.no_grad()
def kernel_gradient(
    points: torch.Tensor,
    coefficients: torch.Tensor,
    nodes: torch.Tensor,
    config: AdaptiveFGDConfig,
) -> torch.Tensor:
    r"""Evaluate G_oi(z) = sum_n a_noi exp(-gamma (z - x_ni)^2).

    points: (N,I), coefficients: (N,O,I), return: (O,I,Q). Both axes
    are chunked, avoiding an unbounded solver-samples x representation matrix.
    """
    result = coefficients.new_zeros(*coefficients.shape[1:], nodes.numel())
    for q in range(0, nodes.numel(), config.node_chunk):
        z = nodes[q : q + config.node_chunk]
        for n in range(0, points.shape[0], config.sample_chunk):
            x = points[n : n + config.sample_chunk]
            a = coefficients[n : n + config.sample_chunk]
            kernels = torch.exp(-config.gamma * (z[:, None, None] - x[None, :, :]).square())
            result[..., q : q + z.numel()] += torch.einsum("qni,noi->oiq", kernels, a)
    return result


def interpolation_error_bound(coefficients: torch.Tensor, intervals: int, gamma: float) -> float:
    r"""Analytic sup-error bound over the WHOLE [-1,1] edge domain.

    A = max_oi sum_n |a_noi|, h = 2 / intervals.
    |G'| <= A sqrt(2 gamma/e); |G''| <= 2 gamma A.
    Linear interpolation has error <= min(L h/2, M h^2/8).
    This bounds representation error, not numerical ODE/roundoff error.
    """
    mass = float(coefficients.abs().sum(dim=0).max())
    h = 2.0 / intervals
    return mass * min(math.sqrt(2 * gamma / math.e) * h / 2, gamma * h * h / 4)


class AdaptiveFGD:
    """Functional-gradient steps with certified refinement and Armijo backtracking.

    ``closure()`` must recompute a deterministic scalar loss with gradients enabled
    and must not call backward itself. Use direct autograd through the ODE, not
    FSA/adjoint or gradient checkpointing. The original RBF coefficients are fixed;
    non-edge parameters receive exact Euclidean gradient steps at the same rate.
    """

    def __init__(self, model: nn.Module, config: AdaptiveFGDConfig):
        self.model = model
        self.config = config
        self.blocks = adaptive_edges(model, initial_intervals=config.initial_intervals)
        self.loss_history: list[float] = []
        if any(b.intervals > config.max_intervals for b in self.blocks.values()):
            raise ValueError("existing representation exceeds max_intervals")

    def state_dict(self) -> dict:
        """State needed to continue the optional nonmonotone line search."""
        return {"loss_history": list(self.loss_history)}

    def load_state_dict(self, state: dict) -> None:
        history = [float(value) for value in state["loss_history"]]
        if any(not math.isfinite(value) for value in history):
            raise ValueError("nonfinite line-search history")
        keep = self.config.line_search_window - 1
        self.loss_history = history[-keep:] if keep else []

    def step(self, closure: Callable[[], torch.Tensor]) -> dict:
        self.model.zero_grad(set_to_none=True)
        for block in self.blocks.values():
            block.samples.clear()
            block.capture = True
        try:
            with torch.enable_grad():
                loss = closure()
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("closure must return a finite scalar loss")
                loss.backward()
        except BaseException:
            for block in self.blocks.values():
                block.samples.clear()
            raise
        finally:
            for block in self.blocks.values():
                block.capture = False

        try:
            return self._update(closure, float(loss.detach()))
        finally:
            for block in self.blocks.values():
                block.samples.clear()
            self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _update(self, closure, loss_before):
        cfg = self.config
        before = {name: b.intervals for name, b in self.blocks.items()}
        counts = dict(before)
        samples = {}
        for name, block in self.blocks.items():
            if block.samples:
                x, a = (torch.cat(parts) for parts in zip(*block.samples))
            else:
                o, i = block.values.shape[:2]
                x, a = block.values.new_zeros(0, i), block.values.new_zeros(0, o, i)
            if not bool(torch.isfinite(x).all() & torch.isfinite(a).all()):
                raise ValueError("nonfinite functional gradient samples")
            samples[name] = (x, a)

        table_ids = {id(b.values) for b in self.blocks.values()}
        exact = [
            (p, p.grad.detach().clone())
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in table_ids and p.grad is not None
        ]
        if any(not bool(torch.isfinite(g).all()) for _, g in exact):
            raise ValueError("nonfinite Euclidean gradient")
        exact_norm = max((float(g.abs().max()) for _, g in exact), default=0.0)
        approximations, bounds = {}, {}
        pending = set(self.blocks)
        while True:
            for name in pending:
                block = self.blocks[name]
                nodes = torch.linspace(
                    -1, 1, counts[name] + 1, dtype=block.values.dtype, device=block.values.device
                )
                x, a = samples[name]
                approximations[name] = kernel_gradient(x, a, nodes, cfg)
                bounds[name] = interpolation_error_bound(a, counts[name], cfg.gamma)
            norm = max(exact_norm, *(float(g.abs().max()) for g in approximations.values()))
            bound = max(bounds.values())
            if not math.isfinite(norm + bound):
                raise ValueError("nonfinite gradient approximation or error bound")
            stationary = norm + bound <= cfg.gradient_atol
            certified = (1 + cfg.epsilon) * bound < cfg.epsilon * norm
            if stationary or certified or not cfg.refine:
                break
            pending = {n for n, u in bounds.items() if (1 + cfg.epsilon) * u >= cfg.epsilon * norm}
            for name in pending:
                if counts[name] * 2 > cfg.max_intervals:
                    raise RefinementLimitError(
                        f"{name}: cannot certify (1+epsilon)*U < epsilon*||g|| "
                        f"at {counts[name]} intervals (U={bound:.6g}, norm={norm:.6g}); "
                        "no update applied. Increase max_intervals or relax epsilon."
                    )
                counts[name] *= 2

        diagnostics = {
            "loss_before": loss_before,
            "loss_after": loss_before,
            "gradient_norm": norm,
            "error_bound": bound,
            "relative_error_bound": bound / norm if norm else 0.0,
            "certified": certified,
            "stationary": stationary,
            "intervals_before": before,
            "intervals_after": dict(before),
            "gradient_intervals": counts,
            "sample_count": {n: int(x.shape[0]) for n, (x, _) in samples.items()},
            "step_size": 0.0,
            "backtracks": 0,
            "directional_derivative": 0.0,
            "line_search_window": cfg.line_search_window,
            "solver_trial_failures": [],
        }
        if stationary:
            return diagnostics

        # Compute DL[g] from the same adjoints; test the actual descent direction
        # because the paper's Banach/Hilbert compatibility is not proven for this ODE.
        slope = sum(
            float((interpolate_edges(approximations[n], x) * a).sum())
            for n, (x, a) in samples.items()
        )
        slope += sum(float(g.square().sum()) for _, g in exact)
        diagnostics["directional_derivative"] = slope
        if not math.isfinite(slope) or slope <= 0:
            raise LineSearchError(
                "certified gradient was not a descent direction; no update applied"
            )

        originals = {n: b.values for n, b in self.blocks.items()}
        refined = {
            n: F.interpolate(
                b.values.detach(), size=counts[n] + 1, mode="linear", align_corners=True
            )
            for n, b in self.blocks.items()
        }
        exact_originals = [p.detach().clone() for p, _ in exact]
        # window=1 is ordinary monotone Armijo. Larger windows permit bounded
        # temporary increases, using training losses only as the reference.
        reference = max([loss_before, *self.loss_history])
        diagnostics["line_search_reference"] = reference
        accepted = False
        try:
            for backtracks in range(cfg.max_backtracks + 1):
                rate = cfg.lr * 0.5**backtracks
                for name, block in self.blocks.items():
                    block.values = nn.Parameter(refined[name] - rate * approximations[name])
                for (p, g), original in zip(exact, exact_originals):
                    p.copy_(original - rate * g)
                changed = any(
                    not torch.equal(block.values.detach(), refined[name])
                    for name, block in self.blocks.items()
                ) or any(
                    not torch.equal(p.detach(), original)
                    for (p, _), original in zip(exact, exact_originals)
                )
                if not changed:
                    raise LineSearchError(
                        "trial update rounded to no parameter change; model restored"
                    )
                try:
                    trial = closure()
                    if trial.ndim != 0:
                        raise ValueError("closure must return a scalar loss")
                    value = float(trial)
                except (AssertionError, FloatingPointError) as exc:
                    # torchdiffeq reports nonfinite/underflowing trial solves as
                    # AssertionError. Other errors surface with state restored.
                    value = math.inf
                    diagnostics["solver_trial_failures"].append(
                        {"rate": rate, "type": type(exc).__name__, "message": str(exc)}
                    )
                threshold = reference - cfg.armijo * rate * slope
                if math.isfinite(value) and value <= threshold:
                    accepted = True
                    keep = cfg.line_search_window - 1
                    self.loss_history = (self.loss_history + [loss_before])[-keep:] if keep else []
                    diagnostics.update(
                        loss_after=value,
                        step_size=rate,
                        backtracks=backtracks,
                        intervals_after=dict(counts),
                        armijo_threshold=threshold,
                        loss_increased=value > loss_before,
                    )
                    return diagnostics
            raise LineSearchError("backtracking exhausted; model restored without an update")
        finally:
            if not accepted:
                for name, block in self.blocks.items():
                    block.values = originals[name]
                for (p, _), original in zip(exact, exact_originals):
                    p.copy_(original)
