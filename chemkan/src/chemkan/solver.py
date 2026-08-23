r"""Numerical integration, kept isolated from the physics and the training loop.

Paper vs. this implementation (documented, NOT claimed equivalent)
-----------------------------------------------------------------
The ChemKAN paper integrates with **Tsit5** and computes gradients with **Forward
Sensitivity Analysis (FSA)**. This PyTorch reproduction:

    method       = "tsit5"            # the SAME Tsit5 method as the paper, provided by
                                      # the pinned GitHub torchdiffeq (see requirements)
    sensitivity  = "direct_autograd"  # backprop THROUGH odeint (NOT odeint_adjoint,
                                      # NOT Forward Sensitivity Analysis)

So the *integrator* now matches the paper (Tsit5 is available in the pinned
``rtqichen/torchdiffeq`` commit; stock PyPI torchdiffeq does not expose it). The
*sensitivity method* does NOT: FSA remains unimplemented, and direct autograd is a
different mechanism that is **not claimed equivalent** to FSA -- this is a remaining
reproduction gap relative to the paper.

``rtol=1e-6`` / ``atol=1e-8`` are implementation choices (not stated paper values); the
experiment scripts pass them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchdiffeq import odeint


@dataclass
class SolverConfig:
    """Explicit solver settings -- every field must be supplied by the caller.

    There are NO defaults: ``method``/``rtol``/``atol``/``sensitivity`` are documented
    PyTorch implementation assumptions (see module docstring), not universal ChemKAN
    values, so the experiment scripts choose them explicitly.
    """

    method: str
    rtol: float
    atol: float
    sensitivity: str

    def __post_init__(self):
        if self.sensitivity != "direct_autograd":
            raise ValueError(
                "Only 'direct_autograd' is supported here; the paper's Forward "
                "Sensitivity Analysis and odeint_adjoint are intentionally not used."
            )


def integrate(func, y0: torch.Tensor, t: torch.Tensor,
              config: SolverConfig) -> torch.Tensor:
    r"""Integrate ``dy/dt = func(t, y)`` from ``y0`` over grid ``t``.

        y0 : (B, dim)   t : (T,)   ->   (T, B, dim)

    ``config`` is required -- there is no fallback solver configuration inside the
    reusable library. Uses ``torchdiffeq.odeint`` with direct autograd (gradients flow
    through the solver); ``odeint_adjoint`` is deliberately not used.
    """
    return odeint(func, y0, t, method=config.method, rtol=config.rtol, atol=config.atol)
