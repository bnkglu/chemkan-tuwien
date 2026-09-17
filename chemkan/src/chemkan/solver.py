r"""Numerical integration, kept isolated from the physics and the training loop.

Paper vs. this implementation
-----------------------------
The ChemKAN paper integrates with **Tsit5** and computes gradients with **Forward
Sensitivity Analysis (FSA)**. This PyTorch reproduction:

    method       = "tsit5"            # the SAME Tsit5 method as the paper, provided by
                                      # the pinned GitHub torchdiffeq (see requirements)
    sensitivity  = "direct_autograd"  # backprop THROUGH odeint (NOT odeint_adjoint,
                                      # NOT Forward Sensitivity Analysis) -- the
                                      # historical backend of every earlier run
                 | "fsa"              # continuous forward sensitivity analysis,
                                      # see ``chemkan.fsa``

``sensitivity`` only selects how TRAINING gradients are formed. ``integrate`` below is
the ordinary state-only solve for both backends and for all inference; the FSA
augmented state/sensitivity solve lives in ``chemkan.fsa``. Direct autograd is a
different mechanism from FSA and is not claimed equivalent to it.

``rtol=1e-6`` / ``atol=1e-8`` are implementation choices (not stated paper values); the
experiment scripts pass them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchdiffeq import odeint


SENSITIVITY_BACKENDS = ("direct_autograd", "fsa")


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
        if self.sensitivity not in SENSITIVITY_BACKENDS:
            raise ValueError(
                f"sensitivity must be one of {SENSITIVITY_BACKENDS}; got "
                f"{self.sensitivity!r} (odeint_adjoint is intentionally not used)."
            )


def integrate(func, y0: torch.Tensor, t: torch.Tensor,
              config: SolverConfig) -> torch.Tensor:
    r"""Integrate ``dy/dt = func(t, y)`` from ``y0`` over grid ``t``.

        y0 : (B, dim)   t : (T,)   ->   (T, B, dim)

    ``config`` is required -- there is no fallback solver configuration inside the
    reusable library. Uses ``torchdiffeq.odeint``; gradients, when requested, flow
    through the solver (direct autograd) and ``odeint_adjoint`` is deliberately not used.
    This is the state-only solve for BOTH sensitivity backends: ``config.sensitivity``
    is not consulted here (FSA training uses ``chemkan.fsa``).
    """
    return odeint(func, y0, t, method=config.method, rtol=config.rtol, atol=config.atol)
