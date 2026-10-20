r"""ODE right-hand sides:  adapt the model math to the solver's (t, state) interface.

These adapters carry NO trainable physics of their own -- they assemble the state the
model expects, apply the pre-KAN input scaling, and forward ``t`` (which the
autonomous ChemKAN ignores).

Representation boundary (see ASSUMPTIONS.md)
-------------------------------------------
The ODE solver always integrates PHYSICAL coordinates and these adapters always
return PHYSICAL derivatives. The ``input_normalizer`` transforms only the COPY of the
current physical state handed to the KAN -- it never touches the integrated state or
the returned derivative. This is the fix for raw dimensional temperatures saturating
the KAN's internal ``tanh`` (``tanh(1000 K) == 1``): the physical state is min-max
scaled to a sane range BEFORE that ``tanh``. ``input_normalizer=None`` is an explicit
raw-input diagnostic/ablation.

* KineticDynamics -- integrates species only (B, m); temperature is supplied
  externally by a temperature provider and concatenated before scaling + kinetic core.
* ChemKANDynamics -- integrates the full state (B, m+1) with the complete ChemKAN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KineticDynamics(nn.Module):
    r"""Biodiesel & hydrogen Stage 1:  state Y (B, m) -> physical dY/dt (B, m).

    Temperature is not integrated; it is provided by ``temperature(t) -> (B, 1)`` in
    physical Kelvin and concatenated to form the physical ``u = [Y, T]`` (B, m+1). That
    full physical state is then optionally scaled before the kinetic core.

    ``input_normalizer`` is REQUIRED (no default): pass a full-state ``MinMaxNormalizer``
    for the main reproduction, or ``None`` for the explicit raw-input ablation.
    """

    def __init__(self, kinetic_core: nn.Module, temperature: nn.Module, *,
                 input_normalizer: nn.Module | None):
        super().__init__()
        self.kinetic = kinetic_core
        self.temperature = temperature
        self.input_normalizer = input_normalizer

    def forward(self, t: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:  # (B, m)->(B, m)
        T = self.temperature(t)                              # (B, 1) physical Kelvin
        u_physical = torch.cat([Y, T], dim=-1)               # (B, m+1) physical
        u_model = (u_physical if self.input_normalizer is None
                   else self.input_normalizer.normalize(u_physical))
        return self.kinetic(u_model)                         # physical dY/dt (B, m)


class ChemKANDynamics(nn.Module):
    r"""Hydrogen Stage 2 / final inference:  physical u=[Y,T] (B, m+1) -> physical (B, m+1).

    ``t`` is accepted for the solver's signature but the full ChemKAN is autonomous.
    ``input_normalizer`` is REQUIRED (no default); see ``KineticDynamics``.
    """

    def __init__(self, model: nn.Module, *, input_normalizer: nn.Module | None):
        super().__init__()
        self.model = model
        self.input_normalizer = input_normalizer

    def forward(self, t: torch.Tensor, u_physical: torch.Tensor) -> torch.Tensor:
        # (B, m+1) physical -> (B, m+1) physical derivative
        u_model = (u_physical if self.input_normalizer is None
                   else self.input_normalizer.normalize(u_physical))
        return self.model(u_model)
