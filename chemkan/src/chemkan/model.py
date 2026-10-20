r"""The ChemKAN model mathematics ONLY (ChemKAN Eq. 13-17, Fig. 2).

This module deliberately contains no ``odeint``, no optimizer, no temperature
interpolation, and no training-stage flags. It maps a thermochemical state to its
time derivative and nothing else. Adapting these to the solver's ``(t, state)``
interface is done in ``dynamics.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .kan.layers import AddKANLayer, LeanKANLayer


class KineticCore(nn.Module):
    r"""Species-rate network  KAN_kin = Psi^lean_1 o Psi^add_0  (Eq. 13, 16).

        input  u = [Y_1..Y_m, T] : (B, m+1)
        output dY/dt             : (B, m)
    """

    def __init__(self, species_dim: int, hidden_dim: int, num_basis: int, n_mu: int,
                 *, use_base_act: bool):
        super().__init__()
        m = species_dim
        self.add = AddKANLayer(m + 1, hidden_dim, num_basis, use_base_act=use_base_act)
        self.lean = LeanKANLayer(hidden_dim, m, n_mu=n_mu, num_basis=num_basis,
                                 use_base_act=use_base_act)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # (B, m+1) -> (B, m)
        return self.lean(self.add(u))


class ThermodynamicSuperstructure(nn.Module):
    r"""Temperature-rate structure (Eq. 14-15, 17):

        dT/dt = Linear(dY/dt) + KAN_cor(u)

    The linear map holds the paper's ``m`` scalar coefficients (~ -h_i / c_p), hence
    ``bias=False``. ``KAN_cor`` is a single-output additive KAN over the full state.
    """

    def __init__(self, species_dim: int, num_basis: int, *, use_base_act: bool):
        super().__init__()
        m = species_dim
        self.linear = nn.Linear(m, 1, bias=False)                       # Eq. 14 coefficients
        self.correction = AddKANLayer(m + 1, 1, num_basis,             # Eq. 17 KAN_cor
                                      use_base_act=use_base_act)

    def forward(self, u: torch.Tensor, dYdt: torch.Tensor) -> torch.Tensor:
        # u: (B, m+1), dYdt: (B, m) -> dT/dt: (B, 1)
        return self.linear(dYdt) + self.correction(u)


class ChemKAN(nn.Module):
    r"""Full ChemKAN:  u = [Y, T] (B, m+1)  ->  [dY/dt, dT/dt] (B, m+1)  (Eq. 13-17).

    Autonomous (no explicit time dependence) and stateless -- there is deliberately
    NO ``use_thermo`` mode flag. Kinetics-only use (biodiesel, hydrogen Stage 1) is
    expressed by driving ``self.kinetic`` directly through ``dynamics.KineticDynamics``.
    """

    def __init__(self, species_dim: int, hidden_dim: int, num_basis: int, n_mu: int,
                 *, use_base_act: bool):
        super().__init__()
        self.kinetic = KineticCore(species_dim, hidden_dim, num_basis, n_mu,
                                   use_base_act=use_base_act)
        self.thermo = ThermodynamicSuperstructure(species_dim, num_basis,
                                                  use_base_act=use_base_act)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # (B, m+1) -> (B, m+1)
        dYdt = self.kinetic(u)
        dTdt = self.thermo(u, dYdt)
        return torch.cat([dYdt, dTdt], dim=-1)
