r"""Isothermal (fixed-temperature) reference chemistry for the kinetic-core diagnostic.

DIAGNOSTIC ONLY. Answers the question raised in the 28.08.2026 supervisor meeting: if the
temperature is held low/fixed, does the learned kinetic core still drive substantial species
reaction, and does it react MORE strongly than the true isothermal chemistry?

The reference here differs from the dataset in exactly one way: the energy equation is
DISABLED, so ``T`` stays at its initial value for the whole integration. Everything else --
mechanism, pressure, species set/order, save grid, solver tolerances -- matches
``data_gen/reactor.integrate_case``.
"""

from __future__ import annotations

import numpy as np

MECH = "h2o2.yaml"
DROP = ("AR",)
FUEL = "H2"
OXIDIZER = {"O2": 1.0, "N2": 3.76}


def isothermal_case(T_fixed: float, phi: float, t, *, mech: str = MECH,
                    pressure: float | None = None, rtol: float = 1e-9,
                    atol: float = 1e-15, drop: tuple[str, ...] = DROP):
    """Cantera species trajectory with the ENERGY EQUATION OFF (T held at ``T_fixed``).

    Returns ``(species_names, Y (len(t), m), T_check (len(t),))``. ``T_check`` is the
    reactor temperature actually observed at each save time -- it must stay flat, and the
    caller should assert that rather than trust the flag.
    """
    import cantera as ct
    if pressure is None:
        pressure = ct.one_atm
    gas = ct.Solution(mech)
    names = [s for s in gas.species_names if s not in drop]
    keep = np.array([gas.species_index(n) for n in names])

    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = float(T_fixed), float(pressure)
    reactor = ct.IdealGasConstPressureReactor(gas)
    reactor.energy_enabled = False                 # <-- the whole point: T is frozen
    net = ct.ReactorNet([reactor])
    net.rtol, net.atol = rtol, atol

    t = np.asarray(t, dtype=float)
    Y = np.empty((len(t), len(keep)))
    T_check = np.empty(len(t))
    Y[0], T_check[0] = gas.Y[keep], reactor.T
    for j in range(1, len(t)):
        net.advance(t[j])
        Y[j], T_check[j] = reactor.Y[keep], reactor.T
    return names, Y, T_check


def initial_composition(phi: float, T: float, *, mech: str = MECH, pressure=None,
                        drop: tuple[str, ...] = DROP):
    """Initial mass fractions for (phi, T), in the repository's species order."""
    import cantera as ct
    if pressure is None:
        pressure = ct.one_atm
    gas = ct.Solution(mech)
    names = [s for s in gas.species_names if s not in drop]
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = float(T), float(pressure)
    return names, np.array([gas.Y[gas.species_index(n)] for n in names])
