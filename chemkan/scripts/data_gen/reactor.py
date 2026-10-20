"""Shared 0-D homogeneous reactor helper (Cantera), used by the H2 and CH4 cases."""

from __future__ import annotations

import cantera as ct
import numpy as np


def species_index(mech: str, drop: tuple[str, ...] = ()) -> tuple[list[str], np.ndarray]:
    """Species kept in the state vector, and their indices in the Cantera phase."""
    gas = ct.Solution(mech)
    names = [s for s in gas.species_names if s not in drop]
    idx = np.array([gas.species_index(n) for n in names])
    return names, idx


def integrate_case(
    mech: str,
    fuel: str,
    oxidizer: dict,
    T0: float,
    phi: float,
    t: np.ndarray,
    pressure: float,
    keep: np.ndarray,
    rtol: float = 1e-9,
    atol: float = 1e-15,
) -> np.ndarray:
    """Generate one trajectory from an adiabatic constant-pressure reactor.

    Sampled on the fixed grid `t`. Returns an array of shape
    (len(t), len(keep) + 1) holding [Y_1, ..., Y_m, T] at each sample time.
    """
    # 0-D homogeneous constant-pressure reactor used for autoignition trajectories.
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = T0, pressure

    # reactor = ct.IdealGasConstPressureReactor(gas, clone = False) # doesn't work for 3.0.0
    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])
    net.rtol, net.atol = rtol, atol

    # Store species mass fractions and temperature over the requested save grid.
    states = np.empty((len(t), len(keep) + 1))
    states[0, :-1] = gas.Y[keep]
    states[0, -1] = T0

    for j in range(1, len(t)):
        net.advance(t[j])
        states[j, :-1] = reactor.Y[keep]
        states[j, -1] = reactor.T

    return states
