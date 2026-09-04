r"""Cantera-derived thermodynamic coefficients for the ChemKAN dT/dt pathway.

DIAGNOSTIC ONLY -- not part of the paper reproduction.

Physics (adiabatic, constant-pressure reactor)
----------------------------------------------
The temperature equation is

    dT/dt = -(1/cp) * sum_k h_k * dY_k/dt

so the coefficient multiplying each species rate ``dY_k/dt`` is

    coefficient_k(Y, T) = -h_k(T) / cp(Y, T)          [K]

with ``h_k`` the species SPECIFIC enthalpy [J/kg] and ``cp`` the mixture specific heat
[J/(kg K)]. Cantera reports ``partial_molar_enthalpies`` in **J/kmol**, so it is divided
by the molecular weight [kg/kmol] EXACTLY ONCE to reach J/kg:

    h_specific_k = partial_molar_enthalpies[k] / molecular_weights[k]

``ChemKAN``'s ``thermo.linear`` (Eq. 14) holds one CONSTANT scalar per species, whereas
the physical coefficients above depend on temperature and mixture. A learned constant is
therefore NOT expected to equal the Cantera value pointwise -- the comparison is a
**magnitude / sign / thermodynamic-structure** diagnostic, and the ``KAN_cor`` term
(Eq. 17) is what is meant to absorb the residual state dependence.
"""

from __future__ import annotations

import numpy as np

MECH = "h2o2.yaml"


def cantera_coefficients(temperature: float, mass_fractions, *, species,
                         mech: str = MECH, pressure: float | None = None):
    """``-h_k/cp`` in KELVIN for each species, in the given ``species`` order.

    ``mass_fractions`` is an array aligned with ``species`` (the ChemKAN state order).
    Returns ``(coeffs (m,), cp_mass)``.
    """
    import cantera as ct

    if pressure is None:
        pressure = ct.one_atm
    gas = ct.Solution(mech)
    comp = {sp: float(y) for sp, y in zip(species, np.asarray(mass_fractions).ravel())}
    gas.TPY = float(temperature), float(pressure), comp

    idx = [gas.species_index(sp) for sp in species]           # map to ChemKAN order
    h_molar = gas.partial_molar_enthalpies                    # J/kmol
    w_molar = gas.molecular_weights                           # kg/kmol
    h_mass = h_molar / w_molar                                # J/kg  (single MW division)
    cp = gas.cp_mass                                          # J/(kg K)
    return np.array([-h_mass[i] / cp for i in idx]), float(cp)


def reference_state_indices(t, ref) -> dict[str, int]:
    """Indices of diagnostic states along a stored reference trajectory.

    ``ref`` is (Nt, m+1) = [Y..., T]. Returns initial / pre_ignition / ignition /
    post_ignition indices, where ignition is the maximum reference dT/dt.
    """
    t = np.asarray(t, dtype=float)
    dT = np.gradient(np.asarray(ref, dtype=float)[:, -1], t)
    k_ign = int(dT.argmax())
    return {
        "initial": 0,
        "pre_ignition": max(0, k_ign // 2),          # midway through induction
        "ignition": k_ign,
        "post_ignition": len(t) - 1,
    }


def coefficients_at_states(t, ref, *, species, mech: str = MECH, pressure=None) -> dict:
    """``{state_name: {"index", "T", "cp_mass", "coeffs"}}`` for the diagnostic states."""
    out = {}
    for name, k in reference_state_indices(t, ref).items():
        T = float(ref[k, -1])
        coeffs, cp = cantera_coefficients(T, ref[k, :len(species)], species=species,
                                          mech=mech, pressure=pressure)
        out[name] = {"index": k, "T": T, "cp_mass": cp, "coeffs": coeffs}
    return out


def resolve_training_ic(train_ics, train_states, temperature: float, phi: float, *,
                        species_dim: int, tol: float = 1e-9):
    """Initial species composition of the EXACT training condition ``(temperature, phi)``.

    Exact match only (within ``tol``) -- never nearest-neighbour, so a mistyped reference
    state fails loudly instead of silently selecting a different condition. Returns the
    composition as float32, matching what the training loader feeds the model.

    NOTE: mass fractions depend on ``phi`` only; ``temperature`` selects the training row
    and is what the caller passes to Cantera for the enthalpy evaluation.
    """
    import numpy as _np

    ics = _np.asarray(train_ics)
    hits = _np.where((_np.abs(ics[:, 0] - float(temperature)) < tol)
                     & (_np.abs(ics[:, 1] - float(phi)) < tol))[0]
    if len(hits) != 1:
        available = ", ".join(f"({t:g},{p:g})" for t, p in ics[:8])
        raise SystemExit(
            f"reference state (T0={temperature:g} K, phi={phi:g}) is not an exact training "
            f"initial condition ({len(hits)} matches). Pass a real training IC. "
            f"First few available: {available} ...")
    return _np.asarray(train_states)[hits[0], 0, :species_dim].astype(_np.float32)
