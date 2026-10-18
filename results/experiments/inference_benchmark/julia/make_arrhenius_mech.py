r"""Arrhenius.jl mechanism files for the mechanism our data were generated with.

Arrhenius.jl reads ``<mech>.yaml`` (species, thermo, reaction types) and ``<mech>.yaml.npz``
(rate data), which its authors produce with ReacTorch. ReacTorch is not installed, and the
copies bundled with Arrhenius.jl differ from our data's mechanism (its ``h2o2.yaml`` has no
N2), so the npz is written here from Cantera with Arrhenius.jl's conventions (checked against
its bundled ``gri30.yaml.npz``): pre-exponential factors in Cantera's SI units (kmol, m, s),
activation energies in cal/mol, Troe parameters as A, T1, T2, T3, falloff low-pressure rates
in ``Arrhenius_A0/b0/Ea0``, third-body efficiencies with Cantera's default efficiency.

    ~/uni_projects/chemkan-venv/bin/python results/experiments/inference_benchmark/julia/make_arrhenius_mech.py h2o2.yaml
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cantera as ct
import numpy as np

OUT = Path(__file__).resolve().parent / "mechanism"
J_PER_KMOL_PER_CAL_PER_MOL = 4184.0


def convert(mech: str) -> None:
    gas = ct.Solution(mech)
    src = Path(gas.source) if Path(gas.source).exists() else Path(ct.__file__).parent / "data" / mech
    OUT.mkdir(exist_ok=True)
    shutil.copy(src, OUT / mech)
    ns, nr = gas.n_species, gas.n_reactions
    arr, eff = np.zeros((nr, 3)), np.zeros((ns, nr))
    low, troe = [], []
    for i, r in enumerate(gas.reactions()):
        rate = r.rate
        high = rate.high_rate if r.reaction_type.startswith("falloff") else rate
        arr[i] = (high.pre_exponential_factor, high.temperature_exponent,
                  high.activation_energy / J_PER_KMOL_PER_CAL_PER_MOL)
        if r.third_body is not None:
            eff[:, i] = r.third_body.default_efficiency
            for name, e in r.third_body.efficiencies.items():
                eff[gas.species_index(name), i] = e
        if r.reaction_type.startswith("falloff"):
            lo = rate.low_rate
            low.append((lo.pre_exponential_factor, lo.temperature_exponent,
                        lo.activation_energy / J_PER_KMOL_PER_CAL_PER_MOL))
            if r.reaction_type == "falloff-Troe":
                c = list(rate.falloff_coeffs)          # Cantera order: A, T3, T1, (T2)
                troe.append((c[0], c[2], c[3] if len(c) > 3 else 0.0, c[1]))
            elif r.reaction_type != "falloff-Lindemann":
                raise SystemExit(f"unsupported falloff type {r.reaction_type}")
        if r.orders:
            raise SystemExit(f"non-default orders in reaction {i}: {r.equation}")
    # A reaction with one named collider ("H + O2 + O2 <=> HO2 + O2") is a three-body
    # reaction in Cantera 3 (collider not in the stoichiometry) but has no "type" key in the
    # YAML, so Arrhenius.jl treats it as elementary: write the collider back as reactant and
    # product (as in Arrhenius.jl's bundled gri30.yaml.npz).
    nu_r, nu_p = gas.reactant_stoich_coeffs.copy(), gas.product_stoich_coeffs.copy()
    for i, r in enumerate(gas.reactions()):
        if r.third_body is not None and "type" not in r.input_data:
            name = r.third_body.name
            if name == "M":
                raise SystemExit(f"generic collider without type key: {r.equation}")
            nu_r[gas.species_index(name), i] += 1
            nu_p[gas.species_index(name), i] += 1
            eff[:, i] = 0.0
    data = dict(molecular_weights=gas.molecular_weights,
                reactant_stoich_coeffs=nu_r,
                product_stoich_coeffs=nu_p,
                reactant_orders=nu_r,
                is_reversible=np.array([r.reversible for r in gas.reactions()], dtype=float),
                Arrhenius_coeffs=arr, efficiencies_coeffs=eff)
    if low:
        lo = np.array(low)
        data |= dict(Arrhenius_A0=lo[:, 0], Arrhenius_b0=lo[:, 1], Arrhenius_Ea0=lo[:, 2])
    if troe:
        tr = np.array(troe)
        data |= dict(Troe_A=tr[:, 0], Troe_T1=tr[:, 1], Troe_T2=tr[:, 2], Troe_T3=tr[:, 3])
    np.savez(OUT / f"{mech}.npz", **data)
    print(f"{mech}: {ns} species, {nr} reactions, {len(low)} falloff ({len(troe)} Troe) -> {OUT}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["h2o2.yaml"]:
        convert(m)
