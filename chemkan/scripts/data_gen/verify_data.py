"""
Sanity checks on generated data. Run this before training anything.

    python verify_data.py ../../data/generated/hydrogen.npz
    python verify_data.py ../../data/generated/biodiesel.npz --system biodiesel

Checks
------
combustion : mass fractions non-negative and summing to 1; element (H/O/N)
             conservation over each trajectory; non-negative temperature;
             normalized states land in [0, 1] on the training set.
biodiesel  : non-negativity; the two conservation laws implied by the
             mechanism, namely
                 TG + DG + MG + GL      = const   (glyceride backbone)
                 ROH_consumed           = R'CO2R produced

The Cantera mechanism used for element conservation is read from the archive's
`mechanism` key (stored by the generator), not guessed from the species count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import normalize

TOL = 1e-6


def report(name: str, value: float, tol: float = TOL) -> bool:
    ok = value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} (tol {tol:.0e})")
    return ok


def check_combustion(d: dict) -> bool:
    # Sanity checks for generated trajectories, not proof of model correctness.
    species = [str(s) for s in d["species"]]
    states = d["states"]
    Y, T = states[..., :-1], states[..., -1]
    ok = True

    ok &= report("max negative mass fraction", max(0.0, -Y.min()), 1e-10)
    ok &= report("max |sum(Y) - 1|", np.abs(Y.sum(-1) - 1.0).max(), 1e-8)
    ok &= report("max negative temperature", max(0.0, -T.min()), 0.0)

    # Element conservation: mass of each element is fixed in a closed reactor.
    # The mechanism is read from the archive rather than inferred.
    mech = str(d["mechanism"]) if "mechanism" in d else "h2o2.yaml"
    try:
        import cantera as ct
        gas = ct.Solution(mech)
        W = np.array([gas.molecular_weights[gas.species_index(s)] for s in species])
        for el in ("H", "O", "N"):
            n = np.array([gas.n_atoms(s, el) if el in gas.element_names else 0.0
                          for s in species])
            if not n.any():
                continue
            Z = (Y * (n * gas.atomic_weight(el) / W)).sum(-1)  # elemental mass fraction
            drift = np.abs(Z - Z[:, :1]).max()
            ok &= report(f"element {el} drift", drift, 1e-8)
    except ImportError:
        print("  [skip] element conservation (cantera not installed)")

    hat = normalize(d["train_states"], d["u_min"], d["u_max"])
    ok &= report("train normalized below 0", max(0.0, -hat.min()), 1e-12)
    ok &= report("train normalized above 1", max(0.0, hat.max() - 1.0), 1e-12)

    tau = d["ignition_delay"]
    print(f"  info: {int(np.isfinite(tau).sum())}/{len(tau)} cases ignited")
    return ok


def check_biodiesel(d: dict) -> bool:
    # These checks verify stoichiometric consistency, not the inferred rate order.
    train = d["train_states"]  # [TG, ROH, DG, MG, GL, RCO2R]
    test = d["test_states"]
    s = np.concatenate([train, test], axis=0)  # check all clean trajectories
    ok = True
    ok &= report("max negative concentration", max(0.0, -s.min()), 1e-10)

    backbone = s[..., [0, 2, 3, 4]].sum(-1)  # TG + DG + MG + GL
    ok &= report("glyceride backbone drift",
                 np.abs(backbone - backbone[:, :1]).max(), 1e-7)

    roh_used = s[:, :1, 1] - s[..., 1]
    ester = s[..., 5]
    ok &= report("ROH consumed vs ester produced", np.abs(roh_used - ester).max(), 1e-7)

    # Normalization check stays train-only (the scaler is fitted on train only).
    hat = normalize(train, d["u_min"], d["u_max"])
    ok &= report("train normalized below 0", max(0.0, -hat.min()), 1e-12)
    ok &= report("train normalized above 1", max(0.0, hat.max() - 1.0), 1e-12)

    # Noise arrays are train-sized, so compare them against the clean training set.
    for lvl in d["noise_levels"]:
        tag = f"{int(round(float(lvl) * 100)):02d}"
        key = f"train_states_noise{tag}"
        if key not in d:
            continue
        resid = d[key] - train
        # empirical relative sigma, ignoring the clean t=0 row
        mask = np.abs(train[:, 1:]) > 1e-3
        if mask.any():
            emp = np.std((resid[:, 1:][mask] / train[:, 1:][mask]))
            print(f"  info: noise {float(lvl):.0%} -> empirical sigma {emp:.3f}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--system", choices=["auto", "combustion", "biodiesel"], default="auto")
    a = p.parse_args()

    with np.load(a.path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}

    system = a.system
    if system == "auto":
        system = "biodiesel" if "noise_levels" in d else "combustion"

    print(f"{a.path.name}  [{system}]")
    if "mechanism" in d:
        print(f"  mechanism: {str(d['mechanism'])}")
    for k in ("train_states", "test_states"):
        if k in d:
            print(f"  {k}: {d[k].shape}")

    ok = check_biodiesel(d) if system == "biodiesel" else check_combustion(d)
    print("  ->", "all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
