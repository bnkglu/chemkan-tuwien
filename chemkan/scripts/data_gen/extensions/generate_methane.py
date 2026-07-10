"""
Methane-air oxidation -- trajectory generator (Cantera). OPTIONAL EXTENSION.

Methane is optional and is not part of the original ChemKAN reproduction. It is
included as a possible harder extension dataset using the same style of 0-D
constant-pressure reactor generation. The setup mirrors the hydrogen generator
where possible, but methane is a larger and different chemical system, not a
controlled one-variable comparison. It does not run as part of generate_all.py;
run this script directly if you want it.

Setup:

    mechanism : full GRI-Mech 3.0 (53 species, 325 reactions)
    T0        : 1400-1650 K -- methane did not ignite within the short
                hydrogen-style time window in preliminary tests, so this
                extension uses a higher temperature range
    phi       : 0.5-1.5, same grid as hydrogen
    window    : 5 ms -- ignition delays here span roughly 0.3-4.2 ms
    held out  : (1550 K, phi = 1.3)

Notes:

1. Argon (AR) is a species in GRI-Mech 3.0 but is a constant-zero, inert
   species in this CH4/(O2 + N2) setup, so it is dropped from the stored state
   vector (DROP below). This leaves 52 species. Set DROP = () to keep it.

2. Full GRI-Mech includes NOx chemistry, so unlike the reduced H2/O2 setup, N2
   is kept in the methane state vector.

3. 52 species is a much wider state than hydrogen's 9. A reduced methane
   mechanism such as DRM19 could be tested later if full GRI-Mech is too large
   -- pass `--mech drm19.yaml` once the file is on Cantera's data path.

Usage
-----
    python generate_methane.py --out ../../../data/generated/methane.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives in scripts/data_gen/extensions/; add the parent data_gen/
# dir to the import path so it can reuse common.py and reactor.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cantera as ct  # noqa: E402
import numpy as np  # noqa: E402

from common import (  # noqa: E402
    fit_minmax, ignition_delay, metadata, save, stiffness_ratio,
)
from reactor import integrate_case, species_index  # noqa: E402

FUEL = "CH4"
OXIDIZER = {"O2": 1.0, "N2": 3.76}
DROP = ("AR",)  # constant-zero inert in CH4/air -> dropped from the state vector

T0_GRID = [1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1650.0]
PHI_GRID = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
TEST_IC = (1550.0, 1.3)


def generate(cfg) -> dict:
    names, keep = species_index(cfg.mech, DROP)
    gas = ct.Solution(cfg.mech)
    print(f"  mechanism {cfg.mech}: {len(names)} species kept "
          f"(AR dropped), {gas.n_reactions} reactions")

    t = np.linspace(0.0, cfg.t_end, cfg.n_points)
    T0s = np.array(cfg.T0s) if cfg.T0s else np.array(T0_GRID)
    phis = np.array(cfg.phis) if cfg.phis else np.array(PHI_GRID)

    ics, states = [], []
    for T0 in T0s:
        for phi in phis:
            states.append(integrate_case(cfg.mech, FUEL, OXIDIZER, T0, phi, t,
                                         cfg.pressure, keep, cfg.rtol, cfg.atol))
            ics.append((T0, phi))
    states = np.stack(states)
    ics = np.array(ics)

    is_test = np.all(np.isclose(ics, np.array(TEST_IC)), axis=1)
    if not is_test.any():
        print("  WARNING: held-out IC not on this grid; all cases marked train")

    train_states, test_states = states[~is_test], states[is_test]
    # Train-only normalization.
    u_min, u_max = fit_minmax(train_states)

    # Ignition delay on a denser diagnostic grid so it is not quantized by the
    # saved sample spacing; saved states keep --n-points.
    t_ign = np.linspace(0.0, cfg.t_end, max(cfg.n_points, cfg.ignition_points))
    if len(t_ign) == len(t):
        tau_ign = np.array([ignition_delay(t, s[:, -1]) for s in states])
    else:
        tau_ign = np.array([
            ignition_delay(
                t_ign,
                integrate_case(cfg.mech, FUEL, OXIDIZER, T0, phi, t_ign,
                               cfg.pressure, keep, cfg.rtol, cfg.atol)[:, -1])
            for T0, phi in ics
        ])
    print(f"  {len(states)} cases | {len(train_states)} train / {len(test_states)} test")
    print(f"  {int(np.isfinite(tau_ign).sum())}/{len(states)} ignited within "
          f"{cfg.t_end * 1e3:.2f} ms")
    print(f"  ignition delay computed on {len(t_ign)}-point diagnostic grid")
    finite = tau_ign[np.isfinite(tau_ign)]
    if finite.size:
        print(f"  ignition delay: {finite.min() * 1e3:.3f}-{finite.max() * 1e3:.3f} ms")
    print(f"  T range: {states[..., -1].min():.0f}-{states[..., -1].max():.0f} K")
    print(f"  stiffness proxy (worst case): {max(stiffness_ratio(t, s) for s in states):.1e}")

    return {
        "t": t,
        "species": np.array(names),
        "mechanism": np.array(cfg.mech),
        "state_layout": np.array("species_then_temperature"),
        "states": states,
        "ics": ics,
        "is_test": is_test,
        "train_states": train_states,
        "test_states": test_states,
        "train_ics": ics[~is_test],
        "test_ics": ics[is_test],
        "u_min": u_min,
        "u_max": u_max,
        "ignition_delay": tau_ign,
        "pressure": np.array(cfg.pressure),
        "metadata": np.array(metadata(
            system="methane (optional extension)",
            generator="extensions/generate_methane.py",
            seed=cfg.seed,
            mechanism=cfg.mech,
            species=names,
            n_points=cfg.n_points,
            t_end_s=cfg.t_end,
            pressure_pa=cfg.pressure,
            normalization="train-only min-max (Eq. 18)",
            ignition_points=cfg.ignition_points,
            ignition_delay_grid="dense diagnostic grid; saved states use n_points",
        )),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/generated/methane.npz"))
    p.add_argument("--seed", type=int, default=0, help="unused; recorded for provenance")
    p.add_argument("--mech", default="gri30.yaml")
    p.add_argument("--t-end", type=float, default=5e-3, help="seconds")
    p.add_argument("--n-points", type=int, default=1001, help="uniform samples => 5 us")
    p.add_argument("--ignition-points", type=int, default=601,
                   help="dense grid used only for ignition-delay diagnostics; "
                        "saved states still use --n-points")
    p.add_argument("--T0s", type=float, nargs="*", default=None)
    p.add_argument("--phis", type=float, nargs="*", default=None)
    p.add_argument("--pressure", type=float, default=ct.one_atm)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-15)
    cfg = p.parse_args()

    print(f"Methane-air [optional extension]: {cfg.n_points} points over "
          f"{cfg.t_end * 1e3:.2f} ms at {cfg.pressure / ct.one_atm:.2f} atm")
    save(cfg.out, **generate(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
