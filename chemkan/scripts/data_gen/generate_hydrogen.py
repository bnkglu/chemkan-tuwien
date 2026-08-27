"""
Hydrogen-air combustion -- trajectory generator (Cantera).

One of the two ChemKAN reproduction datasets. Generates trajectories from
adiabatic, constant-pressure homogeneous (0-D) reactors using the H2/O2
submechanism of GRI-Mech 3.0. Cantera ships this as `h2o2.yaml`; dropping Ar
(inert, zero mole fraction in air) leaves the 9 species / 29 reactions quoted
in the paper:

    H2, H, O, O2, OH, H2O, HO2, H2O2, N2

Grids
-----
coarse (train/test): T0  in {950, 1000, 1050, 1100, 1150, 1200} K   (6 values)
                     phi in {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}          (6 values)
                     -> 36 cases, with (1150 K, 1.3) held out as the test case
                        => 35 train / 1 test, matching the paper's counts.

    Note on phi: ChemKAN reports 36 hydrogen cases and 35 training cases (after
    withholding one case), but the printed list of equivalence ratios contains
    only five values (0.7-1.5). To make the reported case counts work out, this
    implementation adds phi = 0.5 to form a 6 x 6 grid.

fine (generalization): 21 x 21 = 441 cases on the same ranges
                     (T0 step 12.5 K, phi step 0.05). 406 of these are unseen
                     during training, reproducing Fig. 8(A).

State vector: u = [Y_H2, ..., Y_N2, T]  (mass fractions, then temperature).

Time sampling: uniform samples over [0, 0.6 ms]; default 50 follows ChemNODE-style
saved trajectories. Use --n-points 601 for 1 us resolution.

Usage
-----
    python generate_hydrogen.py --out ../../data/generated/hydrogen.npz
    python generate_hydrogen.py --out ../../data/generated/hydrogen_fine.npz --grid fine
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cantera as ct
import numpy as np

from common import fit_minmax, ignition_delay, metadata, save, stiffness_ratio
from reactor import integrate_case, species_index

MECH = "h2o2.yaml"
DROP = ("AR",)  # drop AR to match the 9-species H2/O2 setup (inert, absent from air)
FUEL = "H2"
OXIDIZER = {"O2": 1.0, "N2": 3.76}  # standard air

# Coarse grid from the paper
T0_COARSE = [950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0]
# ChemKAN's prose lists five phi values, but its reported 36 cases and Fig. 8
# imply a six-value grid starting at 0.5 (the omitted value).
PHI_COARSE = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
TEST_IC = (1150.0, 1.3)  # withheld

# Fine grid for the generalization study (Fig. 8A)
N_FINE = 21


def build_grid(kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "coarse":
        return np.array(T0_COARSE), np.array(PHI_COARSE)
    if kind == "fine":
        return (np.linspace(950.0, 1200.0, N_FINE),   # step 12.5 K
                np.linspace(0.5, 1.5, N_FINE))        # step 0.05
    raise ValueError(kind)


def generate(cfg) -> dict:
    names, keep = species_index(MECH, DROP)
    print(f"  mechanism {MECH}: {len(names)} species, "
          f"{ct.Solution(MECH).n_reactions} reactions")
    print(f"  species: {names}")

    # Saved trajectories use 50 points by default; ignition delay uses a denser diagnostic grid.
    t = np.linspace(0.0, cfg.t_end, cfg.n_points)
    T0s, phis = build_grid(cfg.grid)
    if cfg.phis:
        phis = np.array(cfg.phis)

    if len(phis) == 6 and abs(phis[0] - 0.5) < 1e-9:
        print("  Note: using 6 equivalence ratios including phi=0.5; the prose lists 5.\n"
              "        This matches the reported 36 cases and Fig. 8 range.")

    ics, states = [], []
    for T0 in T0s:
        for phi in phis:
            states.append(integrate_case(MECH, FUEL, OXIDIZER, T0, phi, t,
                                         cfg.pressure, keep, cfg.rtol, cfg.atol))
            ics.append((T0, phi))
    states = np.stack(states)  # (n_cases, n_times, n_species + 1)
    ics = np.array(ics)

    # Train/test split: only the coarse grid carries the paper's 35/1 split.
    if cfg.grid == "coarse":
        is_test = np.all(np.isclose(ics, np.array(TEST_IC)), axis=1)
        if not is_test.any():
            print("  WARNING: held-out IC (1150 K, phi=1.3) not on this grid")
    else:
        # "Seen" = the 35 coarse points actually trained on. Everything else --
        # the 405 intermediate points plus the withheld (1150 K, 1.3) -- is
        # unseen, giving the 406 test conditions of Fig. 8(A). The scaler below
        # is therefore fitted on exactly the same 35 cases as the coarse file.
        seen = {(T0, phi) for T0 in T0_COARSE for phi in PHI_COARSE
                if not (abs(T0 - TEST_IC[0]) < 1e-6 and abs(phi - TEST_IC[1]) < 1e-6)}
        is_test = np.array([
            not any(abs(a - T0) < 1e-6 and abs(b - phi) < 1e-6 for T0, phi in seen)
            for a, b in ics
        ])

    train_states = states[~is_test]
    test_states = states[is_test]

    # Fit min/max on clean training data only to avoid test-set leakage.
    u_min, u_max = fit_minmax(train_states if len(train_states) else states)

    # Ignition delay is quantized by the sample spacing, so compute it on a denser
    # diagnostic grid than the saved trajectories. The saved states keep --n-points.
    t_ign = np.linspace(0.0, cfg.t_end, max(cfg.n_points, cfg.ignition_points))
    if len(t_ign) == len(t):
        tau_ign = np.array([ignition_delay(t, s[:, -1]) for s in states])
    else:
        tau_ign = np.array([
            ignition_delay(
                t_ign,
                integrate_case(MECH, FUEL, OXIDIZER, T0, phi, t_ign,
                               cfg.pressure, keep, cfg.rtol, cfg.atol)[:, -1])
            for T0, phi in ics
        ])
    n_ignited = int(np.isfinite(tau_ign).sum())

    print(f"  {len(states)} cases | {len(train_states)} train / {len(test_states)} test")
    print(f"  {n_ignited}/{len(states)} ignited within {cfg.t_end * 1e3:.2f} ms")
    print(f"  ignition delay computed on {len(t_ign)}-point diagnostic grid")
    print(f"  T range: {states[..., -1].min():.0f}-{states[..., -1].max():.0f} K")
    print(f"  stiffness proxy (worst case): {max(stiffness_ratio(t, s) for s in states):.1e}")

    return {
        "t": t,
        "species": np.array(names),
        "mechanism": np.array(MECH),
        "state_layout": np.array("species_then_temperature"),
        "states": states,
        "ics": ics,  # (n_cases, 2) = [T0, phi]
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
            system="hydrogen",
            generator="generate_hydrogen.py",
            seed=cfg.seed,
            mechanism=MECH,
            species=names,
            grid=cfg.grid,
            n_points=cfg.n_points,
            t_end_s=cfg.t_end,
            pressure_pa=cfg.pressure,
            normalization="train-only min-max (Eq. 18)",
            ignition_points=cfg.ignition_points,
            ignition_delay_grid="dense diagnostic grid; saved states use n_points",
        )),
    }


def generate_temperature_only(cfg) -> dict:
    """Dense TEMPERATURE-ONLY cache for the Stage-1 ObservedTemperature provider.

    Reuses the canonical hydrogen setup exactly -- same mechanism / fuel / oxidizer
    / pressure / tolerances / coarse IC grid / 35-1 split / time interval and, most
    importantly, the SAME initial-condition ORDERING as ``generate()``. Only the
    temperature column of each trajectory is kept; no dense species trajectories,
    normalization statistics, or ignition diagnostics are computed or saved.

    This exists so the dense Stage-1 temperature trajectory can be precomputed once
    (supervisor-approved approach) and reused across experiments, instead of calling
    Cantera inside the training loop. The original 50-point ``hydrogen.npz`` remains
    the canonical trajectory/target dataset and is never touched here.
    """
    if cfg.grid != "coarse":
        raise ValueError("--temperature-only requires the coarse grid (the paper's 35/1 split)")

    names, keep = species_index(MECH, DROP)
    t = np.linspace(0.0, cfg.t_end, cfg.n_points)
    T0s, phis = build_grid("coarse")
    if cfg.phis:
        phis = np.array(cfg.phis)

    # Identical loop/order to generate(): T0 outer, phi inner.
    ics, temps = [], []
    for T0 in T0s:
        for phi in phis:
            states = integrate_case(MECH, FUEL, OXIDIZER, T0, phi, t,
                                    cfg.pressure, keep, cfg.rtol, cfg.atol)
            temps.append(states[:, -1])                # temperature column only
            ics.append((T0, phi))
    temps = np.stack(temps)                            # (n_cases, N)
    ics = np.array(ics)                                # (n_cases, 2)

    is_test = np.all(np.isclose(ics, np.array(TEST_IC)), axis=1)
    if not is_test.any():
        raise ValueError("held-out IC (1150 K, phi=1.3) not on this grid")

    # Time-major (N, B, 1) -- the exact layout ObservedTemperature expects --
    # preserving canonical case ordering along B.
    train_T = temps[~is_test].T[:, :, None]            # (N, 35, 1)
    test_T = temps[is_test].T[:, :, None]              # (N,  1, 1)

    print(f"  temperature-only: {len(temps)} cases | "
          f"{train_T.shape[1]} train / {test_T.shape[1]} test | {cfg.n_points} points")
    print(f"  T range: {temps.min():.0f}-{temps.max():.0f} K")

    return {
        "t": t,                                        # (N,)
        "train_T": train_T,                            # (N, 35, 1)
        "test_T": test_T,                              # (N,  1, 1)
        "train_ics": ics[~is_test],                    # (35, 2) = [T0, phi]
        "test_ics": ics[is_test],                      # (1, 2)
        "n_points": np.array(cfg.n_points),
        "t_end": np.array(cfg.t_end),
        "mechanism": np.array(MECH),
        "pressure": np.array(cfg.pressure),
        "rtol": np.array(cfg.rtol),
        "atol": np.array(cfg.atol),
        "species": np.array(names),
        "state_layout": np.array("temperature_only"),
        "metadata": np.array(metadata(
            system="hydrogen-temperature-only",
            generator="generate_hydrogen.py --temperature-only",
            seed=cfg.seed,
            mechanism=MECH,
            species=names,
            grid=cfg.grid,
            n_points=cfg.n_points,
            t_end_s=cfg.t_end,
            pressure_pa=cfg.pressure,
            purpose="dense Stage-1 ObservedTemperature provider (species not saved)",
        )),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/generated/hydrogen.npz"))
    p.add_argument("--seed", type=int, default=0, help="unused; recorded in metadata")
    p.add_argument("--grid", choices=["coarse", "fine"], default="coarse")
    p.add_argument("--t-end", type=float, default=0.6e-3, help="seconds (0.6 ms)")
    p.add_argument("--n-points", type=int, default=50,
                   help="uniform samples over [0, 0.6 ms]; default 50 follows "
                        "ChemNODE-style saved trajectories. Use 601 for 1 us resolution.")
    p.add_argument("--ignition-points", type=int, default=601,
                   help="dense grid used only for ignition-delay diagnostics; "
                        "saved states still use --n-points")
    p.add_argument("--phis", type=float, nargs="*", default=None,
                   help="override the equivalence-ratio grid")
    p.add_argument("--pressure", type=float, default=ct.one_atm)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-15)
    p.add_argument("--temperature-only", action="store_true",
                   help="save ONLY the dense temperature trajectory for the Stage-1 "
                        "ObservedTemperature provider (no dense species / normalization / "
                        "ignition). Requires the coarse grid; use with --n-points 20000 and "
                        "a distinct --out (e.g. hydrogen_temperature_20000.npz).")
    cfg = p.parse_args()

    if cfg.temperature_only:
        if cfg.out.name == "hydrogen.npz":
            raise SystemExit(
                "refusing to overwrite hydrogen.npz in --temperature-only mode; pass a "
                "distinct --out (e.g. --out ../../data/generated/hydrogen_temperature_20000.npz)")
        print(f"Hydrogen-air TEMPERATURE-ONLY cache: coarse grid, {cfg.n_points} points over "
              f"{cfg.t_end * 1e3:.2f} ms at {cfg.pressure / ct.one_atm:.2f} atm")
        save(cfg.out, **generate_temperature_only(cfg))
        return

    print(f"Hydrogen-air: {cfg.grid} grid, {cfg.n_points} points over "
          f"{cfg.t_end * 1e3:.2f} ms at {cfg.pressure / ct.one_atm:.2f} atm")
    save(cfg.out, **generate(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
