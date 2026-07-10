"""
Methane ignition-window diagnostic.

Demonstrates why the H2-analogous temperature range (950-1200 K) and time
window (0.6 ms) fail for methane autoignition, and why the extension uses
1400-1650 K over 5 ms instead.

Three sweeps are run:

    1. H2-analogous conditions   : T0 in {950, 1050, 1150 K}, window = 0.6 ms
       Expected result           : NO ignition -- temperature barely rises.

    2. Intermediate temperatures : T0 in {1200, 1250, 1300, 1350 K}, window = 5 ms
       Expected result           : NO ignition -- ignition delay exceeds 5 ms.

    3. Methane-extension range   : T0 in {1400, 1500, 1600, 1650 K}, window = 5 ms
       Expected result           : ignition observed within the window.

Run from the extensions/ directory (or anywhere on a Cantera 3.x install):

    python methane_ignition_window.py

No output files are written; results are printed to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the parent data_gen/ directory so reactor.py and common.py are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cantera as ct
import numpy as np

from common import ignition_delay
from reactor import integrate_case, species_index

MECH     = "gri30.yaml"
FUEL     = "CH4"
OXIDIZER = {"O2": 1.0, "N2": 3.76}
PHI = 1.0  # stoichiometric reference condition
PRESSURE = ct.one_atm
N_POINTS = 1001         # dense enough to resolve the ignition event

# Species to drop from the state vector (AR is inert in CH4/air).
DROP = ("AR",)

# ---------------------------------------------------------------------------
# Sweep 1 -- H2-analogous conditions
# ---------------------------------------------------------------------------
SWEEP_LOW = {
    "label"    : "H2-analogous conditions  (T0 = 950-1150 K, window = 0.6 ms)",
    "T0s"      : [950.0, 1050.0, 1150.0],
    "t_end_ms" : 0.6,
}

# ---------------------------------------------------------------------------
# Sweep 2 -- intermediate temperatures
# ---------------------------------------------------------------------------
SWEEP_INTERMEDIATE = {
    "label"    : "Intermediate range       (T0 = 1200-1350 K, window = 5 ms)",
    "T0s"      : [1200.0, 1250.0, 1300.0, 1350.0],
    "t_end_ms" : 5.0,
}

# ---------------------------------------------------------------------------
# Sweep 3 -- methane-extension range
# ---------------------------------------------------------------------------
SWEEP_CH4 = {
    "label"    : "Methane-extension range  (T0 = 1400-1650 K, window = 5 ms)",
    "T0s"      : [1400.0, 1500.0, 1600.0, 1650.0],
    "t_end_ms" : 5.0,
}


def run_sweep(sweep: dict, names: list[str], keep: np.ndarray) -> list[tuple]:
    """Run one sweep and return a list of (T0, ignited) tuples."""
    t_end = sweep["t_end_ms"] * 1e-3           # ms -> s
    t     = np.linspace(0.0, t_end, N_POINTS)

    print(f"\n{'=' * 70}")
    print(f"  {sweep['label']}")
    print(f"  phi = {PHI}, pressure = 1 atm, N_points = {N_POINTS}")
    print(f"{'=' * 70}")
    print(f"  {'T0 (K)':>8}  {'T_final (K)':>12}  {'dT (K)':>10}  {'tau_ign (ms)':>14}  {'ignited?':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*8}")

    results = []
    for T0 in sweep["T0s"]:
        states  = integrate_case(MECH, FUEL, OXIDIZER, T0, PHI,
                                 t, PRESSURE, keep, rtol=1e-9, atol=1e-15)
        T_traj  = states[:, -1]          # last column is temperature
        T_final = T_traj[-1]
        dT      = T_final - T0
        # ignition_delay returns nan when the total temperature rise stays below
        # 100 K -- meaning the mixture did not ignite within the time window.
        tau     = ignition_delay(t, T_traj)
        ignited = np.isfinite(tau)

        tau_str = f"{tau * 1e3:>12.3f}" if ignited else f"{'> window':>12}"
        ign_str = "YES" if ignited else "NO"
        print(f"  {T0:>8.0f}  {T_final:>12.1f}  {dT:>10.1f}  {tau_str}  {ign_str:>8}")
        results.append((T0, ignited))

    print()
    return results


def main() -> None:
    print(f"\nMethane ignition-window diagnostic")
    print(f"Mechanism : {MECH}  (GRI-Mech 3.0)")

    names, keep = species_index(MECH, DROP)
    gas = ct.Solution(MECH)
    print(f"Species kept: {len(names)} (AR dropped), reactions: {gas.n_reactions}")

    r1 = run_sweep(SWEEP_LOW,          names, keep)
    r2 = run_sweep(SWEEP_INTERMEDIATE, names, keep)
    r3 = run_sweep(SWEEP_CH4,          names, keep)

    all_results = [
        (SWEEP_LOW,          r1),
        (SWEEP_INTERMEDIATE, r2),
        (SWEEP_CH4,          r3),
    ]

    print("Summary")
    print("-------")
    for sweep, results in all_results:
        for T0, ignited in results:
            status = "ignited" if ignited else "did not ignite"
            print(f"  T0 = {T0:.0f} K, window = {sweep['t_end_ms']:.1f} ms  ->  {status}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
