"""
Biodiesel production (transesterification) -- trajectory generator.

One of the two ChemKAN reproduction datasets. Generates trajectories for a
three-reaction kinetic model (ChemKAN paper Eqs. 19-21):

    TG + ROH --k1--> DG + R'CO2R
    DG + ROH --k2--> MG + R'CO2R
    MG + ROH --k3--> GL + R'CO2R

with Arrhenius rates k_i = A_i exp(-Ea_i / RT) and second-order mass-action
kinetics. The system is isothermal, so temperature is a per-case constant
input rather than a state that evolves.

Paper setup reproduced here:
    Ea      = [14.54, 6.47, 14.42] kcal/mol
    ln(A)   = [18.60, 7.93, 19.13]
    T       ~ U(323, 343) K, isothermal per experiment
    [TG]_0, [ROH]_0 ~ U(0.5, 2.0); DG, MG, GL, R'CO2R start at 0
    20 training sets, 10 testing sets
    30 s window, 30 sampled points

Usage
-----
    python generate_biodiesel.py --out ../../data/generated/biodiesel.npz --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from common import add_noise, fit_minmax, make_rng, metadata, save

# --------------------------------------------------------------------------
# Model definition
# --------------------------------------------------------------------------

SPECIES = ["TG", "ROH", "DG", "MG", "GL", "RCO2R"]  # R'CO2R -> RCO2R (npz-safe)

# ChemKAN reports Ea in kcal/mol, so R must use kcal/(mol*K) (not 8.314 J/(mol*K)).
EA_KCAL = np.array([14.54, 6.47, 14.42])  # kcal / mol
LN_A = np.array([18.60, 7.93, 19.13])  # ln(A), A in 1/(conc * s)
R_KCAL = 1.987204259e-3  # kcal / (mol K)


def rate_constants(T: float) -> np.ndarray:
    """k_i(T) = A_i exp(-Ea_i / RT). Computed in log space for stability."""
    return np.exp(LN_A - EA_KCAL / (R_KCAL * T))


def rhs(t: float, y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """du/dt = f(u) for u = [TG, ROH, DG, MG, GL, R'CO2R].

    Second-order mass action, one glyceride and one methanol per step. This is
    the reading consistent with the stoichiometry as written; the paper does
    not spell out the ODE system itself.
    """
    # The paper gives the reactions but not the explicit ODEs; we implement
    # irreversible second-order mass action (one glyceride + one methanol per step).
    TG, ROH, DG, MG, _GL, _E = y
    r1 = k[0] * TG * ROH
    r2 = k[1] * DG * ROH
    r3 = k[2] * MG * ROH
    return np.array(
        [
            -r1,  # TG
            -(r1 + r2 + r3),  # ROH
            r1 - r2,  # DG
            r2 - r3,  # MG
            r3,  # GL
            r1 + r2 + r3,  # R'CO2R (methyl ester)
        ]
    )


def jac(t: float, y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Analytic Jacobian -- lets the implicit solver take clean steps."""
    TG, ROH, DG, MG, _GL, _E = y
    J = np.zeros((6, 6))
    dr1 = np.array([k[0] * ROH, k[0] * TG, 0.0, 0.0, 0.0, 0.0])
    dr2 = np.array([0.0, k[1] * DG, k[1] * ROH, 0.0, 0.0, 0.0])
    dr3 = np.array([0.0, k[2] * MG, 0.0, k[2] * ROH, 0.0, 0.0])
    J[0] = -dr1
    J[1] = -(dr1 + dr2 + dr3)
    J[2] = dr1 - dr2
    J[3] = dr2 - dr3
    J[4] = dr3
    J[5] = dr1 + dr2 + dr3
    return J


# --------------------------------------------------------------------------
# Trajectory generation
# --------------------------------------------------------------------------


def integrate_case(y0: np.ndarray, T: float, t_eval: np.ndarray) -> np.ndarray:
    k = rate_constants(T)
    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="LSODA",
        jac=jac,
        args=(k,),
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed at T={T:.2f}: {sol.message}")
    return sol.y.T  # (n_times, 6)


def sample_initial_conditions(n: int, rng: np.random.Generator):
    """[TG]_0, [ROH]_0 ~ U(0.5, 2); T ~ U(323, 343) K; others zero."""
    # Biodiesel is isothermal here; T is sampled per trajectory and can be
    # appended later as a constant model input (it is not integrated).
    tg0 = rng.uniform(0.5, 2.0, size=n)
    roh0 = rng.uniform(0.5, 2.0, size=n)
    temps = rng.uniform(323.0, 343.0, size=n)
    y0 = np.zeros((n, len(SPECIES)))
    y0[:, 0] = tg0
    y0[:, 1] = roh0
    return y0, temps


def generate(cfg) -> dict:
    rng = make_rng(cfg.seed)

    # 30 points spanning the 30 s window => ~1 s spacing. See the data-gen README.
    t = np.linspace(0.0, cfg.t_end, cfg.n_points)

    y0_tr, T_tr = sample_initial_conditions(cfg.n_train, rng)
    y0_te, T_te = sample_initial_conditions(cfg.n_test, rng)

    train = np.stack([integrate_case(y0_tr[i], T_tr[i], t) for i in range(cfg.n_train)])
    test = np.stack([integrate_case(y0_te[i], T_te[i], t) for i in range(cfg.n_test)])

    # Non-negativity: LSODA can undershoot by ~1e-14 near depletion.
    train = np.clip(train, 0.0, None)
    test = np.clip(test, 0.0, None)

    # Train-only normalization: scaler fitted on the clean TRAINING set only.
    u_min, u_max = fit_minmax(train)

    out = {
        "t": t,
        "species": np.array(SPECIES),
        "mechanism": np.array("biodiesel_transesterification"),
        # Isothermal: T is a per-case input, not an integrated state. Use
        # common.with_temperature(states, T) to build the ChemKAN input.
        "state_layout": np.array("species_only"),
        "train_states": train,
        "test_states": test,
        "train_T": T_tr,
        "test_T": T_te,
        "u_min": u_min,
        "u_max": u_max,
        "true_Ea_kcal": EA_KCAL,
        "true_lnA": LN_A,
    }

    # Noise is applied after clean trajectory generation; the initial condition
    # stays clean. An independent, seeded stream per level keeps runs reproducible
    # and levels comparable.
    for lvl in cfg.noise_levels:
        tag = f"{int(round(lvl * 100)):02d}"
        rng_tr = make_rng(cfg.seed + 1000 + int(round(lvl * 1000)))
        rng_te = make_rng(cfg.seed + 2000 + int(round(lvl * 1000)))
        out[f"train_states_noise{tag}"] = add_noise(train, lvl, rng_tr, cfg.noise_mode,
                                                    u_min, u_max)
        out[f"test_states_noise{tag}"] = add_noise(test, lvl, rng_te, cfg.noise_mode,
                                                   u_min, u_max)

    out["noise_levels"] = np.array(cfg.noise_levels)
    out["metadata"] = np.array(metadata(
        system="biodiesel",
        generator="generate_biodiesel.py",
        seed=cfg.seed,
        mechanism="biodiesel_transesterification",
        species=SPECIES,
        n_points=cfg.n_points,
        t_end_s=cfg.t_end,
        normalization="train-only min-max (Eq. 18)",
        noise_mode=cfg.noise_mode,
    ))
    return out


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/generated/biodiesel.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--t-end", type=float, default=30.0, help="seconds")
    p.add_argument("--n-points", type=int, default=30)
    p.add_argument("--noise-levels", type=float, nargs="*",
                   default=[0.0, 0.01, 0.02, 0.05, 0.07, 0.10, 0.15])
    p.add_argument("--noise-mode", choices=["multiplicative", "range"],
                   default="multiplicative")
    cfg = p.parse_args()

    print(f"Biodiesel: {cfg.n_train} train + {cfg.n_test} test cases, "
          f"{cfg.n_points} points over {cfg.t_end} s")
    data = generate(cfg)
    save(cfg.out, **data)

    tr = data["train_states"]
    print(f"  state shape (cases, times, species): {tr.shape}  [species only]")
    print(f"  T stored separately in train_T/test_T "
          f"({data['train_T'].min():.1f}-{data['train_T'].max():.1f} K); "
          f"use common.with_temperature() to build model inputs")
    for i, s in enumerate(SPECIES):
        print(f"    {s:>6}: [{tr[..., i].min():.4f}, {tr[..., i].max():.4f}]")


if __name__ == "__main__":
    raise SystemExit(main())
