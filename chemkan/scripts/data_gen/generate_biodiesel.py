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

Defaults follow the authors' released example and clarification (current data,
written to a NEW versioned file, ``biodiesel_v2.npz``):

    --reaction-order released_code   ln(A) = [18.60, 19.13, 7.93], Ea = [14.54, 14.42, 6.47]
                                     (released script lines 61-62)
    --ic-source authors_file         chemkan/data/external/diesel_u0_samples.txt with the
                                     released scaling TG, ROH = x*1.5 + 0.5; DG..R'CO2R = 0;
                                     T = x*20 + 323 K; rows 1-20 train, 21-30 test
    --noise-mode additive_species_max  u + p * max_t(u_i) * N(0,1), clipped at 0 (email B5)

The legacy archive ``biodiesel_legacy.npz`` (formerly ``biodiesel.npz``) was generated with
``--reaction-order paper_text --ic-source sampled --noise-mode multiplicative``
(paper Sec. II D 1: ln(A) = [18.60, 7.93, 19.13], Ea = [14.54, 6.47, 14.42];
[TG]_0, [ROH]_0 ~ U(0.5, 2), T ~ U(323, 343) K from ``--seed``). It is never overwritten:
this script refuses an existing output file unless ``--force`` is given.

Both: 20 training + 10 testing sets, 30 s window, 30 sampled points; clean
trajectories (train_states / test_states -- test_states is the noise-free test copy);
per noise level p, noisy train and test observations from a fixed, recorded seed
(train: seed + 1000 + round(1000 p); test: seed + 2000 + round(1000 p)).

Normalization: train-only min-max. ``u_min``/``u_max`` are fitted on the clean
training trajectories; ``u_min_noiseXX``/``u_max_noiseXX`` on the noisy training
trajectories of that level (the data a model at that level trains on).

Usage
-----
    python generate_biodiesel.py                       # -> chemkan/data/generated/biodiesel_v2.npz
    python generate_biodiesel.py --reaction-order paper_text --ic-source sampled \\
        --noise-mode multiplicative --out <new file>     # legacy recipe
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from common import NOISE_MODES, add_noise, fit_minmax, make_rng, metadata, save

# --------------------------------------------------------------------------
# Model definition
# --------------------------------------------------------------------------

SPECIES = ["TG", "ROH", "DG", "MG", "GL", "RCO2R"]  # R'CO2R -> RCO2R (npz-safe)

# ChemKAN reports Ea in kcal/mol, so R must use kcal/(mol*K) (not 8.314 J/(mol*K)).
R_KCAL = 1.987204259e-3  # kcal / (mol K)
KINETICS = {
    # released example, ChemKAN_biodiesel_example.jl lines 61-62 (DENG-MIT/ChemKAN d7aa5ab)
    "released_code": {"ln_A": np.array([18.60, 19.13, 7.93]),
                      "Ea_kcal": np.array([14.54, 14.42, 6.47])},
    # paper Sec. II D 1 (the legacy archive's kinetics)
    "paper_text": {"ln_A": np.array([18.60, 7.93, 19.13]),
                   "Ea_kcal": np.array([14.54, 6.47, 14.42])},
}
# Legacy names (paper-text order), kept for existing imports.
EA_KCAL = KINETICS["paper_text"]["Ea_kcal"]
LN_A = KINETICS["paper_text"]["ln_A"]

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
AUTHORS_IC_FILE = DATA_ROOT / "external" / "diesel_u0_samples.txt"
DEFAULT_OUT = DATA_ROOT / "generated" / "biodiesel_v2.npz"
DEFAULT_NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]   # Fig. 5 levels


def rate_constants(T: float, reaction_order: str = "paper_text") -> np.ndarray:
    """k_i(T) = A_i exp(-Ea_i / RT). Computed in log space for stability."""
    kin = KINETICS[reaction_order]
    return np.exp(kin["ln_A"] - kin["Ea_kcal"] / (R_KCAL * T))


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


def integrate_case(y0: np.ndarray, T: float, t_eval: np.ndarray,
                   reaction_order: str = "paper_text") -> np.ndarray:
    k = rate_constants(T, reaction_order)
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
    """[TG]_0, [ROH]_0 ~ U(0.5, 2); T ~ U(323, 343) K; others zero (legacy ICs)."""
    # Biodiesel is isothermal here; T is sampled per trajectory and can be
    # appended later as a constant model input (it is not integrated).
    tg0 = rng.uniform(0.5, 2.0, size=n)
    roh0 = rng.uniform(0.5, 2.0, size=n)
    temps = rng.uniform(323.0, 343.0, size=n)
    y0 = np.zeros((n, len(SPECIES)))
    y0[:, 0] = tg0
    y0[:, 1] = roh0
    return y0, temps


def authors_initial_conditions(path: Path = AUTHORS_IC_FILE):
    """The released example's ICs: ``diesel_u0_samples.txt`` (30 rows of 7 values in [0, 1])
    with its scaling TG, ROH = x*1.5 + 0.5; DG..R'CO2R = 0; T = x*20 + 323 K.

    Returns (y0 (30, 6), temps (30,)); rows 1-20 are train, 21-30 test.
    """
    x = np.loadtxt(path)
    if x.shape != (30, len(SPECIES) + 1):
        raise ValueError(f"{path}: expected 30 x 7 samples, got {x.shape}")
    y0 = np.zeros((x.shape[0], len(SPECIES)))
    y0[:, :2] = x[:, :2] * 1.5 + 0.5
    return y0, x[:, len(SPECIES)] * 20.0 + 323.0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noise_seeds(seed: int, level: float) -> tuple[int, int]:
    """(train, test) RNG seeds of one noise level -- fixed by the level, recorded."""
    return seed + 1000 + int(round(level * 1000)), seed + 2000 + int(round(level * 1000))


def generate(cfg) -> dict:
    # 30 points spanning the 30 s window => ~1 s spacing. See the data-gen README.
    t = np.linspace(0.0, cfg.t_end, cfg.n_points)

    if cfg.ic_source == "authors_file":
        if (cfg.n_train, cfg.n_test) != (20, 10):
            raise ValueError("--ic-source authors_file fixes the split at 20 train + 10 test")
        y0, temps = authors_initial_conditions(cfg.ic_file)
        y0_tr, T_tr, y0_te, T_te = y0[:20], temps[:20], y0[20:30], temps[20:30]
        ic_record = {"ic_source": "authors_file", "ic_file": str(cfg.ic_file.name),
                     "ic_file_sha256": file_sha256(cfg.ic_file),
                     "ic_scaling": "TG, ROH = x*1.5 + 0.5; DG..RCO2R = 0; T = x*20 + 323 K",
                     "ic_split": "rows 1-20 train, 21-30 test"}
    else:
        rng = make_rng(cfg.seed)
        y0_tr, T_tr = sample_initial_conditions(cfg.n_train, rng)
        y0_te, T_te = sample_initial_conditions(cfg.n_test, rng)
        ic_record = {"ic_source": "sampled",
                     "ic_sampling": "TG0, ROH0 ~ U(0.5, 2), T ~ U(323, 343) K from --seed"}

    order = cfg.reaction_order
    train = np.stack([integrate_case(y0_tr[i], T_tr[i], t, order) for i in range(len(T_tr))])
    test = np.stack([integrate_case(y0_te[i], T_te[i], t, order) for i in range(len(T_te))])

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
        "test_states": test,                  # clean: the noise-free test copy
        "train_T": T_tr,
        "test_T": T_te,
        "u_min": u_min,
        "u_max": u_max,
        "true_Ea_kcal": KINETICS[order]["Ea_kcal"],
        "true_lnA": KINETICS[order]["ln_A"],
    }

    # Noise is applied after clean trajectory generation. An independent, seeded stream
    # per level keeps runs reproducible and levels comparable.
    seeds = {}
    for lvl in cfg.noise_levels:
        tag = f"{int(round(lvl * 100)):02d}"
        s_tr, s_te = noise_seeds(cfg.seed, lvl)
        seeds[tag] = {"train": s_tr, "test": s_te, "noise_mode": cfg.noise_mode}
        noisy_train = add_noise(train, lvl, make_rng(s_tr), cfg.noise_mode, u_min, u_max)
        out[f"train_states_noise{tag}"] = noisy_train
        out[f"test_states_noise{tag}"] = add_noise(test, lvl, make_rng(s_te), cfg.noise_mode,
                                                   u_min, u_max)
        if cfg.per_level_stats:
            out[f"u_min_noise{tag}"], out[f"u_max_noise{tag}"] = fit_minmax(noisy_train)

    out["noise_levels"] = np.array(cfg.noise_levels)
    out["noise_seeds"] = np.array([[seeds[f"{int(round(lvl * 100)):02d}"]["train"],
                                    seeds[f"{int(round(lvl * 100)):02d}"]["test"]]
                                   for lvl in cfg.noise_levels])
    out["metadata"] = np.array(metadata(
        system="biodiesel",
        generator="generate_biodiesel.py",
        seed=cfg.seed,
        mechanism="biodiesel_transesterification",
        species=SPECIES,
        n_points=cfg.n_points,
        t_end_s=cfg.t_end,
        reaction_order=order,
        ln_A=KINETICS[order]["ln_A"].tolist(),
        Ea_kcal=KINETICS[order]["Ea_kcal"].tolist(),
        **ic_record,
        normalization="train-only min-max (Eq. 18): u_min/u_max on the clean training "
                      "set" + ("; u_min_noiseXX/u_max_noiseXX on the noisy training set of "
                           "each level" if cfg.per_level_stats else ""),
        noise_mode=cfg.noise_mode,
        noise_levels=list(cfg.noise_levels),
        noise_seeds=seeds,
        noise_free_test="test_states",
    ))
    return out


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true", help="overwrite an existing --out file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reaction-order", choices=list(KINETICS), default="released_code")
    p.add_argument("--ic-source", choices=["authors_file", "sampled"], default="authors_file")
    p.add_argument("--ic-file", type=Path, default=AUTHORS_IC_FILE)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--t-end", type=float, default=30.0, help="seconds")
    p.add_argument("--n-points", type=int, default=30)
    p.add_argument("--noise-levels", type=float, nargs="*", default=DEFAULT_NOISE_LEVELS)
    p.add_argument("--noise-mode", choices=list(NOISE_MODES), default="additive_species_max")
    p.add_argument("--no-per-level-stats", dest="per_level_stats", action="store_false",
                   help="omit u_min_noiseXX/u_max_noiseXX (the legacy archive has none)")
    return p


def main():
    cfg = build_parser().parse_args()
    if cfg.out.exists() and not cfg.force:
        raise SystemExit(f"{cfg.out} exists; choose a new versioned --out (or pass --force)")

    print(f"Biodiesel: {cfg.n_train} train + {cfg.n_test} test cases, "
          f"{cfg.n_points} points over {cfg.t_end} s; reaction order {cfg.reaction_order}, "
          f"ICs {cfg.ic_source}, noise {cfg.noise_mode} {cfg.noise_levels}")
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
