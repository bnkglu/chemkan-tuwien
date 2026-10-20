r"""Two versioned author-recipe biodiesel datasets that differ ONLY in the reaction 2/3 rates.

Same recipe as ``build_dataset.recipe()`` (released example, DENG-MIT/ChemKAN d7aa5ab):

    ICs      diesel_u0_samples.txt; TG, ROH = x*1.5 + 0.5; T = x*20 + 323 K; others 0
    split    rows 1-20 train, 21-30 test; rows 21-30 duplicated as the noise-free test block
    grid     [0, 30] s, 30 equally spaced points; noise 0; clipped at 0
    solver   scipy LSODA, rtol 1e-12, atol 1e-14
    stats    per-species min/max over all 40 trajectories and times; T min/max over all ICs

    code order   lnA [18.60, 19.13, 7.93], Ea [14.54, 14.42, 6.47]  (released script lines 61-62)
    paper order  lnA [18.60, 7.93, 19.13], Ea [14.54, 6.47, 14.42]  (paper Sec. II D 1)

Writes ``<EXP>/data/author_recipe_{code,paper}_order_v1.npz`` (the layout ``train_author_repo_match
--data-file`` reads), the Fig. 3 case (row 24) ground truth under both orders as JSON and PNG, and
prints the code-order dataset's difference from the Julia arrays as a sanity check.

    python build_reaction_order_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _author_match import EXP, N_TEST, N_TRAIN, SPECIES, load_julia_reference  # noqa: E402
from build_dataset import AUTHORS, R_KCAL  # noqa: E402

VERSION = "v1"
ORDERS = {
    "code": {"lnA": [18.60, 19.13, 7.93], "Ea": [14.54, 14.42, 6.47],
             "source": "released script ChemKAN_biodiesel_example.jl lines 61-62"},
    "paper": {"lnA": [18.60, 7.93, 19.13], "Ea": [14.54, 6.47, 14.42],
              "source": "paper Sec. II D 1"},
}
FIG3_ROW = 24                          # 1-based row of diesel_u0_samples.txt (a test row)
FIG3_YLIM = {"TG": 2.0, "ROH": 2.0, "DG": 0.6, "MG": 0.2, "GL": 0.2, "RCO2R": 2.0}
OUT_DIR = EXP / "reaction_order"


def initial_conditions() -> np.ndarray:
    x = np.loadtxt(AUTHORS / "diesel_u0_samples.txt")
    u0 = np.zeros((x.shape[0], 7))
    u0[:, :2] = x[:, :2] * 1.5 + 0.5
    u0[:, 6] = x[:, 6] * 20.0 + 323.0
    return u0


def solve(y0: np.ndarray, T: float, lnA, Ea, t_eval: np.ndarray) -> np.ndarray:
    """(6,) initial species -> (6, len(t_eval)), clipped at 0."""
    k = np.exp(np.asarray(lnA) - np.asarray(Ea) / (R_KCAL * T))

    def rhs(_t, y):
        r1, r2, r3 = k[0] * y[0] * y[1], k[1] * y[2] * y[1], k[2] * y[3] * y[1]
        return [-r1, -r1 - r2 - r3, r1 - r2, r2 - r3, r3, r1 + r2 + r3]

    sol = solve_ivp(rhs, (0.0, 30.0), y0, t_eval=t_eval, method="LSODA", rtol=1e-12, atol=1e-14)
    return np.clip(sol.y, 0.0, None)


def build(order: str, u0: np.ndarray, t: np.ndarray) -> dict:
    n = N_TRAIN + N_TEST
    kin = ORDERS[order]
    data = np.stack([solve(u0[i, :6], u0[i, 6], kin["lnA"], kin["Ea"], t) for i in range(n)])
    all40 = np.concatenate([data, data[N_TRAIN:n]])           # + duplicated noise-free test block
    ymin, ymax = all40.min(axis=(0, 2)), all40.max(axis=(0, 2))
    u040 = np.concatenate([u0[:n], u0[N_TRAIN:n]])
    T_min, T_max = u040[:, 6].min(), u040[:, 6].max()
    u0_norm = u0[:n].copy()
    u0_norm[:, :6] = (u0[:n, :6] - ymin) / (ymax - ymin)
    u0_norm[:, 6] = (u0[:n, 6] - T_min) / (T_max - T_min)
    normdata = (data - ymin[None, :, None]) / (ymax - ymin)[None, :, None]
    return {"t": t, "species": np.array(SPECIES), "u0_raw": u0[:n], "u0_norm": u0_norm,
            "ode_data": data, "normdata": normdata, "ymin": ymin, "ymax": ymax,
            "T_minmax": np.array([T_min, T_max]), "n_train": N_TRAIN, "n_test": N_TEST}


def main() -> int:
    ref = load_julia_reference()
    u0, t = initial_conditions(), np.linspace(0.0, 30.0, 30)
    n = N_TRAIN + N_TEST
    (EXP / "data").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = {}
    for order in ORDERS:
        d = built[order] = build(order, u0, t)
        meta = {"version": VERSION, "order": order, **ORDERS[order], "noise": 0.0,
                "recipe": "released example (d7aa5ab); see build_reaction_order_datasets.py",
                "normalization": "species min/max over all 40 trajectories (train, test, "
                                 "duplicated test block) and times; T min/max over all ICs",
                "solver": "scipy LSODA rtol 1e-12 atol 1e-14"}
        path = EXP / "data" / f"author_recipe_{order}_order_{VERSION}.npz"
        np.savez(path, **d, metadata=np.array(json.dumps(meta)))
        print(f"wrote {path.relative_to(EXP.parents[2])}")
        print(f"  {order} ymin {np.round(d['ymin'], 5).tolist()}")
        print(f"  {order} ymax {np.round(d['ymax'], 5).tolist()}  T {d['T_minmax'].round(3).tolist()}")

    diff = np.abs(built["code"]["ode_data"] - ref["ode_data"][:n])
    rng = built["code"]["ymax"] - built["code"]["ymin"]
    print(f"sanity (code order vs Julia arrays): max |diff| {diff.max():.3e}, "
          f"max |diff| / species range {(diff.max(axis=(0, 2)) / rng).max():.3e}")

    # Fig. 3 case: ground truth under both orders, dense and on the 30-point grid.
    i = FIG3_ROW - 1
    t_dense = np.linspace(0.0, 30.0, 301)
    case = {"row": FIG3_ROW, "TG0": u0[i, 0], "ROH0": u0[i, 1], "T_K": u0[i, 6],
            "t": t.tolist(), "t_dense": t_dense.tolist(), "orders": {}}
    for order, kin in ORDERS.items():
        dense = solve(u0[i, :6], u0[i, 6], kin["lnA"], kin["Ea"], t_dense)
        case["orders"][order] = {"lnA": kin["lnA"], "Ea": kin["Ea"],
                                 "grid": dict(zip(SPECIES, built[order]["ode_data"][i].tolist())),
                                 "dense": dict(zip(SPECIES, dense.tolist())),
                                 "peak": dict(zip(SPECIES, dense.max(axis=1).tolist())),
                                 "final": dict(zip(SPECIES, dense[:, -1].tolist()))}
    (OUT_DIR / "fig3_case_trajectories.json").write_text(json.dumps(case, indent=1) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
    for ax, s in zip(axes.flat, SPECIES):
        for order, style in (("paper", "-"), ("code", "--")):
            ax.plot(t_dense, case["orders"][order]["dense"][s], style, label=f"{order} order")
        ax.set(title=s, xlim=(0, 30), ylim=(0, FIG3_YLIM[s]), xlabel="t [s]")
        if case["orders"]["paper"]["peak"][s] > FIG3_YLIM[s] or \
                case["orders"]["code"]["peak"][s] > FIG3_YLIM[s]:
            ax.text(0.98, 0.95, "exceeds axis", ha="right", va="top", transform=ax.transAxes,
                    fontsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Fig. 3 case (row {FIG3_ROW}): TG0 {u0[i, 0]:.3f}, ROH0 {u0[i, 1]:.3f}, "
                 f"T {u0[i, 6]:.2f} K; ground truth under both reaction orders")
    fig.savefig(OUT_DIR / "fig3_case_trajectories.png", dpi=150)
    for order in ORDERS:
        print(f"fig3 {order} peaks", {s: round(v, 4) for s, v in case["orders"][order]["peak"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
