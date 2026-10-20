r"""Experiment-local dataset for the authors-repo-matched biodiesel run.

The arrays are the released example's OWN data (exported from Julia by
``export_julia_reference.jl``), so the PyTorch run trains on exactly what the Julia run
trained on. The recipe is re-derived independently here (scipy, tight tolerances) and
compared, so the dataset is documented rather than just copied:

    ICs      diesel_u0_samples.txt; TG, ROH = x*1.5 + 0.5; T = x*20 + 323 K; others 0
    split    rows 1-20 train, 21-30 test (the example's rows 31-40 duplicate 21-30)
    kinetics logA = [18.60, 19.13, 7.93], Ea = [14.54, 14.42, 6.47] kcal/mol for
             r1 = k1 TG ROH, r2 = k2 DG ROH, r3 = k3 MG ROH
    grid     [0, 30] s, 30 equally spaced points; zero noise; clipped at 0
    stats    per-species min/max over ALL experiments and times; T min/max over all ICs

    python build_dataset.py      # writes <EXP>/data/author_repo_match_biodiesel.npz
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _author_match import (EXP, N_TEST, N_TRAIN, ROOT, SPECIES,  # noqa: E402
                           load_julia_reference)

AUTHORS = ROOT.parent / "chemkan-authors"
LOG_A = np.array([18.60, 19.13, 7.93])
EA = np.array([14.54, 14.42, 6.47])
R_KCAL = 1.98720425864083e-3


def recipe() -> dict:
    """The example's data recipe, solved to rtol 1e-12 (independent of Julia)."""
    x = np.loadtxt(AUTHORS / "diesel_u0_samples.txt")
    u0 = np.zeros((x.shape[0], 7))
    u0[:, :2] = x[:, :2] * 1.5 + 0.5
    u0[:, 6] = x[:, 6] * 20.0 + 323.0
    t = np.linspace(0.0, 30.0, 30)

    def rhs(_t, y, k):
        r1, r2, r3 = k[0] * y[0] * y[1], k[1] * y[2] * y[1], k[2] * y[3] * y[1]
        return [-r1, -r1 - r2 - r3, r1 - r2, r2 - r3, r3, r1 + r2 + r3]

    data = np.stack([solve_ivp(rhs, (0, 30), u0[i, :6], t_eval=t, method="LSODA",
                               rtol=1e-12, atol=1e-14,
                               args=(np.exp(LOG_A - EA / (R_KCAL * u0[i, 6])),)).y
                     for i in range(len(u0))])
    data = np.clip(data, 0.0, None)
    return {"u0": u0, "t": t, "data": data,
            "ymin": data.min(axis=(0, 2)), "ymax": data.max(axis=(0, 2))}


def main() -> int:
    ref = load_julia_reference()
    n = N_TRAIN + N_TEST
    ode, norm = ref["ode_data"][:n], ref["normdata"][:n]
    u0n = ref["u0_norm"][:n]
    rec = recipe()

    # The duplicated block (rows 31-40) is identical at zero noise and adds no information.
    assert np.array_equal(ref["ode_data"][n:], ref["ode_data"][N_TRAIN:n])
    check = {
        "raw_data_max_abs_diff_julia_vs_recipe": float(np.abs(ode - rec["data"]).max()),
        "raw_data_max_rel_diff_to_species_range": float(
            (np.abs(ode - rec["data"]).max(axis=(0, 2)) / (rec["ymax"] - rec["ymin"])).max()),
        "ymin_max_abs_diff": float(np.abs(ref["ymin"] - rec["ymin"]).max()),
        "ymax_max_abs_diff": float(np.abs(ref["ymax"] - rec["ymax"]).max()),
        "ic_raw_max_abs_diff": float(np.abs(ref["u0_raw"][:n] - rec["u0"]).max()),
        "time_grid_max_abs_diff": float(np.abs(ref["t"] - rec["t"]).max()),
        "note": "Julia generated the data with AutoTsit5(Rosenbrock23) at library-default "
                "tolerances and stored it as Float32; the recipe uses LSODA at rtol 1e-12. "
                "Training uses the Julia arrays.",
    }
    commit = subprocess.run(["git", "-C", str(AUTHORS), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "source": "authors' released biodiesel example (DENG-MIT/ChemKAN), exported arrays",
        "authors_commit": commit,
        "split": {"train": "rows 1-20", "test": "rows 21-30"},
        "normalization": "per-species min/max over all 40 example experiments (train, test "
                         "and the duplicated test block) and all times; T min/max over all ICs",
        "noise": 0.0,
        "logA": LOG_A.tolist(), "Ea_kcal_per_mol": EA.tolist(),
        "recipe_check": check,
    }
    out = EXP / "data"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "author_repo_match_biodiesel.npz",
             t=ref["t"], species=np.array(SPECIES),
             u0_raw=ref["u0_raw"][:n], u0_norm=u0n,
             ode_data=ode, normdata=norm,
             ymin=ref["ymin"], ymax=ref["ymax"], T_minmax=ref["T_minmax"],
             n_train=N_TRAIN, n_test=N_TEST, metadata=np.array(json.dumps(meta)))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
